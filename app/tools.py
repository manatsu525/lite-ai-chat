"""模型无关的外部工具：web_search + scrape_url。"""
import asyncio
import json
import logging
import re
from html import unescape
from html.parser import HTMLParser
from typing import Any, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from .config import (
    HTTP_TIMEOUT,
    MAX_SEARCH_RESULTS,
    SCRAPER_URL,
    SEARXNG_URL,
)

logger = logging.getLogger("lite-ai-chat.tools")

# 单源超时（秒）— 远小于串行 30s*N
_SEARCH_TIMEOUT = min(float(HTTP_TIMEOUT), 8.0)
_CONNECT_TIMEOUT = 3.0
_SCRAPER_EXT_TIMEOUT = 3.0
_MAX_SEARCH_SNIPPET_CHARS = 500
_MAX_SCRAPE_CHARS = 8000
_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _timeout(total: float = None) -> httpx.Timeout:
    t = total if total is not None else _SEARCH_TIMEOUT
    return httpx.Timeout(t, connect=_CONNECT_TIMEOUT)


def _trim_search_results(results: list, limit: int) -> list:
    """统一限制搜索卡片长度，避免提供商返回异常长摘要。"""
    trimmed = []
    for item in (results or [])[:limit]:
        if not isinstance(item, dict):
            continue
        value = dict(item)
        value["title"] = str(value.get("title") or "")[:300]
        value["url"] = str(value.get("url") or "")[:2000]
        value["snippet"] = str(value.get("snippet") or "")[
            :_MAX_SEARCH_SNIPPET_CHARS
        ]
        trimmed.append(value)
    return trimmed


# OpenAI function calling 工具定义
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "搜索互联网，返回相关网页、链接和检索摘要。适合查找最新信息、事实核实、资料检索。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词或问题",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": f"返回结果数量，默认 {MAX_SEARCH_RESULTS}，最大 10",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scrape_url",
            "description": "抓取指定 URL 的网页正文，转为 markdown。适合阅读搜索结果中的具体页面。",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "要抓取的完整 URL",
                    },
                },
                "required": ["url"],
            },
        },
    },
]


class _HTMLToText(HTMLParser):
    """极简 HTML → 近似 markdown，零额外依赖。"""

    SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "header"}

    def __init__(self):
        super().__init__()
        self._parts = []
        self._skip = 0
        self._href = None

    def handle_starttag(self, tag, attrs):
        t = tag.lower()
        if t in self.SKIP:
            self._skip += 1
            return
        if self._skip:
            return
        if t in ("p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4"):
            self._parts.append("\n")
        if t in ("h1", "h2", "h3", "h4"):
            self._parts.append("\n" + "#" * int(t[1]) + " ")
        if t == "li":
            self._parts.append("- ")
        if t == "a":
            self._href = dict(attrs).get("href")
            self._parts.append("[")
        if t in ("b", "strong"):
            self._parts.append("**")
        if t in ("i", "em"):
            self._parts.append("*")

    def handle_endtag(self, tag):
        t = tag.lower()
        if t in self.SKIP:
            self._skip = max(0, self._skip - 1)
            return
        if self._skip:
            return
        if t == "a":
            href = self._href or ""
            self._parts.append(f"]({href})")
            self._href = None
        if t in ("b", "strong"):
            self._parts.append("**")
        if t in ("i", "em"):
            self._parts.append("*")
        if t in ("p", "div", "li", "h1", "h2", "h3", "h4"):
            self._parts.append("\n")

    def handle_data(self, data):
        if self._skip:
            return
        text = data.strip()
        if text:
            self._parts.append(text + " ")

    def get_text(self) -> str:
        raw = "".join(self._parts)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        raw = re.sub(r"[ \t]{2,}", " ", raw)
        return raw.strip()


def _strip_tags(s: str) -> str:
    s = unescape(re.sub(r"<[^>]+>", "", s or ""))
    return re.sub(r"\s+", " ", s).strip()


def _unwrap_ddg_url(href: str) -> str:
    href = (href or "").replace("&amp;", "&")
    if href.startswith("//"):
        href = "https:" + href
    if "uddg=" in href:
        qs = parse_qs(urlparse(href).query)
        if qs.get("uddg"):
            return unquote(qs["uddg"][0])
    return href


# ---------- 各搜索源（均为模型之外的独立服务） ----------


_WEATHER_RE = re.compile(
    r"(天气|气温|温度|体感|降雨|下雨|下雪|风速|weather|temperature|forecast|rain|snow)",
    re.I,
)

_WMO_WEATHER = {
    0: "晴",
    1: "大致晴朗",
    2: "局部多云",
    3: "阴天",
    45: "雾",
    48: "雾凇",
    51: "小毛毛雨",
    53: "中等毛毛雨",
    55: "强毛毛雨",
    56: "轻微冻毛毛雨",
    57: "强冻毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    66: "轻微冻雨",
    67: "强冻雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    77: "米雪",
    80: "小阵雨",
    81: "中阵雨",
    82: "强阵雨",
    85: "小阵雪",
    86: "强阵雪",
    95: "雷暴",
    96: "雷暴伴小冰雹",
    99: "雷暴伴强冰雹",
}


def _weather_location(query: str) -> str:
    """从常见中英文天气问题里提取地点；非天气问题返回空。"""
    text = (query or "").strip()
    if not text or not _WEATHER_RE.search(text):
        return ""

    english = re.search(
        r"(?:weather|temperature|forecast)\s+(?:for|in|at)\s+"
        r"(.+?)(?:\s+(?:today|tomorrow|now|this week))?$",
        text,
        re.I,
    )
    if english:
        return english.group(1).strip(" ?.,")

    cleaned = re.sub(
        r"(请|帮我|查询|查一下|看看|告诉我|今天|今日|现在|当前|实时|"
        r"明天|后天|未来[一二三四五六七八九十\d]*天|本周|这周|"
        r"天气预报|天气|气温|温度|体感|降雨|下雨|下雪|风速|"
        r"weather|temperature|forecast|today|tomorrow|now|"
        r"怎么样|怎样|如何|多少|会不会|是否|会|吗|呢|的)",
        " ",
        text,
        flags=re.I,
    )
    cleaned = re.sub(r"[\s，。！？、,.!?;；:：]+", " ", cleaned).strip()
    return cleaned[:80]


async def _open_meteo_weather(query: str, n: int) -> list:
    """天气查询走结构化实时 API，避免依赖搜索引擎摘要。"""
    location = _weather_location(query)
    if not location:
        return []

    timeout = _timeout(7)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        geo = await client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={
                "name": location,
                "count": 1,
                # 英文地名用英文检索，避免 "New York" 在中文结果中
                # 被模糊匹配为 Nebraska 的 York。
                "language": "zh" if re.search(r"[\u4e00-\u9fff]", location) else "en",
                "format": "json",
            },
            headers={"User-Agent": "LiteAIChat/1.0"},
        )
        if geo.status_code != 200:
            return []
        places = (geo.json() or {}).get("results") or []
        if not places:
            return []
        place = places[0]
        latitude = place.get("latitude")
        longitude = place.get("longitude")
        if latitude is None or longitude is None:
            return []

        forecast = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": (
                    "temperature_2m,apparent_temperature,relative_humidity_2m,"
                    "precipitation,weather_code,wind_speed_10m"
                ),
                "daily": (
                    "weather_code,temperature_2m_max,temperature_2m_min,"
                    "precipitation_probability_max"
                ),
                "timezone": "auto",
                "forecast_days": 3,
            },
            headers={"User-Agent": "LiteAIChat/1.0"},
        )
        if forecast.status_code != 200:
            return []
        weather = forecast.json() or {}

    current = weather.get("current") or {}
    daily = weather.get("daily") or {}
    name_parts = [
        place.get("name"),
        place.get("admin1"),
        place.get("country"),
    ]
    display_name = "，".join(str(x) for x in name_parts if x)
    condition = _WMO_WEATHER.get(current.get("weather_code"), "未知天气")
    snippet = (
        f"{display_name}当前（{current.get('time', '')}）：{condition}，"
        f"温度 {current.get('temperature_2m', '未知')}°C，"
        f"体感 {current.get('apparent_temperature', '未知')}°C，"
        f"湿度 {current.get('relative_humidity_2m', '未知')}%，"
        f"降水 {current.get('precipitation', '未知')} mm，"
        f"风速 {current.get('wind_speed_10m', '未知')} km/h。"
    )
    days = []
    dates = daily.get("time") or []
    highs = daily.get("temperature_2m_max") or []
    lows = daily.get("temperature_2m_min") or []
    codes = daily.get("weather_code") or []
    rain_prob = daily.get("precipitation_probability_max") or []
    for i, date in enumerate(dates[:3]):
        days.append(
            {
                "date": date,
                "condition": _WMO_WEATHER.get(codes[i], "未知")
                if i < len(codes)
                else "未知",
                "max_c": highs[i] if i < len(highs) else None,
                "min_c": lows[i] if i < len(lows) else None,
                "precipitation_probability_max": rain_prob[i]
                if i < len(rain_prob)
                else None,
            }
        )
    return [
        {
            "title": f"{display_name}实时天气与未来 3 天预报",
            "url": (
                "https://open-meteo.com/en/docs"
                f"?latitude={latitude}&longitude={longitude}"
            ),
            "snippet": snippet,
            "current": current,
            "daily": days,
            "source": "Open-Meteo",
        }
    ][:n]


async def _searx_search(query: str, n: int) -> list:
    # 聚合多个引擎通常需要 5-8 秒，不能沿用单一搜索源的短超时。
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(12.0, connect=_CONNECT_TIMEOUT),
        follow_redirects=True,
    ) as client:
        r = await client.get(
            f"{SEARXNG_URL}/search",
            params={"q": query, "format": "json", "categories": "general"},
            headers={"Accept": "application/json"},
        )
        ct = r.headers.get("content-type", "")
        if r.status_code != 200 or "json" not in ct:
            return []
        data = r.json()
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            return []
        out = []
        for item in data["results"][:n]:
            url = item.get("url") or item.get("link") or ""
            if not url:
                continue
            out.append(
                {
                    "title": item.get("title") or "",
                    "url": url,
                    "snippet": item.get("content") or item.get("snippet") or "",
                    "engines": item.get("engines") or [],
                }
            )
        return out


async def _mojeek_search(query: str, n: int) -> list:
    async with httpx.AsyncClient(timeout=_timeout(), follow_redirects=True) as client:
        r = await client.get(
            "https://www.mojeek.com/search",
            params={"q": query},
            headers={"User-Agent": _UA, "Accept": "text/html"},
        )
        if r.status_code >= 400:
            return []
        html = r.text
    blocks = re.findall(
        r'<h2[^>]*>\s*<a[^>]+href="(https?://[^"]+)"[^>]*>(.*?)</a>\s*</h2>\s*<p class="s">(.*?)</p>',
        html,
        re.I | re.S,
    )
    if not blocks:
        blocks = [
            (u, t, "")
            for u, t in re.findall(
                r'<h2[^>]*>\s*<a[^>]+href="(https?://[^"]+)"[^>]*>(.*?)</a>',
                html,
                re.I | re.S,
            )
        ]
    out = []
    for url, title, snip in blocks:
        if "mojeek.com" in url:
            continue
        out.append({"title": _strip_tags(title), "url": url, "snippet": _strip_tags(snip)})
        if len(out) >= n:
            break
    return out


async def _wikipedia_search(query: str, n: int) -> list:
    out: List[dict] = []
    headers = {"User-Agent": "LiteAIChat/1.0"}
    async with httpx.AsyncClient(timeout=_timeout(6), follow_redirects=True) as client:
        for base in ("https://en.wikipedia.org", "https://zh.wikipedia.org"):
            try:
                r = await client.get(
                    f"{base}/w/api.php",
                    params={
                        "action": "opensearch",
                        "search": query,
                        "limit": n,
                        "namespace": 0,
                        "format": "json",
                    },
                    headers=headers,
                )
                if r.status_code != 200:
                    continue
                data = r.json()
                titles = data[1] if len(data) > 1 else []
                descs = data[2] if len(data) > 2 else []
                urls = data[3] if len(data) > 3 else []
                for i, title in enumerate(titles):
                    out.append(
                        {
                            "title": title,
                            "url": urls[i] if i < len(urls) else "",
                            "snippet": descs[i] if i < len(descs) else "",
                        }
                    )
                    if len(out) >= n:
                        return out
            except Exception:
                continue
    return out


async def _google_news_rss(query: str, n: int) -> list:
    headers = {"User-Agent": "LiteAIChat/1.0"}
    async with httpx.AsyncClient(timeout=_timeout(6), follow_redirects=True) as client:
        r = await client.get(
            "https://news.google.com/rss/search",
            params={"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"},
            headers=headers,
        )
        if r.status_code >= 400:
            r = await client.get(
                "https://news.google.com/rss/search",
                params={"q": query, "hl": "zh-CN", "gl": "CN", "ceid": "CN:zh-Hans"},
                headers=headers,
            )
        if r.status_code >= 400:
            return []
        xml = r.text

    def _rss_field(block: str, tag: str) -> str:
        m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", block, re.I | re.S)
        if not m:
            return ""
        return _strip_tags(m.group(1).replace("<![CDATA[", "").replace("]]>", ""))

    out = []
    for item in re.findall(r"<item>(.*?)</item>", xml, re.I | re.S):
        title = _rss_field(item, "title")
        if not title:
            continue
        out.append(
            {
                "title": title,
                "url": _rss_field(item, "link"),
                "snippet": _rss_field(item, "description")[:300],
            }
        )
        if len(out) >= n:
            break
    return out


async def _bing_rss_search(query: str, n: int) -> list:
    """Bing 的 RSS 搜索接口，作为无需额外密钥的通用备用源。"""
    async with httpx.AsyncClient(timeout=_timeout(7), follow_redirects=True) as client:
        r = await client.get(
            "https://www.bing.com/search",
            params={"q": query, "format": "rss"},
            headers={"User-Agent": _UA, "Accept": "application/rss+xml,application/xml"},
        )
        if r.status_code != 200:
            return []
        xml = r.text

    out = []
    for item in re.findall(r"<item>(.*?)</item>", xml, re.I | re.S):
        title_m = re.search(r"<title[^>]*>(.*?)</title>", item, re.I | re.S)
        link_m = re.search(r"<link[^>]*>(.*?)</link>", item, re.I | re.S)
        desc_m = re.search(r"<description[^>]*>(.*?)</description>", item, re.I | re.S)
        title = _strip_tags(title_m.group(1)) if title_m else ""
        url = _strip_tags(link_m.group(1)) if link_m else ""
        snippet = _strip_tags(desc_m.group(1)) if desc_m else ""
        if not title or not url.startswith(("http://", "https://")):
            continue
        out.append({"title": title, "url": url, "snippet": snippet[:500]})
        if len(out) >= n:
            break
    return out


async def _ddg_lite_search(query: str, n: int) -> list:
    """仅尝试一次 lite GET，避免多端点串行拖死。"""
    async with httpx.AsyncClient(timeout=_timeout(5), follow_redirects=True) as client:
        r = await client.get(
            "https://lite.duckduckgo.com/lite/",
            params={"q": query},
            headers={"User-Agent": _UA, "Accept": "text/html"},
        )
        if r.status_code not in (200, 202):
            return []
        html = r.text
    links = re.findall(
        r'<a[^>]+rel="nofollow"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        html,
        re.I | re.S,
    )
    out = []
    for href, title in links:
        url = _unwrap_ddg_url(href)
        if not url.startswith("http"):
            continue
        if "duckduckgo.com" in url:
            continue
        out.append({"title": _strip_tags(title), "url": url, "snippet": ""})
        if len(out) >= n:
            break
    return out


async def _run_backend(name: str, coro) -> Tuple[str, list]:
    try:
        res = await asyncio.wait_for(coro, timeout=_SEARCH_TIMEOUT + 1)
        return name, res or []
    except Exception as e:
        logger.info("search backend %s fail: %s", name, type(e).__name__)
        return name, []


async def web_search(query: str, num_results: int = None) -> str:
    """调用本机独立 SearXNG；仅在其无结果时使用外部备用源。"""
    n = num_results or MAX_SEARCH_RESULTS
    n = max(1, min(int(n), 10))
    if not (query or "").strip():
        return json.dumps({"error": "query 不能为空"}, ensure_ascii=False)

    # 实时天气属于结构化数据，优先使用稳定的天气 API。
    if _WEATHER_RE.search(query):
        try:
            weather_results = await asyncio.wait_for(
                _open_meteo_weather(query, n),
                timeout=_SEARCH_TIMEOUT + 1,
            )
            if weather_results:
                weather_results = _trim_search_results(weather_results, n)
                return json.dumps(
                    {
                        "query": query,
                        "results": weather_results,
                        "source": "open_meteo",
                    },
                    ensure_ascii=False,
                )
        except Exception as e:
            logger.info("weather backend fail: %s", type(e).__name__)

    # SearXNG 是独立于所选大模型的统一搜索层。每次工具调用只执行当前
    # 查询；是否改写查询并继续下一轮，由外层 agent 根据结果决定。
    try:
        searx_results = await asyncio.wait_for(
            _searx_search(query, n),
            timeout=13.0,
        )
        if searx_results:
            searx_results = _trim_search_results(searx_results, n)
            return json.dumps(
                {
                    "query": query,
                    "results": searx_results,
                    "source": "searxng_external",
                },
                ensure_ascii=False,
            )
    except Exception as e:
        logger.info("searxng backend fail: %s", type(e).__name__)

    backends = [
        ("bing", _bing_rss_search(query, n)),
        ("wikipedia", _wikipedia_search(query, n)),
        ("mojeek", _mojeek_search(query, n)),
        ("google_news", _google_news_rss(query, n)),
        ("duckduckgo", _ddg_lite_search(query, n)),
    ]

    tasks = [asyncio.create_task(_run_backend(name, coro)) for name, coro in backends]
    source = None
    results: list = []
    errors = []

    try:
        # 总竞速窗口
        done_names = []
        for fut in asyncio.as_completed(tasks, timeout=_SEARCH_TIMEOUT + 2):
            try:
                name, res = await fut
            except Exception as e:
                errors.append(str(type(e).__name__))
                continue
            done_names.append(name)
            if res:
                source = name
                results = res
                break
            errors.append(f"{name}: empty")
    except asyncio.TimeoutError:
        errors.append("race: timeout")
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        # 回收 cancel
        await asyncio.gather(*tasks, return_exceptions=True)

    if not results:
        return json.dumps(
            {
                "query": query,
                "results": [],
                "message": "无搜索结果（各搜索源未返回有效结果）",
                "errors": errors,
            },
            ensure_ascii=False,
        )
    results = _trim_search_results(results, n)
    return json.dumps(
        {"query": query, "results": results, "source": source},
        ensure_ascii=False,
    )


# ---------- 抓取 ----------


async def _builtin_scrape(url: str) -> str:
    headers = {
        "User-Agent": "LiteAIChat/1.0 (+local-scraper)",
        "Accept": "text/html,application/xhtml+xml",
    }
    async with httpx.AsyncClient(timeout=_timeout(10), follow_redirects=True) as client:
        r = await client.get(url, headers=headers)
        r.raise_for_status()
        html = r.text
    title_m = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
    title = re.sub(r"\s+", " ", title_m.group(1)).strip() if title_m else ""
    parser = _HTMLToText()
    try:
        parser.feed(html)
        text = parser.get_text()
    except Exception:
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
    if len(text) > _MAX_SCRAPE_CHARS:
        text = text[:_MAX_SCRAPE_CHARS] + "\n\n...[内容已截断]"
    return json.dumps(
        {"url": url, "title": title, "markdown": text, "source": "builtin"},
        ensure_ascii=False,
    )


async def scrape_url(url: str) -> str:
    """优先外部 Firecrawl；短超时失败后本地回退。"""
    if not url or not url.startswith(("http://", "https://")):
        return json.dumps({"error": "无效 URL，需以 http:// 或 https:// 开头"}, ensure_ascii=False)

    payload = {"url": url, "formats": ["markdown"], "onlyMainContent": True}
    try:
        async with httpx.AsyncClient(timeout=_timeout(_SCRAPER_EXT_TIMEOUT), follow_redirects=True) as client:
            r = await client.post(
                f"{SCRAPER_URL}/v1/scrape",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            if r.status_code < 400:
                ct = r.headers.get("content-type", "")
                if "application/json" in ct:
                    data = r.json()
                    if isinstance(data, dict):
                        if data.get("data") and isinstance(data["data"], dict):
                            md = data["data"].get("markdown") or data["data"].get("content") or ""
                            title = (data["data"].get("metadata") or {}).get("title", "")
                        else:
                            md = data.get("markdown") or data.get("content") or data.get("data") or ""
                            title = data.get("title") or ""
                        text = str(md)
                        if len(text) > _MAX_SCRAPE_CHARS:
                            text = text[:_MAX_SCRAPE_CHARS] + "\n\n...[内容已截断]"
                        return json.dumps(
                            {"url": url, "title": title, "markdown": text},
                            ensure_ascii=False,
                        )
                text = r.text
                if len(text) > _MAX_SCRAPE_CHARS:
                    text = text[:_MAX_SCRAPE_CHARS] + "\n\n...[内容已截断]"
                return json.dumps({"url": url, "markdown": text}, ensure_ascii=False)
    except Exception:
        pass

    try:
        return await _builtin_scrape(url)
    except Exception as e:
        return json.dumps({"error": f"抓取失败: {type(e).__name__}: {e}"}, ensure_ascii=False)


async def execute_tool(name: str, arguments: Any) -> str:
    """根据工具名分发执行，失败时返回错误 JSON 给模型。"""
    try:
        if isinstance(arguments, str):
            args = json.loads(arguments) if arguments.strip() else {}
        else:
            args = arguments or {}
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"参数 JSON 解析失败: {e}"}, ensure_ascii=False)

    try:
        if name == "web_search":
            return await web_search(
                query=args.get("query", ""),
                num_results=args.get("num_results"),
            )
        if name == "scrape_url":
            return await scrape_url(url=args.get("url", ""))
        return json.dumps({"error": f"未知工具: {name}"}, ensure_ascii=False)
    except Exception as e:
        logger.exception("tool %s failed", name)
        return json.dumps({"error": f"工具执行异常: {type(e).__name__}: {e}"}, ensure_ascii=False)
