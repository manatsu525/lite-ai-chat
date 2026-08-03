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
    SEARXNG_ENGINES,
    SEARXNG_URL,
    get_searxng_enabled_engines,
)

logger = logging.getLogger("lite-ai-chat.tools")

# 单源超时（秒）— 远小于串行 30s*N
_SEARCH_TIMEOUT = min(float(HTTP_TIMEOUT), 8.0)
_CONNECT_TIMEOUT = 3.0
_SCRAPER_EXT_TIMEOUT = 3.0
_MAX_SEARCH_SNIPPET_CHARS = 500
_MAX_SCRAPE_CHARS = 8000
_MAX_HTML_BYTES = 2_500_000
_READABILITY_MAX_CHARS = 1_000_000
_READABILITY_MAX_TAGS = 8_000
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
            "description": (
                "搜索互联网，返回相关网页、链接和检索摘要。适合查找最新信息、"
                "事实核实、资料检索。中文提问的通用主题应分别使用中文查询和"
                "真正翻译后的英文查询，以获得中英文来源；强中国本地问题除外。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "搜索关键词或问题；需要跨语言检索时，每次调用只传一种"
                            "语言的自然查询，并用另一次调用搜索另一种语言"
                        ),
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
            "description": (
                "抓取指定 URL 的网页正文，转为 markdown。适合阅读搜索结果中的具体页面。"
                "通用主题不要只读中文结果，应同时参考可访问的英文官网、英文文档"
                "或其他英文可靠来源；优先选择无需登录、验证码且可公开读取的官网或媒体来源；"
                "抓取器可能在该 URL 所属的同一轮搜索结果内自动换源，返回时必须以"
                "resolved_url 作为正文实际来源，不要把正文归到 requested_url；"
                "同等信息下避免知乎、百度搜索、百度贴吧、抖音、头条、"
                "小红书、什么值得买；百度百科、百度文库等未被固定屏蔽的"
                "百度子站仍可正常尝试。"
            ),
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

    SKIP = {
        "script",
        "style",
        "noscript",
        "svg",
        "nav",
        "footer",
        "header",
        "aside",
        "form",
        "button",
    }

    def __init__(self):
        super().__init__()
        self._parts = []
        self._skip = 0
        self._href = None

    def handle_starttag(self, tag, attrs):
        t = tag.lower()
        attr_map = dict(attrs)
        if t in self.SKIP:
            self._skip += 1
            return
        if self._skip:
            return
        if t in ("p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4"):
            self._parts.append("\n")
        if t in ("td", "th"):
            self._parts.append(" | ")
        if t in ("h1", "h2", "h3", "h4"):
            self._parts.append("\n" + "#" * int(t[1]) + " ")
        if t == "li":
            self._parts.append("- ")
        if t == "a":
            self._href = attr_map.get("href")
            self._parts.append("[")
        if t == "img":
            alt = (attr_map.get("alt") or attr_map.get("title") or "").strip()
            if alt:
                self._parts.append(alt + " ")
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
        if t in ("td", "th"):
            self._parts.append(" | ")
        if t in ("p", "div", "tr", "li", "h1", "h2", "h3", "h4"):
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


_BOILERPLATE_RE = re.compile(
    r"^(登录|注册|打开APP|下载APP|返回首页|首页|更多|展开全文|"
    r"相关推荐|热门推荐|猜你喜欢|免责声明|版权声明|扫码|关注我们|"
    r"广告|推广|上一篇|下一篇)(?:\s|$|[:：])",
    re.I,
)
_ERROR_PAGE_RE = re.compile(
    r"(404|页面不存在|访问验证|安全验证|请输入验证码|访问过于频繁|"
    r"access denied|just a moment|captcha)",
    re.I,
)


def _clean_extracted_text(text: str) -> str:
    """删除正文提取后残留的短导航、广告和重复行。"""
    cleaned = []
    seen = set()
    for raw_line in str(text or "").splitlines():
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if not line:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue
        plain = re.sub(r"\[([^\]]*)\]\([^)]+\)", r"\1", line)
        link_count = len(re.findall(r"\[[^\]]*\]\([^)]+\)", line))
        if link_count >= 3 and len(plain) < 240:
            continue
        if len(plain) <= 100 and _BOILERPLATE_RE.search(plain):
            continue
        signature = re.sub(r"\s+", "", plain).lower()
        if len(signature) <= 240 and signature in seen:
            continue
        seen.add(signature)
        cleaned.append(line)
    text = "\n".join(cleaned).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _css_pixel(style: str, key: str) -> Optional[float]:
    match = re.search(
        rf"(?:^|;)\s*{re.escape(key)}\s*:\s*(-?[0-9.]+)px",
        style or "",
        re.I,
    )
    return float(match.group(1)) if match else None


def _resource_name(element) -> str:
    names = {
        "gold_": "Gold",
        "wood_": "Wood",
        "ore_": "Ore",
        "mercury_": "Mercury",
        "sulfur_": "Sulfur",
        "crystal_": "Crystal",
        "gem_": "Gem",
    }
    found = []
    for image in element.xpath(".//img"):
        src = str(image.get("src") or "").lower().rsplit("/", 1)[-1]
        for marker, label in names.items():
            if marker in src and label not in found:
                found.append(label)
    return "+".join(found)


def _visual_grid_value(element) -> str:
    text = " ".join(element.text_content().split())
    resource = _resource_name(element)
    if resource and text and text != "-":
        return f"{text} {resource}"
    if resource and not text:
        return resource
    return text


def _mediawiki_positioned_grid(container) -> str:
    """把 MediaWiki 用绝对坐标绘制的“视觉表格”还原为 Markdown 表格。"""
    absolute = container.xpath('.//*[contains(@style, "position:absolute")]')

    def column(key: str, value: float) -> List[Tuple[float, str]]:
        out = []
        for element in absolute:
            position = _css_pixel(element.get("style") or "", key)
            top = _css_pixel(element.get("style") or "", "top")
            if position is None or top is None or abs(position - value) > 0.5:
                continue
            text = " ".join(element.text_content().split())
            if text:
                out.append((top, text))
        return sorted(out)

    buildings = column("left", 156)
    creatures = column("left", 584)
    max_week = column("right", 74)
    if min(len(buildings), len(creatures), len(max_week)) < 2:
        return ""

    row_count = min(len(buildings), len(creatures))
    creature_tops = [item[0] for item in creatures[:row_count]]

    def nearest_row(top: float) -> Optional[int]:
        if not creature_tops:
            return None
        index = min(
            range(len(creature_tops)),
            key=lambda item: abs(creature_tops[item] - top),
        )
        return index if abs(creature_tops[index] - top) <= 24 else None

    cost_rows: List[List[str]] = [[] for _ in range(row_count)]
    for element in absolute:
        style = element.get("style") or ""
        top = _css_pixel(style, "top")
        width = _css_pixel(style, "width")
        if top is None or width is None or not element.xpath("./p"):
            continue
        index = nearest_row(top)
        if index is None:
            continue
        value = _visual_grid_value(element)
        if value and value != "-":
            cost_rows[index].append(value)

    cost_week_rows: List[List[str]] = [[] for _ in range(row_count)]
    for element in absolute:
        style = element.get("style") or ""
        if _css_pixel(style, "right") != 6:
            continue
        top = _css_pixel(style, "top")
        if top is None:
            continue
        index = nearest_row(top)
        if index is None:
            continue
        value = _visual_grid_value(element)
        if value and value != "-":
            cost_week_rows[index].append(value)

    lines = [
        "| Building | Cost | Creature | Max/Wk | Cost/Wk |",
        "|---|---:|---|---:|---:|",
    ]
    for index in range(row_count):
        max_index = min(index // 2, len(max_week) - 1)
        values = (
            buildings[index][1],
            " + ".join(cost_rows[index]) if cost_rows[index] else "-",
            creatures[index][1],
            max_week[max_index][1],
            (
                " + ".join(cost_week_rows[index])
                if cost_week_rows[index]
                else "-"
            ),
        )
        escaped = [str(value).replace("|", r"\|") for value in values]
        lines.append("| " + " | ".join(escaped) + " |")
    return "\n".join(lines)


def _mediawiki_main_markdown(html: str) -> Optional[Tuple[str, str, str]]:
    """MediaWiki 正文和表格保留器；普通文章继续交给 Readability。"""
    if "mw-content-text" not in html:
        return None
    try:
        from lxml import etree
        from lxml import html as lxml_html

        document = lxml_html.fromstring(html)
        roots = document.xpath('//*[@id="mw-content-text"]')
        if not roots:
            return None
        root = roots[0]
        heading = document.xpath('//*[@id="firstHeading"]')
        title = (
            " ".join(heading[0].text_content().split())
            if heading
            else ""
        )
        grids = []
        for container in root.xpath('.//*[contains(@style, "position:relative")]'):
            grid = _mediawiki_positioned_grid(container)
            if not grid:
                continue
            grids.append(grid)
            parent = container.getparent()
            if parent is not None:
                parent.remove(container)

        remove_xpaths = (
            './/*[contains(concat(" ", normalize-space(@class), " "), " mw-editsection ")]',
            './/*[contains(concat(" ", normalize-space(@class), " "), " mw-jump-link ")]',
            './/*[contains(concat(" ", normalize-space(@class), " "), " navbox ")]',
            './/*[contains(concat(" ", normalize-space(@class), " "), " printfooter ")]',
            './/*[@id="toc"]',
            ".//script",
            ".//style",
            ".//noscript",
        )
        for xpath in remove_xpaths:
            for element in root.xpath(xpath):
                parent = element.getparent()
                if parent is not None:
                    parent.remove(element)

        # 典型侧边导航表：大量单单元格行。视觉数据表已单独还原，
        # 这种目录表只会挤占正文预算。
        if grids:
            for table in root.xpath(".//table"):
                rows = table.xpath("./tr|./tbody/tr|./thead/tr")
                single_cell_rows = sum(
                    len(row.xpath("./th|./td")) <= 1 for row in rows
                )
                if len(rows) >= 10 and single_cell_rows / len(rows) >= 0.8:
                    parent = table.getparent()
                    if parent is not None:
                        parent.remove(table)

        remaining_html = etree.tostring(
            root,
            encoding="unicode",
            method="html",
        )
        parser = _HTMLToText()
        parser.feed(remaining_html)
        remaining = _clean_extracted_text(parser.get_text())
        sections = []
        if title:
            sections.append("# " + title)
        sections.extend(grids)
        if remaining:
            sections.append(remaining)
        text = "\n\n".join(sections).strip()
        if not text:
            return None
        if len(text) > _MAX_SCRAPE_CHARS:
            text = text[:_MAX_SCRAPE_CHARS] + "\n\n...[正文内容已截断]"
        return title[:300], text, "mediawiki"
    except Exception as exc:
        logger.info("mediawiki extraction fallback: %s", type(exc).__name__)
        return None


def _html_to_main_markdown(html: str) -> Tuple[str, str, str]:
    """Readability 先选正文节点，再转为轻量 Markdown。"""
    raw_html = str(html or "")
    mediawiki = _mediawiki_main_markdown(raw_html)
    if mediawiki is not None:
        return mediawiki
    title_m = re.search(r"<title[^>]*>(.*?)</title>", raw_html, re.I | re.S)
    fallback_title = _strip_tags(title_m.group(1)) if title_m else ""
    # 先用不构建 DOM 的方式删除最容易膨胀的区域，既减少导航广告，也避免
    # lxml 面对数千节点时瞬间占满小内存 VPS。
    selected_html = re.sub(
        r"<(script|style|noscript|svg|nav|header|footer|aside|form)\b[^>]*>"
        r".*?</\1\s*>",
        "",
        raw_html,
        flags=re.I | re.S,
    )
    selected_html = re.sub(r"<!--.*?-->", "", selected_html, flags=re.S)
    title = fallback_title
    extraction = "fallback"
    try:
        from readability import Document

        if (
            len(selected_html) > _READABILITY_MAX_CHARS
            or selected_html.count("<") > _READABILITY_MAX_TAGS
        ):
            raise RuntimeError("HTML 过大，使用低内存流式正文提取")
        document = Document(selected_html)
        summary = document.summary(html_partial=True)
        if summary and len(_strip_tags(summary)) >= 20:
            selected_html = summary
            title = _strip_tags(document.short_title()) or fallback_title
            extraction = "readability"
    except Exception as exc:
        logger.info("readability fallback: %s", type(exc).__name__)

    parser = _HTMLToText()
    try:
        parser.feed(selected_html)
        text = parser.get_text()
    except Exception:
        text = _strip_tags(selected_html)
    text = _clean_extracted_text(text)

    if _ERROR_PAGE_RE.search(title or "") and len(text) < 1000:
        raise RuntimeError(f"页面返回验证或错误页：{title or 'unknown'}")
    normalized_text = re.sub(r"\s+", "", text)
    normalized_title = re.sub(r"\s+", "", title)
    if (
        not text
        or len(text) < 80
        or (
            normalized_title
            and normalized_text.strip("#") == normalized_title
        )
    ):
        raise RuntimeError("网页 HTML 只有标题/导航，正文可能需要 JavaScript")
    if len(text) > _MAX_SCRAPE_CHARS:
        text = text[:_MAX_SCRAPE_CHARS] + "\n\n...[正文内容已截断]"
    return title[:300], text, extraction


def _decode_html(content: bytes, encoding: Optional[str] = None) -> str:
    for candidate in (encoding, "utf-8", "gb18030"):
        if not candidate:
            continue
        try:
            return content.decode(candidate)
        except (LookupError, UnicodeDecodeError):
            continue
    return content.decode("utf-8", errors="replace")


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


def _url_host_for_log(value: str) -> str:
    try:
        return (urlparse(value).hostname or "")[:200]
    except ValueError:
        return ""


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


async def _searx_payload(query: str, engines: list[str]) -> dict:
    # 聚合多个引擎通常需要 5-8 秒，不能沿用单一搜索源的短超时。
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(12.0, connect=_CONNECT_TIMEOUT),
        follow_redirects=True,
    ) as client:
        r = await client.get(
            f"{SEARXNG_URL}/search",
            params={
                "q": query,
                "format": "json",
                "engines": ",".join(engines),
            },
            headers={"Accept": "application/json"},
        )
        ct = r.headers.get("content-type", "")
        if r.status_code != 200 or "json" not in ct:
            raise RuntimeError(f"SearXNG HTTP {r.status_code}")
        data = r.json()
        if not isinstance(data, dict):
            raise RuntimeError("SearXNG 返回格式错误")
        return data


def _searx_results_from_payload(data: dict, n: int) -> list:
    out = []
    for item in data.get("results") or []:
        if not isinstance(item, dict):
            continue
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
        if len(out) >= n:
            return out

    # SearXNG 的 Wikipedia 引擎返回 infoboxes 而非 results；把它转换成
    # 普通搜索卡片，否则界面显示“可用”但聊天搜索永远拿不到 Wikipedia。
    for item in data.get("infoboxes") or []:
        if not isinstance(item, dict):
            continue
        urls = item.get("urls") or []
        first_url = urls[0].get("url") if urls and isinstance(urls[0], dict) else ""
        url = item.get("url") or item.get("id") or first_url or ""
        if not str(url).startswith(("http://", "https://")):
            continue
        out.append(
            {
                "title": item.get("infobox") or item.get("title") or "Wikipedia",
                "url": url,
                "snippet": item.get("content") or "",
                "engines": item.get("engines") or [item.get("engine") or "wikipedia"],
            }
        )
        if len(out) >= n:
            break
    return out


async def _searx_search(query: str, n: int) -> list:
    engines = get_searxng_enabled_engines()
    if not engines:
        return []
    data = await _searx_payload(query, engines)
    return _searx_results_from_payload(data, n)


async def test_searxng_engine(engine: str, query: str = "OpenAI") -> dict:
    """管理员设置页使用的真实单引擎测试，不受当前开关状态影响。"""
    allowed = {item["id"] for item in SEARXNG_ENGINES}
    if engine not in allowed:
        raise ValueError("不支持的 SearXNG 引擎")
    data = await _searx_payload(query.strip() or "OpenAI", [engine])
    results = _trim_search_results(_searx_results_from_payload(data, 3), 3)
    unresponsive = data.get("unresponsive_engines") or []
    return {
        "engine": engine,
        "ok": bool(results),
        "result_count": len(results),
        "results": results,
        "unresponsive_engines": unresponsive,
        "message": (
            f"测试成功，取得 {len(results)} 条结果"
            if results
            else "请求完成，但没有解析到有效结果"
        ),
    }


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

    source = None
    results: list = []
    errors = []

    # 确定性优先级：先单独尝试实测可直接访问的 DuckDuckGo；中间三个
    # 并行以免串行拖慢，但仍按固定顺序选结果；不稳定且曾返回错位结果的
    # Bing RSS 只在所有其他来源都为空时最后尝试。
    name, res = await _run_backend("duckduckgo", _ddg_lite_search(query, n))
    if res:
        source, results = name, res
    else:
        errors.append("duckduckgo: empty")

    if not results:
        middle = await asyncio.gather(
            _run_backend("wikipedia", _wikipedia_search(query, n)),
            _run_backend("mojeek", _mojeek_search(query, n)),
            _run_backend("google_news", _google_news_rss(query, n)),
        )
        for name, res in middle:
            if res and not results:
                source, results = name, res
            elif not res:
                errors.append(f"{name}: empty")

    if not results:
        name, res = await _run_backend("bing", _bing_rss_search(query, n))
        if res:
            source, results = name, res
        else:
            errors.append("bing: empty")

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


class _ScrapeStatusError(RuntimeError):
    def __init__(self, status_code: int):
        self.status_code = int(status_code)
        super().__init__(f"HTTP {self.status_code}")


def _scrape_document_result(
    url: str,
    html: str,
    source: str,
    input_truncated: bool = False,
) -> str:
    title, text, extraction = _html_to_main_markdown(html)
    return json.dumps(
        {
            "url": url,
            "title": title,
            "markdown": text,
            "source": source,
            "extraction": extraction,
            "input_truncated": bool(input_truncated),
        },
        ensure_ascii=False,
    )


async def _builtin_scrape(url: str) -> str:
    headers = {
        "User-Agent": _UA,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.7",
    }
    async with httpx.AsyncClient(timeout=_timeout(10), follow_redirects=True) as client:
        async with client.stream("GET", url, headers=headers) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            if content_type and not any(
                kind in content_type
                for kind in ("text/", "html", "xhtml", "xml")
            ):
                raise RuntimeError(f"不支持的网页类型：{content_type[:100]}")
            body = bytearray()
            input_truncated = False
            async for chunk in response.aiter_bytes():
                remaining = _MAX_HTML_BYTES - len(body)
                if remaining <= 0:
                    input_truncated = True
                    break
                body.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    input_truncated = True
                    break
            html = _decode_html(bytes(body), response.encoding)
    return await asyncio.to_thread(
        _scrape_document_result,
        url,
        html,
        "httpx",
        input_truncated,
    )


async def _curl_cffi_scrape(url: str) -> str:
    """失败后用浏览器 TLS/HTTP2 指纹重试，不执行 JavaScript。"""
    from curl_cffi.requests import AsyncSession

    chunks = []
    received = 0
    input_truncated = False

    def receive(chunk: bytes) -> int:
        nonlocal received, input_truncated
        remaining = _MAX_HTML_BYTES - received
        if remaining > 0:
            kept = chunk[:remaining]
            chunks.append(kept)
            received += len(kept)
        if len(chunk) > max(remaining, 0):
            input_truncated = True
        return len(chunk)

    async with AsyncSession(
        impersonate="chrome",
        timeout=12,
    ) as session:
        response = await session.get(
            url,
            allow_redirects=True,
            headers={
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.7",
            },
            content_callback=receive,
        )
    if response.status_code >= 400:
        raise _ScrapeStatusError(response.status_code)
    content_type = str(response.headers.get("content-type") or "").lower()
    if content_type and not any(
        kind in content_type
        for kind in ("text/", "html", "xhtml", "xml")
    ):
        raise RuntimeError(f"不支持的网页类型：{content_type[:100]}")
    html = _decode_html(
        b"".join(chunks),
        getattr(response, "encoding", None),
    )
    return await asyncio.to_thread(
        _scrape_document_result,
        url,
        html,
        "curl_cffi",
        input_truncated,
    )


def _scrape_failure(url: str, error: Exception) -> str:
    if isinstance(error, _ScrapeStatusError):
        status_code = error.status_code
    elif isinstance(error, httpx.HTTPStatusError):
        status_code = error.response.status_code
    else:
        status_code = None
    if status_code in (401, 403, 407, 429, 451):
        return json.dumps(
            {
                "url": url,
                "error": f"源站返回 HTTP {status_code}，拒绝自动抓取",
                "status_code": status_code,
                "blocked": True,
            },
            ensure_ascii=False,
        )
    if status_code is not None:
        return json.dumps(
            {
                "url": url,
                "error": f"网页请求失败（HTTP {status_code}）",
                "status_code": status_code,
            },
            ensure_ascii=False,
        )
    message = str(error).strip()
    detail = f"{type(error).__name__}: {message}" if message else type(error).__name__
    return json.dumps(
        {"url": url, "error": f"抓取失败: {detail[:500]}"},
        ensure_ascii=False,
    )


async def scrape_url(url: str) -> str:
    """Firecrawl → httpx → curl_cffi；所有正文先做主内容提取。"""
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
    except Exception as primary_error:
        logger.info(
            "httpx scrape failed for %s: %s",
            _url_host_for_log(url),
            type(primary_error).__name__,
        )
    try:
        return await _curl_cffi_scrape(url)
    except Exception as curl_error:
        logger.info(
            "curl_cffi scrape failed for %s: %s",
            _url_host_for_log(url),
            type(curl_error).__name__,
        )
        return _scrape_failure(url, curl_error or primary_error)


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
