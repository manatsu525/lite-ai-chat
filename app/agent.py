"""多轮 tool-calling agent 循环 + SSE 流式输出。

针对 Groq/Llama 常见的 tool_use_failed（模型输出 <function=...> XML）做了容错解析。
"""
import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from datetime import date
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx

from .config import LLM_TIMEOUT, MAX_TOOL_ROUNDS, MODEL_NAME, get_model_config
from .tools import TOOLS_SCHEMA, execute_tool

logger = logging.getLogger("lite-ai-chat.agent")

_TOOL_EXEC_TIMEOUT = 30.0
_WEB_SEARCH_TIMEOUT = 60.0
_HEARTBEAT_INTERVAL = 5.0
_MAX_EXTERNAL_SEARCHES = 5
_MAX_EXTERNAL_SCRAPES = 5
_MAX_TOOL_RESULT_CHARS = 60000
_RECENT_CONVERSATION_ROUNDS = 10
_OLDER_HISTORY_SUMMARY_CHARS = 6000
_SCRAPE_AVOID_DOMAINS = (
    "douyin.com",
    "smzdm.com",
    "tieba.baidu.com",
    "toutiao.com",
    "xiaohongshu.com",
    "zhihu.com",
)
_SCRAPE_AVOID_PATHS = (
    # 只固定避开实测触发安全验证的百度搜索，不扩大到百度其他页面或子站。
    ("www.baidu.com", "/s"),
)


@dataclass(frozen=True)
class AgentBudget:
    max_tool_rounds: int
    max_tool_calls: Optional[int]
    max_searches: int
    max_search_results: int
    max_scrapes: int


_DEEP_BUDGET = AgentBudget(
    max_tool_rounds=MAX_TOOL_ROUNDS,
    max_tool_calls=None,
    max_searches=_MAX_EXTERNAL_SEARCHES,
    max_search_results=10,
    max_scrapes=_MAX_EXTERNAL_SCRAPES,
)
_NORMAL_BUDGET = AgentBudget(
    max_tool_rounds=min(MAX_TOOL_ROUNDS, 6),
    max_tool_calls=6,
    max_searches=3,
    max_search_results=5,
    max_scrapes=_MAX_EXTERNAL_SCRAPES,
)


def _budget_for(reasoning_depth: str) -> AgentBudget:
    return _NORMAL_BUDGET if reasoning_depth == "normal" else _DEEP_BUDGET


def _system_prompt(budget: AgentBudget) -> str:
    tool_limit = (
        f"所有工具累计最多调用 {budget.max_tool_calls} 次。"
        if budget.max_tool_calls is not None
        else ""
    )
    return f"""你是一个有用的 AI 助手，可以使用工具获取最新网络信息。

当前日期：{date.today().isoformat()}。

工具使用原则：
1. 需要实时或外部信息时，调用 web_search。它是独立于当前模型的外部搜索服务。
2. 每轮搜索后先判断结果是否真正回答了问题；若无关、不完整、互相冲突或缺少权威来源，不要仓促回答，要换一个明显不同且更精确的查询继续搜索。
3. 改写查询时可加入准确年份、关键实体、官方域名或 site: 限定。最多进行 {budget.max_searches} 次 web_search，每次最多返回 {budget.max_search_results} 条结果；不要重复完全相同的查询。
4. 找到关键结果后，可用 scrape_url 阅读具体页面，每个回答最多深入抓取 {budget.max_scrapes} 个网页。优先政府、学校、机构官网等一手来源；同等信息下避开知乎、百度搜索、百度贴吧、抖音、头条、小红书、什么值得买等已经实测会要求验证、拒绝 VPS 访问或只返回 JavaScript 空壳的站点，改选搜索结果中可公开读取的来源。百度百科、百度文库等未被固定屏蔽的百度子站仍应正常尝试读取。若结果标记 partial 或 search_index_fallback，表示源站阻止抓取、当前只有搜索索引摘要；不得把它当作网页全文，也不能推断摘要未提及的细节。此类失败抓取不占抓取及工具调用配额，应在还有工具轮数时改抓其他相关搜索结果。{tool_limit}
5. 回答具体日期时，正文必须明确说明该日期对应用户所问事件；网页的“发布时间/更新时间”不能当成开学、放假等事件日期。若抓到的只是列表页、图片页或正文没有答案，应继续换查询搜索，不能猜测。
6. 不要编造链接或事实；最终回答必须基于工具结果，并附上实际来源链接。达到搜索上限仍无可靠答案时，要如实说明未核实到。
7. 使用与用户相同的语言回答。
8. 只通过 API 的 function calling 调用工具，不要输出 XML 或伪代码。
9. “今年”“今天”等相对日期必须以上面的当前日期为准。"""


SYSTEM_PROMPT = _system_prompt(_DEEP_BUDGET)


def _llm_round_timeout(_model: str) -> float:
    """所有提供商使用同一个模型响应上限。"""
    return max(5.0, float(LLM_TIMEOUT))

# 匹配 Groq failed_generation 里的畸形调用，例如：
# <function=web_search{"query":"x"}</function>
# <function=web_search {"query": "x", "num_results": 5}</function>
# <function=web_search>{"query":"x"}</function>
# 兼容多种畸形写法，包括:
# <function=web_search {"q":1}></function>  (多一个 >)
# <function=web_search{"q":1}</function>
_FAILED_FN_RE = re.compile(
    r"<function\s*=\s*([a-zA-Z0-9_]+)\s*>?\s*(\{.*?\})\s*>?\s*</function>",
    re.I | re.S,
)
_FAILED_FN_RE2 = re.compile(
    r"<function\s*=\s*([a-zA-Z0-9_]+)\s*(?:>(\{.*?\})|(\{.*?\}))\s*</function>",
    re.I | re.S,
)
_FAILED_FN_RE3 = re.compile(
    r"function\s*=\s*([a-zA-Z0-9_]+)\s*(\{.*?\})",
    re.I | re.S,
)
_FAILED_FN_BLOCK_RE = re.compile(
    r"\s*<function\s*=\s*[a-zA-Z0-9_]+\s*>?\s*\{.*?\}\s*>?\s*</function>\s*",
    re.I | re.S,
)
_UNCERTAIN_FINAL_RE = re.compile(
    r"(无法(?:确定|核实|找到|获取)|未(?:找到|查到|检索到|公布)|"
    r"没有(?:找到|查到)|建议.{0,20}(?:关注|咨询|查询)|"
    r"unable to (?:determine|verify|find)|could not find|not (?:found|available))",
    re.I | re.S,
)


def _headers(api_key: str) -> dict:
    h = {"Content-Type": "application/json"}
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    return h


def _compact_conversation_history(messages: List[dict]) -> List[dict]:
    """保留最近十轮原文，更早内容用本地摘录摘要代替。"""
    user_indexes = [
        index for index, item in enumerate(messages) if item.get("role") == "user"
    ]
    if len(user_indexes) <= _RECENT_CONVERSATION_ROUNDS:
        return messages

    cut_at = user_indexes[-_RECENT_CONVERSATION_ROUNDS]
    older = [
        item
        for item in messages[:cut_at]
        if item.get("role") in ("user", "assistant") and item.get("content")
    ]
    summary_lines = []
    used = 0
    for item in older:
        text = re.sub(r"\s+", " ", str(item.get("content") or "")).strip()
        if not text:
            continue
        excerpt = text[:400]
        label = "用户" if item.get("role") == "user" else "助手"
        line = f"- {label}：{excerpt}"
        remaining = _OLDER_HISTORY_SUMMARY_CHARS - used
        if remaining <= 0:
            break
        line = line[:remaining]
        summary_lines.append(line)
        used += len(line) + 1

    system_messages = [
        item for item in messages[:cut_at] if item.get("role") == "system"
    ]
    compacted = list(system_messages)
    if summary_lines:
        compacted.append(
            {
                "role": "system",
                "content": (
                    "以下是更早对话的本地压缩摘要，仅用于延续上下文；"
                    "最近十轮对话仍保留原文：\n" + "\n".join(summary_lines)
                ),
            }
        )
    compacted.extend(messages[cut_at:])
    return compacted


def _normalize_messages(
    messages: List[dict],
    system_prompt: str = SYSTEM_PROMPT,
) -> List[dict]:
    """确保有 system 提示，并清理 Groq 不喜欢的字段。"""
    out = []
    for m in messages:
        item = {k: v for k, v in dict(m).items() if v is not None}
        # content 为空字符串时保留；tool 消息必须有 content
        if item.get("role") == "assistant" and "content" not in item:
            if item.get("tool_calls"):
                item["content"] = ""
        out.append(item)
    out = _compact_conversation_history(out)
    if not any(m.get("role") == "system" for m in out):
        out.insert(0, {"role": "system", "content": system_prompt})
    elif not any((m.get("content") or "") == system_prompt for m in out):
        out.insert(0, {"role": "system", "content": system_prompt})
    return out


def _parse_failed_generation(text: str) -> List[dict]:
    """把 Groq tool_use_failed 的 failed_generation 解析成 tool_calls。"""
    if not text:
        return []
    calls = []
    for rx in (_FAILED_FN_RE, _FAILED_FN_RE2, _FAILED_FN_RE3):
        for m in rx.finditer(text):
            name = m.group(1)
            # 不同正则参数位置略有差异
            args = ""
            for g in m.groups()[1:]:
                if g and g.strip().startswith("{"):
                    args = g.strip()
                    break
            if not name:
                continue
            # 校验 JSON
            try:
                json.loads(args or "{}")
            except json.JSONDecodeError:
                # 尝试修正常见尾部问题
                fixed = (args or "{}").strip()
                fixed = re.sub(r",\s*}", "}", fixed)
                try:
                    json.loads(fixed)
                    args = fixed
                except json.JSONDecodeError:
                    args = json.dumps({"raw": args}, ensure_ascii=False)
            calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:10]}",
                    "type": "function",
                    "function": {"name": name, "arguments": args or "{}"},
                }
            )
        if calls:
            break
    # 去重（同名+同参）
    seen = set()
    uniq = []
    for c in calls:
        key = (c["function"]["name"], c["function"]["arguments"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq


def _clean_failed_generation(text: str) -> str:
    """从正常回答中移除 Groq 偶尔夹带的 XML 风格伪工具调用。"""
    return _FAILED_FN_BLOCK_RE.sub("", text or "").strip()


def _extract_error_payload(resp_text: str) -> dict:
    try:
        return json.loads(resp_text)
    except Exception:
        return {"error": {"message": resp_text[:800]}}


def _has_image_content(messages: List[dict]) -> bool:
    for message in messages:
        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(item, dict) and item.get("type") == "image_url"
            for item in content
        ):
            return True
    return False


class ToolUseFailed(Exception):
    """Groq 返回 tool_use_failed，已解析出 synthetic tool_calls。"""

    def __init__(self, tool_calls: List[dict], raw: str):
        super().__init__(raw[:300])
        self.tool_calls = tool_calls
        self.raw = raw


async def _call_llm_with_model(messages: List[dict], model: str, with_tools: bool = True) -> dict:
    model_config = get_model_config(model)
    if not model_config:
        raise RuntimeError(f"模型未配置或不可用: {model}")

    body: Dict[str, Any] = {
        "model": model_config["model_id"],
        "messages": messages,
        "stream": False,
    }
    if model_config["provider"] == "deepseek":
        # V4 默认使用思考模式；工具轮必须保留 reasoning_content。
        body["thinking"] = {"type": "enabled"}
    else:
        body["temperature"] = 0.3 if with_tools else 0.5
    if with_tools:
        body["tools"] = TOOLS_SCHEMA
        # DeepSeek V4 思考模式不接受 tool_choice。
        if model_config["provider"] != "deepseek":
            body["tool_choice"] = "auto"

    url = f"{model_config['api_base']}/chat/completions"
    async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
        r = await client.post(url, headers=_headers(model_config["api_key"]), json=body)
        if r.status_code >= 400:
            payload = _extract_error_payload(r.text)
            err = payload.get("error") or {}
            code = err.get("code") or ""
            failed_gen = err.get("failed_generation") or ""
            # 部分实现把 failed_generation 放在顶层
            if not failed_gen:
                failed_gen = payload.get("failed_generation") or ""

            if r.status_code == 400 and (code == "tool_use_failed" or failed_gen):
                calls = _parse_failed_generation(failed_gen)
                logger.warning(
                    "tool_use_failed parsed=%s gen=%s",
                    len(calls),
                    failed_gen[:200],
                )
                if calls:
                    raise ToolUseFailed(calls, failed_gen)
            if _has_image_content(messages) and r.status_code in (400, 404, 415, 422):
                raise RuntimeError(
                    "所选模型或接口不接受图片输入，请改用支持视觉的多模态模型。"
                    f"（上游 HTTP {r.status_code}）"
                )
            raise RuntimeError(f"LLM HTTP {r.status_code}: {r.text[:800]}")
        return r.json()


async def _apply_tool_calls(
    msgs: List[dict],
    tool_calls: List[dict],
    assistant_message: Optional[dict] = None,
    max_search_results: int = 10,
) -> List[dict]:
    """把工具结果加入模型上下文，并返回适合前端展示的精简事件。"""
    # 模型可能主动请求 10 条；普通思考模式必须在执行层强制压到 5 条。
    for tc in tool_calls:
        fn = tc.get("function") or {}
        if fn.get("name") != "web_search":
            continue
        raw_args = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        requested = args.get("num_results", max_search_results)
        try:
            requested = int(requested)
        except (TypeError, ValueError):
            requested = max_search_results
        args["num_results"] = max(1, min(requested, max_search_results))
        fn["arguments"] = json.dumps(args, ensure_ascii=False)

    events = []
    assistant = {
        "role": "assistant",
        "content": (assistant_message or {}).get("content") or "",
        "tool_calls": tool_calls,
    }
    reasoning_content = (assistant_message or {}).get("reasoning_content")
    if reasoning_content is not None:
        assistant["reasoning_content"] = reasoning_content
    msgs.append(assistant)
    for tc in tool_calls:
        fn = tc.get("function") or {}
        name = fn.get("name") or ""
        args = fn.get("arguments") or "{}"
        routing = None
        if name == "scrape_url":
            replacement = _replacement_scrape_source(msgs, args)
            if replacement:
                original_args = _tool_arguments(tc)
                requested_url = str(original_args.get("url") or "")
                replacement_url = str(replacement.get("url") or "")
                routing = {
                    "requested_url": requested_url,
                    "replacement_url": replacement_url,
                    "replacement_title": str(
                        replacement.get("title") or ""
                    )[:300],
                }
                fn["arguments"] = json.dumps(
                    {"url": replacement_url},
                    ensure_ascii=False,
                )
                args = fn["arguments"]
        tc_id = tc.get("id") or f"call_{uuid.uuid4().hex[:8]}"
        logger.info("exec tool %s args=%s", name, str(args)[:200])
        try:
            result = await asyncio.wait_for(
                execute_tool(name, args),
                timeout=_WEB_SEARCH_TIMEOUT if name == "web_search" else _TOOL_EXEC_TIMEOUT,
            )
        except asyncio.TimeoutError:
            timeout = _WEB_SEARCH_TIMEOUT if name == "web_search" else _TOOL_EXEC_TIMEOUT
            logger.warning("tool %s timed out after %.1fs", name, timeout)
            result = json.dumps(
                {"error": f"工具执行超过 {timeout:.0f} 秒，已停止"},
                ensure_ascii=False,
            )
        if name == "scrape_url":
            result = _annotate_scrape_routing(result, routing)
            result = _scrape_search_index_fallback(msgs, args, result)
        msgs.append(
            {
                "role": "tool",
                "tool_call_id": tc_id,
                "content": result,
            }
        )
        _enforce_tool_result_budget(msgs)
        events.append(_tool_trace_event(tc, result, state="done"))
    return events


def _normalized_url(value: str) -> str:
    return re.sub(r"[#?].*$", "", str(value or "")).rstrip("/").lower()


def _url_host(value: str) -> str:
    try:
        return (urlparse(str(value or "")).hostname or "").lower()
    except ValueError:
        return ""


def _avoid_scrape_url(value: str) -> bool:
    try:
        parsed = urlparse(str(value or ""))
        host = (parsed.hostname or "").lower()
        path = parsed.path.rstrip("/") or "/"
    except ValueError:
        return False
    if any(
        host == domain or host.endswith("." + domain)
        for domain in _SCRAPE_AVOID_DOMAINS
    ):
        return True
    return any(host == blocked_host and path == blocked_path for blocked_host, blocked_path in _SCRAPE_AVOID_PATHS)


def _replacement_scrape_source(
    messages: List[dict],
    arguments: Any,
) -> Optional[dict]:
    """已知强反爬来源改抓同一搜索轮中的可访问候选。"""
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    target_url = str(args.get("url") or "")
    if not _avoid_scrape_url(target_url):
        return None
    normalized_target = _normalized_url(target_url)
    used_urls = set()
    for message in messages:
        if message.get("role") != "tool":
            continue
        try:
            payload = json.loads(message.get("content") or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        for key in ("url", "requested_url", "replacement_url"):
            normalized = _normalized_url(payload.get(key))
            if normalized:
                used_urls.add(normalized)

    for message in reversed(messages):
        if message.get("role") != "tool":
            continue
        try:
            payload = json.loads(message.get("content") or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list):
            continue
        if not any(
            _normalized_url(item.get("url")) == normalized_target
            for item in results
            if isinstance(item, dict)
        ):
            continue
        for item in results:
            if not isinstance(item, dict):
                continue
            candidate = str(item.get("url") or "")
            normalized = _normalized_url(candidate)
            if (
                not candidate.startswith(("http://", "https://"))
                or not normalized
                or normalized == normalized_target
                or normalized in used_urls
                or _avoid_scrape_url(candidate)
            ):
                continue
            return item
    return None


def _annotate_scrape_routing(
    result: str,
    routing: Optional[dict],
) -> str:
    if not routing:
        return result
    try:
        payload = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return result
    if not isinstance(payload, dict):
        return result
    payload["requested_url"] = routing["requested_url"]
    payload["replacement_url"] = routing["replacement_url"]
    payload["routing_note"] = (
        "原来源经常阻止 VPS 抓取，已自动改用同一轮搜索中的可访问候选："
        + (routing.get("replacement_title") or routing["replacement_url"])
    )
    return json.dumps(payload, ensure_ascii=False)


def _scrape_search_index_fallback(
    messages: List[dict],
    arguments: Any,
    result: str,
) -> str:
    """源站反爬时复用先前搜索摘要，避免把 403 当成唯一工具结果。"""
    try:
        payload = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return result
    if not isinstance(payload, dict) or not payload.get("error"):
        return result
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
    except (json.JSONDecodeError, TypeError, ValueError):
        return result
    target_url = str(args.get("url") or payload.get("url") or "")
    normalized_target = _normalized_url(target_url)
    if not normalized_target:
        return result

    for message in reversed(messages):
        if message.get("role") != "tool":
            continue
        try:
            search_payload = json.loads(message.get("content") or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(search_payload, dict):
            continue
        for item in search_payload.get("results") or []:
            if not isinstance(item, dict):
                continue
            if _normalized_url(item.get("url")) != normalized_target:
                continue
            title = str(item.get("title") or "")[:300]
            snippet = str(
                item.get("snippet") or item.get("key_point") or ""
            )[:1000]
            if not title and not snippet:
                continue
            warning = str(payload.get("error") or "")[:500]
            markdown = (
                "源站阻止了自动读取。以下内容来自此前搜索结果的索引摘要，"
                "不是网页全文，不能据此推断摘要中未提及的细节。\n\n"
            )
            if title:
                markdown += f"标题：{title}\n\n"
            if snippet:
                markdown += f"索引摘要：{snippet}"
            return json.dumps(
                {
                    "url": target_url,
                    "title": title,
                    "markdown": markdown,
                    "source": "search_index_fallback",
                    "partial": True,
                    "warning": warning,
                },
                ensure_ascii=False,
            )
    return result


def _tool_arguments(tool_call: dict) -> dict:
    raw = (tool_call.get("function") or {}).get("arguments") or "{}"
    try:
        value = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _tool_trace_event(
    tool_call: dict,
    result: Optional[str] = None,
    state: str = "running",
) -> dict:
    """只暴露工具参数和有限结果；不把大段网页正文塞给轮询接口。"""
    fn = tool_call.get("function") or {}
    name = str(fn.get("name") or "tool")
    args = _tool_arguments(tool_call)
    event_id = "tool-" + str(tool_call.get("id") or uuid.uuid4().hex[:12])[:80]
    event = {
        "id": event_id,
        "kind": "tool",
        "tool": name,
        "state": state,
    }
    if name == "web_search":
        query = str(args.get("query") or "")[:500]
        event.update(
            {
                "kind": "search",
                "title": f"搜索：{query}" if query else "搜索网页",
                "query": query,
                "results": [],
            }
        )
    elif name == "scrape_url":
        url = str(args.get("url") or "")[:2000]
        event.update(
            {
                "kind": "page",
                "title": "读取网页",
                "url": url,
            }
        )
    else:
        event["title"] = f"调用工具：{name}"

    if result is None:
        return event
    try:
        payload = json.loads(result) if isinstance(result, str) else result
    except (json.JSONDecodeError, TypeError):
        payload = {"content": str(result or "")}
    if not isinstance(payload, dict):
        payload = {"content": str(payload)}

    if payload.get("error"):
        event["error"] = str(payload["error"])[:500]
        event["state"] = "error"
    if payload.get("warning"):
        event["warning"] = str(payload["warning"])[:500]
    if payload.get("routing_note"):
        event["routing_note"] = str(payload["routing_note"])[:500]
    if payload.get("partial"):
        event["partial"] = True
    if name == "web_search":
        event["source"] = str(payload.get("source") or "")[:100]
        event["results"] = [
            {
                "title": str(item.get("title") or "")[:300],
                "url": str(item.get("url") or "")[:2000],
                "snippet": str(item.get("snippet") or "")[:500],
            }
            for item in (payload.get("results") or [])[:10]
            if isinstance(item, dict)
        ]
        event["title"] = (
            f"搜索：{event.get('query') or ''}（{len(event['results'])} 条）"
        )
    elif name == "scrape_url":
        event["url"] = str(payload.get("url") or event.get("url") or "")[:2000]
        event["source"] = str(payload.get("source") or "")[:100]
        event["extraction"] = str(payload.get("extraction") or "")[:100]
        page_title = str(payload.get("title") or "")[:300]
        event["title"] = f"读取网页：{page_title}" if page_title else "读取网页"
        body = payload.get("markdown")
        if body is None:
            body = payload.get("content")
        event["excerpt"] = str(body or "")[:2000]
    return event


def _refund_failed_scrape_quota(
    events: List[dict],
    tool_calls_used: int,
    scrape_calls: int,
) -> Tuple[int, int]:
    """失败或仅返回搜索摘要的抓取不占工具及抓取配额。"""
    refundable = 0
    for event in events:
        if event.get("tool") != "scrape_url":
            continue
        if (
            event.get("state") == "error"
            or event.get("source") == "search_index_fallback"
            or event.get("partial") is True
        ):
            event["counts_toward_tool_limit"] = False
            refundable += 1
    return (
        max(0, tool_calls_used - refundable),
        max(0, scrape_calls - refundable),
    )


def _tool_running_events(tool_calls: List[dict]) -> List[dict]:
    return [_tool_trace_event(tc, state="running") for tc in tool_calls]


def _tool_signature(tool_call: dict) -> Tuple[str, str]:
    """用于阻止模型连续发出完全相同的工具调用。"""
    fn = tool_call.get("function") or {}
    name = str(fn.get("name") or "")
    raw_args = fn.get("arguments") or "{}"
    try:
        parsed = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        args = json.dumps(parsed, ensure_ascii=False, sort_keys=True)
    except Exception:
        args = str(raw_args).strip()
    return name, args


def _compact_tool_result(content: str) -> str:
    """把旧工具结果压缩为来源卡片或关键摘录。"""
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return str(content)[:1200] + "\n...[旧工具结果已压缩]"
    if not isinstance(payload, dict):
        return str(content)[:1200] + "\n...[旧工具结果已压缩]"

    if isinstance(payload.get("results"), list):
        results = []
        for item in payload["results"][:10]:
            if not isinstance(item, dict):
                continue
            results.append(
                {
                    "title": str(item.get("title") or "")[:200],
                    "url": str(item.get("url") or "")[:1000],
                    "key_point": str(
                        item.get("snippet") or item.get("key_point") or ""
                    )[:220],
                }
            )
        compacted = {
            "query": str(payload.get("query") or "")[:300],
            "results": results,
            "note": "旧搜索结果已压缩，仅保留标题、URL和关键结论",
        }
    elif payload.get("markdown") is not None or payload.get("content") is not None:
        body = payload.get("markdown")
        if body is None:
            body = payload.get("content")
        compacted = {
            "title": str(payload.get("title") or "")[:300],
            "url": str(payload.get("url") or "")[:2000],
            "key_excerpt": str(body or "")[:1200],
            "note": "旧网页正文已压缩为关键摘录",
        }
    else:
        compacted = dict(payload)
        compacted["note"] = "旧工具结果已压缩"
    return json.dumps(compacted, ensure_ascii=False)


def _enforce_tool_result_budget(messages: List[dict]) -> None:
    """工具正文总量超限时，从最旧结果开始压缩。"""
    tool_messages = [item for item in messages if item.get("role") == "tool"]
    total = sum(len(str(item.get("content") or "")) for item in tool_messages)
    if total <= _MAX_TOOL_RESULT_CHARS:
        return

    # 最新结果最可能与下一步判断相关，优先保留；从最旧项开始压缩。
    for item in tool_messages[:-1]:
        old = str(item.get("content") or "")
        if "旧搜索结果已压缩" in old or "旧网页正文已压缩" in old:
            continue
        compacted = _compact_tool_result(old)
        if len(compacted) >= len(old):
            continue
        item["content"] = compacted
        total -= len(old) - len(compacted)
        if total <= _MAX_TOOL_RESULT_CHARS:
            return

    # 极端情况下仍超限，继续硬裁剪最旧结果，确保预算是真正的硬上限。
    for item in tool_messages:
        if total <= _MAX_TOOL_RESULT_CHARS:
            break
        old = str(item.get("content") or "")
        suffix = "\n...[工具总预算裁剪]"
        excess = total - _MAX_TOOL_RESULT_CHARS
        keep = max(0, len(old) - excess - len(suffix))
        if keep >= len(old):
            continue
        new = old[:keep] + suffix
        item["content"] = new
        total -= len(old) - len(new)


def _limit_external_tools(
    tool_calls: List[dict],
    searches_used: int,
    scrapes_used: int,
    tool_calls_used: int,
    budget: AgentBudget,
) -> List[dict]:
    """按当前思考深度强制裁剪搜索、抓取和工具调用总数。"""
    searches_remaining = max(0, budget.max_searches - searches_used)
    scrapes_remaining = max(0, budget.max_scrapes - scrapes_used)
    calls_remaining = (
        len(tool_calls)
        if budget.max_tool_calls is None
        else max(0, budget.max_tool_calls - tool_calls_used)
    )
    kept = []
    for tc in tool_calls:
        if calls_remaining <= 0:
            break
        name = (tc.get("function") or {}).get("name")
        if name == "web_search":
            if searches_remaining <= 0:
                continue
            searches_remaining -= 1
        elif name == "scrape_url":
            if scrapes_remaining <= 0:
                continue
            scrapes_remaining -= 1
        kept.append(tc)
        calls_remaining -= 1
    return kept


def _should_refine_search(
    content: str,
    search_calls: int,
    tool_calls_used: int,
    budget: AgentBudget,
) -> bool:
    """模型准备以“没查到”收尾时，若还有额度则要求改写查询继续检索。"""
    return (
        0 < search_calls < budget.max_searches
        and (
            budget.max_tool_calls is None
            or tool_calls_used < budget.max_tool_calls
        )
        and bool(_UNCERTAIN_FINAL_RE.search(content or ""))
    )


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _chunk(
    content: str = None,
    finish_reason: str = None,
    model: str = None,
    cid: str = None,
) -> dict:
    delta: Dict[str, Any] = {}
    if content is not None:
        delta["content"] = content
    return {
        "id": cid or f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model or MODEL_NAME,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


async def _resolve_round(
    msgs: List[dict], model: str
) -> Tuple[Optional[dict], List[dict], Optional[str]]:
    """
    请求一轮模型。
    返回 (message_dict_or_None, tool_calls, error_text)
    tool_calls 可能来自正常响应或 failed_generation 解析。
    """
    try:
        data = await _call_llm_with_model(msgs, model=model, with_tools=True)
    except ToolUseFailed as e:
        return None, e.tool_calls, None
    except Exception as e:
        return None, [], str(e)

    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    tool_calls = message.get("tool_calls") or []
    content = message.get("content") or ""
    if not tool_calls and content:
        embedded_calls = _parse_failed_generation(content)
        if embedded_calls:
            cleaned = _clean_failed_generation(content)
            # 首轮只有伪调用时把它当真实工具调用；已有工具结果且已有
            # 实质答案时，保留答案并丢弃末尾重复的伪调用。
            has_tool_result = any(m.get("role") == "tool" for m in msgs)
            if has_tool_result and cleaned:
                message["content"] = cleaned
            else:
                message["content"] = cleaned
                tool_calls = embedded_calls
    return message, tool_calls, None


async def run_agent_stream(
    messages: List[dict],
    model: Optional[str] = None,
    reasoning_depth: str = "deep",
) -> AsyncIterator[str]:
    use_model = model or MODEL_NAME
    budget = _budget_for(reasoning_depth)
    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    msgs = _normalize_messages(messages, _system_prompt(budget))

    if not get_model_config(use_model):
        yield _sse(_chunk(content="错误: 所选模型未配置或不可用", finish_reason="stop", model=use_model, cid=cid))
        yield "data: [DONE]\n\n"
        return

    # 立刻推送心跳，避免前端首包前长时间空白
    boot = _chunk(model=use_model, cid=cid)
    boot["status"] = {
        "type": "thinking",
        "message": "正在分析问题…",
        "trace_events": [
            {
                "id": "thinking-1",
                "kind": "thinking",
                "title": "分析问题并判断是否需要使用工具",
                "state": "running",
            }
        ],
    }
    yield _sse(boot)

    seen_tool_calls = set()
    search_calls = 0
    scrape_calls = 0
    tool_calls_used = 0
    for round_i in range(budget.max_tool_rounds + 1):
        if round_i > 0:
            thinking = _chunk(model=use_model, cid=cid)
            thinking["status"] = {
                "type": "thinking",
                "message": f"正在结合第 {round_i} 轮工具结果继续分析…",
                "trace_events": [
                    {
                        "id": f"thinking-{round_i + 1}",
                        "kind": "thinking",
                        "title": f"结合第 {round_i} 轮工具结果继续分析",
                        "state": "running",
                    }
                ],
            }
            yield _sse(thinking)
        # 上游偶尔会长时间不返回；等待期间持续发 SSE 心跳，并设置整轮硬上限。
        resolve_task = asyncio.create_task(_resolve_round(msgs, use_model))
        started = time.monotonic()
        round_timeout = _llm_round_timeout(use_model)
        try:
            while True:
                try:
                    message, tool_calls, err = await asyncio.wait_for(
                        asyncio.shield(resolve_task),
                        timeout=_HEARTBEAT_INTERVAL,
                    )
                    break
                except asyncio.TimeoutError:
                    elapsed = time.monotonic() - started
                    if elapsed >= round_timeout:
                        resolve_task.cancel()
                        await asyncio.gather(resolve_task, return_exceptions=True)
                        message, tool_calls = None, []
                        err = f"上游模型响应超过 {round_timeout:.0f} 秒"
                        break
                    heartbeat = _chunk(model=use_model, cid=cid)
                    heartbeat["status"] = {
                        "type": "thinking",
                        "message": f"仍在等待模型响应…（{int(elapsed)} 秒）",
                    }
                    yield _sse(heartbeat)
        except asyncio.CancelledError:
            # 后台任务被用户停止时，也要取消被 shield 保护的上游 HTTP 请求。
            resolve_task.cancel()
            await asyncio.gather(resolve_task, return_exceptions=True)
            raise
        if err:
            yield _sse(_chunk(content=f"\n[上游模型错误] {err}", finish_reason="stop", model=use_model, cid=cid))
            yield "data: [DONE]\n\n"
            return

        if tool_calls and round_i < budget.max_tool_rounds:
            new_tool_calls = [
                tc for tc in tool_calls if _tool_signature(tc) not in seen_tool_calls
            ]
            new_tool_calls = _limit_external_tools(
                new_tool_calls,
                search_calls,
                scrape_calls,
                tool_calls_used,
                budget,
            )
            if not new_tool_calls:
                logger.warning("repeated/excess tool call stopped at round %s", round_i + 1)
                msgs.append(
                    {
                        "role": "user",
                        "content": "工具调用已重复，或外部搜索/网页抓取已达上限。请立即基于已有结果给出最终回答，不要再调用工具。",
                    }
                )
                status = _chunk(model=use_model, cid=cid)
                status["status"] = {
                    "type": "tool_done",
                    "tools": [tc.get("function", {}).get("name", "?") for tc in tool_calls],
                    "round": round_i + 1,
                    "message": "检测到重复工具调用，正在生成最终回答…",
                    "trace_events": [
                        {
                            "id": f"thinking-{round_i + 1}",
                            "kind": "thinking",
                            "title": "工具预算已达上限，基于已有资料整理回答",
                            "state": "done",
                        }
                    ],
                }
                yield _sse(status)
                try:
                    async for piece in _call_llm_final_stream(msgs, use_model, cid):
                        yield piece
                except Exception as e:
                    yield _sse(
                        _chunk(content=f"\n[最终回答错误] {e}", finish_reason="stop", model=use_model, cid=cid)
                    )
                    yield "data: [DONE]\n\n"
                return

            tool_calls = new_tool_calls
            tool_calls_used += len(tool_calls)
            search_calls += sum(
                1
                for tc in tool_calls
                if (tc.get("function") or {}).get("name") == "web_search"
            )
            scrape_calls += sum(
                1
                for tc in tool_calls
                if (tc.get("function") or {}).get("name") == "scrape_url"
            )
            seen_tool_calls.update(_tool_signature(tc) for tc in tool_calls)
            names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
            status = _chunk(model=use_model, cid=cid)
            status["status"] = {
                "type": "tool_start",
                "tools": names,
                "round": round_i + 1,
                "trace_events": [
                    {
                        "id": f"thinking-{round_i + 1}",
                        "kind": "thinking",
                        "title": "已决定调用外部工具",
                        "state": "done",
                    },
                    *_tool_running_events(tool_calls),
                ],
            }
            yield _sse(status)

            # 工具可能稍慢：执行前再推一条，保持连接活跃
            running = _chunk(model=use_model, cid=cid)
            running["status"] = {
                "type": "tool_running",
                "tools": names,
                "round": round_i + 1,
                "message": "正在执行工具…",
            }
            yield _sse(running)

            tool_events = await _apply_tool_calls(
                msgs,
                tool_calls,
                message,
                budget.max_search_results,
            )
            tool_calls_used, scrape_calls = _refund_failed_scrape_quota(
                tool_events,
                tool_calls_used,
                scrape_calls,
            )

            done = _chunk(model=use_model, cid=cid)
            done["status"] = {
                "type": "tool_done",
                "tools": names,
                "round": round_i + 1,
                "trace_events": tool_events,
            }
            yield _sse(done)
            continue

        # 达上限仍想调工具：执行最后一轮工具后强制总结
        if tool_calls and round_i >= budget.max_tool_rounds:
            tool_calls = _limit_external_tools(
                tool_calls,
                search_calls,
                scrape_calls,
                tool_calls_used,
                budget,
            )
            if tool_calls:
                tool_calls_used += len(tool_calls)
                tool_events = await _apply_tool_calls(
                    msgs,
                    tool_calls,
                    message,
                    budget.max_search_results,
                )
                tool_calls_used, scrape_calls = _refund_failed_scrape_quota(
                    tool_events,
                    tool_calls_used,
                    scrape_calls,
                )
                tool_done = _chunk(model=use_model, cid=cid)
                tool_done["status"] = {
                    "type": "tool_done",
                    "tools": [
                        tc.get("function", {}).get("name", "?")
                        for tc in tool_calls
                    ],
                    "round": round_i + 1,
                    "trace_events": tool_events,
                }
                yield _sse(tool_done)
            msgs.append(
                {
                    "role": "user",
                    "content": "请基于已有工具结果直接给出最终回答，不要再调用工具。",
                }
            )
            final_status = _chunk(model=use_model, cid=cid)
            final_status["status"] = {
                "type": "thinking",
                "message": "工具轮数已达上限，正在整理最终回答…",
                "trace_events": [
                    {
                        "id": f"thinking-{round_i + 1}",
                        "kind": "thinking",
                        "title": "工具轮数已达上限，整理最终回答",
                        "state": "done",
                    }
                ],
            }
            yield _sse(final_status)
            try:
                async for piece in _call_llm_final_stream(msgs, use_model, cid):
                    yield piece
            except Exception as e:
                yield _sse(
                    _chunk(content=f"\n[最终回答错误] {e}", finish_reason="stop", model=use_model, cid=cid)
                )
                yield "data: [DONE]\n\n"
            return

        # 正常文本回答
        final_content = (message or {}).get("content") or ""
        if final_content:
            if _should_refine_search(
                final_content,
                search_calls,
                tool_calls_used,
                budget,
            ):
                msgs.append({"role": "assistant", "content": final_content})
                msgs.append(
                    {
                        "role": "user",
                        "content": (
                            "你当前的草稿表明信息仍未核实，且外部搜索额度尚未用完。"
                            "请分析缺口，换一个更精确且明显不同的查询继续调用 web_search；"
                            "可使用 site: 官方域名、完整年份和关键事件词。"
                        ),
                    }
                )
                status = _chunk(model=use_model, cid=cid)
                status["status"] = {
                    "type": "thinking",
                    "message": "现有结果不足，正在改写查询继续搜索…",
                    "trace_events": [
                        {
                            "id": f"thinking-{round_i + 1}",
                            "kind": "thinking",
                            "title": "现有资料不足，准备改写关键词继续检索",
                            "state": "done",
                        }
                    ],
                }
                yield _sse(status)
                continue
            final_status = _chunk(model=use_model, cid=cid)
            final_status["status"] = {
                "type": "thinking",
                "message": "回答生成完成",
                "trace_events": [
                    {
                        "id": f"thinking-{round_i + 1}",
                        "kind": "thinking",
                        "title": "完成分析并生成回答",
                        "state": "done",
                    }
                ],
            }
            yield _sse(final_status)
            step = 24
            for i in range(0, len(final_content), step):
                yield _sse(_chunk(content=final_content[i : i + step], model=use_model, cid=cid))
            yield _sse(_chunk(finish_reason="stop", model=use_model, cid=cid))
            yield "data: [DONE]\n\n"
            return

        # 空内容：无 tools 再拉一次
        try:
            async for piece in _call_llm_final_stream(msgs, use_model, cid):
                yield piece
        except Exception as e:
            yield _sse(_chunk(content=f"\n[流式错误] {e}", finish_reason="stop", model=use_model, cid=cid))
            yield "data: [DONE]\n\n"
        return


async def _call_llm_final_stream(messages: List[dict], model: str, cid: str) -> AsyncIterator[str]:
    """最终轮：不带 tools，流式输出。"""
    model_config = get_model_config(model)
    if not model_config:
        raise RuntimeError(f"模型未配置或不可用: {model}")

    # 清理历史里可能让上游困惑的 tool 结构：保留，但去掉 tools 参数
    body = {
        "model": model_config["model_id"],
        "messages": messages,
        "stream": True,
    }
    if model_config["provider"] == "deepseek":
        body["thinking"] = {"type": "enabled"}
    else:
        body["temperature"] = 0.5
    url = f"{model_config['api_base']}/chat/completions"
    async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
        async with client.stream(
            "POST",
            url,
            headers=_headers(model_config["api_key"]),
            json=body,
        ) as resp:
            if resp.status_code >= 400:
                text = (await resp.aread()).decode("utf-8", errors="replace")
                # 流式失败时退回非流式
                logger.warning("final stream failed: %s", text[:300])
                data = await _call_llm_with_model(messages, model=model, with_tools=False)
                content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
                if content:
                    yield _sse(_chunk(content=content, model=model, cid=cid))
                yield _sse(_chunk(finish_reason="stop", model=model, cid=cid))
                yield "data: [DONE]\n\n"
                return
            async for line in resp.aiter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        yield "data: [DONE]\n\n"
                        return
                    try:
                        obj = json.loads(payload)
                        obj["id"] = cid
                        obj["model"] = model
                        # DeepSeek 会把内部 reasoning_content 逐 token 流出；
                        # 前端不展示这些内容，不应浪费带宽和浏览器解析开销。
                        useful = False
                        for choice in obj.get("choices") or []:
                            delta = choice.get("delta") or {}
                            delta.pop("reasoning_content", None)
                            if delta.get("content") is None:
                                delta.pop("content", None)
                            if delta.get("content") or choice.get("finish_reason") is not None:
                                useful = True
                        if not useful:
                            continue
                        yield _sse(obj)
                    except json.JSONDecodeError:
                        continue
            yield "data: [DONE]\n\n"


async def run_agent_sync(
    messages: List[dict],
    model: Optional[str] = None,
    reasoning_depth: str = "deep",
) -> dict:
    use_model = model or MODEL_NAME
    budget = _budget_for(reasoning_depth)
    msgs = _normalize_messages(messages, _system_prompt(budget))
    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    if not get_model_config(use_model):
        return _completion_obj(cid, use_model, "错误: 所选模型未配置或不可用")

    search_calls = 0
    scrape_calls = 0
    tool_calls_used = 0
    for round_i in range(budget.max_tool_rounds + 1):
        message, tool_calls, err = await _resolve_round(msgs, use_model)
        if err:
            return _completion_obj(cid, use_model, f"[上游模型错误] {err}")

        if tool_calls and round_i < budget.max_tool_rounds:
            tool_calls = _limit_external_tools(
                tool_calls,
                search_calls,
                scrape_calls,
                tool_calls_used,
                budget,
            )
            if not tool_calls:
                msgs.append(
                    {
                        "role": "user",
                        "content": "外部搜索已达上限。请基于已有结果直接给出最终回答，不要再调用工具。",
                    }
                )
                try:
                    data = await _call_llm_with_model(msgs, model=use_model, with_tools=False)
                except Exception as e:
                    return _completion_obj(cid, use_model, f"[最终回答错误] {e}")
                content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
                return _completion_obj(cid, use_model, content)
            tool_calls_used += len(tool_calls)
            search_calls += sum(
                1
                for tc in tool_calls
                if (tc.get("function") or {}).get("name") == "web_search"
            )
            scrape_calls += sum(
                1
                for tc in tool_calls
                if (tc.get("function") or {}).get("name") == "scrape_url"
            )
            tool_events = await _apply_tool_calls(
                msgs,
                tool_calls,
                message,
                budget.max_search_results,
            )
            tool_calls_used, scrape_calls = _refund_failed_scrape_quota(
                tool_events,
                tool_calls_used,
                scrape_calls,
            )
            continue

        if tool_calls and round_i >= budget.max_tool_rounds:
            tool_calls = _limit_external_tools(
                tool_calls,
                search_calls,
                scrape_calls,
                tool_calls_used,
                budget,
            )
            if tool_calls:
                tool_calls_used += len(tool_calls)
                tool_events = await _apply_tool_calls(
                    msgs,
                    tool_calls,
                    message,
                    budget.max_search_results,
                )
                tool_calls_used, scrape_calls = _refund_failed_scrape_quota(
                    tool_events,
                    tool_calls_used,
                    scrape_calls,
                )
            msgs.append(
                {
                    "role": "user",
                    "content": "请基于已有工具结果直接给出最终回答，不要再调用工具。",
                }
            )
            try:
                data = await _call_llm_with_model(msgs, model=use_model, with_tools=False)
            except Exception as e:
                return _completion_obj(cid, use_model, f"[最终回答错误] {e}")
            content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            return _completion_obj(cid, use_model, content)

        content = (message or {}).get("content") or ""
        if _should_refine_search(
            content,
            search_calls,
            tool_calls_used,
            budget,
        ):
            msgs.append({"role": "assistant", "content": content})
            msgs.append(
                {
                    "role": "user",
                    "content": (
                        "当前结论仍未核实且搜索额度尚未用完。请换一个更精确、"
                        "明显不同的查询继续调用 web_search，优先 site: 官方域名。"
                    ),
                }
            )
            continue
        return _completion_obj(cid, use_model, content)

    return _completion_obj(cid, use_model, "")


def _completion_obj(cid: str, model: str, content: str) -> dict:
    return {
        "id": cid,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
