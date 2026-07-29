"""多轮 tool-calling agent 循环 + SSE 流式输出。

针对 Groq/Llama 常见的 tool_use_failed（模型输出 <function=...> XML）做了容错解析。
"""
import asyncio
import json
import logging
import re
import time
import uuid
from datetime import date
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx

from .config import LLM_TIMEOUT, MAX_TOOL_ROUNDS, MODEL_NAME, get_model_config
from .tools import TOOLS_SCHEMA, execute_tool

logger = logging.getLogger("lite-ai-chat.agent")

_TOOL_EXEC_TIMEOUT = 15.0
_WEB_SEARCH_TIMEOUT = 60.0
_HEARTBEAT_INTERVAL = 5.0
_MAX_EXTERNAL_SEARCHES = 5

SYSTEM_PROMPT = f"""你是一个有用的 AI 助手，可以使用工具获取最新网络信息。

当前日期：{date.today().isoformat()}。

工具使用原则：
1. 需要实时或外部信息时，调用 web_search。它是独立于当前模型的外部搜索服务。
2. 每轮搜索后先判断结果是否真正回答了问题；若无关、不完整、互相冲突或缺少权威来源，不要仓促回答，要换一个明显不同且更精确的查询继续搜索。
3. 改写查询时可加入准确年份、关键实体、官方域名或 site: 限定。最多进行 {_MAX_EXTERNAL_SEARCHES} 次 web_search；不要重复完全相同的查询。
4. 找到关键结果后，可用 scrape_url 阅读具体页面。优先政府、学校、机构官网等一手来源。
5. 回答具体日期时，正文必须明确说明该日期对应用户所问事件；网页的“发布时间/更新时间”不能当成开学、放假等事件日期。若抓到的只是列表页、图片页或正文没有答案，应继续换查询搜索，不能猜测。
6. 不要编造链接或事实；最终回答必须基于工具结果，并附上实际来源链接。达到搜索上限仍无可靠答案时，要如实说明未核实到。
7. 用简洁中文回答（除非用户使用其他语言）。
8. 只通过 API 的 function calling 调用工具，不要输出 XML 或伪代码。
9. “今年”“今天”等相对日期必须以上面的当前日期为准。"""


def _llm_round_timeout(model: str) -> float:
    config = get_model_config(model)
    provider_cap = 90.0 if config and config["provider"] == "deepseek" else 30.0
    return max(5.0, min(float(LLM_TIMEOUT), provider_cap))

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


def _normalize_messages(messages: List[dict]) -> List[dict]:
    """确保有 system 提示，并清理 Groq 不喜欢的字段。"""
    out = []
    for m in messages:
        item = {k: v for k, v in dict(m).items() if v is not None}
        # content 为空字符串时保留；tool 消息必须有 content
        if item.get("role") == "assistant" and "content" not in item:
            if item.get("tool_calls"):
                item["content"] = ""
        out.append(item)
    if not any(m.get("role") == "system" for m in out):
        out.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
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
            raise RuntimeError(f"LLM HTTP {r.status_code}: {r.text[:800]}")
        return r.json()


async def _apply_tool_calls(
    msgs: List[dict],
    tool_calls: List[dict],
    assistant_message: Optional[dict] = None,
) -> List[str]:
    """把 assistant tool_calls + tool 结果追加到 msgs，返回工具名列表。"""
    names = []
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
        tc_id = tc.get("id") or f"call_{uuid.uuid4().hex[:8]}"
        names.append(name or "?")
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
        msgs.append(
            {
                "role": "tool",
                "tool_call_id": tc_id,
                "content": result,
            }
        )
    return names


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


def _limit_external_searches(tool_calls: List[dict], used: int) -> List[dict]:
    """每个回答最多允许三次外部搜索，抓取工具不计入此额度。"""
    remaining = max(0, _MAX_EXTERNAL_SEARCHES - used)
    kept = []
    for tc in tool_calls:
        name = (tc.get("function") or {}).get("name")
        if name == "web_search":
            if remaining <= 0:
                continue
            remaining -= 1
        kept.append(tc)
    return kept


def _should_refine_search(content: str, search_calls: int) -> bool:
    """模型准备以“没查到”收尾时，若还有额度则要求改写查询继续检索。"""
    return (
        0 < search_calls < _MAX_EXTERNAL_SEARCHES
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


async def run_agent_stream(messages: List[dict], model: Optional[str] = None) -> AsyncIterator[str]:
    use_model = model or MODEL_NAME
    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    msgs = _normalize_messages(messages)

    if not get_model_config(use_model):
        yield _sse(_chunk(content="错误: 所选模型未配置或不可用", finish_reason="stop", model=use_model, cid=cid))
        yield "data: [DONE]\n\n"
        return

    # 立刻推送心跳，避免前端首包前长时间空白
    boot = _chunk(model=use_model, cid=cid)
    boot["status"] = {"type": "thinking", "message": "思考中…"}
    yield _sse(boot)

    seen_tool_calls = set()
    search_calls = 0
    for round_i in range(MAX_TOOL_ROUNDS + 1):
        # 上游偶尔会长时间不返回；等待期间持续发 SSE 心跳，并设置整轮硬上限。
        resolve_task = asyncio.create_task(_resolve_round(msgs, use_model))
        started = time.monotonic()
        round_timeout = _llm_round_timeout(use_model)
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
        if err:
            yield _sse(_chunk(content=f"\n[上游模型错误] {err}", finish_reason="stop", model=use_model, cid=cid))
            yield "data: [DONE]\n\n"
            return

        if tool_calls and round_i < MAX_TOOL_ROUNDS:
            new_tool_calls = [
                tc for tc in tool_calls if _tool_signature(tc) not in seen_tool_calls
            ]
            new_tool_calls = _limit_external_searches(new_tool_calls, search_calls)
            if not new_tool_calls:
                logger.warning("repeated/excess tool call stopped at round %s", round_i + 1)
                msgs.append(
                    {
                        "role": "user",
                        "content": "工具调用已重复或外部搜索已达上限。请立即基于已有结果给出最终回答，不要再调用工具。",
                    }
                )
                status = _chunk(model=use_model, cid=cid)
                status["status"] = {
                    "type": "tool_done",
                    "tools": [tc.get("function", {}).get("name", "?") for tc in tool_calls],
                    "round": round_i + 1,
                    "message": "检测到重复工具调用，正在生成最终回答…",
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
            search_calls += sum(
                1
                for tc in tool_calls
                if (tc.get("function") or {}).get("name") == "web_search"
            )
            seen_tool_calls.update(_tool_signature(tc) for tc in tool_calls)
            names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
            status = _chunk(model=use_model, cid=cid)
            status["status"] = {"type": "tool_start", "tools": names, "round": round_i + 1}
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

            await _apply_tool_calls(msgs, tool_calls, message)

            done = _chunk(model=use_model, cid=cid)
            done["status"] = {"type": "tool_done", "tools": names, "round": round_i + 1}
            yield _sse(done)
            continue

        # 达上限仍想调工具：执行最后一轮工具后强制总结
        if tool_calls and round_i >= MAX_TOOL_ROUNDS:
            tool_calls = _limit_external_searches(tool_calls, search_calls)
            if tool_calls:
                await _apply_tool_calls(msgs, tool_calls, message)
            msgs.append(
                {
                    "role": "user",
                    "content": "请基于已有工具结果直接给出最终回答，不要再调用工具。",
                }
            )
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
            if _should_refine_search(final_content, search_calls):
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
                }
                yield _sse(status)
                continue
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


async def run_agent_sync(messages: List[dict], model: Optional[str] = None) -> dict:
    use_model = model or MODEL_NAME
    msgs = _normalize_messages(messages)
    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    if not get_model_config(use_model):
        return _completion_obj(cid, use_model, "错误: 所选模型未配置或不可用")

    search_calls = 0
    for round_i in range(MAX_TOOL_ROUNDS + 1):
        message, tool_calls, err = await _resolve_round(msgs, use_model)
        if err:
            return _completion_obj(cid, use_model, f"[上游模型错误] {err}")

        if tool_calls and round_i < MAX_TOOL_ROUNDS:
            tool_calls = _limit_external_searches(tool_calls, search_calls)
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
            search_calls += sum(
                1
                for tc in tool_calls
                if (tc.get("function") or {}).get("name") == "web_search"
            )
            await _apply_tool_calls(msgs, tool_calls, message)
            continue

        if tool_calls and round_i >= MAX_TOOL_ROUNDS:
            tool_calls = _limit_external_searches(tool_calls, search_calls)
            if tool_calls:
                await _apply_tool_calls(msgs, tool_calls, message)
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
        if _should_refine_search(content, search_calls):
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
