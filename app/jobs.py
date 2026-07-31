"""与浏览器连接解耦的后台聊天任务。"""
import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional

from . import attachments, users
from .agent import run_agent_stream

_tasks: Dict[str, asyncio.Task] = {}
_start_lock = asyncio.Lock()
_attachment_job_lock = asyncio.Lock()


def clean_trace(value: Any) -> List[dict]:
    """限制过程记录体积，避免轮询和 SQLite 被网页正文拖大。"""
    if not isinstance(value, list):
        return []
    cleaned = []
    for raw in value[-40:]:
        if not isinstance(raw, dict):
            continue
        item = {
            "id": str(raw.get("id") or uuid.uuid4().hex[:12])[:100],
            "kind": str(raw.get("kind") or "thinking")[:30],
            "title": str(raw.get("title") or "处理中")[:300],
            "state": str(raw.get("state") or "done")[:20],
        }
        for key, limit in (
            ("tool", 80),
            ("query", 500),
            ("url", 2000),
            ("source", 100),
            ("excerpt", 2000),
            ("error", 500),
            ("warning", 500),
        ):
            if raw.get(key):
                item[key] = str(raw[key])[:limit]
        results = []
        for result in (raw.get("results") or [])[:10]:
            if not isinstance(result, dict):
                continue
            results.append(
                {
                    "title": str(result.get("title") or "")[:300],
                    "url": str(result.get("url") or "")[:2000],
                    "snippet": str(result.get("snippet") or "")[:500],
                }
            )
        if results or "results" in raw:
            item["results"] = results
        cleaned.append(item)
    return cleaned


def _merge_trace_events(trace: List[dict], events: Any) -> List[dict]:
    incoming = clean_trace(events)
    if not incoming:
        return trace
    positions = {item.get("id"): index for index, item in enumerate(trace)}
    for event in incoming:
        event_id = event.get("id")
        if event_id in positions:
            trace[positions[event_id]] = event
        else:
            positions[event_id] = len(trace)
            trace.append(event)
    return clean_trace(trace)


def _status_label(status: dict) -> str:
    if status.get("message"):
        return str(status["message"])
    kind = status.get("type")
    tools = ", ".join(status.get("tools") or [])
    if kind == "thinking":
        return "思考中…"
    if kind == "tool_start":
        return f"调用工具：{tools}"
    if kind == "tool_running":
        return f"执行工具：{tools}"
    if kind == "tool_done":
        return f"工具完成：{tools}"
    return ""


def _persist_answer(
    user_id: int,
    conversation_id: str,
    title: str,
    messages: List[dict],
    content: str,
    trace: Optional[List[dict]] = None,
) -> None:
    conversation_messages = []
    for item in messages:
        if item.get("role") not in ("user", "assistant"):
            continue
        message = {"role": item["role"], "content": item.get("content") or ""}
        old_trace = clean_trace(item.get("trace"))
        if old_trace and item.get("role") == "assistant":
            message["trace"] = old_trace
        conversation_messages.append(message)
    if content:
        answer = {"role": "assistant", "content": content}
        current_trace = clean_trace(trace)
        if current_trace:
            answer["trace"] = current_trace
        conversation_messages.append(answer)
    users.save_conversation(
        user_id,
        conversation_id,
        title,
        conversation_messages[-100:],
    )


async def _run_job(
    job_id: str,
    user_id: int,
    conversation_id: str,
    title: str,
    messages: List[dict],
    model: str,
    attachment_records: List[dict],
    reasoning_depth: str,
) -> None:
    content = ""
    trace: List[dict] = []
    status_message = "思考中…"
    last_persist = 0.0
    attachment_lock_acquired = False
    users.update_chat_job(job_id, "running", content, status_message, trace=trace)
    try:
        model_messages = [
            {
                "role": item["role"],
                "content": item.get("content") or "",
            }
            for item in messages
            if item.get("role") in ("user", "assistant")
        ]
        if attachment_records:
            status_message = "正在等待附件任务…"
            users.update_chat_job(job_id, "running", content, status_message)
            await _attachment_job_lock.acquire()
            attachment_lock_acquired = True
            status_message = "正在读取附件…"
            users.update_chat_job(job_id, "running", content, status_message)
            model_messages = await asyncio.to_thread(
                attachments.build_model_messages,
                model_messages,
                attachment_records,
            )
        async for frame in run_agent_stream(
            model_messages,
            model=model,
            reasoning_depth=reasoning_depth,
        ):
            force_persist = False
            for line in frame.splitlines():
                if not line.startswith("data: "):
                    continue
                raw = line[6:].strip()
                if not raw or raw == "[DONE]":
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if payload.get("status"):
                    status = payload["status"]
                    status_message = _status_label(status)
                    trace_events = status.get("trace_events")
                    trace = _merge_trace_events(
                        trace,
                        trace_events,
                    )
                    force_persist = force_persist or bool(trace_events)
                for choice in payload.get("choices") or []:
                    content += (choice.get("delta") or {}).get("content") or ""

            now = time.monotonic()
            if force_persist or now - last_persist >= 0.75:
                users.update_chat_job(
                    job_id,
                    "running",
                    content,
                    status_message,
                    trace=trace,
                )
                last_persist = now

        if not content:
            content = "(无内容)"
        _persist_answer(
            user_id,
            conversation_id,
            title,
            messages,
            content,
            trace,
        )
        users.update_chat_job(job_id, "completed", content, "", trace=trace)
    except asyncio.CancelledError:
        stopped_content = content
        if stopped_content:
            stopped_content += "\n\n[已停止生成]"
        else:
            stopped_content = "[已停止生成]"
        _persist_answer(
            user_id,
            conversation_id,
            title,
            messages,
            stopped_content,
            trace,
        )
        users.update_chat_job(job_id, "stopped", stopped_content, "", trace=trace)
    except Exception as exc:
        error = f"{type(exc).__name__}: {str(exc)[:500]}"
        failed_content = content or f"[后台生成失败] {error}"
        _persist_answer(
            user_id,
            conversation_id,
            title,
            messages,
            failed_content,
            trace,
        )
        users.update_chat_job(
            job_id,
            "failed",
            failed_content,
            "",
            error,
            trace,
        )
    finally:
        if attachment_lock_acquired:
            _attachment_job_lock.release()
        if attachment_records:
            deleted = users.delete_attachments(
                user_id,
                [record["id"] for record in attachment_records],
            )
            await asyncio.to_thread(attachments.delete_files, deleted)
        _tasks.pop(job_id, None)


async def start_job(
    user_id: int,
    conversation_id: str,
    title: str,
    messages: List[dict],
    model: str,
    attachment_records: Optional[List[dict]] = None,
    reasoning_depth: str = "deep",
) -> dict:
    attachment_records = attachment_records or []
    async with _start_lock:
        active = users.get_active_chat_job(user_id)
        if active:
            return {"existing": True, **active}
        job_id = uuid.uuid4().hex
        users.create_chat_job(job_id, user_id, conversation_id, model)
        task = asyncio.create_task(
            _run_job(
                job_id,
                user_id,
                conversation_id,
                title,
                messages,
                model,
                attachment_records,
                reasoning_depth,
            )
        )
        _tasks[job_id] = task
        return {
            "id": job_id,
            "conversation_id": conversation_id,
            "model": model,
            "reasoning_depth": reasoning_depth,
            "status": "queued",
            "content": "",
            "status_message": "已提交，等待运行…",
            "trace": [],
            "error": "",
        }


def get_job(user_id: int, job_id: str) -> Optional[dict]:
    return users.get_chat_job(user_id, job_id)


def get_active_job(user_id: int) -> Optional[dict]:
    return users.get_active_chat_job(user_id)


async def stop_job(user_id: int, job_id: str) -> Optional[dict]:
    job = users.get_chat_job(user_id, job_id)
    if not job:
        return None
    task = _tasks.get(job_id)
    if task and not task.done():
        users.update_chat_job(
            job_id,
            "stopping",
            job.get("content") or "",
            "正在停止…",
        )
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    return users.get_chat_job(user_id, job_id)


async def stop_active_job_for_user(user_id: int) -> None:
    """管理员删除账号前停止该账号正在运行的上游请求。"""
    active = users.get_active_chat_job(user_id)
    if active:
        await stop_job(user_id, active["id"])
