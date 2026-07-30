"""与浏览器连接解耦的后台聊天任务。"""
import asyncio
import json
import time
import uuid
from typing import Dict, List, Optional

from . import users
from .agent import run_agent_stream

_tasks: Dict[str, asyncio.Task] = {}
_start_lock = asyncio.Lock()


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
) -> None:
    conversation_messages = [
        {"role": item["role"], "content": item.get("content") or ""}
        for item in messages
        if item.get("role") in ("user", "assistant")
    ]
    if content:
        conversation_messages.append({"role": "assistant", "content": content})
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
) -> None:
    content = ""
    status_message = "思考中…"
    last_persist = 0.0
    users.update_chat_job(job_id, "running", content, status_message)
    try:
        async for frame in run_agent_stream(messages, model=model):
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
                    status_message = _status_label(payload["status"])
                for choice in payload.get("choices") or []:
                    content += (choice.get("delta") or {}).get("content") or ""

            now = time.monotonic()
            if now - last_persist >= 0.75:
                users.update_chat_job(
                    job_id,
                    "running",
                    content,
                    status_message,
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
        )
        users.update_chat_job(job_id, "completed", content, "")
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
        )
        users.update_chat_job(job_id, "stopped", stopped_content, "")
    except Exception as exc:
        error = f"{type(exc).__name__}: {str(exc)[:500]}"
        failed_content = content or f"[后台生成失败] {error}"
        _persist_answer(
            user_id,
            conversation_id,
            title,
            messages,
            failed_content,
        )
        users.update_chat_job(
            job_id,
            "failed",
            failed_content,
            "",
            error,
        )
    finally:
        _tasks.pop(job_id, None)


async def start_job(
    user_id: int,
    conversation_id: str,
    title: str,
    messages: List[dict],
    model: str,
) -> dict:
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
            )
        )
        _tasks[job_id] = task
        return {
            "id": job_id,
            "conversation_id": conversation_id,
            "model": model,
            "status": "queued",
            "content": "",
            "status_message": "已提交，等待运行…",
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
