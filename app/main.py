"""
极致轻量 AI 聊天后端
- 登录 / 首次注册
- OpenAI 兼容 POST /v1/chat/completions（含 SSE 流式 + 多轮工具）
- 静态前端
"""
from pathlib import Path
import re
from typing import Any, List, Optional
from urllib.parse import urlparse

import httpx
from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import jobs, users
from .agent import run_agent_stream, run_agent_sync
from .auth import create_token, require_user
from .config import (
    HOST,
    JWT_EXPIRE_DAYS,
    PORT,
    default_model_id,
    delete_provider_config,
    get_model_config,
    get_provider_secret,
    provider_settings_public,
    public_model_options,
    save_provider_config,
)

users.init_db()

app = FastAPI(title="Lite AI Chat", version="1.0.0", docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


# ---------- 请求体 ----------
class AuthBody(BaseModel):
    username: str
    password: str


class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[Any] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    # 忽略多余字段（tools 等由后端内置）
    max_tokens: Optional[int] = None


class ChatJobBody(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage] = Field(min_length=1, max_length=100)
    conversation_id: str = Field(min_length=8, max_length=64)
    title: str = Field(default="新对话", max_length=100)


class ProviderBody(BaseModel):
    provider_id: str = Field(min_length=1, max_length=40, pattern=r"^[a-zA-Z0-9_-]+$")
    provider_label: str = Field(min_length=1, max_length=80)
    api_base: str = Field(min_length=8, max_length=500)
    api_key: str = Field(default="", max_length=1000)
    selected_models: List[str] = Field(default_factory=list, max_length=500)


class ConversationBody(BaseModel):
    title: str = Field(default="新对话", max_length=100)
    messages: List[ChatMessage] = Field(default_factory=list, max_length=100)


def _provider_credentials(body: ProviderBody):
    parsed = urlparse(body.api_base)
    if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.username:
        raise HTTPException(status_code=400, detail="API 地址必须是有效的 http/https URL")
    api_key = body.api_key.strip()
    existing = get_provider_secret(body.provider_id)
    if not api_key and existing:
        api_key = existing.get("api_key") or ""
    if not api_key:
        raise HTTPException(status_code=400, detail="请输入 API Key")
    return body.api_base.rstrip("/"), api_key


async def _fetch_provider_models(api_base: str, api_key: str):
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(25.0, connect=5.0),
            follow_redirects=True,
        ) as client:
            response = await client.get(
                f"{api_base}/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"无法连接模型接口：{type(exc).__name__}",
        )
    if response.status_code >= 400:
        try:
            payload = response.json()
            message = (payload.get("error") or {}).get("message") or payload.get("detail")
        except Exception:
            message = ""
        raise HTTPException(
            status_code=400,
            detail=f"API 测试失败（HTTP {response.status_code}）"
            + (f"：{str(message)[:300]}" if message else ""),
        )
    try:
        payload = response.json()
    except Exception:
        raise HTTPException(status_code=502, detail="模型接口未返回有效 JSON")
    models = []
    for item in payload.get("data") or []:
        model_id = str(item.get("id") or "").strip()
        if model_id and model_id not in models:
            models.append(model_id)
    if not models:
        raise HTTPException(status_code=400, detail="API 测试成功，但未读取到模型列表")
    return sorted(models)[:500]


def _clean_model_ids(values: List[str]):
    models = list(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))
    if any(len(item) > 300 for item in models):
        raise HTTPException(status_code=400, detail="模型名称过长")
    return models


async def _test_provider_model(api_base: str, api_key: str, model_id: str):
    """对未出现在 /models 的手填模型发起最小聊天请求验证。"""
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(45.0, connect=5.0),
            follow_redirects=True,
        ) as client:
            response = await client.post(
                f"{api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": "Reply OK"}],
                    "max_tokens": 1,
                    "stream": False,
                },
            )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"手填模型 {model_id} 连接失败：{type(exc).__name__}",
        )
    if response.status_code >= 400:
        try:
            payload = response.json()
            message = (payload.get("error") or {}).get("message") or payload.get("detail")
        except Exception:
            message = ""
        raise HTTPException(
            status_code=400,
            detail=f"手填模型 {model_id} 测试失败（HTTP {response.status_code}）"
            + (f"：{str(message)[:300]}" if message else ""),
        )


# ---------- 认证 ----------
@app.get("/api/auth/status")
def auth_status():
    """前端据此判断是否需要首次注册。"""
    return {
        "has_users": users.user_count() > 0,
        "jwt_expire_days": JWT_EXPIRE_DAYS,
    }


@app.post("/api/auth/register")
def register(body: AuthBody, response: Response):
    """
    注册：
    - 若尚无任何用户 → 允许创建第一个管理员账号
    - 若已有用户 → 拒绝（极简单机，不开放公开注册）
    """
    if users.user_count() > 0:
        raise HTTPException(status_code=403, detail="已初始化，禁止公开注册")
    try:
        u = users.create_user(body.username, body.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    token = create_token(u["id"], u["username"])
    response.set_cookie(
        key="token",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=JWT_EXPIRE_DAYS * 86400,
        path="/",
    )
    return {"token": token, "username": u["username"], "expire_days": JWT_EXPIRE_DAYS}


@app.post("/api/auth/login")
def login(body: AuthBody, response: Response):
    u = users.verify_password(body.username, body.password)
    if not u:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = create_token(u["id"], u["username"])
    response.set_cookie(
        key="token",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=JWT_EXPIRE_DAYS * 86400,
        path="/",
    )
    return {"token": token, "username": u["username"], "expire_days": JWT_EXPIRE_DAYS}


@app.post("/api/auth/logout")
def logout(response: Response):
    response.delete_cookie("token", path="/")
    return {"ok": True}


@app.get("/api/auth/me")
def me(user: dict = Depends(require_user)):
    return {"id": user["id"], "username": user["username"]}


# ---------- API 与模型设置 ----------
@app.get("/api/settings/providers")
def provider_settings(user: dict = Depends(require_user)):
    return {"providers": provider_settings_public()}


@app.post("/api/settings/providers/test")
async def test_provider(body: ProviderBody, user: dict = Depends(require_user)):
    api_base, api_key = _provider_credentials(body)
    manual_models = _clean_model_ids(body.selected_models)
    models_warning = ""
    try:
        models = await _fetch_provider_models(api_base, api_key)
    except HTTPException as exc:
        if not manual_models:
            raise
        models = []
        models_warning = str(exc.detail)
    if len(manual_models) > 20:
        raise HTTPException(status_code=400, detail="一次最多测试 20 个手填模型")
    for model_id in manual_models:
        await _test_provider_model(api_base, api_key, model_id)
    combined = list(dict.fromkeys([*models, *manual_models]))
    return {
        "ok": True,
        "models": combined,
        "manual_tested": manual_models,
        "models_warning": models_warning,
    }


@app.post("/api/settings/providers")
async def save_provider(body: ProviderBody, user: dict = Depends(require_user)):
    api_base, api_key = _provider_credentials(body)
    selected = _clean_model_ids(body.selected_models)
    if not selected:
        raise HTTPException(status_code=400, detail="请至少选择一个模型")
    try:
        available_models = await _fetch_provider_models(api_base, api_key)
    except HTTPException:
        available_models = []
    manual_models = [model for model in selected if model not in available_models]
    if len(manual_models) > 20:
        raise HTTPException(status_code=400, detail="一次最多验证 20 个手填模型")
    for model_id in manual_models:
        await _test_provider_model(api_base, api_key, model_id)
    save_provider_config(
        {
            "provider_id": body.provider_id,
            "provider_label": body.provider_label.strip(),
            "provider": "deepseek" if body.provider_id == "deepseek" else body.provider_id,
            "api_base": api_base,
            "api_key": api_key,
            "models": selected,
        }
    )
    return {"ok": True, "models": public_model_options()}


@app.delete("/api/settings/providers/{provider_id}")
def delete_provider(provider_id: str, user: dict = Depends(require_user)):
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,40}", provider_id):
        raise HTTPException(status_code=400, detail="无效的提供商标识")
    delete_provider_config(provider_id)
    return {"ok": True, "models": public_model_options()}


# ---------- 最近 10 个对话 ----------
@app.get("/api/conversations")
def conversations(user: dict = Depends(require_user)):
    return {"data": users.list_conversations(user["id"], limit=10)}


@app.put("/api/conversations/{conversation_id}")
def save_conversation(
    conversation_id: str,
    body: ConversationBody,
    user: dict = Depends(require_user),
):
    if not re.fullmatch(r"[a-zA-Z0-9_-]{8,64}", conversation_id):
        raise HTTPException(status_code=400, detail="无效的对话标识")
    messages = []
    total_chars = 0
    for message in body.messages[-100:]:
        if message.role not in ("user", "assistant"):
            continue
        content = message.content or ""
        total_chars += len(content)
        if total_chars > 1_000_000:
            raise HTTPException(status_code=413, detail="对话内容过大")
        messages.append({"role": message.role, "content": content})
    title = body.title.strip()[:100] or "新对话"
    users.save_conversation(user["id"], conversation_id, title, messages)
    return {"ok": True}


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(
    conversation_id: str,
    user: dict = Depends(require_user),
):
    if not re.fullmatch(r"[a-zA-Z0-9_-]{8,64}", conversation_id):
        raise HTTPException(status_code=400, detail="无效的对话标识")
    users.delete_conversation(user["id"], conversation_id)
    return {"ok": True}


# ---------- 与页面连接解耦的后台生成 ----------
@app.post("/api/chat/jobs")
async def create_chat_job(body: ChatJobBody, user: dict = Depends(require_user)):
    if not re.fullmatch(r"[a-zA-Z0-9_-]{8,64}", body.conversation_id):
        raise HTTPException(status_code=400, detail="无效的对话标识")
    clean = []
    total_chars = 0
    for message in body.messages:
        if message.role not in ("user", "assistant"):
            continue
        content = message.content or ""
        total_chars += len(content)
        if total_chars > 1_000_000:
            raise HTTPException(status_code=413, detail="对话内容过大")
        clean.append({"role": message.role, "content": content})
    if not clean:
        raise HTTPException(status_code=400, detail="messages 不能为空")
    model = body.model or default_model_id()
    if not get_model_config(model):
        raise HTTPException(status_code=400, detail=f"模型不可用: {model}")
    title = body.title.strip()[:100] or "新对话"
    return await jobs.start_job(
        user["id"],
        body.conversation_id,
        title,
        clean,
        model,
    )


@app.get("/api/chat/jobs/active")
def active_chat_job(user: dict = Depends(require_user)):
    return {"job": jobs.get_active_job(user["id"])}


@app.get("/api/chat/jobs/{job_id}")
def chat_job(job_id: str, user: dict = Depends(require_user)):
    if not re.fullmatch(r"[a-f0-9]{32}", job_id):
        raise HTTPException(status_code=400, detail="无效的任务标识")
    job = jobs.get_job(user["id"], job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    return job


@app.post("/api/chat/jobs/{job_id}/stop")
async def stop_chat_job(job_id: str, user: dict = Depends(require_user)):
    if not re.fullmatch(r"[a-f0-9]{32}", job_id):
        raise HTTPException(status_code=400, detail="无效的任务标识")
    job = await jobs.stop_job(user["id"], job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    return job


# ---------- OpenAI 兼容接口 ----------
@app.post("/v1/chat/completions")
async def chat_completions(body: ChatRequest, user: dict = Depends(require_user)):
    messages = [m.model_dump(exclude_none=True) for m in body.messages]
    # 过滤非法 role
    clean = []
    for m in messages:
        if m.get("role") in ("system", "user", "assistant", "tool"):
            clean.append(m)
    if not clean:
        raise HTTPException(status_code=400, detail="messages 不能为空")

    model = body.model or default_model_id()
    if not get_model_config(model):
        raise HTTPException(status_code=400, detail=f"模型不可用: {model}")

    if body.stream:
        async def gen():
            # 逐块 yield，配合 X-Accel-Buffering 避免中间层缓冲
            async for chunk in run_agent_stream(clean, model=model):
                yield chunk

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    result = await run_agent_sync(clean, model=model)
    return JSONResponse(result)


@app.get("/v1/models")
def list_models(user: dict = Depends(require_user)):
    return {
        "object": "list",
        "default": default_model_id(),
        "data": public_model_options(),
    }


@app.get("/health")
def health():
    return {"ok": True}


# ---------- 静态前端 ----------
if STATIC_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=str(STATIC_DIR)), name="assets")


@app.get("/")
def index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse({"message": "Lite AI Chat API", "docs": "见 README"})


def main():
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=PORT,
        workers=1,
        log_level="info",
        # 低内存：关闭 reload
        reload=False,
    )


if __name__ == "__main__":
    main()
