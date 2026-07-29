"""
极致轻量 AI 聊天后端
- 登录 / 首次注册
- OpenAI 兼容 POST /v1/chat/completions（含 SSE 流式 + 多轮工具）
- 静态前端
"""
from pathlib import Path
from typing import Any, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import users
from .agent import run_agent_stream, run_agent_sync
from .auth import create_token, require_user
from .config import (
    HOST,
    JWT_EXPIRE_DAYS,
    MODEL_NAME,
    PORT,
    get_model_config,
    public_model_options,
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

    model = body.model or MODEL_NAME
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
        "default": MODEL_NAME,
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
