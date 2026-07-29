"""环境变量配置 — 全部可覆盖，默认值面向极简部署。"""
import os
import json
import secrets
import threading
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# OpenAI 兼容 API
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.groq.com/openai/v1").rstrip("/")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")

# DeepSeek（OpenAI 兼容接口）
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

PROVIDERS_FILE = DATA_DIR / "providers.json"
_providers_lock = threading.Lock()

# 首次运行兼容 .env；前端保存后由 providers.json 覆盖对应提供商。
_ENV_PROVIDERS = (
    {
        "id": MODEL_NAME,
        "provider_id": "groq",
        "provider_label": "Groq",
        "provider": "groq",
        "api_base": OPENAI_API_BASE,
        "api_key": OPENAI_API_KEY,
        "models": [MODEL_NAME],
    },
    {
        "provider_id": "deepseek",
        "provider_label": "DeepSeek",
        "provider": "deepseek",
        "api_base": DEEPSEEK_API_BASE,
        "api_key": DEEPSEEK_API_KEY,
        "models": ["deepseek-v4-flash", "deepseek-v4-pro"],
    },
)


def _configured_providers():
    providers = {
        item["provider_id"]: dict(item)
        for item in _ENV_PROVIDERS
        if item.get("api_key")
    }
    try:
        payload = json.loads(PROVIDERS_FILE.read_text())
        for provider_id in payload.get("disabled_providers") or []:
            providers.pop(str(provider_id), None)
        for item in payload.get("providers") or []:
            provider_id = str(item.get("provider_id") or "").strip()
            if provider_id and item.get("api_key"):
                providers[provider_id] = item
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return list(providers.values())


def provider_settings_public():
    known = {
        "groq": {
            "provider_id": "groq",
            "provider_label": "Groq",
            "api_base": "https://api.groq.com/openai/v1",
        },
        "deepseek": {
            "provider_id": "deepseek",
            "provider_label": "DeepSeek",
            "api_base": "https://api.deepseek.com",
        },
    }
    for item in _configured_providers():
        provider_id = item["provider_id"]
        known[provider_id] = {
            "provider_id": provider_id,
            "provider_label": item.get("provider_label") or provider_id,
            "api_base": item.get("api_base") or "",
            "configured": True,
            "api_key_masked": "••••••••" + str(item.get("api_key") or "")[-4:],
            "models": item.get("models") or [],
        }
    return list(known.values())


def get_provider_secret(provider_id: str):
    for item in _configured_providers():
        if item.get("provider_id") == provider_id:
            return item
    return None


def save_provider_config(provider: dict):
    with _providers_lock:
        try:
            payload = json.loads(PROVIDERS_FILE.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            payload = {"providers": [], "disabled_providers": []}
        providers = [
            item
            for item in payload.get("providers") or []
            if item.get("provider_id") != provider["provider_id"]
        ]
        providers.append(provider)
        disabled = [
            item
            for item in payload.get("disabled_providers") or []
            if item != provider["provider_id"]
        ]
        tmp = PROVIDERS_FILE.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(
                {"providers": providers, "disabled_providers": disabled},
                ensure_ascii=False,
                indent=2,
            )
        )
        os.chmod(str(tmp), 0o600)
        tmp.replace(PROVIDERS_FILE)


def delete_provider_config(provider_id: str):
    with _providers_lock:
        try:
            payload = json.loads(PROVIDERS_FILE.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            payload = {"providers": [], "disabled_providers": []}
        providers = [
            item
            for item in payload.get("providers") or []
            if item.get("provider_id") != provider_id
        ]
        disabled = list(dict.fromkeys(
            [*(payload.get("disabled_providers") or []), provider_id]
        ))
        tmp = PROVIDERS_FILE.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(
                {"providers": providers, "disabled_providers": disabled},
                ensure_ascii=False,
                indent=2,
            )
        )
        os.chmod(str(tmp), 0o600)
        tmp.replace(PROVIDERS_FILE)


def _selector_id(provider_id: str, model_id: str) -> str:
    return f"{provider_id}::{model_id}"


def get_model_config(model_id=None):
    selected = model_id or default_model_id()
    for provider in _configured_providers():
        for upstream_model in provider.get("models") or []:
            selector = _selector_id(provider["provider_id"], upstream_model)
            # 接受旧版未带 provider 前缀的已保存选择。
            if selected in (selector, upstream_model):
                return {
                    "id": selector,
                    "model_id": upstream_model,
                    "label": f"{provider.get('provider_label') or provider['provider_id']} · {upstream_model}",
                    "provider": provider.get("provider") or provider["provider_id"],
                    "api_base": str(provider["api_base"]).rstrip("/"),
                    "api_key": provider["api_key"],
                }
    return None


def public_model_options():
    options = []
    for provider in _configured_providers():
        for model_id in provider.get("models") or []:
            options.append(
                {
                    "id": _selector_id(provider["provider_id"], model_id),
                    "object": "model",
                    "owned_by": provider["provider_id"],
                    "label": f"{provider.get('provider_label') or provider['provider_id']} · {model_id}",
                }
            )
    return options


def default_model_id():
    options = public_model_options()
    return options[0]["id"] if options else MODEL_NAME


# 本地搜索 / 抓取
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://127.0.0.1:8080").rstrip("/")
SCRAPER_URL = os.getenv("SCRAPER_URL", "http://127.0.0.1:3002").rstrip("/")

# Agent 限制
MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS", "10"))
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "10"))

# 认证
_jwt_file = DATA_DIR / ".jwt_secret"
if os.getenv("JWT_SECRET"):
    JWT_SECRET = os.getenv("JWT_SECRET")
elif _jwt_file.exists():
    JWT_SECRET = _jwt_file.read_text().strip()
else:
    JWT_SECRET = secrets.token_hex(32)
    _jwt_file.write_text(JWT_SECRET)

JWT_EXPIRE_DAYS = int(os.getenv("JWT_EXPIRE_DAYS", "60"))
JWT_ALGORITHM = "HS256"

# 用户库
DB_PATH = Path(os.getenv("DB_PATH", DATA_DIR / "users.db"))

# HTTP 超时（秒）— 工具侧默认偏短，避免串行卡死
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "8"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "180"))

# 服务
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
