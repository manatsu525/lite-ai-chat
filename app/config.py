"""环境变量配置 — 全部可覆盖，默认值面向极简部署。"""
import os
import secrets
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

# 只允许使用这里声明的模型，避免客户端借服务端密钥调用任意模型。
MODEL_OPTIONS = (
    {
        "id": MODEL_NAME,
        "label": "Llama 3.3 70B (Groq)",
        "provider": "groq",
        "api_base": OPENAI_API_BASE,
        "api_key": OPENAI_API_KEY,
    },
    {
        "id": "deepseek-v4-flash",
        "label": "DeepSeek V4 Flash",
        "provider": "deepseek",
        "api_base": DEEPSEEK_API_BASE,
        "api_key": DEEPSEEK_API_KEY,
    },
    {
        "id": "deepseek-v4-pro",
        "label": "DeepSeek V4 Pro",
        "provider": "deepseek",
        "api_base": DEEPSEEK_API_BASE,
        "api_key": DEEPSEEK_API_KEY,
    },
)


def get_model_config(model_id=None):
    selected = model_id or MODEL_NAME
    for option in MODEL_OPTIONS:
        if option["id"] == selected and option["api_key"]:
            return option
    return None


def public_model_options():
    return [
        {
            "id": option["id"],
            "object": "model",
            "owned_by": option["provider"],
            "label": option["label"],
        }
        for option in MODEL_OPTIONS
        if option["api_key"]
    ]


# 本地搜索 / 抓取
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://127.0.0.1:8080").rstrip("/")
SCRAPER_URL = os.getenv("SCRAPER_URL", "http://127.0.0.1:3002").rstrip("/")

# Agent 限制
MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS", "5"))
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "5"))

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
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "90"))

# 服务
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
