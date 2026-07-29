"""极简 SQLite 用户存储。"""
import json
import sqlite3
import threading
from typing import Optional

import bcrypt

from .config import DB_PATH

_lock = threading.Lock()


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    with _lock:
        with _conn() as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TEXT DEFAULT (datetime('now'))
                )
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT NOT NULL,
                    user_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    messages_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                        DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    PRIMARY KEY (id, user_id)
                )
                """
            )
            c.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversations_user_updated
                ON conversations(user_id, updated_at DESC)
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_jobs (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    status TEXT NOT NULL,
                    content TEXT NOT NULL DEFAULT '',
                    status_message TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                        DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    updated_at TEXT NOT NULL
                        DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                )
                """
            )
            c.execute(
                """
                UPDATE chat_jobs
                SET status = 'interrupted',
                    error = '服务重启，后台任务已中断',
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE status IN ('queued', 'running', 'stopping')
                """
            )
            c.commit()


def user_count() -> int:
    with _conn() as c:
        row = c.execute("SELECT COUNT(*) AS n FROM users").fetchone()
        return int(row["n"])


def get_user(username: str) -> Optional[dict]:
    with _conn() as c:
        row = c.execute(
            "SELECT id, username, password_hash FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        return dict(row) if row else None


def create_user(username: str, password: str) -> dict:
    """创建用户，密码 bcrypt 加盐哈希。"""
    username = username.strip()
    if not username or len(username) < 2:
        raise ValueError("用户名至少 2 个字符")
    if not password or len(password) < 4:
        raise ValueError("密码至少 4 个字符")

    pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=10)).decode("utf-8")
    with _lock:
        with _conn() as c:
            try:
                cur = c.execute(
                    "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                    (username, pw_hash),
                )
                c.commit()
                return {"id": cur.lastrowid, "username": username}
            except sqlite3.IntegrityError:
                raise ValueError("用户名已存在")


def verify_password(username: str, password: str) -> Optional[dict]:
    user = get_user(username)
    if not user:
        return None
    ok = bcrypt.checkpw(password.encode("utf-8"), user["password_hash"].encode("utf-8"))
    if not ok:
        return None
    return {"id": user["id"], "username": user["username"]}


def list_conversations(user_id: int, limit: int = 10) -> list:
    with _conn() as c:
        rows = c.execute(
            """
            SELECT id, title, messages_json, updated_at
            FROM conversations
            WHERE user_id = ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (user_id, max(1, min(limit, 10))),
        ).fetchall()
    out = []
    for row in rows:
        try:
            messages = json.loads(row["messages_json"])
        except (json.JSONDecodeError, TypeError):
            messages = []
        out.append(
            {
                "id": row["id"],
                "title": row["title"],
                "messages": messages,
                "updated_at": row["updated_at"],
            }
        )
    return out


def save_conversation(
    user_id: int,
    conversation_id: str,
    title: str,
    messages: list,
) -> None:
    payload = json.dumps(messages, ensure_ascii=False)
    with _lock:
        with _conn() as c:
            c.execute(
                """
                INSERT INTO conversations
                    (id, user_id, title, messages_json, updated_at)
                VALUES
                    (?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                ON CONFLICT(id, user_id) DO UPDATE SET
                    title = excluded.title,
                    messages_json = excluded.messages_json,
                    updated_at = excluded.updated_at
                """,
                (conversation_id, user_id, title, payload),
            )
            old_rows = c.execute(
                """
                SELECT id FROM conversations
                WHERE user_id = ?
                ORDER BY updated_at DESC
                LIMIT -1 OFFSET 10
                """,
                (user_id,),
            ).fetchall()
            if old_rows:
                c.executemany(
                    "DELETE FROM conversations WHERE user_id = ? AND id = ?",
                    [(user_id, row["id"]) for row in old_rows],
                )
            c.commit()


def delete_conversation(user_id: int, conversation_id: str) -> None:
    with _lock:
        with _conn() as c:
            c.execute(
                "DELETE FROM conversations WHERE user_id = ? AND id = ?",
                (user_id, conversation_id),
            )
            c.commit()


def create_chat_job(
    job_id: str,
    user_id: int,
    conversation_id: str,
    model: str,
) -> None:
    with _lock:
        with _conn() as c:
            c.execute(
                """
                INSERT INTO chat_jobs
                    (id, user_id, conversation_id, model, status)
                VALUES (?, ?, ?, ?, 'queued')
                """,
                (job_id, user_id, conversation_id, model),
            )
            c.execute(
                """
                DELETE FROM chat_jobs
                WHERE user_id = ? AND id NOT IN (
                    SELECT id FROM chat_jobs
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                    LIMIT 30
                )
                """,
                (user_id, user_id),
            )
            c.commit()


def update_chat_job(
    job_id: str,
    status: str,
    content: str = "",
    status_message: str = "",
    error: str = "",
) -> None:
    with _lock:
        with _conn() as c:
            c.execute(
                """
                UPDATE chat_jobs
                SET status = ?, content = ?, status_message = ?, error = ?,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE id = ?
                """,
                (status, content, status_message, error, job_id),
            )
            c.commit()


def get_chat_job(user_id: int, job_id: str) -> Optional[dict]:
    with _conn() as c:
        row = c.execute(
            """
            SELECT id, conversation_id, model, status, content,
                   status_message, error, created_at, updated_at
            FROM chat_jobs
            WHERE user_id = ? AND id = ?
            """,
            (user_id, job_id),
        ).fetchone()
    return dict(row) if row else None


def get_active_chat_job(user_id: int) -> Optional[dict]:
    with _conn() as c:
        row = c.execute(
            """
            SELECT id, conversation_id, model, status, content,
                   status_message, error, created_at, updated_at
            FROM chat_jobs
            WHERE user_id = ? AND status IN ('queued', 'running', 'stopping')
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (user_id,),
        ).fetchone()
    return dict(row) if row else None
