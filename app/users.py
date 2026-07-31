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
                    is_admin INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now'))
                )
                """
            )
            columns = {
                row["name"] for row in c.execute("PRAGMA table_info(users)").fetchall()
            }
            if "is_admin" not in columns:
                c.execute(
                    "ALTER TABLE users ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0"
                )
            # 兼容旧安装：若还没有管理员，将最早创建的账号提升为管理员。
            c.execute(
                """
                UPDATE users
                SET is_admin = 1
                WHERE id = (SELECT MIN(id) FROM users)
                  AND NOT EXISTS (SELECT 1 FROM users WHERE is_admin = 1)
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
                CREATE TABLE IF NOT EXISTS attachments (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    conversation_id TEXT NOT NULL,
                    original_name TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    original_size INTEGER NOT NULL,
                    processed_size INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                        DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                )
                """
            )
            c.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_attachments_user_conversation
                ON attachments(user_id, conversation_id, created_at)
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
            """
            SELECT id, username, password_hash, is_admin, created_at
            FROM users
            WHERE username = ?
            """,
            (username,),
        ).fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: int) -> Optional[dict]:
    with _conn() as c:
        row = c.execute(
            """
            SELECT id, username, password_hash, is_admin, created_at
            FROM users
            WHERE id = ?
            """,
            (user_id,),
        ).fetchone()
        return dict(row) if row else None


def list_users() -> list:
    with _conn() as c:
        rows = c.execute(
            """
            SELECT id, username, is_admin, created_at
            FROM users
            ORDER BY id ASC
            """
        ).fetchall()
    return [
        {
            "id": int(row["id"]),
            "username": row["username"],
            "is_admin": bool(row["is_admin"]),
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def create_user(
    username: str,
    password: str,
    *,
    is_admin: bool = False,
    max_users: Optional[int] = None,
) -> dict:
    """创建用户，密码 bcrypt 加盐哈希。"""
    username = username.strip()
    if not username or len(username) < 2:
        raise ValueError("用户名至少 2 个字符")
    if len(username) > 40:
        raise ValueError("用户名最多 40 个字符")
    if not password or len(password) < 4:
        raise ValueError("密码至少 4 个字符")
    if len(password.encode("utf-8")) > 72:
        raise ValueError("密码 UTF-8 编码后不能超过 72 字节")

    pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=10)).decode("utf-8")
    with _lock:
        with _conn() as c:
            count = int(c.execute("SELECT COUNT(*) FROM users").fetchone()[0])
            if max_users is not None and count >= max_users:
                raise ValueError(f"账号数量已达上限（{max_users} 个）")
            if is_admin and count > 0:
                raise ValueError("管理员账号已存在")
            try:
                cur = c.execute(
                    """
                    INSERT INTO users (username, password_hash, is_admin)
                    VALUES (?, ?, ?)
                    """,
                    (username, pw_hash, 1 if is_admin else 0),
                )
                c.commit()
                return {
                    "id": cur.lastrowid,
                    "username": username,
                    "is_admin": bool(is_admin),
                }
            except sqlite3.IntegrityError:
                raise ValueError("用户名已存在")


def verify_password(username: str, password: str) -> Optional[dict]:
    user = get_user(username)
    if not user:
        return None
    ok = bcrypt.checkpw(password.encode("utf-8"), user["password_hash"].encode("utf-8"))
    if not ok:
        return None
    return {
        "id": user["id"],
        "username": user["username"],
        "is_admin": bool(user["is_admin"]),
    }


def delete_managed_user(user_id: int) -> None:
    """删除普通账号及其全部隔离数据；管理员账号不可删除。"""
    with _lock:
        with _conn() as c:
            row = c.execute(
                "SELECT is_admin FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
            if not row:
                raise ValueError("账号不存在")
            if row["is_admin"]:
                raise ValueError("不能删除管理员账号")
            c.execute("DELETE FROM conversations WHERE user_id = ?", (user_id,))
            c.execute("DELETE FROM chat_jobs WHERE user_id = ?", (user_id,))
            c.execute("DELETE FROM attachments WHERE user_id = ?", (user_id,))
            c.execute("DELETE FROM users WHERE id = ?", (user_id,))
            c.commit()


def create_attachment(
    attachment_id: str,
    user_id: int,
    conversation_id: str,
    original_name: str,
    kind: str,
    media_type: str,
    stored_path: str,
    original_size: int,
    processed_size: int,
) -> dict:
    with _lock:
        with _conn() as c:
            c.execute(
                """
                INSERT INTO attachments (
                    id, user_id, conversation_id, original_name, kind,
                    media_type, stored_path, original_size, processed_size
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attachment_id,
                    user_id,
                    conversation_id,
                    original_name,
                    kind,
                    media_type,
                    stored_path,
                    int(original_size),
                    int(processed_size),
                ),
            )
            c.commit()
    return get_attachment(user_id, attachment_id)


def get_attachment(user_id: int, attachment_id: str) -> Optional[dict]:
    with _conn() as c:
        row = c.execute(
            """
            SELECT id, conversation_id, original_name, kind, media_type,
                   stored_path, original_size, processed_size, created_at
            FROM attachments
            WHERE user_id = ? AND id = ?
            """,
            (user_id, attachment_id),
        ).fetchone()
    return dict(row) if row else None


def get_attachments(
    user_id: int,
    conversation_id: str,
    attachment_ids: Optional[list] = None,
) -> list:
    with _conn() as c:
        if attachment_ids is None:
            rows = c.execute(
                """
                SELECT id, conversation_id, original_name, kind, media_type,
                       stored_path, original_size, processed_size, created_at
                FROM attachments
                WHERE user_id = ? AND conversation_id = ?
                ORDER BY created_at ASC
                """,
                (user_id, conversation_id),
            ).fetchall()
        else:
            unique_ids = list(dict.fromkeys(str(item) for item in attachment_ids))
            if not unique_ids:
                return []
            placeholders = ",".join("?" for _ in unique_ids)
            rows = c.execute(
                f"""
                SELECT id, conversation_id, original_name, kind, media_type,
                       stored_path, original_size, processed_size, created_at
                FROM attachments
                WHERE user_id = ? AND conversation_id = ?
                  AND id IN ({placeholders})
                ORDER BY created_at ASC
                """,
                (user_id, conversation_id, *unique_ids),
            ).fetchall()
    return [dict(row) for row in rows]


def get_all_attachments_for_user(user_id: int) -> list:
    with _conn() as c:
        rows = c.execute(
            """
            SELECT id, conversation_id, original_name, kind, media_type,
                   stored_path, original_size, processed_size, created_at
            FROM attachments
            WHERE user_id = ?
            ORDER BY created_at ASC
            """,
            (user_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def attachment_usage(user_id: int, conversation_id: str) -> dict:
    with _conn() as c:
        row = c.execute(
            """
            SELECT COUNT(*) AS count, COALESCE(SUM(original_size), 0) AS bytes
            FROM attachments
            WHERE user_id = ? AND conversation_id = ?
            """,
            (user_id, conversation_id),
        ).fetchone()
    return {"count": int(row["count"]), "bytes": int(row["bytes"])}


def delete_attachments(user_id: int, attachment_ids: list) -> list:
    unique_ids = list(dict.fromkeys(str(item) for item in attachment_ids))
    if not unique_ids:
        return []
    placeholders = ",".join("?" for _ in unique_ids)
    with _lock:
        with _conn() as c:
            rows = c.execute(
                f"""
                SELECT id, stored_path FROM attachments
                WHERE user_id = ? AND id IN ({placeholders})
                """,
                (user_id, *unique_ids),
            ).fetchall()
            c.execute(
                f"""
                DELETE FROM attachments
                WHERE user_id = ? AND id IN ({placeholders})
                """,
                (user_id, *unique_ids),
            )
            c.commit()
    return [dict(row) for row in rows]


def cleanup_expired_attachments(max_age_hours: int = 24) -> list:
    """返回已从数据库移除的附件路径，由调用方删除文件。"""
    hours = max(1, min(int(max_age_hours), 168))
    with _lock:
        with _conn() as c:
            rows = c.execute(
                """
                SELECT id, stored_path FROM attachments
                WHERE julianday(created_at) < julianday('now', ?)
                """,
                (f"-{hours} hours",),
            ).fetchall()
            if rows:
                c.executemany(
                    "DELETE FROM attachments WHERE id = ?",
                    [(row["id"],) for row in rows],
                )
            c.commit()
    return [dict(row) for row in rows]


def all_attachment_paths() -> set:
    with _conn() as c:
        rows = c.execute("SELECT stored_path FROM attachments").fetchall()
    return {str(row["stored_path"]) for row in rows}


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


def count_conversations(user_id: int) -> int:
    with _conn() as c:
        row = c.execute(
            "SELECT COUNT(*) AS n FROM conversations WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    return int(row["n"])


def list_conversation_summaries(
    user_id: int,
    limit: int = 10,
    offset: int = 0,
) -> list:
    """只返回历史选择器需要的元数据，避免一次传输全部对话正文。"""
    with _conn() as c:
        rows = c.execute(
            """
            SELECT id, title, updated_at
            FROM conversations
            WHERE user_id = ?
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
            """,
            (user_id, max(1, min(limit, 10)), max(0, int(offset))),
        ).fetchall()
    return [
        {
            "id": row["id"],
            "title": row["title"],
            "updated_at": row["updated_at"],
        }
        for row in rows
    ]


def get_conversation(user_id: int, conversation_id: str) -> Optional[dict]:
    with _conn() as c:
        row = c.execute(
            """
            SELECT id, title, messages_json, updated_at
            FROM conversations
            WHERE user_id = ? AND id = ?
            """,
            (user_id, conversation_id),
        ).fetchone()
    if not row:
        return None
    try:
        messages = json.loads(row["messages_json"])
    except (json.JSONDecodeError, TypeError):
        messages = []
    return {
        "id": row["id"],
        "title": row["title"],
        "messages": messages,
        "updated_at": row["updated_at"],
    }


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
                LIMIT -1 OFFSET 100
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


def cleanup_storage(
    max_conversations: int = 100,
    max_jobs: int = 30,
) -> dict:
    """限制每个账号的持久化数据，并在值得时回收 SQLite 文件空间。"""
    max_conversations = max(1, min(int(max_conversations), 100))
    max_jobs = max(1, min(int(max_jobs), 30))
    active_statuses = ("queued", "running", "stopping")
    removed_conversations = 0
    removed_jobs = 0
    vacuumed = False

    with _lock:
        with _conn() as c:
            user_ids = [
                int(row["id"])
                for row in c.execute("SELECT id FROM users").fetchall()
            ]

            # 先移除已不存在账号留下的非活动任务。
            cursor = c.execute(
                """
                DELETE FROM chat_jobs
                WHERE user_id NOT IN (SELECT id FROM users)
                  AND status NOT IN (?, ?, ?)
                """,
                active_statuses,
            )
            removed_jobs += max(0, cursor.rowcount)

            for user_id in user_ids:
                old_rows = c.execute(
                    """
                    SELECT id
                    FROM conversations
                    WHERE user_id = ?
                    ORDER BY updated_at DESC
                    LIMIT -1 OFFSET ?
                    """,
                    (user_id, max_conversations),
                ).fetchall()
                for row in old_rows:
                    conversation_id = row["id"]
                    active = c.execute(
                        """
                        SELECT 1 FROM chat_jobs
                        WHERE user_id = ? AND conversation_id = ?
                          AND status IN (?, ?, ?)
                        LIMIT 1
                        """,
                        (user_id, conversation_id, *active_statuses),
                    ).fetchone()
                    if active:
                        continue
                    cursor = c.execute(
                        """
                        DELETE FROM chat_jobs
                        WHERE user_id = ? AND conversation_id = ?
                        """,
                        (user_id, conversation_id),
                    )
                    removed_jobs += max(0, cursor.rowcount)
                    cursor = c.execute(
                        """
                        DELETE FROM conversations
                        WHERE user_id = ? AND id = ?
                        """,
                        (user_id, conversation_id),
                    )
                    removed_conversations += max(0, cursor.rowcount)

                # 已完成任务只保留最近 30 条；活动任务永不被维护任务误删。
                old_jobs = c.execute(
                    """
                    SELECT id FROM chat_jobs
                    WHERE user_id = ?
                      AND status NOT IN (?, ?, ?)
                    ORDER BY created_at DESC
                    LIMIT -1 OFFSET ?
                    """,
                    (user_id, *active_statuses, max_jobs),
                ).fetchall()
                if old_jobs:
                    cursor = c.executemany(
                        "DELETE FROM chat_jobs WHERE user_id = ? AND id = ?",
                        [(user_id, row["id"]) for row in old_jobs],
                    )
                    removed_jobs += max(0, cursor.rowcount)

            # 清除已没有对应窗口的非活动任务。
            cursor = c.execute(
                """
                DELETE FROM chat_jobs
                WHERE status NOT IN (?, ?, ?)
                  AND NOT EXISTS (
                    SELECT 1 FROM conversations
                    WHERE conversations.user_id = chat_jobs.user_id
                      AND conversations.id = chat_jobs.conversation_id
                  )
                """,
                active_statuses,
            )
            removed_jobs += max(0, cursor.rowcount)
            c.commit()

            page_count = int(c.execute("PRAGMA page_count").fetchone()[0])
            free_pages = int(c.execute("PRAGMA freelist_count").fetchone()[0])
            page_size = int(c.execute("PRAGMA page_size").fetchone()[0])
            # 文件至少 1 MiB 且超过 25% 是空页时才压缩，避免小鸡频繁 VACUUM。
            if (
                page_count > 0
                and page_count * page_size >= 1024 * 1024
                and free_pages / page_count >= 0.25
            ):
                c.execute("VACUUM")
                vacuumed = True

    return {
        "removed_conversations": removed_conversations,
        "removed_jobs": removed_jobs,
        "vacuumed": vacuumed,
    }


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
