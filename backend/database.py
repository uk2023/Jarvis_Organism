# -*- coding: utf-8 -*-
"""
All SQLite access lives here: connection setup, the write-serialization
lock (the actual fix for "database is locked" / pin-rename-delete not
working), and every read/write query used by the HTTP + websocket routes.
"""
import os
import sqlite3
import threading
import time
import traceback

from . import config
from .ws_manager import debug_log, broadcast_to_clients

# ---------------------------------------------------------------------------
# Global DB write-serialization lock + WAL mode.
#
# ROOT CAUSE of pin/rename/delete "not working" and sessions not
# auto-refreshing: multiple threads (FastAPI request handlers, the
# websocket message handler running the executor in a thread, and the
# 1-second file-watcher loop) were all opening their own sqlite3
# connections and writing concurrently. On Android's emulated storage this
# very easily produces "database is locked" (sqlite3.OperationalError),
# which the old code caught, logged quietly, and returned as a plain 500 --
# so from the UI it just looked like "nothing happened" with zero feedback.
#
# Fix: (a) turn on WAL journal mode + a busy_timeout so sqlite itself waits
# instead of failing immediately, (b) serialize ALL writes behind one
# process-wide lock so two write transactions never race each other, and
# (c) wrap writes in a small retry helper as a second safety net.
# ---------------------------------------------------------------------------
db_write_lock = threading.RLock()


def get_db_connection():
    os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH, check_same_thread=False, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        # WAL lets readers and a single writer coexist without "database is
        # locked" errors, and busy_timeout makes sqlite retry internally for
        # up to 8s instead of throwing immediately if a write is briefly held.
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=8000;")
        conn.execute("PRAGMA synchronous=NORMAL;")
    except Exception:
        pass
    return conn


def db_write(fn, *args, retries: int = 5, base_delay: float = 0.15, **kwargs):
    """Run a DB-writing function under the global write lock, retrying a
    couple of times if sqlite still reports 'database is locked'."""
    last_err = None
    for attempt in range(retries):
        try:
            with db_write_lock:
                return fn(*args, **kwargs)
        except sqlite3.OperationalError as e:
            last_err = e
            if "locked" in str(e).lower() or "busy" in str(e).lower():
                time.sleep(base_delay * (attempt + 1))
                continue
            raise
    raise last_err


def init_db():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at TEXT
                );
            """)

        cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    sender TEXT,
                    text TEXT,
                    source TEXT,
                    timestamp TEXT,
                    trace_log TEXT
                );
            """)

        # --- Lightweight migration: add pinned / updated_at columns if missing ---
        cursor.execute("PRAGMA table_info(chat_sessions)")
        existing_cols = {col[1] for col in cursor.fetchall()}
        if "pinned" not in existing_cols:
            cursor.execute(
                "ALTER TABLE chat_sessions ADD COLUMN pinned INTEGER DEFAULT 0"
            )
        if "updated_at" not in existing_cols:
            cursor.execute("ALTER TABLE chat_sessions ADD COLUMN updated_at TEXT")
            cursor.execute(
                "UPDATE chat_sessions SET updated_at = created_at WHERE"
                " updated_at IS NULL"
            )

        cursor.execute("SELECT COUNT(*) FROM chat_sessions")
        if cursor.fetchone()[0] == 0:
            now = config.get_local_ist_timestamp()
            cursor.execute(
                "INSERT INTO chat_sessions (session_id, title, created_at,"
                " updated_at, pinned) VALUES (?, ?, ?, ?, 0)",
                ("main_session", "General Conversation", now, now),
            )

        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB INIT ERROR] {e}")


def _derive_title_from_text(text: str, max_len: int = 40) -> str:
    """Turns the first user message into a clean, ChatGPT/Gemini-style title."""
    clean = " ".join((text or "").strip().split())
    if not clean:
        return "New Conversation"
    if len(clean) <= max_len:
        return clean
    truncated = clean[:max_len].rsplit(" ", 1)[0].strip()
    return (truncated or clean[:max_len]) + "..."


def _save_message_to_db_impl(
    session_id: str,
    sender: str,
    text: str,
    source: str = "web",
    trace_log: str = None,
):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT session_id, title FROM chat_sessions WHERE session_id = ?",
            (session_id,),
        )
        existing_session = cursor.fetchone()
        current_time = config.get_local_ist_timestamp()
        auto_title = None

        if not existing_session:
            title = (
                _derive_title_from_text(text) if sender == "user" else "New Conversation"
            )
            cursor.execute(
                "INSERT INTO chat_sessions (session_id, title, created_at,"
                " updated_at, pinned) VALUES (?, ?, ?, ?, 0)",
                (session_id, title, current_time, current_time),
            )
        else:
            if sender == "user" and (
                existing_session["title"] or ""
            ).strip() in config.DEFAULT_SESSION_TITLES:
                auto_title = _derive_title_from_text(text)
                cursor.execute(
                    "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE"
                    " session_id = ?",
                    (auto_title, current_time, session_id),
                )
            else:
                cursor.execute(
                    "UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?",
                    (current_time, session_id),
                )

        cursor.execute(
            """
                INSERT INTO chat_messages
                (session_id, sender, text, source, timestamp, trace_log)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, sender, text, source, current_time, trace_log),
        )
        conn.commit()
        return auto_title
    finally:
        conn.close()


def save_message_to_db(
    session_id: str,
    sender: str,
    text: str,
    source: str = "web",
    trace_log: str = None,
):
    try:
        auto_title = db_write(
            _save_message_to_db_impl,
            session_id=session_id,
            sender=sender,
            text=text,
            source=source,
            trace_log=trace_log,
        )
        if auto_title:
            broadcast_to_clients({
                "type": "session_renamed",
                "session_id": session_id,
                "title": auto_title,
            })
    except Exception as e:
        err_str = traceback.format_exc()
        debug_log(f"DB Save Error: {e}", "bold red")
        broadcast_to_clients({
            "type": "system_error",
            "source": "save_message_to_db",
            "error": str(e),
            "traceback": err_str,
        })


def list_sessions():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
            SELECT s.session_id, s.title, s.created_at, s.updated_at,
                   s.pinned, COUNT(m.id) as msg_count
            FROM chat_sessions s
            LEFT JOIN chat_messages m ON s.session_id = m.session_id
            GROUP BY s.session_id
            ORDER BY s.pinned DESC, COALESCE(s.updated_at, s.created_at) DESC
        """)
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_history_rows(session_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
            SELECT id, session_id, sender, text, source, timestamp, trace_log
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY id ASC
        """,
        (session_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def _create_new_session_impl(session_id: str):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        now = config.get_local_ist_timestamp()
        cursor.execute(
            "INSERT INTO chat_sessions (session_id, title, created_at,"
            " updated_at, pinned) VALUES (?, ?, ?, ?, 0)",
            (session_id, "New Conversation", now, now),
        )
        conn.commit()
    finally:
        conn.close()


def create_new_session(session_id: str):
    db_write(_create_new_session_impl, session_id)


def _rename_session_impl(session_id: str, new_title: str) -> int:
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE session_id = ?",
            (new_title, config.get_local_ist_timestamp(), session_id),
        )
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()


def rename_session(session_id: str, new_title: str) -> int:
    return db_write(_rename_session_impl, session_id, new_title)


def _pin_session_impl(session_id: str, payload: dict):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT pinned FROM chat_sessions WHERE session_id = ?",
            (session_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None

        if payload and "pinned" in payload:
            new_pinned = 1 if payload["pinned"] else 0
        else:
            new_pinned = 0 if row["pinned"] else 1  # toggle

        cursor.execute(
            "UPDATE chat_sessions SET pinned = ?, updated_at = ? WHERE session_id = ?",
            (new_pinned, config.get_local_ist_timestamp(), session_id),
        )
        conn.commit()
        return new_pinned
    finally:
        conn.close()


def pin_session(session_id: str, payload: dict):
    return db_write(_pin_session_impl, session_id, payload)


def _delete_session_impl(session_id: str) -> int:
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
        deleted = cursor.rowcount
        conn.commit()
        return deleted
    finally:
        conn.close()


def delete_session(session_id: str) -> int:
    return db_write(_delete_session_impl, session_id)
