"""
All project persistence lives here — one SQLite database, one module.
No other module may open a database connection directly.
"""

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional

import config

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        with _lock:
            if _conn is None:
                Path(config.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
                _conn = sqlite3.connect(
                    config.DB_PATH,
                    check_same_thread=False,
                    detect_types=sqlite3.PARSE_DECLTYPES,
                )
                _conn.row_factory = sqlite3.Row
                _conn.execute("PRAGMA journal_mode=WAL")
                _conn.execute("PRAGMA foreign_keys=ON")
    return _conn


def _execute(sql: str, params: tuple = ()) -> sqlite3.Cursor:
    with _lock:
        return _get_conn().execute(sql, params)


def _executemany(sql: str, params_seq: list) -> sqlite3.Cursor:
    with _lock:
        return _get_conn().executemany(sql, params_seq)


def _commit() -> None:
    with _lock:
        _get_conn().commit()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create all tables if they do not already exist."""
    conn = _get_conn()
    with _lock:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                role    TEXT    NOT NULL,
                content TEXT    NOT NULL,
                ts      TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS knowledge (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT    NOT NULL,
                key      TEXT    NOT NULL,
                value    TEXT    NOT NULL,
                source   TEXT    NOT NULL DEFAULT '',
                ts       TEXT    NOT NULL DEFAULT (datetime('now')),
                UNIQUE(category, key)
            );

            CREATE TABLE IF NOT EXISTS tasks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                title       TEXT    NOT NULL,
                description TEXT    NOT NULL DEFAULT '',
                done_check  TEXT    NOT NULL DEFAULT '',
                status      TEXT    NOT NULL DEFAULT 'pending',
                created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
                updated_at  TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS objectives (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                text       TEXT    NOT NULL,
                metric     TEXT    NOT NULL DEFAULT '',
                target     TEXT    NOT NULL DEFAULT '',
                status     TEXT    NOT NULL DEFAULT 'pending',
                created_at TEXT    NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS metrics (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_json TEXT    NOT NULL,
                ts            TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS state (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)
        conn.commit()
    logger.debug("Database initialised at %s", config.DB_PATH)


# ---------------------------------------------------------------------------
# conversations
# ---------------------------------------------------------------------------

def save_message(role: str, content: str) -> None:
    """Append a message to the conversation history."""
    _execute(
        "INSERT INTO conversations (role, content) VALUES (?, ?)",
        (role, content),
    )
    _commit()


def load_history(limit: int = 100) -> list[dict]:
    """Return the most recent *limit* messages, oldest first."""
    cur = _execute(
        "SELECT role, content, ts FROM conversations ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    rows = cur.fetchall()
    return [dict(r) for r in reversed(rows)]


def count_history() -> int:
    """Return the total number of stored messages."""
    cur = _execute("SELECT COUNT(*) FROM conversations", ())
    return cur.fetchone()[0]


def clear_history() -> None:
    """Delete all conversation messages."""
    _execute("DELETE FROM conversations", ())
    _commit()


# ---------------------------------------------------------------------------
# knowledge
# ---------------------------------------------------------------------------

def save_knowledge(category: str, key: str, value: str, source: str = "") -> None:
    """Insert or replace a knowledge entry (upsert on category+key)."""
    _execute(
        """
        INSERT INTO knowledge (category, key, value, source, ts)
        VALUES (?, ?, ?, ?, datetime('now'))
        ON CONFLICT(category, key) DO UPDATE SET
            value  = excluded.value,
            source = excluded.source,
            ts     = excluded.ts
        """,
        (category, key, value, source),
    )
    _commit()


def search_knowledge(query: str, cat: Optional[str] = None) -> list[dict]:
    """Full-text search over knowledge values (LIKE).

    Filter by *cat* when provided.
    """
    pattern = f"%{query}%"
    if cat:
        cur = _execute(
            "SELECT * FROM knowledge WHERE category = ? AND value LIKE ? ORDER BY ts DESC",
            (cat, pattern),
        )
    else:
        cur = _execute(
            "SELECT * FROM knowledge WHERE value LIKE ? ORDER BY ts DESC",
            (pattern,),
        )
    return [dict(r) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------

_VALID_TASK_STATUSES = {"pending", "in_progress", "done", "failed", "needs_review"}


def add_task(title: str, desc: str, done_check: str) -> int:
    """Insert a new task and return its id."""
    cur = _execute(
        "INSERT INTO tasks (title, description, done_check) VALUES (?, ?, ?)",
        (title, desc, done_check),
    )
    _commit()
    return cur.lastrowid  # type: ignore[return-value]


def update_task_status(task_id: int, status: str) -> None:
    """Update a task's status. Raises ValueError for unknown statuses."""
    if status not in _VALID_TASK_STATUSES:
        raise ValueError(f"Unknown task status '{status}'. Valid: {_VALID_TASK_STATUSES}")
    _execute(
        "UPDATE tasks SET status = ?, updated_at = datetime('now') WHERE id = ?",
        (status, task_id),
    )
    _commit()


def get_tasks(status: Optional[str] = None) -> list[dict]:
    """Return tasks, optionally filtered by *status*."""
    if status:
        cur = _execute(
            "SELECT * FROM tasks WHERE status = ? ORDER BY created_at",
            (status,),
        )
    else:
        cur = _execute("SELECT * FROM tasks ORDER BY created_at", ())
    return [dict(r) for r in cur.fetchall()]


def archive_done() -> int:
    """Mark done tasks as archived by deleting them; returns count removed."""
    cur = _execute("DELETE FROM tasks WHERE status = 'done'", ())
    _commit()
    return cur.rowcount


# ---------------------------------------------------------------------------
# objectives
# ---------------------------------------------------------------------------

_VALID_OBJ_STATUSES = {"pending", "in_progress", "done", "failed"}


def add_objective(text: str, metric: str, target: str) -> int:
    """Insert a new objective and return its id."""
    cur = _execute(
        "INSERT INTO objectives (text, metric, target) VALUES (?, ?, ?)",
        (text, metric, target),
    )
    _commit()
    return cur.lastrowid  # type: ignore[return-value]


def update_objective(obj_id: int, status: str) -> None:
    """Update an objective's status."""
    if status not in _VALID_OBJ_STATUSES:
        raise ValueError(f"Unknown objective status '{status}'. Valid: {_VALID_OBJ_STATUSES}")
    _execute(
        "UPDATE objectives SET status = ?, updated_at = datetime('now') WHERE id = ?",
        (status, obj_id),
    )
    _commit()


def get_objectives(status: Optional[str] = None) -> list[dict]:
    """Return objectives, optionally filtered by *status*."""
    if status:
        cur = _execute(
            "SELECT * FROM objectives WHERE status = ? ORDER BY created_at",
            (status,),
        )
    else:
        cur = _execute("SELECT * FROM objectives ORDER BY created_at", ())
    return [dict(r) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def save_metrics(snapshot: dict) -> None:
    """Persist a metrics snapshot as JSON."""
    _execute(
        "INSERT INTO metrics (snapshot_json) VALUES (?)",
        (json.dumps(snapshot),),
    )
    _commit()


def latest_metrics() -> dict:
    """Return the most recent metrics snapshot, or {} if none exist."""
    cur = _execute(
        "SELECT snapshot_json FROM metrics ORDER BY id DESC LIMIT 1", ()
    )
    row = cur.fetchone()
    return json.loads(row[0]) if row else {}


# ---------------------------------------------------------------------------
# state  (generic key/value for self-model and misc runtime data)
# ---------------------------------------------------------------------------

def set_state(key: str, value: Any) -> None:
    """Upsert a state entry. *value* is JSON-serialised."""
    _execute(
        """
        INSERT INTO state (key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, json.dumps(value)),
    )
    _commit()


def get_state(key: str, default: Any = None) -> Any:
    """Return the deserialised value for *key*, or *default* if absent."""
    cur = _execute("SELECT value FROM state WHERE key = ?", (key,))
    row = cur.fetchone()
    return json.loads(row[0]) if row else default


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

init_db()
