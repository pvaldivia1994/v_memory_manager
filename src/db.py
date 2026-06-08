from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Optional

from .models import LongTermMemory, Message

_ACTIVE = "_active"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    role        TEXT NOT NULL CHECK(role IN ('user','assistant')),
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS prompts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    content     TEXT NOT NULL,
    orden       INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS long_term_memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    content     TEXT NOT NULL,
    tags        TEXT NOT NULL DEFAULT '',
    weight      REAL NOT NULL DEFAULT 1.0,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS configurations (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS semantic_memories (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id       TEXT NOT NULL UNIQUE,
    chroma_id       TEXT UNIQUE,
    namespace       TEXT NOT NULL DEFAULT 'normal',
    scope           TEXT NOT NULL DEFAULT 'user',
    content         TEXT NOT NULL,
    original_text   TEXT NOT NULL DEFAULT '',
    tags            TEXT NOT NULL DEFAULT '',
    confidence      REAL NOT NULL DEFAULT 1.0,
    importance      REAL NOT NULL DEFAULT 0.5,
    memory_type     TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'active',
    source          TEXT NOT NULL DEFAULT 'auto',
    source_message_ids TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
    owner_type      TEXT NOT NULL DEFAULT '',
    character_id    TEXT NOT NULL DEFAULT '',
    source_role     TEXT NOT NULL DEFAULT '',
    canon_status    TEXT NOT NULL DEFAULT 'canon',
    fact_key        TEXT NOT NULL DEFAULT '',
    fact_value      TEXT NOT NULL DEFAULT '',
    scene_id        TEXT NOT NULL DEFAULT '',
    world_id        TEXT NOT NULL DEFAULT '',
    expires_scope   TEXT NOT NULL DEFAULT 'never'
);
"""


def _conn(db_path: str) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.commit()


def verify_schema(conn: sqlite3.Connection) -> None:
    required = {"messages", "prompts", "configurations"}
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name IN (?, ?, ?)",
        ("messages", "prompts", "configurations"),
    ).fetchall()
    found = {r[0] for r in rows}
    missing = required - found
    if missing:
        raise RuntimeError(f"DB corrupta — tablas faltantes: {missing}")
    _migrate_schema(conn)


def _migrate_schema(conn: sqlite3.Connection) -> None:
    cols = {r[1] for r in conn.execute("PRAGMA table_info(prompts)").fetchall()}
    if "orden" not in cols:
        conn.execute("ALTER TABLE prompts ADD COLUMN orden INTEGER NOT NULL DEFAULT 0")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS long_term_memories (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            content     TEXT NOT NULL,
            tags        TEXT NOT NULL DEFAULT '',
            weight      REAL NOT NULL DEFAULT 1.0,
            created_at  TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS semantic_memories (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id       TEXT NOT NULL UNIQUE,
            chroma_id       TEXT UNIQUE,
            namespace       TEXT NOT NULL DEFAULT 'normal',
            scope           TEXT NOT NULL DEFAULT 'user',
            content         TEXT NOT NULL,
            original_text   TEXT NOT NULL DEFAULT '',
            tags            TEXT NOT NULL DEFAULT '',
            confidence      REAL NOT NULL DEFAULT 1.0,
            importance      REAL NOT NULL DEFAULT 0.5,
            memory_type     TEXT NOT NULL DEFAULT '',
            status          TEXT NOT NULL DEFAULT 'active',
            source          TEXT NOT NULL DEFAULT 'auto',
            source_message_ids TEXT NOT NULL DEFAULT '',
            created_at      TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
            owner_type      TEXT NOT NULL DEFAULT '',
            character_id    TEXT NOT NULL DEFAULT '',
            source_role     TEXT NOT NULL DEFAULT '',
            canon_status    TEXT NOT NULL DEFAULT 'canon',
            fact_key        TEXT NOT NULL DEFAULT '',
            fact_value      TEXT NOT NULL DEFAULT '',
            scene_id        TEXT NOT NULL DEFAULT '',
            world_id        TEXT NOT NULL DEFAULT '',
            expires_scope   TEXT NOT NULL DEFAULT 'never'
        )
    """)
    conn.commit()


def read_default_system() -> str:
    path = Path(__file__).parent / "def_system.md"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return "Eres un asistente útil y natural."


# ── Messages ─────────────────────────────────────────────────


def add_message(conn: sqlite3.Connection, role: str, content: str) -> None:
    conn.execute(
        "INSERT INTO messages (role, content) VALUES (?, ?)",
        (role, content),
    )
    conn.commit()


def get_all_messages(conn: sqlite3.Connection) -> list[Message]:
    rows = conn.execute(
        "SELECT id, role, content FROM messages ORDER BY id"
    ).fetchall()
    return [Message(id=r[0], role=r[1], content=r[2]) for r in rows]


def count_messages(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]


def clear_messages(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM messages")
    conn.commit()


# ── Prompts ───────────────────────────────────────────────────


def upsert_prompt(conn: sqlite3.Connection, name: str, content: str, orden: int = 0) -> None:
    conn.execute(
        "INSERT INTO prompts (name, content, orden) VALUES (?, ?, ?) "
        "ON CONFLICT(name) DO UPDATE SET content = excluded.content, "
        "orden = excluded.orden, created_at = datetime('now')",
        (name, content, orden),
    )
    conn.commit()


def get_all_prompts_ordered(conn: sqlite3.Connection) -> list[tuple[str, str, int]]:
    rows = conn.execute(
        "SELECT name, content, orden FROM prompts ORDER BY orden ASC, name ASC"
    ).fetchall()
    return [(r[0], r[1], r[2]) for r in rows]


def get_prompt(conn: sqlite3.Connection, name: str) -> Optional[str]:
    row = conn.execute(
        "SELECT content FROM prompts WHERE name = ?", (name,)
    ).fetchone()
    return row[0] if row else None


def list_prompts(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM prompts WHERE name != ? ORDER BY name", (_ACTIVE,)
    ).fetchall()
    return [r[0] for r in rows]


def delete_prompt(conn: sqlite3.Connection, name: str) -> None:
    conn.execute("DELETE FROM prompts WHERE name = ?", (name,))
    conn.commit()


# ── Configurations ────────────────────────────────────────────


def get_config(conn: sqlite3.Connection, key: str, default: Any = None) -> Any:
    row = conn.execute(
        "SELECT value FROM configurations WHERE key = ?", (key,)
    ).fetchone()
    return row[0] if row else default


def set_config(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO configurations (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
    conn.commit()


def all_configs(conn: sqlite3.Connection) -> dict:
    rows = conn.execute(
        "SELECT key, value FROM configurations ORDER BY key"
    ).fetchall()
    return {r[0]: r[1] for r in rows}


# ── Long-term memories ─────────────────────────────────────


def add_long_term_memory(
    conn: sqlite3.Connection, content: str, tags: str = "", weight: float = 1.0
) -> None:
    conn.execute(
        "INSERT INTO long_term_memories (content, tags, weight) VALUES (?, ?, ?)",
        (content, tags, weight),
    )
    conn.commit()


def get_all_long_term_memories(
    conn: sqlite3.Connection,
    tag: Optional[str] = None,
    min_weight: Optional[float] = None,
) -> list[LongTermMemory]:
    query = "SELECT id, content, tags, weight FROM long_term_memories"
    conditions = []
    params: list[Any] = []
    if tag:
        conditions.append("tags LIKE ?")
        params.append(f"%{tag}%")
    if min_weight is not None:
        conditions.append("weight >= ?")
        params.append(min_weight)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY weight ASC, created_at ASC"
    rows = conn.execute(query, params).fetchall()
    return [LongTermMemory(id=r[0], content=r[1], tags=r[2], weight=r[3]) for r in rows]


def get_long_term_memory(conn: sqlite3.Connection, memory_id: int) -> Optional[LongTermMemory]:
    row = conn.execute(
        "SELECT id, content, tags, weight FROM long_term_memories WHERE id = ?",
        (memory_id,),
    ).fetchone()
    return LongTermMemory(id=row[0], content=row[1], tags=row[2], weight=row[3]) if row else None


def delete_long_term_memory(conn: sqlite3.Connection, memory_id: int) -> None:
    conn.execute("DELETE FROM long_term_memories WHERE id = ?", (memory_id,))
    conn.commit()


def count_long_term_memories(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM long_term_memories").fetchone()[0]
