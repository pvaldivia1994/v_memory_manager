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
    user_id         TEXT NOT NULL DEFAULT 'default',
    canon_status    TEXT NOT NULL DEFAULT 'canon',
    fact_key        TEXT NOT NULL DEFAULT '',
    fact_value      TEXT NOT NULL DEFAULT '',
    scene_id        TEXT NOT NULL DEFAULT '',
    world_id        TEXT NOT NULL DEFAULT '',
    expires_scope   TEXT NOT NULL DEFAULT 'never'
);

CREATE TABLE IF NOT EXISTS conversation_summary_state (
    conversation_id             TEXT NOT NULL,
    user_id                     TEXT NOT NULL DEFAULT 'default',

    summary                     TEXT NOT NULL DEFAULT '',
    last_summarized_message_id  INTEGER NOT NULL DEFAULT 0,
    last_summarized_created_at  TEXT NOT NULL DEFAULT '',

    summary_version             INTEGER NOT NULL DEFAULT 1,
    status                      TEXT NOT NULL DEFAULT 'active',

    summary_error_count         INTEGER NOT NULL DEFAULT 0,
    last_error                  TEXT NOT NULL DEFAULT '',

    created_at                  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at                  TEXT NOT NULL DEFAULT (datetime('now')),

    PRIMARY KEY (conversation_id, user_id)
);

CREATE TABLE IF NOT EXISTS books (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL DEFAULT 'default',
    title           TEXT NOT NULL DEFAULT '',
    author          TEXT NOT NULL DEFAULT '',
    source_path     TEXT NOT NULL,
    source_file_hash TEXT NOT NULL,
    source_text_hash TEXT NOT NULL DEFAULT '',
    source_type     TEXT NOT NULL DEFAULT '',
    source_layout   TEXT NOT NULL DEFAULT '',
    source_text_path TEXT NOT NULL DEFAULT '',
    total_pages     INTEGER NOT NULL DEFAULT 0,
    total_chapters  INTEGER NOT NULL DEFAULT 0,
    total_chunks    INTEGER NOT NULL DEFAULT 0,
    language        TEXT NOT NULL DEFAULT 'es',
    status          TEXT NOT NULL DEFAULT 'pending',
    error_message   TEXT NOT NULL DEFAULT '',
    embedding_model TEXT NOT NULL DEFAULT '',
    embedding_dim   INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(source_file_hash, source_text_hash)
);

CREATE TABLE IF NOT EXISTS book_chunks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id        TEXT NOT NULL UNIQUE,
    book_id         TEXT NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    parent_chunk_id TEXT REFERENCES book_chunks(chunk_id),
    level           TEXT NOT NULL CHECK(level IN ('chapter', 'section')),
    chapter         TEXT NOT NULL DEFAULT '',
    chapter_index   INTEGER NOT NULL DEFAULT 0,
    section_index   INTEGER NOT NULL DEFAULT 0,
    page_start      INTEGER NOT NULL DEFAULT 0,
    page_end        INTEGER NOT NULL DEFAULT 0,
    chunk_text      TEXT NOT NULL,
    char_count      INTEGER NOT NULL DEFAULT 0,
    chunk_hash      TEXT NOT NULL DEFAULT '',
    embedding       BLOB,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_book_chunks_book_id ON book_chunks(book_id);
CREATE INDEX IF NOT EXISTS idx_book_chunks_parent ON book_chunks(parent_chunk_id);
CREATE INDEX IF NOT EXISTS idx_book_chunks_book_chapter ON book_chunks(book_id, chapter_index);
CREATE UNIQUE INDEX IF NOT EXISTS idx_book_chunks_book_section ON book_chunks(book_id, chapter_index, section_index);
CREATE INDEX IF NOT EXISTS idx_book_chunks_hash ON book_chunks(chunk_hash);

CREATE TABLE IF NOT EXISTS book_chapters (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    book_id         TEXT NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    chapter_index   INTEGER NOT NULL,
    title           TEXT NOT NULL DEFAULT '',
    page_start      INTEGER NOT NULL DEFAULT 0,
    page_end        INTEGER NOT NULL DEFAULT 0,
    char_count      INTEGER NOT NULL DEFAULT 0,
    chunk_count     INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(book_id, chapter_index)
);"""



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
            user_id         TEXT NOT NULL DEFAULT 'default',
            canon_status    TEXT NOT NULL DEFAULT 'canon',
            fact_key        TEXT NOT NULL DEFAULT '',
            fact_value      TEXT NOT NULL DEFAULT '',
            scene_id        TEXT NOT NULL DEFAULT '',
            world_id        TEXT NOT NULL DEFAULT '',
            expires_scope   TEXT NOT NULL DEFAULT 'never'
        )
    """)

    # Migration: add user_id if missing (pre-v0.4.1 databases)
    sem_cols = {r[1] for r in conn.execute("PRAGMA table_info(semantic_memories)").fetchall()}
    if "user_id" not in sem_cols:
        conn.execute("ALTER TABLE semantic_memories ADD COLUMN user_id TEXT NOT NULL DEFAULT 'default'")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversation_summary_state (
            conversation_id             TEXT NOT NULL,
            user_id                     TEXT NOT NULL DEFAULT 'default',

            summary                     TEXT NOT NULL DEFAULT '',
            last_summarized_message_id  INTEGER NOT NULL DEFAULT 0,
            last_summarized_created_at  TEXT NOT NULL DEFAULT '',

            summary_version             INTEGER NOT NULL DEFAULT 1,
            status                      TEXT NOT NULL DEFAULT 'active',

            summary_error_count         INTEGER NOT NULL DEFAULT 0,
            last_error                  TEXT NOT NULL DEFAULT '',

            created_at                  TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at                  TEXT NOT NULL DEFAULT (datetime('now')),

            PRIMARY KEY (conversation_id, user_id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS books (
            id              TEXT PRIMARY KEY,
            user_id         TEXT NOT NULL DEFAULT 'default',
            title           TEXT NOT NULL DEFAULT '',
            author          TEXT NOT NULL DEFAULT '',
            source_path     TEXT NOT NULL,
            source_file_hash TEXT NOT NULL,
            source_text_hash TEXT NOT NULL DEFAULT '',
            source_type     TEXT NOT NULL DEFAULT '',
            source_layout   TEXT NOT NULL DEFAULT '',
            source_text_path TEXT NOT NULL DEFAULT '',
            total_pages     INTEGER NOT NULL DEFAULT 0,
            total_chapters  INTEGER NOT NULL DEFAULT 0,
            total_chunks    INTEGER NOT NULL DEFAULT 0,
            language        TEXT NOT NULL DEFAULT 'es',
            status          TEXT NOT NULL DEFAULT 'pending',
            error_message   TEXT NOT NULL DEFAULT '',
            embedding_model TEXT NOT NULL DEFAULT '',
            embedding_dim   INTEGER NOT NULL DEFAULT 0,
            created_at      TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(source_file_hash, source_text_hash)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS book_chunks (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id        TEXT NOT NULL UNIQUE,
            book_id         TEXT NOT NULL REFERENCES books(id) ON DELETE CASCADE,
            parent_chunk_id TEXT REFERENCES book_chunks(chunk_id),
            level           TEXT NOT NULL CHECK(level IN ('chapter', 'section')),
            chapter         TEXT NOT NULL DEFAULT '',
            chapter_index   INTEGER NOT NULL DEFAULT 0,
            section_index   INTEGER NOT NULL DEFAULT 0,
            page_start      INTEGER NOT NULL DEFAULT 0,
            page_end        INTEGER NOT NULL DEFAULT 0,
            chunk_text      TEXT NOT NULL,
            char_count      INTEGER NOT NULL DEFAULT 0,
            chunk_hash      TEXT NOT NULL DEFAULT '',
            embedding       BLOB,
            created_at      TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_book_chunks_book_id ON book_chunks(book_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_book_chunks_parent ON book_chunks(parent_chunk_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_book_chunks_book_chapter ON book_chunks(book_id, chapter_index)")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_book_chunks_book_section ON book_chunks(book_id, chapter_index, section_index)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_book_chunks_hash ON book_chunks(chunk_hash)")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS book_chapters (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            book_id         TEXT NOT NULL REFERENCES books(id) ON DELETE CASCADE,
            chapter_index   INTEGER NOT NULL,
            title           TEXT NOT NULL DEFAULT '',
            page_start      INTEGER NOT NULL DEFAULT 0,
            page_end        INTEGER NOT NULL DEFAULT 0,
            char_count      INTEGER NOT NULL DEFAULT 0,
            chunk_count     INTEGER NOT NULL DEFAULT 0,
            created_at      TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(book_id, chapter_index)
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


# ── Conversation summary state ─────────────────────────────

CONV_SUMMARY_COLS = [
    "conversation_id", "user_id", "summary", "last_summarized_message_id",
    "last_summarized_created_at", "summary_version", "status",
    "summary_error_count", "last_error", "created_at", "updated_at",
]


def get_conv_summary_state(
    conn: sqlite3.Connection, conversation_id: str, user_id: str = "default"
) -> Optional[dict]:
    row = conn.execute(
        "SELECT * FROM conversation_summary_state WHERE conversation_id=? AND user_id=?",
        (conversation_id, user_id),
    ).fetchone()
    return dict(row) if row else None


def upsert_conv_summary_state(
    conn: sqlite3.Connection,
    conversation_id: str,
    user_id: str = "default",
    **kwargs,
) -> None:
    existing = get_conv_summary_state(conn, conversation_id, user_id)
    if existing:
        sets = ", ".join(f"{k}=?" for k in kwargs)
        sets += ", updated_at=datetime('now')"
        params = list(kwargs.values()) + [conversation_id, user_id]
        conn.execute(
            f"UPDATE conversation_summary_state SET {sets} WHERE conversation_id=? AND user_id=?",
            params,
        )
    else:
        fields = {**kwargs, "conversation_id": conversation_id, "user_id": user_id}
        cols = ", ".join(fields.keys())
        placeholders = ", ".join("?" for _ in fields)
        conn.execute(
            f"INSERT INTO conversation_summary_state ({cols}) VALUES ({placeholders})",
            list(fields.values()),
        )
    conn.commit()


def delete_conv_summary_state(
    conn: sqlite3.Connection, conversation_id: str, user_id: str = "default"
) -> None:
    conn.execute(
        "DELETE FROM conversation_summary_state WHERE conversation_id=? AND user_id=?",
        (conversation_id, user_id),
    )
    conn.commit()


def get_messages_range(
    conn: sqlite3.Connection, from_id: int, to_id: int
) -> list[dict]:
    rows = conn.execute(
        "SELECT id, role, content, created_at FROM messages WHERE id >= ? AND id <= ? ORDER BY id",
        (from_id, to_id),
    ).fetchall()
    return [
        {"id": r[0], "role": r[1], "content": r[2], "created_at": r[3]}
        for r in rows
    ]


# ── Books ────────────────────────────────────────────────────

BOOK_COLS = [
    "id", "user_id", "title", "author", "source_path",
    "source_file_hash", "source_text_hash",
    "source_type", "source_layout", "source_text_path",
    "total_pages", "total_chapters", "total_chunks", "language",
    "status", "error_message", "embedding_model", "embedding_dim",
    "created_at", "updated_at",
]

BOOK_CHUNK_COLS = [
    "id", "chunk_id", "book_id", "parent_chunk_id", "level",
    "chapter", "chapter_index", "section_index",
    "page_start", "page_end", "chunk_text",
    "char_count", "chunk_hash", "embedding", "created_at",
]


def insert_book(conn: sqlite3.Connection, book_id: str, source_path: str, source_file_hash: str,
                user_id: str = "default", **kwargs) -> None:
    fields = {
        "id": book_id,
        "user_id": user_id,
        "source_path": source_path,
        "source_file_hash": source_file_hash,
        **{k: v for k, v in kwargs.items() if k in BOOK_COLS},
    }
    cols = ", ".join(fields.keys())
    placeholders = ", ".join("?" for _ in fields)
    conn.execute(f"INSERT INTO books ({cols}) VALUES ({placeholders})", list(fields.values()))
    conn.commit()


def update_book(conn: sqlite3.Connection, book_id: str, **kwargs) -> None:
    sets = ", ".join(f"{k}=?" for k in kwargs if k in BOOK_COLS)
    if not sets:
        return
    sets += ", updated_at=datetime('now')"
    params = list(kwargs.values()) + [book_id]
    conn.execute(f"UPDATE books SET {sets} WHERE id=?", params)
    conn.commit()


def get_book(conn: sqlite3.Connection, book_id: str) -> Optional[dict]:
    row = conn.execute("SELECT * FROM books WHERE id=?", (book_id,)).fetchone()
    return dict(row) if row else None


def get_book_by_hashes(conn: sqlite3.Connection, source_file_hash: str,
                       source_text_hash: str) -> Optional[dict]:
    row = conn.execute(
        "SELECT * FROM books WHERE source_file_hash=? AND source_text_hash=?",
        (source_file_hash, source_text_hash),
    ).fetchone()
    return dict(row) if row else None


def get_book_by_file_hash(conn: sqlite3.Connection, source_file_hash: str) -> Optional[dict]:
    row = conn.execute(
        "SELECT * FROM books WHERE source_file_hash=?", (source_file_hash,)
    ).fetchone()
    return dict(row) if row else None


def list_books(conn: sqlite3.Connection, user_id: str = "default") -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM books WHERE user_id=? ORDER BY created_at DESC", (user_id,)
    ).fetchall()
    return [dict(r) for r in rows]


def delete_book(conn: sqlite3.Connection, book_id: str) -> None:
    conn.execute("DELETE FROM book_chunks WHERE book_id=?", (book_id,))
    conn.execute("DELETE FROM books WHERE id=?", (book_id,))
    conn.commit()


def count_books(conn: sqlite3.Connection, user_id: str = "default") -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM books WHERE user_id=?", (user_id,)
    ).fetchone()[0]


def has_books(conn: sqlite3.Connection, user_id: str = "default") -> bool:
    row = conn.execute(
        "SELECT COUNT(*) FROM books WHERE status='indexed' AND user_id=?", (user_id,)
    ).fetchone()
    return row[0] > 0 if row else False


def insert_book_chunk(conn: sqlite3.Connection, chunk_id: str, book_id: str,
                      level: str, chapter_index: int, section_index: int,
                      chunk_text: str, chunk_hash: str, **kwargs) -> None:
    fields = {
        "chunk_id": chunk_id, "book_id": book_id,
        "level": level, "chapter_index": chapter_index,
        "section_index": section_index,
        "chunk_text": chunk_text, "chunk_hash": chunk_hash,
        "char_count": len(chunk_text),
        **{k: v for k, v in kwargs.items() if k in BOOK_CHUNK_COLS},
    }
    cols = ", ".join(fields.keys())
    placeholders = ", ".join("?" for _ in fields)
    conn.execute(f"INSERT OR REPLACE INTO book_chunks ({cols}) VALUES ({placeholders})", list(fields.values()))


def get_book_chunks(conn: sqlite3.Connection, book_id: str, limit: int = 1000) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM book_chunks WHERE book_id=? ORDER BY chapter_index ASC, section_index ASC LIMIT ?",
        (book_id, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def get_book_chunk_by_chunk_id(conn: sqlite3.Connection, chunk_id: str) -> Optional[dict]:
    row = conn.execute("SELECT * FROM book_chunks WHERE chunk_id=?", (chunk_id,)).fetchone()
    return dict(row) if row else None


def delete_book_chunks(conn: sqlite3.Connection, book_id: str) -> None:
    conn.execute("DELETE FROM book_chunks WHERE book_id=?", (book_id,))
    conn.commit()


def count_book_chunks(conn: sqlite3.Connection, book_id: str) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM book_chunks WHERE book_id=?", (book_id,)
    ).fetchone()[0]


def count_book_chapters(conn: sqlite3.Connection, book_id: str) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM book_chunks WHERE book_id=? AND level='chapter'", (book_id,)
    ).fetchone()[0]


def chunk_exists_by_hash(conn: sqlite3.Connection, book_id: str, chunk_hash: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM book_chunks WHERE book_id=? AND chunk_hash=? LIMIT 1",
        (book_id, chunk_hash),
    ).fetchone()
    return row is not None


def get_parent_by_chunk_id(conn: sqlite3.Connection, chunk_id: str) -> Optional[dict]:
    row = conn.execute("""
        SELECT p.* FROM book_chunks p
        JOIN book_chunks c ON c.parent_chunk_id = p.chunk_id
        WHERE c.chunk_id = ?
    """, (chunk_id,)).fetchone()
    return dict(row) if row else None


def get_chapter_chunks(conn: sqlite3.Connection, book_id: str,
                       chapter_index: int) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM book_chunks WHERE book_id=? AND chapter_index=? AND level='section' ORDER BY section_index",
        (book_id, chapter_index),
    ).fetchall()
    return [dict(r) for r in rows]


def list_chapters(conn: sqlite3.Connection, book_id: str) -> list[dict]:
    rows = conn.execute("""
        SELECT chapter_index, title, page_start, page_end, char_count, chunk_count
        FROM book_chapters
        WHERE book_id=?
        ORDER BY chapter_index
    """, (book_id,)).fetchall()
    if rows:
        return [dict(r) for r in rows]
    rows = conn.execute("""
        SELECT chunk_id, chapter, chapter_index, page_start, page_end, char_count
        FROM book_chunks
        WHERE book_id=? AND level='chapter'
        ORDER BY chapter_index
    """, (book_id,)).fetchall()
    return [dict(r) for r in rows]


def upsert_book_chapter(conn: sqlite3.Connection, book_id: str, chapter_index: int,
                        title: str, page_start: int, page_end: int,
                        char_count: int, chunk_count: int) -> None:
    conn.execute("""
        INSERT INTO book_chapters (book_id, chapter_index, title, page_start, page_end, char_count, chunk_count)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(book_id, chapter_index) DO UPDATE SET
            title=excluded.title, page_start=excluded.page_start,
            page_end=excluded.page_end, char_count=excluded.char_count,
            chunk_count=excluded.chunk_count
    """, (book_id, chapter_index, title, page_start, page_end, char_count, chunk_count))
    conn.commit()


def get_first_window_message_id(
    conn: sqlite3.Connection, window_size: int
) -> Optional[int]:
    rows = conn.execute(
        "SELECT id, role FROM messages ORDER BY id"
    ).fetchall()
    if not rows:
        return None
    if window_size < 1:
        return None

    window = list(rows)[-window_size:]

    while window and window[0][1] != "user":
        window.pop(0)
    while window and window[-1][1] == "user":
        window.pop()

    return window[0][0] if window else None
