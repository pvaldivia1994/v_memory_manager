from __future__ import annotations

import sqlite3
from typing import Optional

from . import db
from .models import Message


def get_visible_window(
    conn: sqlite3.Connection, window_size: int
) -> list[Message]:
    """Devuelve los mensajes visibles en el sliding window (sin system prompt)."""
    msgs = db.get_all_messages(conn)

    if window_size < 1:
        return []

    window = msgs[-window_size:]

    # El primer mensaje despues de system DEBE ser user. Si arranca con
    # assistant, se descarta (el sliding-window agarro un resto de par).
    while window and window[0].role != "user":
        window.pop(0)

    # Si el ultimo es user huerfano (turno incompleto), se descarta:
    # el user actual se pasa por separado en llm.chat(user=...)
    while window and window[-1].role == "user":
        window.pop()

    return window


def build_history(
    conn: sqlite3.Connection,
    max_messages: int = 10,
    system_prompt: Optional[str] = None,
) -> list[Message]:
    if system_prompt is None:
        system = db.get_prompt(conn, db._ACTIVE)
    else:
        system = system_prompt

    window_size = max_messages - 1
    window = get_visible_window(conn, window_size)

    result: list[Message] = []
    if system:
        result.append(Message(role="system", content=system))
    result.extend(window)
    return result
