from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any, Optional

from . import db, deque
from .conversation_summary import ConversationSummaryMemory
from .models import LongTermMemory, Message


class MemoryManager:
    def __init__(self):
        self._conn: Optional[sqlite3.Connection] = None
        self._db_path: str = ""

    @property
    def conn(self) -> Optional[sqlite3.Connection]:
        return self._conn

    # ── Lifecycle ──────────────────────────────────────────────

    def create_memory_db(
        self, path: str, default_system_path: Optional[str] = None
    ) -> None:
        parent = Path(path).parent
        if not parent.exists():
            raise FileNotFoundError(
                f"El directorio padre no existe: {parent}"
            )

        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._db_path = path
        db.init_db(self._conn)

        if default_system_path:
            sp_path = Path(default_system_path)
            if not sp_path.exists():
                raise FileNotFoundError(
                    f"Archivo de system prompt no encontrado: {sp_path}"
                )
            content = sp_path.read_text(encoding="utf-8").strip()
        else:
            content = db.read_default_system()

        if content:
            db.upsert_prompt(self._conn, db._ACTIVE, content)

    def load_memory_db(self, path: str) -> None:
        if not Path(path).exists():
            raise FileNotFoundError(f"DB no encontrada: {path}")

        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._db_path = path
        db.verify_schema(self._conn)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def drop_memory_db(self) -> None:
        self.close()
        if self._db_path and Path(self._db_path).exists():
            os.remove(self._db_path)
            self._db_path = ""

    def clear_memory_db(self) -> None:
        """Limpia mensajes y memorias semanticas. No borra BookMemory."""
        self._require_conn()
        db.clear_messages(self._conn)
        self._conn.execute("DELETE FROM semantic_memories")
        self._conn.commit()

    def clear_all_memory(self) -> None:
        """Limpia TODO: mensajes, semantic_memories, books, book_chunks."""
        self._require_conn()
        db.clear_messages(self._conn)
        self._conn.execute("DELETE FROM semantic_memories")
        self._conn.execute("DELETE FROM book_chunks")
        self._conn.execute("DELETE FROM books")
        self._conn.commit()

    # ── Messages ───────────────────────────────────────────────

    def add_message(self, role: str, content: str) -> None:
        self._require_conn()
        if role == "system":
            db.upsert_prompt(self._conn, db._ACTIVE, content)
        else:
            db.add_message(self._conn, role, content)

    def get_history(self, max_messages: int = 10) -> list[Message]:
        self._require_conn()
        system_prompt = self.build_system_prompt()
        return deque.build_history(self._conn, max_messages, system_prompt)

    def get_system_prompt(self) -> str:
        self._require_conn()
        return db.get_prompt(self._conn, db._ACTIVE) or ""

    def build_system_prompt(self) -> str:
        self._require_conn()
        parts = []

        core = db.get_prompt(self._conn, db._ACTIVE)
        if core:
            parts.append(core)

        rules_lines = [
            "[USO DE MEMORIA]",
            "- USER_MEMORY describe al usuario que está conversando.",
            "- ASSISTANT_MEMORY describe al asistente.",
            "- PROJECT_MEMORY describe proyectos o contexto técnico.",
            "- CONVERSATION_SUMMARY resume partes anteriores de esta conversación que ya no están en el sliding window.",
            "- El sliding window tiene más detalle reciente que CONVERSATION_SUMMARY.",
            "- Si CONVERSATION_SUMMARY contradice los últimos mensajes, usar los últimos mensajes.",
            "- Si USER_MEMORY contradice CONVERSATION_SUMMARY, usar USER_MEMORY.",
            "- El contexto adicional en el mensaje del usuario tiene prioridad sobre CONVERSATION_SUMMARY si contradicen.",
            "- No tratar CONVERSATION_SUMMARY como una cita exacta; es una compresión.",
            "- Si el usuario pregunta por 'mi', 'me', 'yo', 'mis gustos', 'mi nombre' o 'mi favorito', revisa USER_MEMORY primero.",
            "- Si USER_MEMORY contiene la respuesta, responde directamente usando esa memoria.",
            "- No digas 'como modelo de lenguaje no tengo preferencias' cuando el usuario pregunta por sus propias preferencias.",
            "- No respondas con explicaciones genéricas si hay una memoria relevante en USER_MEMORY.",
        ]
        parts.append("\n".join(rules_lines))

        all_prompts = db.get_all_prompts_ordered(self._conn)
        extra = [(n, c, o) for n, c, o in all_prompts if n != db._ACTIVE]
        for _, content, _ in extra:
            parts.append(content)

        return "\n\n".join(parts)

    def build_user_message(self, user_input: str, book_context: str = "",
                           semantic_memory: Any = None,
                           conv_summary_memory: Any = None,
                           extra_context: str = "") -> str:
        parts = [user_input]
        context_lines = []

        if book_context:
            context_lines.append(book_context)

        if semantic_memory:
            try:
                user_mem = semantic_memory.search(user_input, n_results=3, scope="user")
                if user_mem:
                    lines = "\n".join(f"- {m.content}" for m in user_mem)
                    context_lines.append(f"[USER_MEMORY]\n{lines}")
            except Exception:
                pass
            try:
                asst_mem = semantic_memory.search(user_input, n_results=3, scope="assistant")
                if asst_mem:
                    lines = "\n".join(f"- {m.content}" for m in asst_mem)
                    context_lines.append(f"[ASSISTANT_MEMORY]\n{lines}")
            except Exception:
                pass

        if conv_summary_memory:
            try:
                block = conv_summary_memory.build_context_block()
                if block:
                    context_lines.append(block)
            except Exception:
                pass

        if extra_context:
            context_lines.append(f"[USER_MEMORY]\n{extra_context}")

        if context_lines:
            parts.append("--- Contexto adicional ---")
            parts.extend(context_lines)

        return "\n\n".join(parts)

    def count_messages(self) -> int:
        self._require_conn()
        return db.count_messages(self._conn)

    # ── Prompts ────────────────────────────────────────────────

    def save_prompt(self, name: str, content: str, orden: int = 0) -> None:
        self._require_conn()
        db.upsert_prompt(self._conn, name, content, orden)

    def load_prompt(self, name: str) -> Optional[str]:
        self._require_conn()
        return db.get_prompt(self._conn, name)

    def list_prompts(self) -> list[str]:
        self._require_conn()
        return db.list_prompts(self._conn)

    def delete_prompt(self, name: str) -> None:
        self._require_conn()
        db.delete_prompt(self._conn, name)

    # ── Long-term memories ─────────────────────────────────────

    def add_long_term_memory(
        self, content: str, tags: str = "", weight: float = 1.0
    ) -> None:
        self._require_conn()
        db.add_long_term_memory(self._conn, content, tags, weight)

    def get_long_term_memories(
        self,
        tag: Optional[str] = None,
        min_weight: Optional[float] = None,
    ) -> list[LongTermMemory]:
        self._require_conn()
        return db.get_all_long_term_memories(self._conn, tag, min_weight)

    def get_long_term_memory(self, memory_id: int) -> Optional[LongTermMemory]:
        self._require_conn()
        return db.get_long_term_memory(self._conn, memory_id)

    def delete_long_term_memory(self, memory_id: int) -> None:
        self._require_conn()
        db.delete_long_term_memory(self._conn, memory_id)

    def count_long_term_memories(self) -> int:
        self._require_conn()
        return db.count_long_term_memories(self._conn)

    # ── Configurations ─────────────────────────────────────────

    def get_config(self, key: str, default: Any = None) -> Any:
        self._require_conn()
        return db.get_config(self._conn, key, default)

    def set_config(self, key: str, value: str) -> None:
        self._require_conn()
        db.set_config(self._conn, key, value)

    def all_configs(self) -> dict:
        self._require_conn()
        return db.all_configs(self._conn)

    # ── Internal ───────────────────────────────────────────────

    def _require_conn(self) -> None:
        if self._conn is None:
            raise RuntimeError(
                "No hay DB cargada. Llama a create_memory_db() o load_memory_db() primero."
            )
