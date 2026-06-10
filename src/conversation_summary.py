from __future__ import annotations

import logging
import sqlite3
from typing import Any, Callable, Optional

from . import db

log = logging.getLogger("conv_summary")

SummarizerFn = Callable[[str, list[dict], int], str]
"""
Args:
    old_summary: Resumen anterior (o "" si es el primero).
    messages: Lista de dicts con id, role, content, created_at.
    max_chars: Máximo de caracteres permitido.
Returns:
    Nuevo resumen como string.
"""


class ConversationSummaryMemory:
    """
    Mantiene un resumen vivo de la parte de la conversación
    que ya quedó fuera del sliding window.

    No reemplaza SemanticMemory.
    No guarda hechos permanentes.
    No borra mensajes.
    Solo compacta continuidad conversacional.
    """

    def __init__(
        self,
        sqlite_conn: sqlite3.Connection,
        conversation_id: str = "default",
        user_id: str = "default",
        *,
        max_messages: int = 10,
        reserved_system_messages: int = 1,
        summarize_margin: int = 4,
        max_summary_chars: int = 3000,
        summarizer: Optional[SummarizerFn] = None,
        model: Any = None,
    ):
        self.conn = sqlite_conn
        self.conversation_id = conversation_id
        self.user_id = user_id
        self.window_size = max_messages - reserved_system_messages
        self.summarize_margin = summarize_margin
        self.max_summary_chars = max_summary_chars
        self.model = model
        self.summarizer = summarizer or self._make_default_summarizer()

    def _make_default_summarizer(self) -> Optional[SummarizerFn]:
        if self.model is None:
            return None
        model_ref = self.model

        def _summarize(old_summary: str, messages: list[dict], max_chars: int) -> str:
            texto = "\n".join(
                f"{m['role']}: {m['content'].strip()}" for m in messages
            )
            prompt = (
                "Eres un resumidor de continuidad conversacional.\n\n"
                "Tu tarea es actualizar un resumen existente con nuevos mensajes.\n"
                "No guardes gustos permanentes del usuario (eso es SemanticMemory).\n"
                "No inventes datos. No incluyas saludos, relleno ni detalles irrelevantes.\n"
                "Mantén el resumen útil para continuar la conversación.\n\n"
                "Devuelve SOLO el resumen actualizado en este formato:\n"
                "Tema actual:\n- ...\n\n"
                "Estado actual:\n- ...\n\n"
                "Decisiones tomadas:\n- ...\n\n"
                "Detalles técnicos importantes:\n- ...\n\n"
                "Pendientes:\n- ...\n\n"
                f"Resumen anterior:\n{old_summary}\n\n"
                f"Nuevos mensajes:\n{texto}"
            )
            res = model_ref.chat(
                system="Eres un asistente que resume conversaciones.",
                user=prompt,
                history=[],
            )
            content = res.content if hasattr(res, "content") else str(res)
            return content[:max_chars]

        return _summarize

    # ── Lectura ───────────────────────────────────────────────────

    def get_summary(self) -> str:
        state = db.get_conv_summary_state(self.conn, self.conversation_id, self.user_id)
        return state["summary"] if state else ""

    def get_last_summarized_message_id(self) -> int:
        state = db.get_conv_summary_state(self.conn, self.conversation_id, self.user_id)
        return int(state["last_summarized_message_id"]) if state else 0

    def get_state(self) -> dict:
        state = db.get_conv_summary_state(self.conn, self.conversation_id, self.user_id)
        if not state:
            return {
                "conversation_id": self.conversation_id,
                "user_id": self.user_id,
                "summary": "",
                "last_summarized_message_id": 0,
                "last_summarized_created_at": "",
                "summary_version": 0,
                "status": "active",
                "summary_error_count": 0,
                "last_error": "",
                "created_at": "",
                "updated_at": "",
            }
        return dict(state)

    def build_context_block(self) -> str:
        state = db.get_conv_summary_state(self.conn, self.conversation_id, self.user_id)
        if not state:
            return "[CONVERSATION_SUMMARY]\n- Sin resumen previo."

        if state.get("status") != "active":
            return "[CONVERSATION_SUMMARY]\n- Resumen desactivado."

        summary = state.get("summary", "").strip()
        if not summary:
            return "[CONVERSATION_SUMMARY]\n- Sin resumen previo."

        return f"[CONVERSATION_SUMMARY]\n{summary}"

    # ── Resumen ───────────────────────────────────────────────────

    def maybe_update(self) -> bool:
        if self.summarizer is None:
            return False

        state = db.get_conv_summary_state(self.conn, self.conversation_id, self.user_id)
        if state and state.get("status") != "active":
            return False

        first_id = db.get_first_window_message_id(self.conn, self.window_size)
        if first_id is None:
            return False

        cutoff_id = first_id - 1
        from_id = self.get_last_summarized_message_id() + 1

        if cutoff_id < from_id:
            return False

        messages = db.get_messages_range(self.conn, from_id, cutoff_id)
        messages = [m for m in messages if m.get("role") != "system"]

        if len(messages) < self.summarize_margin:
            return False

        self.update_summary(messages)
        return True

    def update_summary(self, messages: list[dict]) -> None:
        if self.summarizer is None:
            log.warning("No summarizer configured — skipping summary update")
            return

        old_summary = self.get_summary()
        try:
            new_summary = self.summarizer(
                old_summary=old_summary,
                messages=messages,
                max_chars=self.max_summary_chars,
            )
        except Exception as e:
            self._increment_error(str(e))
            log.exception("summarizer failed")
            return

        if not new_summary or not new_summary.strip():
            self._increment_error("summarizer returned empty")
            return

        if len(new_summary) > self.max_summary_chars:
            new_summary = new_summary[: self.max_summary_chars].rstrip()
            new_summary += "\n- [Resumen truncado por límite]"

        last_msg = messages[-1]
        db.upsert_conv_summary_state(
            self.conn,
            self.conversation_id,
            self.user_id,
            summary=new_summary,
            last_summarized_message_id=last_msg["id"],
            last_summarized_created_at=last_msg.get("created_at", ""),
            summary_error_count=0,
            last_error="",
        )

    # ── Reset ─────────────────────────────────────────────────────

    def reset(self) -> None:
        db.upsert_conv_summary_state(
            self.conn,
            self.conversation_id,
            self.user_id,
            summary="",
            last_summarized_message_id=0,
            last_summarized_created_at="",
            summary_version=1,
            status="active",
            summary_error_count=0,
            last_error="",
        )

    # ── Internals ─────────────────────────────────────────────────

    def _increment_error(self, error_msg: str) -> None:
        state = db.get_conv_summary_state(self.conn, self.conversation_id, self.user_id)
        count = (state["summary_error_count"] if state else 0) + 1
        db.upsert_conv_summary_state(
            self.conn,
            self.conversation_id,
            self.user_id,
            summary_error_count=count,
            last_error=error_msg[:500],
        )
