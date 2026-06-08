from __future__ import annotations

from typing import Optional

from .memory_models import MemoryRecord
from .roleplay_memory import RoleplaySemanticMemory
from .semantic_memory import SemanticMemory


class MemoryRouter:
    def __init__(
        self,
        semantic: SemanticMemory,
        roleplay: RoleplaySemanticMemory,
        roleplay_enabled: bool = False,
    ):
        self.semantic = semantic
        self.roleplay = roleplay
        self._roleplay_enabled = roleplay_enabled

    @property
    def roleplay_enabled(self) -> bool:
        return self._roleplay_enabled

    def set_roleplay_enabled(self, enabled: bool) -> None:
        self._roleplay_enabled = enabled

    def remember_user(
        self, text: str, msg_id: str = ""
    ) -> Optional[str]:
        if self._roleplay_enabled:
            ids = self.roleplay.remember(text, source_role="user", source="auto")
            return ids[0] if ids else None
        return self.semantic.remember(text, msg_ids=msg_id)

    def remember_assistant(
        self, text: str, msg_id: str = ""
    ) -> Optional[str]:
        if self._roleplay_enabled:
            ids = self.roleplay.remember(text, source_role="assistant", source="auto")
            return ids[0] if ids else None
        return None

    def remember_assistant_raw(
        self, text: str, msg_id: str = ""
    ) -> list[str]:
        if self._roleplay_enabled:
            return self.roleplay.remember(text, source_role="assistant", source="auto")
        return []

    def build_context(self, query: str, n_results: int = 5) -> str:
        if self._roleplay_enabled:
            return self.roleplay.build_context(query, n_results)
        results = self.semantic.search(query, n_results)
        if not results:
            return ""
        lines = "\n".join(f"- {m.content}" for m in results)
        return f"[USER_MEMORY]\n{lines}"

    def get_memory(self, memory_id: str) -> Optional[MemoryRecord]:
        rec = self.semantic.get_memory(memory_id)
        if rec:
            return rec
        return self.roleplay.get_memory(memory_id)

    def forget(self, memory_id: str) -> None:
        self.semantic.forget(memory_id)
        self.roleplay.forget(memory_id)

    def list_memories(self, limit: int = 50) -> list[MemoryRecord]:
        sem = self.semantic.list_memories(limit)
        rp = self.roleplay.list_memories(limit)
        combined = sem + rp
        combined.sort(key=lambda m: m.created_at, reverse=True)
        return combined[:limit]
