from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AnalysisResult:
    should_remember: bool = False
    reason: str = ""
    confidence: float = 0.0
    content: str = ""
    memory_type: str = ""
    tags: list[str] = field(default_factory=list)
    importance: float = 0.5


@dataclass
class MemoryRecord:
    memory_id: str = ""
    chroma_id: str = ""
    content: str = ""
    tags: list[str] = field(default_factory=list)
    confidence: float = 0.0
    importance: float = 0.5
    memory_type: str = ""
    namespace: str = "normal"
    scope: str = "user"
    status: str = "active"
    source: str = "auto"
    original_text: str = ""
    source_message_ids: str = ""
    created_at: str = ""
    updated_at: str = ""
    owner_type: str = ""
    character_id: str = ""
    source_role: str = ""
    canon_status: str = "canon"
    fact_key: str = ""
    fact_value: str = ""
    scene_id: str = ""
    world_id: str = ""
    expires_scope: str = "never"


@dataclass
class RoleplayAnalysisResult:
    should_remember: bool = False
    reason: str = ""
    confidence: float = 0.0
    content: str = ""
    memory_type: str = ""
    tags: list[str] = field(default_factory=list)
    importance: float = 0.5
    owner_type: str = ""
    character_id: str = ""
    source_role: str = ""
    canon_status: str = "canon"
    fact_key: str = ""
    fact_value: str = ""
    scene_id: str = ""
    world_id: str = ""
    expires_scope: str = "never"
