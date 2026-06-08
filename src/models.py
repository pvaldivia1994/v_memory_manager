from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Message:
    id: int = 0
    role: str = ""
    content: str = ""


@dataclass
class LongTermMemory:
    id: int = 0
    content: str = ""
    tags: str = ""
    weight: float = 1.0
