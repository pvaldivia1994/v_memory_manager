from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BookChunk:
    chunk_id: str = ""
    book_id: str = ""
    book_title: str = ""
    parent_chunk_id: Optional[str] = None
    level: str = "section"
    chapter: str = ""
    chapter_index: int = 0
    section_index: int = 0
    page_start: int = 0
    page_end: int = 0
    text: str = ""
    char_count: int = 0
    distance: float = 0.0


@dataclass
class BookInfo:
    book_id: str = ""
    title: str = ""
    author: str = ""
    user_id: str = "default"
    source_path: str = ""
    source_file_hash: str = ""
    source_text_hash: str = ""
    source_type: str = ""
    source_layout: str = ""
    source_text_path: str = ""
    total_pages: int = 0
    total_chapters: int = 0
    total_chunks: int = 0
    language: str = "es"
    status: str = ""
    embedding_model: str = ""
    embedding_dim: int = 0
    created_at: str = ""


@dataclass
class ChapterInfo:
    chapter_index: int = 0
    chapter: str = ""
    page_start: int = 0
    page_end: int = 0
    char_count: int = 0
    chunk_count: int = 0
    chunk_id: str = ""
