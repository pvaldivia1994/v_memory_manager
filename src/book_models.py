from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BookChunk:
    chunk_id: str = ""
    book_id: str = ""
    book_title: str = ""
    chapter: str = ""
    page_start: int = 0
    page_end: int = 0
    chunk_index: int = 0
    text: str = ""
    char_count: int = 0
    distance: float = 0.0


@dataclass
class BookInfo:
    book_id: str = ""
    title: str = ""
    author: str = ""
    source_path: str = ""
    source_hash: str = ""
    total_pages: int = 0
    total_chunks: int = 0
    status: str = ""
    embedding_model: str = ""
    created_at: str = ""
