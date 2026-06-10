from __future__ import annotations

# pyrefly: ignore [missing-import]
from .book_memory import BookMemory, should_search_books
# pyrefly: ignore [missing-import]
from .book_models import BookChunk, BookInfo
# pyrefly: ignore [missing-import]
from .conversation_summary import ConversationSummaryMemory
# pyrefly: ignore [missing-import]
from .memory import MemoryManager
# pyrefly: ignore [missing-import]
from .memory_models import AnalysisResult, MemoryRecord
# pyrefly: ignore [missing-import]
from .models import LongTermMemory, Message
# pyrefly: ignore [missing-import]
from .semantic_memory import (
    SemanticMemory,
    analyze_assistant_text,
    migrate_long_term_to_semantic,
)

__version__ = "0.5.0"

__all__ = [
    "BookChunk",
    "BookInfo",
    "BookMemory",
    "ConversationSummaryMemory",
    "MemoryManager",
    "Message",
    "LongTermMemory",
    "SemanticMemory",
    "AnalysisResult",
    "MemoryRecord",
    "analyze_assistant_text",
    "migrate_long_term_to_semantic",
    "should_search_books",
    "__version__",
]
