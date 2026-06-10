"""Smoke tests for BookMemory."""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.book_memory import (
    BookMemory,
    chunk_text,
    should_search_books,
    _generate_book_id,
    _make_chunk_id,
)
from src import MemoryManager


def test_should_search_books():
    assert should_search_books("que dice el libro") == True
    assert should_search_books("hola como estas") == False
    assert should_search_books("/book principios SOLID") == True
    assert should_search_books("/lore clean architecture") == True
    assert should_search_books("segun el autor") == True
    assert should_search_books("que menciona el capitulo 3") == True
    print("[OK] should_search_books")


def test_chunk_id_format():
    bid = _generate_book_id()
    assert bid.startswith("book_")
    assert len(bid) == 17
    cid = _make_chunk_id(bid, 42)
    assert cid == f"{bid}_chunk_000042"
    print(f"[OK] chunk_id format: {cid}")


def test_chunk_text_with_pages():
    text = """[PAGE 1]
Chapter 1: Introduction
First paragraph of the book. Contains important information.
And this is the second paragraph of the page.

[PAGE 2]
Chapter 2: Development
Here we start with the development content.
More details about the main topic."""

    chunks = chunk_text(text, chunk_size_chars=200, chunk_overlap_chars=50)
    assert len(chunks) >= 2
    print(f"[OK] chunk_text with pages: {len(chunks)} chunks")
    for i, c in enumerate(chunks):
        print(f"  chunk {i}: chapter={c['chapter']}, pages={c['page_start']}-{c['page_end']}, len={len(c['text'])}")


def test_chunk_text_simple():
    text = """Chapter 1: Start
Text of the first chapter. Contains relevant information.

Chapter 2: Middle
Text of the second chapter. More important content.

Chapter 3: End
Text of the third chapter. Concludes the text."""

    chunks = chunk_text(text, chunk_size_chars=100, chunk_overlap_chars=20)
    assert len(chunks) >= 2
    print(f"[OK] chunk_text simple: {len(chunks)} chunks")


def test_book_memory_ingest_text():
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test.db")
    chroma_dir = os.path.join(tmpdir, "chroma")

    mem = MemoryManager()
    mem.create_memory_db(db_path)

    bm = BookMemory(
        persist_dir=chroma_dir,
        sqlite_conn=mem._conn,
        chunk_size_chars=300,
        chunk_overlap_chars=50,
    )

    assert bm.has_books() == False

    text = """Chapter 1: Inception
Inception is a 2010 science fiction film directed by Christopher Nolan.
The film stars Leonardo DiCaprio as Dom Cobb, a professional thief.

Chapter 2: Dream Layers
The film explores the concept of dreams within dreams.
Each layer of the dream has its own unique time dilation.

Chapter 3: The Kick
A kick is a mechanism used to wake someone from a dream.
Falling in a dream is a common trigger for a kick."""

    book_id = bm.ingest_text(text, title="Inception Explained", author="Film Analysis")
    assert book_id.startswith("book_")
    assert bm.has_books() == True
    print(f"[OK] ingest_text: {book_id}")

    book = bm.get_book(book_id)
    assert book is not None
    assert book["title"] == "Inception Explained"
    assert book["status"] == "indexed"
    print(f"[OK] book metadata: title={book['title']} status={book['status']}")

    chunks = bm.get_chunks(book_id)
    assert len(chunks) > 0
    print(f"[OK] get_chunks: {len(chunks)} chunks")

    stats = bm.get_stats()
    assert stats["total_books"] == 1
    assert stats["total_chunks"] > 0
    print(f"[OK] stats: {stats}")

    results = bm.search("dream layers time dilation", n_results=3)
    assert len(results) > 0
    print(f"[OK] search: {len(results)} results")
    for r in results:
        print(f"  dist={r.distance:.3f} chapter={r.chapter} preview={r.text[:50]}")

    context = bm.build_context("how does the kick mechanism work", max_chars=2000)
    assert "[BOOK_CONTEXT]" in context
    assert "Inception Explained" in context
    assert "The Kick" in context or "Chapter 3" in context
    print(f"[OK] build_context: {len(context)} chars")

    # Empty results should return empty string
    context_empty = bm.build_context("xyznonexistentcontent12345", max_chars=2000)
    assert context_empty == ""
    print("[OK] build_context empty = ''")

    # Dedup: same text should return same book_id
    same_id = bm.ingest_text(text, title="Duplicate")
    assert same_id == book_id
    print(f"[OK] ingest_text dedup: same_id")

    # Force reingest
    new_id = bm.ingest_text(text, title="Forced", force=True)
    assert new_id != book_id
    print(f"[OK] ingest_text force: new_id={new_id}")

    # Delete
    bm.delete_book(new_id)
    assert bm.get_book(new_id) is None
    print(f"[OK] delete_book: {new_id} removed")

    # List books (the force=True deleted the original, then created new, then we deleted new -> 0 books)
    books = bm.list_books()
    print(f"[OK] list_books: {len(books)} book(s) after cleanup")

    # Validate and reindex with a fresh book
    fresh_id = bm.ingest_text(text, title="Validation Test")
    validation = bm.validate_index(fresh_id)
    assert validation.get("exists", False) == True
    assert validation["match"] == True
    print(f"[OK] validate_index: sql={validation['sql_chunks']} chroma={validation['chroma_chunks']} match={validation['match']}")

    initial_chunks = bm.get_chunks(fresh_id)
    bm.reindex_book(fresh_id)
    after_chunks = bm.get_chunks(fresh_id)
    assert len(initial_chunks) == len(after_chunks)
    print(f"[OK] reindex_book: {len(after_chunks)} chunks preserved")

    bm.delete_book(fresh_id)
    bm.close()
    mem.close()
    print("\n=== ALL BOOK_MEMORY TESTS PASSED ===")


if __name__ == "__main__":
    test_should_search_books()
    test_chunk_id_format()
    test_chunk_text_with_pages()
    test_chunk_text_simple()
    test_book_memory_ingest_text()
