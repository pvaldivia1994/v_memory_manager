"""Smoke tests for BookMemory (parent-child)."""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.book_memory import (
    BookMemory,
    _hash_text,
    chunk_text,
    should_search_books,
    _generate_book_id,
    _make_parent_chunk_id,
    _make_child_chunk_id,
    clean_extracted_text,
    detect_chapters,
)
from src import MemoryManager


def _fake_embed(text: str) -> list[float]:
    import hashlib
    h = hashlib.md5(text.encode()).hexdigest()
    return [int(h[i:i + 2], 16) / 255.0 for i in range(0, 32, 2)]


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
    pid = _make_parent_chunk_id(bid, 0)
    assert pid == f"{bid}_ch_0000"
    cid = _make_child_chunk_id(bid, 0, 42)
    assert cid == f"{bid}_ch_0000_s_0042"
    print(f"[OK] chunk_id format: {pid}, {cid}")


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
        print(f"  chunk {i}: pages={c['page_start']}-{c['page_end']}, len={len(c['text'])}")


def test_detect_chapters():
    text = """Some intro text that is long enough to pass the minimum chapter size filter.
Chapter 1: First Chapter
Content of chapter one. This has enough text to be a valid chapter with more than one hundred characters total.
Chapter 2: Second Chapter
Content of chapter two. Also long enough to be a real chapter with substantial content in it."""
    chapters = detect_chapters(text)
    assert len(chapters) >= 2
    print(f"[OK] detect_chapters: {len(chapters)} chapters")
    for s, e, n in chapters:
        print(f"  {n}: {e - s}c")

    text_no_markers = "Just a long block of text without any chapter markers. " * 500
    chapters2 = detect_chapters(text_no_markers)
    assert len(chapters2) >= 1
    print(f"[OK] detect_chapters fallback: {len(chapters2)} chapters")


def test_book_memory_ingest_text():
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test.db")

    mem = MemoryManager()
    mem.create_memory_db(db_path)

    bm = BookMemory(
        sqlite_conn=mem._conn,
        embed_fn=_fake_embed,
        chunk_size_chars=300,
        chunk_overlap_chars=50,
    )

    assert bm.has_books() == False

    text = """[PAGE 1]
Chapter 1: Inception
Inception is a 2010 science fiction film directed by Christopher Nolan.
The film stars Leonardo DiCaprio as Dom Cobb, a professional thief.

[PAGE 2]
Chapter 2: Dream Layers
The film explores the concept of dreams within dreams.
Each layer of the dream has its own unique time dilation.

[PAGE 3]
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
    assert book["source_type"] == "text"
    assert book["total_chapters"] > 0
    print(f"[OK] book metadata: title={book['title']} status={book['status']} chapters={book['total_chapters']}")

    chunks = bm.get_chunks(book_id)
    assert len(chunks) > 0
    print(f"[OK] get_chunks: {len(chunks)} chunks")

    chapters = bm.list_chapters(book_id)
    assert len(chapters) > 0
    print(f"[OK] list_chapters: {len(chapters)} chapters")
    for ch in chapters:
        print(f"  [{ch.chapter_index}] {ch.chapter} ({ch.char_count}c)")

    stats = bm.get_stats()
    assert stats["total_books"] == 1
    assert stats["total_chunks"] > 0
    print(f"[OK] stats: {stats}")

    results = bm.search("dream layers time dilation", book_id=book_id, n_results=3)
    assert len(results) > 0
    print(f"[OK] search: {len(results)} results")

    context = bm.build_context("how does the kick mechanism work", book_id=book_id, max_chars=2000)
    assert "[BOOK_CONTEXT]" in context
    print(f"[OK] build_context: {len(context)} chars")

    context_unknown = bm.build_context("xyznonexistentcontent12345", book_id=book_id, max_chars=2000)
    print(f"[OK] build_context unlikely query: got {len(context_unknown)} chars")

    same_id = bm.ingest_text(text, title="Duplicate")
    assert same_id == book_id
    print(f"[OK] ingest_text dedup: same_id")

    new_id = bm.ingest_text(text, title="Forced", force=True)
    assert new_id != book_id
    print(f"[OK] ingest_text force: new_id={new_id}")

    bm.delete_book(new_id)
    assert bm.get_book(new_id) is None
    print(f"[OK] delete_book: {new_id} removed")

    books = bm.list_books()
    print(f"[OK] list_books: {len(books)} book(s) after cleanup")

    fresh_id = bm.ingest_text(text, title="Validation Test")
    validation = bm.validate_index(fresh_id)
    assert validation.get("exists", False) == True
    print(f"[OK] validate_index: sections={validation['section_chunks']} total={validation['total_chunks']}")

    chapter = bm.get_chapter(fresh_id, 2)
    assert chapter is not None
    assert "kick" in chapter["chunk_text"].lower() or "Kick" in chapter["chunk_text"]
    print(f"[OK] get_chapter: {chapter['chapter']}")

    chapter_ctx = bm.build_chapter_context(fresh_id, 2, max_chars=5000)
    assert "[BOOK_CONTEXT]" in chapter_ctx
    print(f"[OK] build_chapter_context: {len(chapter_ctx)} chars")

    bm.delete_book(fresh_id)
    bm.close()
    mem.close()
    print("\n=== ALL BOOK_MEMORY TESTS PASSED ===")


def test_dual_hash():
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test.db")

    mem = MemoryManager()
    mem.create_memory_db(db_path)
    bm = BookMemory(sqlite_conn=mem._conn, chunk_size_chars=300, chunk_overlap_chars=50)

    text_a = "Chapter 1: Cats\nCats are domestic animals. They are very independent."
    text_b = "Chapter 1: Cats\nCats are domestic animals of the family Felidae."

    id_a = bm.ingest_text(text_a, title="Cats A")
    assert id_a != ""
    print(f"[OK] dual_hash: first book id={id_a}")

    id_b = bm.ingest_text(text_b, title="Cats B")
    assert id_b != id_a
    print(f"[OK] dual_hash: different text -> different id")

    id_a2 = bm.ingest_text(text_a, title="Cats A dup")
    assert id_a2 == id_a
    print(f"[OK] dual_hash: same text -> dedup")

    id_a3 = bm.ingest_text(text_a, title="Cats A forced", force=True)
    assert id_a3 != id_a
    print(f"[OK] dual_hash: force -> new id")

    book = bm.get_book(id_a3)
    assert book["source_text_hash"] == _hash_text(text_a)
    assert book["source_type"] == "text"

    bm.delete_book(id_a3)
    bm.delete_book(id_b)
    bm.close()
    mem.close()
    print("[OK] test_dual_hash")


def test_extract_and_ingest_no_ingest():
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test.db")

    mem = MemoryManager()
    mem.create_memory_db(db_path)
    bm = BookMemory(sqlite_conn=mem._conn)

    txt_path = os.path.join(tmpdir, "test_book.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("[PAGE 1]\nChapter 1\nThis is a test book for extraction.\n")

    cache_result = bm.extract_and_ingest(
        txt_path, cache_dir=tmpdir, title="Test Book",
        ingest=False,
    )
    assert os.path.exists(cache_result)
    assert bm.has_books() == False

    book_id = bm.ingest(txt_path, title="Test Book")
    assert book_id != ""
    assert bm.has_books() == True

    bm.delete_book(book_id)
    bm.close()
    mem.close()
    print("[OK] test_extract_and_ingest_no_ingest")


def test_truncate_centered():
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test.db")

    mem = MemoryManager()
    mem.create_memory_db(db_path)
    bm = BookMemory(sqlite_conn=mem._conn, embed_fn=_fake_embed)

    text = """[PAGE 1]
Chapter 1: The Beginning
This is the start of a very long chapter.
It contains multiple sections of content.
The middle section has the most relevant information.
And then it continues with more text.
And even more text after that.
Finally it reaches the end of the chapter.
This is the last paragraph before the next chapter."""

    book_id = bm.ingest_text(text, title="Test Truncation")
    chapter = bm.get_chapter(book_id, 0)

    child_rows = [
        {"parent_chunk_id": chapter["chunk_id"], "chunk_text": "middle section has the most relevant"},
    ]

    result = bm.truncate_centered(chapter, child_rows, max_chars=200)
    assert "[...]" in result
    print(f"[OK] truncate_centered: {len(result)} chars, has markers")

    full = bm.truncate_centered(chapter, child_rows, max_chars=10000)
    assert "[...]" not in full
    print(f"[OK] truncate_centered no truncation: {len(full)} chars")

    bm.delete_book(book_id)
    bm.close()
    mem.close()
    print("[OK] test_truncate_centered")


def test_handle_language_mismatch():
    bm = BookMemory()
    q = bm.handle_language_mismatch("hello", "en", strategy="passthrough")
    assert q == "hello"
    q2 = bm.handle_language_mismatch("hello", "en", strategy="translate")
    assert q2 == "hello"
    print("[OK] test_handle_language_mismatch")


def test_toc_dot_leader_columns():
    """TOC page: y-only sort, all entries present. 12+ entries triggers _page_is_toc."""
    from src.book_memory import _extract_page_blocks

    class MockPage:
        rect = type("rect", (), {"width": 612, "height": 792})()

        def get_text(self, kind):
            entries = []
            y = 60
            for title in [
                "About this Book...................................2",
                "Preface..................................................3",
                "Ch. 1: The World of Grimnir...................4",
                "History of Grimnir.................................4",
                "Traveling To Grimnir.............................8",
                "Ch. 2: Creating Heroes..........................22",
                "Grimnir Settlers...................................22",
                "Tools of the Raiders.............................22",
                "Ch. 3: Playable Races............................37",
                "Beastborn.............................................37",
                "Ch. 4: Class Archetypes..........................46",
                "Ranger: Wolf Rider...............................59",
            ]:
                x = 72 if entries.count(0) % 2 == 0 else 360
                entries.append((x, y, x+400, y+12, title, 0, 0))
                y += 30
                if y > 200:
                    y = 60
                    x = 360
            return entries

    page = MockPage()
    result = _extract_page_blocks(page, 1)

    lines = [l for l in result.split("\n") if l.strip() and not l.startswith("[PAGE")]
    texts = [l.strip() for l in lines]

    assert len(texts) >= 10, f"Expected 10+ lines, got {len(texts)}"
    assert any("About" in t for t in texts)
    assert any("Preface" in t for t in texts)
    assert any("Ch. 1" in t for t in texts)
    assert any("Ch. 2" in t for t in texts)
    assert any("Ch. 3" in t for t in texts)
    assert any("Ch. 4" in t for t in texts)
    print(f"[OK] TOC y-sort: {len(texts)} entries")


def test_three_column_toc():
    """TOC page with 3 columns — y-only sort, all entries present."""
    from src.book_memory import _extract_page_blocks

    class MockPage:
        rect = type("rect", (), {"width": 612, "height": 792})()

        def get_text(self, kind):
            return [
                (200, 20, 400, 35, "Contents", 0, 0),
                (72, 60, 540, 72, "About this Book...................................2", 0, 0),
                (260, 60, 500, 72, "Ch. 2: Creating Heroes..........................22", 0, 0),
                (430, 60, 560, 72, "Ch. 4: Class Archetypes..........................46", 0, 0),
                (72, 90, 540, 102, "Preface..................................................3", 0, 0),
                (260, 90, 500, 102, "Ch. 3: Playable Races............................37", 0, 0),
                (430, 90, 560, 102, "Ranger: Wolf Rider...............................59", 0, 0),
                (72, 120, 540, 132, "Ch. 1: The World of Grimnir...................4", 0, 0),
                (260, 120, 500, 132, "Beastborn............................................37", 0, 0),
                (430, 120, 560, 132, "Rogue: True Believer.............................63", 0, 0),
                (72, 600, 540, 612, "Ch. 5: Ships of the Sea...........................74", 0, 0),
                (430, 600, 560, 612, "Ch. 5 subentries here.........................76", 0, 0),
            ]

    page = MockPage()
    result = _extract_page_blocks(page, 1)

    lines = [l for l in result.split("\n") if l.strip() and not l.startswith("[PAGE")]
    texts = [l.strip() for l in lines]

    print(f"[DEBUG] 3-col TOC y-sort ({len(texts)} lines):")
    for t in texts:
        print(f"  {t[:60]}")

    assert any("About" in t for t in texts), "Missing About"
    assert any("Ch. 1" in t for t in texts), "Missing Ch. 1"
    assert any("Ch. 2" in t for t in texts), "Missing Ch. 2"
    assert any("Ch. 3" in t for t in texts), "Missing Ch. 3"
    assert any("Ch. 4" in t for t in texts), "Missing Ch. 4"
    assert any("Ch. 5" in t for t in texts), "Missing Ch. 5"
    print(f"[OK] 3-col TOC y-sort: {len(texts)} entries, all present")


def test_spanned_title_with_two_columns():
    """Page with a title spanning full width + 2-column content below."""
    from src.book_memory import _extract_page_blocks

    class MockPage:
        rect = type("rect", (), {"width": 612, "height": 792})()

        def get_text(self, kind):
            return [
                (50, 20, 562, 40, "Chapter Title That Spans Full Width", 0, 0),
                (72, 60, 300, 120, "Left column paragraph one. Has enough text to be a real block with content.",
                 0, 0),
                (360, 60, 560, 120, "Right column paragraph one. Also has enough text to be read properly.",
                 0, 0),
                (72, 140, 300, 200, "More text in the left column. Continues the discussion of the topic.",
                 0, 0),
                (360, 140, 560, 200, "Right column continues with related content for the section.",
                 0, 0),
            ]

    page = MockPage()
    result = _extract_page_blocks(page, 1)

    lines = [l for l in result.split("\n") if l.strip() and not l.startswith("[PAGE")]
    texts = [l.strip() for l in lines]

    print(f"[DEBUG] Spanned title + 2 columns ({len(texts)} lines):")
    for t in texts:
        print(f"  {t[:60]}")

        assert texts[0].startswith("Chapter Title That Spans"), f"Title first: {texts[0][:40]}"
        assert texts[1].startswith("Left column paragraph"), f"Left col 1: {texts[1][:40]}"
        assert texts[2].startswith("More text in the left"), f"Left col 2: {texts[2][:40]}"
        assert texts[3].startswith("Right column paragraph"), f"Right col 1: {texts[3][:40]}"
    print(f"[OK] Spanned title → left col → right col")


def test_clean_markdown_pdf_text():
    """Post-processor debe partir TOC merges y quitar bold markdown."""
    from src.book_memory import clean_markdown_pdf_text

    text = (
        "**About this Book** ........................2 "
        "**Preface** ........................3 "
        "**Ch. 1: The World of Grimnir** ........................4\n"
        "History of Grimnir ........................4\n"
    )
    result = clean_markdown_pdf_text(text)
    assert "**" not in result, f"Bold not removed: {result}"
    lines = [l for l in result.split("\n") if l.strip()]
    assert any("About this Book" in l for l in lines), f"Missing About: {lines}"
    assert any("Ch. 1" in l for l in lines), f"Missing Ch. 1: {lines}"
    assert any("History of Grimnir" in l for l in lines)
    print(f"[OK] clean_markdown_pdf_text: {len(lines)} lines")


def test_clean_extracted_text():
    text = "soft\u00ad\nhyphen\u00adned\n\n\n\n\nword"
    cleaned = clean_extracted_text(text)
    assert "\u00ad" not in cleaned
    assert "\n\n\n" not in cleaned
    print(f"[OK] clean_extracted_text: {cleaned}")


if __name__ == "__main__":
    test_should_search_books()
    test_chunk_id_format()
    test_chunk_text_with_pages()
    test_detect_chapters()
    test_book_memory_ingest_text()
    test_dual_hash()
    test_extract_and_ingest_no_ingest()
    test_truncate_centered()
    test_handle_language_mismatch()
    test_clean_extracted_text()
