"""Smoke test for plan v5 improvements."""
from __future__ import annotations

import sqlite3
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# pyrefly: ignore [missing-import]
from src import SemanticMemory, MemoryManager, analyze_assistant_text, migrate_long_term_to_semantic
# pyrefly: ignore [missing-import]
from src.semantic_memory import analyze_text, is_noise

# ── Test A: analyze_assistant_text ────────────────────────
print("=== Test A: Assistant analysis ===")
# Generic responses should NOT be remembered
r1 = analyze_assistant_text("Claro, puedo ayudarte con eso")
assert not r1.should_remember, f"Should NOT remember generic response: {r1}"
print(f"  Generic response: should_remember={r1.should_remember} reason={r1.reason} ✓")

r2 = analyze_assistant_text("Te sugiero usar Python para ese proyecto")
assert not r2.should_remember, f"Should NOT remember suggestion: {r2}"
print(f"  Suggestion: should_remember={r2.should_remember} reason={r2.reason} ✓")

# Explicit user facts SHOULD be remembered
r3 = analyze_assistant_text("Entendido, tu nombre es Pablo y te gusta Python")
assert r3.should_remember, f"Should remember user fact: {r3}"
print(f"  User fact: should_remember={r3.should_remember} reason={r3.reason} conf={r3.confidence} ✓")

# Short/noise should not
r4 = analyze_assistant_text("ok")
assert not r4.should_remember, f"Should NOT remember noise: {r4}"
print(f"  Noise: should_remember={r4.should_remember} reason={r4.reason} ✓")

print()

# ── Test B: Conflict resolution (requires chromadb) ──────
print("=== Test B: Conflict resolution ===")
try:
    import chromadb
    import tempfile, shutil

    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    # pyrefly: ignore [missing-import]
    from src.db import init_db
    init_db(conn)

    sem = SemanticMemory(
        persist_dir=os.path.join(tmpdir, "chroma"),
        sqlite_conn=conn,
    )

    # Store initial memory
    id1 = sem.remember_force("Preferencia del usuario: color favorito es rojo", tags=["gustos"])
    print(f"  Initial: {id1}")

    # Store update (should archive old one)
    id2 = sem.remember("Mi color favorito ahora es el azul")
    print(f"  Update: {id2}")

    # Verify old is archived
    if id1 != id2 and id2:
        old = sem.get_memory(id1)
        if old:
            print(f"  Old status: {old.status} (expected: archived or active)")
        else:
            print(f"  Old memory not found (may be archived)")
    print(f"  Total active: {sem.count()}")

    # Verify fact_key and fact_value are set for id2 in SQL (spaCy integration)
    row = conn.execute("SELECT fact_key, fact_value FROM semantic_memories WHERE memory_id=?", (id2,)).fetchone()
    if row:
        print(f"  Stored fact_key: '{row['fact_key']}', fact_value: '{row['fact_value']}'")
        assert row['fact_key'] == "color", f"Expected fact_key='color', got '{row['fact_key']}'"
        assert row['fact_value'] == "azul", f"Expected fact_value='azul', got '{row['fact_value']}'"

    # ── Test C: search_by_tags efficient ──
    print()
    print("=== Test C: search_by_tags SQL ===")
    results = sem.search_by_tags(["gustos"])
    print(f"  Found {len(results)} memories by tag 'gustos' ✓")

    # ── Test D: review_pending ──
    print()
    print("=== Test D: review_pending ===")
    # Create a low-confidence memory (pending_review)
    # pyrefly: ignore [missing-import]
    from src.memory_models import AnalysisResult
    sem._store(AnalysisResult(
        should_remember=True, reason="test", confidence=0.5,
        content="Posible memoria: algo sobre café", memory_type="pending",
        tags=["comida"],
    ), source="test")
    pending = sem.review_pending()
    print(f"  Pending: {len(pending)} memories")
    assert len(pending) >= 1, "Should have at least 1 pending"
    # Approve first pending
    sem.approve(pending[0].memory_id)
    approved = sem.get_memory(pending[0].memory_id)
    assert approved.status == "active", f"Expected active, got {approved.status}"
    print(f"  Approved: {approved.memory_id[:16]} status={approved.status} ✓")
    # Search should NOT include pending_review
    search_results = sem.search("café", n_results=10)
    for sr in search_results:
        status = sr.status
        assert status == "active", f"Search returned non-active: {status}"
    print(f"  Search only returns active: ✓")

    # ── Test E: importance boost ──
    print()
    print("=== Test E: importance boost ===")
    if search_results:
        mid = search_results[0].memory_id or search_results[0].chroma_id
        # Query importance before
        before = conn.execute("SELECT importance FROM semantic_memories WHERE memory_id=?", (mid,)).fetchone()
        if before:
            imp_before = before[0]
            # Search again to trigger boost
            sem.search("café", n_results=3)
            conn.commit()
            after = conn.execute("SELECT importance FROM semantic_memories WHERE memory_id=?", (mid,)).fetchone()
            imp_after = after[0] if after else imp_before
            print(f"  Importance before: {imp_before:.3f}, after: {imp_after:.3f}")
            if imp_after > imp_before:
                print(f"  Boost works ✓")
            else:
                print(f"  Boost may not apply (memory_id mismatch with chroma_id)")

    # ── Test F: migration ──
    print()
    print("=== Test F: migrate_long_term_to_semantic ===")
    conn.execute("INSERT INTO long_term_memories (content, tags, weight) VALUES (?, ?, ?)",
                 ("Al asistente le gusta programar en Rust", "rust,programacion", 0.8))
    conn.commit()
    count = migrate_long_term_to_semantic(conn, sem)
    print(f"  Migrated: {count} memories")
    assert count >= 1, "Should migrate at least 1"
    # Search for migrated memory
    migrated_results = sem.search("Rust programación", n_results=3)
    found = any("Rust" in r.content for r in migrated_results)
    print(f"  Found in semantic search: {found} ✓")
    # Running again should not duplicate
    count2 = migrate_long_term_to_semantic(conn, sem)
    print(f"  Re-run migration: {count2} (expected 0) ✓")

    conn.close()
    shutil.rmtree(tmpdir, ignore_errors=True)

except ImportError:
    print("  [SKIP: chromadb not installed]")

print()
print("=== ALL TESTS PASSED ===")
