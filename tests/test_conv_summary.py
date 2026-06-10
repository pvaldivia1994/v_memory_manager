"""Regression tests for ConversationSummaryMemory."""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import ConversationSummaryMemory, MemoryManager
from src import db

ok = 0
fail = 0


def check(label: str, got, expected) -> None:
    global ok, fail
    if got == expected:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL  {label}: got={got} expected={expected}")


def make_conv(max_messages: int = 10, margin: int = 4):
    mem = MemoryManager()
    db_path = os.path.join(tempfile.mkdtemp(), "test.db")
    mem.create_memory_db(db_path)

    conv = ConversationSummaryMemory(
        sqlite_conn=mem._conn,
        conversation_id="test",
        max_messages=max_messages,
        summarize_margin=margin,
    )

    def dummy(old_summary, messages, max_chars):
        return (
            "Tema actual:\n- Test\n\n"
            "Estado actual:\n- Probando\n\n"
            "Decisiones tomadas:\n- Usar summary\n\n"
            "Pendientes:\n- Tests"
        )

    conv.summarizer = dummy
    return mem, conv, db_path


def add_n_messages(mem: MemoryManager, n_pairs: int) -> None:
    for i in range(n_pairs):
        mem.add_message("user", f"mensaje usuario {i+1}")
        mem.add_message("assistant", f"respuesta asistente {i+1}")


def cleanup(mem, db_path):
    mem.close()
    try:
        os.remove(db_path)
    except PermissionError:
        pass


# ── 1. Poco historial → no resume ───────────────────────────
print("=== 1. Poco historial ===")
mem, conv, db_path = make_conv(max_messages=10)
add_n_messages(mem, 4)
assert mem.count_messages() == 8
check("no resume si no hay suficientes", conv.maybe_update(), False)
cleanup(mem, db_path)

# ── 2. Resumen con 15 mensajes ──────────────────────────────
print("\n=== 2. Resumen con 15 mensajes ===")
mem, conv, db_path = make_conv(max_messages=10, margin=2)
add_n_messages(mem, 7)
mem.add_message("user", "msg extra 15")
assert mem.count_messages() == 15
updated = conv.maybe_update()
check("resume cuando hay fuera", updated, True)
summary = conv.get_summary()
check("summary no vacio", len(summary) > 0, True)
check("formato estructurado", "Tema actual" in summary, True)
check("formato estado actual", "Estado actual" in summary, True)
check("formato decisiones", "Decisiones tomadas" in summary, True)
cleanup(mem, db_path)

# ── 3. Ya resumido ──────────────────────────────────────────
print("\n=== 3. Ya resumido ===")
mem, conv, db_path = make_conv(max_messages=10, margin=2)
add_n_messages(mem, 7)
conv.maybe_update()
check("segunda llamada no resume", conv.maybe_update(), False)
cleanup(mem, db_path)

# ── 4. Summarizer vacio ─────────────────────────────────────
print("\n=== 4. Summarizer vacío ===")
mem, conv, db_path = make_conv(max_messages=10, margin=2)
add_n_messages(mem, 7)
conv.summarizer = lambda old_summary, messages, max_chars: ""
conv.maybe_update()
check("summary sigue vacio", conv.get_summary(), "")
check("error count", conv.get_state()["summary_error_count"], 1)
cleanup(mem, db_path)

# ── 5. Summarizer falla → no rompe ──────────────────────────
print("\n=== 5. Summarizer falla ===")
mem, conv, db_path = make_conv(max_messages=10, margin=2)
add_n_messages(mem, 7)
conv.summarizer = lambda old_summary, messages, max_chars: 1 / 0  # type: ignore
conv.maybe_update()
check("summary vacio tras error", conv.get_summary(), "")
check("error count incrementado", conv.get_state()["summary_error_count"], 1)
cleanup(mem, db_path)

# ── 6. Reset ─────────────────────────────────────────────────
print("\n=== 6. Reset ===")
mem, conv, db_path = make_conv(max_messages=10, margin=2)
add_n_messages(mem, 7)
conv.maybe_update()
assert conv.get_summary() != ""
conv.summarizer = lambda old_summary, messages, max_chars: 1 / 0  # type: ignore
conv.update_summary([{"id": 1, "role": "user", "content": "x", "created_at": ""}])
assert conv.get_state()["summary_error_count"] > 0
conv.reset()
check("summary vacio", conv.get_summary(), "")
check("last_id 0", conv.get_last_summarized_message_id(), 0)
check("errores 0", conv.get_state()["summary_error_count"], 0)
check("status active", conv.get_state()["status"], "active")
check("last_error vacio", conv.get_state()["last_error"], "")
check("last_summarized_created_at vacio", conv.get_state()["last_summarized_created_at"], "")
cleanup(mem, db_path)

# ── 7. build_context_block sin resumen ──────────────────────
print("\n=== 7. build_context_block sin resumen ===")
mem, conv, db_path = make_conv(max_messages=10)
block = conv.build_context_block()
check("contiene Sin resumen previo", "Sin resumen previo" in block, True)
check("empieza con tag", block.startswith("[CONVERSATION_SUMMARY]"), True)
cleanup(mem, db_path)

# ── 8. Excluir system ───────────────────────────────────────
print("\n=== 8. Excluir system ===")
mem, conv, db_path = make_conv(max_messages=10, margin=1)
mem.add_message("system", "Eres un asistente util")
add_n_messages(mem, 6)
conv.maybe_update()
check("summary generado pese a system", conv.get_summary() != "", True)
cleanup(mem, db_path)

# ── 9. Truncado ─────────────────────────────────────────────
print("\n=== 9. Truncado ===")
mem, conv, db_path = make_conv(max_messages=10, margin=2)
conv.max_summary_chars = 30
add_n_messages(mem, 7)
conv.summarizer = lambda old_summary, messages, max_chars: "a" * 100
conv.maybe_update()
summary = conv.get_summary()
marker = "\n- [Resumen truncado por límite]"
check("truncado <= max+marker", len(summary) <= 30 + len(marker), True)
check("marca de truncado", "truncado" in summary, True)
cleanup(mem, db_path)

# ── 10. No huecos ───────────────────────────────────────────
print("\n=== 10. No huecos ===")
mem, conv, db_path = make_conv(max_messages=10, margin=2)
add_n_messages(mem, 7)
mem.add_message("user", "extra 1")
mem.add_message("assistant", "extra 2")
conv.maybe_update()
first_id = db.get_first_window_message_id(conv.conn, conv.window_size)
last_id = conv.get_last_summarized_message_id()
if first_id is not None:
    check("sin hueco: last_summarized == first_window - 1", last_id, first_id - 1)
else:
    check("no hay ventana (no deberia pasar)", False, True)
cleanup(mem, db_path)

# ── 11. No resumir visibles ─────────────────────────────────
print("\n=== 11. No resumir visibles ===")
mem, conv, db_path = make_conv(max_messages=10, margin=2)
add_n_messages(mem, 7)
conv.maybe_update()
first_id = db.get_first_window_message_id(conv.conn, conv.window_size)
last_id = conv.get_last_summarized_message_id()
if first_id is not None:
    check("ultimo resumido antes del primer visible", last_id < first_id, True)
cleanup(mem, db_path)

# ── 12. Disabled ────────────────────────────────────────────
print("\n=== 12. Disabled ===")
mem, conv, db_path = make_conv(max_messages=10, margin=2)
add_n_messages(mem, 7)
db.upsert_conv_summary_state(conv.conn, conv.conversation_id, conv.user_id, status="disabled")
check("maybe_update false si disabled", conv.maybe_update(), False)
block = conv.build_context_block()
check("build_context_block dice desactivado", "desactivado" in block, True)
cleanup(mem, db_path)


total = ok + fail
print(f"\n=== Results: {ok} passed, {fail} failed ===")
if fail:
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
