from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(Path(__file__).with_suffix(".log"), encoding="utf-8"),
        logging.StreamHandler(sys.stderr),
    ],
)
log = logging.getLogger("chat")

_ROOT = Path(__file__).resolve().parent.parent
_VLLAMA = _ROOT.parent / "v_llama"
_CHROMA_DIR = _ROOT / ".chatdb" / "chroma"

# ── importar v_llama ─────────────────────────────────────
if not _VLLAMA.exists():
    print(f"ERROR: no se encuentra v_llama en {_VLLAMA}", file=sys.stderr)
    sys.exit(1)

sys.path.insert(0, str(_VLLAMA))
from src import VLLaMA
sys.path.remove(str(_VLLAMA))

# ── importar v_memory_manager ────────────────────────────
sys.path.insert(0, str(_ROOT))
for k in list(sys.modules):
    if k.startswith("src"):
        del sys.modules[k]
from src import BookMemory, ConversationSummaryMemory, MemoryManager, SemanticMemory, should_search_books

DIM = "\033[90m"
CYAN = "\033[36m"
GREEN = "\033[32m"
RESET = "\033[0m"


def _show_stats(r, msg_ids: list[int] | None = None,
                mem_count: int | None = None,
                book_chars: int = 0) -> None:
    total = r.prompt_tokens + r.completion_tokens
    remaining = r.context_remaining
    pct = round((total / r.context_limit) * 100, 1) if r.context_limit else 0

    print()
    print(
        f"{DIM}"
        f"[{r.duration_ms}ms | {r.tokens_per_second} tok/s]",
        end=""
    )
    if mem_count is not None:
        print(f"  memorias {mem_count}", end="")
    if book_chars:
        print(f"  libro {book_chars}c", end="")
    print()
    print(
        f"{DIM}"
        f"  prompt  {r.prompt_tokens:>6}  (system {r.system_tokens}, "
        f"history {r.history_tokens}, user {r.prompt_tokens - r.system_tokens - r.history_tokens})\n"
    )
    if msg_ids is not None:
        print(f"  messages {msg_ids}")
    print(
        f"{DIM}"
        f"  output  {r.completion_tokens:>6}\n"
        f"  total   {total:>6}  ({pct}% de {r.context_limit})\n"
        f"  libre   {remaining:>6}\n"
        f"  config: t={r.temperature}, p={r.top_p}, k={r.top_k}, "
        f"rp={r.repeat_penalty}, max={r.max_tokens}"
        f"{RESET}"
    )


def _replay_history(mem) -> None:
    hist = mem.get_history()
    for m in hist:
        if m.role == "system":
            print(f"{DIM}[System]{RESET} {m.content}")
        elif m.role == "user":
            print(f"{CYAN}Tu:{RESET} {m.content}")
        elif m.role == "assistant":
            print(f"{GREEN}Asistente:{RESET} {m.content}")


def _pick_model(llm) -> str:
    models = llm.list_models()
    if not models:
        print("No se encontraron modelos GGUF en el directorio configurado.")
        sys.exit(1)

    print(f"{DIM}Modelos disponibles:{RESET}")
    for i, m in enumerate(models, 1):
        print(f"  {i}. {m}")

    while True:
        try:
            idx = int(input(f"{CYAN}Seleccione modelo (1-{len(models)}):{RESET} ").strip())
            if 1 <= idx <= len(models):
                return models[idx - 1]
        except (ValueError, EOFError, KeyboardInterrupt):
            pass
        print(f"  Opción inválida. Elija entre 1 y {len(models)}.")


def _ask_continue(db_path: Path, mem: MemoryManager) -> None:
    if not db_path.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)
        mem.create_memory_db(str(db_path))
        print(f"[Memoria creada: {db_path.name}]")
        return

    mem.load_memory_db(str(db_path))
    print(f"[DB existente: {db_path.name}]")

    if mem.count_messages() == 0:
        return

    resp = input("¿Cargar conversación anterior? (s/N): ").strip().lower()
    if resp == "s":
        print(f"{DIM}── Historial cargado ──{RESET}")
        _replay_history(mem)
        print(f"{DIM}──────────────────────{RESET}")
    else:
        mem.clear_memory_db()
        print("[DB reiniciada]")


def _ask_system_prompt(mem: MemoryManager) -> str:
    current = mem.get_system_prompt()
    print(f"\n{DIM}System prompt actual:{RESET} {current or '(ninguno)'}")
    resp = input("¿Desea cambiarlo? (s/N): ").strip().lower()
    if resp != "s":
        return current or ""

    while True:
        nuevo = input(f"{CYAN}Nuevo system prompt:{RESET} ").strip()
        if nuevo:
            mem.add_message("system", nuevo)
            print("[System prompt actualizado]\n")
            return nuevo
        print("  El prompt no puede estar vacío.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Chat por consola con v_llama + v_memory_manager")
    parser.add_argument("--db", default=str(_ROOT / ".chatdb" / "chat.db"), help="Ruta a la DB de memoria")
    parser.add_argument("--config", help="Ruta al config.json de v_llama")
    parser.add_argument("--model", help="Nombre o ruta del modelo GGUF")
    parser.add_argument("--no-stream", action="store_true", help="Deshabilitar streaming")
    parser.add_argument("--book", help="Ruta a PDF/EPUB/TXT para ingestar al iniciar")
    parser.add_argument("--index", help="Archivo de indice (titulo|pagina) para definir capitulos")
    parser.add_argument("--extract", help="Solo extraer texto a TXT (no indexar)")
    parser.add_argument("--pdf-layout", default="",
                        choices=["", "plain", "blocks", "two_columns", "columns", "pymupdf4llm"],
                        help="Layout: '' (pymupdf4llm + fallback), plain, blocks, columns")
    args = parser.parse_args()

    # ── v_llama ────────────────────────────────────────────────
    llm = VLLaMA(config_path=args.config, auto_load=False)

    if args.model:
        model_name = args.model
    else:
        model_name = _pick_model(llm)

    llm.load_model(model_name)

    print(f"\n{DIM}Modelo:{RESET} {llm.model_name}")

    # ── Memoria ────────────────────────────────────────────────
    mem = MemoryManager()
    _ask_continue(Path(args.db), mem)
    mem.set_config("model_name", llm.model_name)

    try:
        sem = SemanticMemory(
            persist_dir=str(_CHROMA_DIR),
            sqlite_conn=mem._conn,
            allow_assistant_memory=True,
        )
    except Exception as e:
        print(f"{DIM}[SemanticMemory no disponible: {e}]{RESET}")
        sem = None

    if args.extract:
        try:
            bm_temp = BookMemory(sqlite_conn=mem._conn)
            out = bm_temp.extract_to_txt(
                args.extract,
                str(_ROOT / ".chatdb" / "extracted" / Path(args.extract).with_suffix(".txt").name),
                pdf_layout=args.pdf_layout,
            )
            print(f"[Extraccion completada. Revisa el archivo antes de indexar con --book]")
        except Exception as e:
            print(f"[Error al extraer: {e}]")
        return

    embed_fn = None
    try:
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        def embed_fn(text: str) -> list[float]:
            return _EMBED_MODEL.encode(text).tolist()
    except ImportError:
        print(f"{DIM}[Embedding no disponible: pip install sentence-transformers]{RESET}")

    try:
        book_mem = BookMemory(
            sqlite_conn=mem._conn,
            embed_fn=embed_fn,
        )
        if args.book:
            book_path = Path(args.book)
            if not book_path.exists():
                print(f"{DIM}[Libro no encontrado: {args.book}]{RESET}")
            else:
                try:
                    extracted_dir = _ROOT / ".chatdb" / "extracted"
                    extracted_dir.mkdir(parents=True, exist_ok=True)
                    save_path = str(extracted_dir / book_path.with_suffix(".txt").name)
                    ingest_kw = dict(
                        pdf_layout=args.pdf_layout,
                        save_extracted_to=save_path,
                    )
                    if args.index:
                        ingest_kw["index_path"] = args.index
                    bid = book_mem.ingest(str(book_path), **ingest_kw)
                    b = book_mem.get_book(bid)
                    print(f"[Libro cargado: {b['title']} ({bid}) - {b['total_chunks']} chunks]")
                    print(f"[Texto extraido en: {save_path}]")
                except Exception as e:
                    print(f"{DIM}[Error al cargar libro: {e}]{RESET}")
        print(f"{DIM}[BookMemory: {book_mem.has_books()} libros]{RESET}")
    except Exception as e:
        print(f"{DIM}[BookMemory no disponible: {e}]{RESET}")
        book_mem = None

    summarizer_model = None
    try:
        summarizer_model = VLLaMA(model="Gemma-3-1B.gguf", auto_load=True)
        print(f"{DIM}[Summarizer: {summarizer_model.model_name}]{RESET}")
    except Exception as e:
        print(f"{DIM}[Summarizer no disponible: {e}]{RESET}")

    conv_summary = ConversationSummaryMemory(
        sqlite_conn=mem._conn,
        conversation_id="default",
        max_messages=10,
        model=summarizer_model,
    )

    system_prompt = _ask_system_prompt(mem)

    print(f"\nComandos: /clear  /prompt  /save  /load  /show_prompt  /remember  /memories  /search  /forget  /review  /book  /exit\n")

    stream = not args.no_stream

    while True:
        try:
            user = input(f"{CYAN}Tu:{RESET} ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user.strip():
            continue

        if user.startswith("/"):
            parts = user.split(maxsplit=1)
            cmd = parts[0].lower()

            if cmd == "/exit":
                break

            elif cmd == "/clear":
                mem.clear_memory_db()
                if sem:
                    _clear_chroma(_CHROMA_DIR)
                system_prompt = mem.get_system_prompt()
                print("[Historial limpiado, ChromaDB y system prompt conservados]")
                continue

            elif cmd == "/prompt":
                if len(parts) < 2:
                    print("[Uso: /prompt <nuevo system prompt>]")
                    continue
                mem.add_message("system", parts[1])
                mem.clear_memory_db()
                system_prompt = mem.get_system_prompt()
                print("[System prompt actualizado, historial limpiado]")
                continue

            elif cmd == "/save":
                if len(parts) < 2:
                    print("[Uso: /save <nombre>]")
                    continue
                mem.save_prompt(parts[1], system_prompt)
                print(f"[Prompt guardado como '{parts[1]}']")
                continue

            elif cmd == "/load":
                if len(parts) < 2:
                    print("[Uso: /load <nombre>]")
                    continue
                loaded = mem.load_prompt(parts[1])
                if loaded is None:
                    print(f"[No existe el prompt '{parts[1]}']")
                    continue
                mem.add_message("system", loaded)
                mem.clear_memory_db()
                system_prompt = mem.get_system_prompt()
                print(f"[Prompt '{parts[1]}' cargado, historial limpiado]")
                continue

            elif cmd == "/show_prompt":
                print(f"\n{DIM}── System prompt construido ──{RESET}")
                print(mem.build_system_prompt())
                print(f"{DIM}─────────────────────────────{RESET}")
                continue

            elif cmd in ("/remember", "/mem"):
                if not sem:
                    print("[SemanticMemory no disponible]")
                    continue
                if len(parts) < 2:
                    print("[Uso: /remember <texto>]")
                    continue
                mid = sem.remember_force(parts[1])
                print(f"[Memoria guardada: {mid}]")
                continue

            elif cmd == "/forget":
                if not sem:
                    print("[SemanticMemory no disponible]")
                    continue
                if len(parts) < 2:
                    print("[Uso: /forget <id>]")
                    continue
                sem.forget(parts[1])
                print(f"[Memoria eliminada: {parts[1]}]")
                continue

            elif cmd == "/memories":
                if not sem:
                    print("[SemanticMemory no disponible]")
                    continue
                memories = sem.list_memories(limit=50)
                if not memories:
                    print("[No hay memorias guardadas]")
                    continue
                print(f"\n{DIM}── Memorias ({len(memories)}) ──{RESET}")
                for m in memories:
                    tag_str = f" [{', '.join(m.tags)}]" if m.tags else ""
                    status_str = f" ({m.status})" if m.status != "active" else ""
                    print(f"  {m.memory_id[:16]}: {m.content[:80]}{tag_str}{status_str}")
                print(f"{DIM}─────────────────────────{RESET}")
                continue

            elif cmd == "/search":
                if not sem:
                    print("[SemanticMemory no disponible]")
                    continue
                if len(parts) < 2:
                    print("[Uso: /search <consulta>]")
                    continue
                ctx = sem.search(parts[1], n_results=5)
                if not ctx:
                    print("[Sin resultados]")
                    continue
                print(f"\n{DIM}── Resultados para: {parts[1]} ──{RESET}")
                for m in ctx:
                    tag_str = f" [{', '.join(m.tags)}]" if m.tags else ""
                    print(f"  {m.content[:100]}{tag_str}")
                print(f"{DIM}──────────────────────────{RESET}")
                continue

            elif cmd == "/review":
                if not sem:
                    print("[SemanticMemory no disponible]")
                    continue
                pending = sem.review_pending(limit=10)
                if not pending:
                    print("[No hay memorias pendientes de revisión]")
                    continue
                print(f"\n{DIM}── Memorias pendientes ({len(pending)}) ──{RESET}")
                for i, m in enumerate(pending, 1):
                    tag_str = f" [{', '.join(m.tags)}]" if m.tags else ""
                    print(f"  {i}. {m.memory_id[:16]}: {m.content[:80]}{tag_str} (conf={m.confidence:.2f})")
                print(f"{DIM}──────────────────────────{RESET}")
                action = input(f"{CYAN}Acción (a=aprobar, r=rechazar, #=número, enter=saltar):{RESET} ").strip().lower()
                if not action:
                    continue
                for token in action.split():
                    parts_review = token.split(":", 1) if ":" in token else [token, ""]
                    try:
                        idx = int(parts_review[0]) - 1
                        if 0 <= idx < len(pending):
                            mid = pending[idx].memory_id
                            act = parts_review[1] if len(parts_review) > 1 else "a"
                            if act in ("r", "reject"):
                                sem.reject(mid)
                                print(f"  [Rechazada: {mid[:16]}]")
                            else:
                                sem.approve(mid)
                                print(f"  [Aprobada: {mid[:16]}]")
                    except (ValueError, IndexError):
                        if token == "a":
                            for m in pending:
                                sem.approve(m.memory_id)
                            print(f"  [Todas aprobadas ({len(pending)})]")
                            break
                        elif token == "r":
                            for m in pending:
                                sem.reject(m.memory_id)
                            print(f"  [Todas rechazadas ({len(pending)})]")
                            break
                continue

            elif cmd == "/book":
                if not book_mem:
                    print("[BookMemory no disponible]")
                    continue
                if len(parts) < 2:
                    print("[Uso: /book <comando>]")
                    print("  /book list                    - Listar libros indexados")
                    print("  /book ingest <ruta>           - Ingestar un PDF/TXT")
                    print("  /book search <consulta>       - Buscar en libros")
                    print("  /book chapters <book_id>      - Listar capitulos")
                    print("  /book chapter <id> <N>        - Inyectar capitulo completo")
                    print("  /book caps <book_id>          - Listar caps (indice)")
                    print("  /book cap <id> <ch> <cap>    - Inyectar cap del indice")
                    print("  /book delete <book_id>        - Eliminar libro")
                    print("  /book stats                   - Estadisticas")
                    continue
                sub = parts[1].split(maxsplit=1)
                sub_cmd = sub[0].lower()
                sub_arg = sub[1] if len(sub) > 1 else ""

                if sub_cmd == "list":
                    books = book_mem.list_books()
                    if not books:
                        print("[No hay libros indexados]")
                        continue
                    print(f"\n{DIM}── Libros ({len(books)}) ──{RESET}")
                    for b in books:
                        print(f"  {b['id']}: {b['title']} ({b['status']}, {b['total_chunks']} chunks, {b['total_chapters']} capitulos)")
                    print(f"{DIM}─────────────────────{RESET}")

                elif sub_cmd == "ingest":
                    parts_cmd = sub_arg.split(maxsplit=1)
                    ingest_path = parts_cmd[0]
                    ingest_index = parts_cmd[1] if len(parts_cmd) > 1 else None
                    if not ingest_path:
                        print("[Uso: /book ingest <ruta> [archivo_indice]")
                        print("  /book ingest libro.pdf")
                        print("  /book ingest libro.pdf indice.txt")
                        continue
                    try:
                        bid = book_mem.ingest(ingest_path, index_path=ingest_index)
                        book = book_mem.get_book(bid)
                        print(f"[Libro indexado: {book['title']} ({bid}) - {book['total_chunks']} chunks, {book['total_chapters']} capitulos]")
                    except Exception as e:
                        print(f"[Error al ingestar: {e}]")

                elif sub_cmd in ("search", "s"):
                    if not sub_arg:
                        print("[Uso: /book search <consulta>]")
                        continue
                    ctx = book_mem.build_context(sub_arg, n_results=5, max_chars=5000)
                    if not ctx:
                        print("[Sin resultados en libros]")
                        continue
                    print(f"\n{DIM}── Resultados en libros ──{RESET}")
                    print(ctx)
                    print(f"{DIM}─────────────────────────{RESET}")

                elif sub_cmd == "chapters":
                    if not sub_arg:
                        print("[Uso: /book chapters <book_id>]")
                        print("  /book list  - para ver los IDs")
                        continue
                    chapters = book_mem.list_chapters(sub_arg)
                    if not chapters:
                        print("[No hay capitulos en este libro]")
                        continue
                    print(f"\n{DIM}── Capitulos de {sub_arg} ──{RESET}")
                    for ch in chapters:
                        print(f"  [{ch.chapter_index}] {ch.chapter} ({ch.char_count}c, pag {ch.page_start}-{ch.page_end})")
                    print(f"{DIM}──────────────────────────{RESET}")

                elif sub_cmd == "chapter":
                    ch_parts = sub_arg.split(maxsplit=1)
                    if len(ch_parts) < 2:
                        print("[Uso: /book chapter <book_id> <N>]")
                        continue
                    ch_book_id = ch_parts[0]
                    try:
                        ch_index = int(ch_parts[1])
                    except ValueError:
                        print("[El indice debe ser un numero. Usa /book chapters <book_id>]")
                        continue
                    ctx = book_mem.build_chapter_context(ch_book_id, ch_index, max_chars=30000)
                    if not ctx:
                        print("[Capitulo no encontrado]")
                        continue
                    print(f"\n{DIM}── Capitulo {ch_index} ──{RESET}")
                    print(ctx)
                    print(f"{DIM}─────────────────────────{RESET}")

                elif sub_cmd == "caps":
                    caps = book_mem.list_caps(sub_arg)
                    if not caps:
                        print("[No hay caps para este libro (usa --index)]")
                        continue
                    print(f"\n{DIM}── Caps de {sub_arg} ({len(caps)}) ──{RESET}")
                    for c in caps:
                        print(f"  [ch={c['chapter_index']} cap={c['cap_index']}] {c['chapter'][:50]} (p{c['page_start']}-{c['page_end']})")
                    print(f"{DIM}──────────────────────────{RESET}")

                elif sub_cmd == "cap":
                    cap_parts = sub_arg.split(maxsplit=2)
                    if len(cap_parts) < 3:
                        print("[Uso: /book cap <book_id> <ch_index> <cap_index>]")
                        print("  /book caps <book_id>  - para ver los indices")
                        continue
                    cap_book_id = cap_parts[0]
                    try:
                        cap_ch = int(cap_parts[1])
                        cap_idx = int(cap_parts[2])
                    except ValueError:
                        print("[Los indices deben ser numeros]")
                        continue
                    ctx = book_mem.build_cap_context(cap_book_id, cap_ch, cap_idx, max_chars=30000)
                    if not ctx:
                        print("[Cap no encontrado]")
                        continue
                    print(f"\n{DIM}── Cap (ch={cap_ch}, cap={cap_idx}) ──{RESET}")
                    print(ctx)
                    print(f"{DIM}─────────────────────────{RESET}")

                elif sub_cmd == "delete":
                    if not sub_arg:
                        print("[Uso: /book delete <book_id>]")
                        print("  /book list  - para ver los IDs")
                        continue
                    book = book_mem.get_book(sub_arg)
                    if not book:
                        print(f"[Libro no encontrado: {sub_arg}]")
                        continue
                    resp = input(f"Borrar '{book['title']}' ({book['total_chunks']} chunks)? (s/N): ").strip().lower()
                    if resp == "s":
                        book_mem.delete_book(sub_arg)
                        print(f"[Libro eliminado: {sub_arg}]")
                    else:
                        print("[Cancelado]")

                elif sub_cmd == "stats":
                    stats = book_mem.get_stats()
                    print(f"[Libros: {stats['total_books']}, Chunks: {stats['total_chunks']}]")

                else:
                    print(f"[Subcomando desconocido: {sub_cmd}]")
                continue

            else:
                print(f"[Comando desconocido: {cmd}]")
                continue

        # ── Chat ────────────────────────────────────────────

        conv_summary.maybe_update()

        extra_context = ""
        mem_count = 0
        book_chars = 0
        if sem:
            results = sem.search(user, n_results=3)
            if results:
                mem_count = len(results)
                lines = "\n".join(f"- {m.content}" for m in results)
                extra_context = lines
                print(f"{DIM}[{mem_count} memorias recuperadas]{RESET}")

        book_context = ""
        if book_mem and book_mem.has_books():
            book_context = book_mem.build_context(user, n_results=3, max_chars=3000)
            if book_context:
                book_chars = len(book_context)
                print(f"{DIM}[{book_chars}c de contexto de libros recuperados]{RESET}")

        full_system = mem.build_system_prompt(
            extra_context=extra_context,
            conv_summary_memory=conv_summary,
            book_context=book_context,
        )
        if extra_context:
            log.warning("System prompt con memorias (%d chars):\n%s", len(full_system), full_system)

        raw = [m for m in mem.get_history(extra_context=extra_context) if m.role != "system"]
        msg_ids = [m.id for m in raw]
        history = [{"role": m.role, "content": m.content} for m in raw]
        cleaned = []
        for m in history:
            if cleaned and cleaned[-1]["role"] == m["role"]:
                continue
            cleaned.append(m)
        if len(cleaned) != len(history):
            history = cleaned

        print("Asistente: ", end="", flush=True)

        if stream:
            full = ""
            stats = None
            for chunk in llm.chat(system=full_system, user=user, history=history, stream=True):
                if not chunk.success:
                    log.error("Stream error: %s | user=%r | system_len=%d", chunk.error, user[:100], len(full_system))
                    print(f"\n{DIM}[ERROR: {chunk.error}]{RESET}")
                    break
                print(chunk.content, end="", flush=True)
                full += chunk.content
                stats = chunk
            print()
            if stats:
                if not full:
                    log.warning("Stream empty response | user=%r | system_len=%d | prompt_tokens=%d | finish=%s",
                                user[:100], len(full_system), stats.prompt_tokens, stats.finish_reason)
                _show_stats(stats, msg_ids, mem_count, book_chars)
            if full:
                mem.add_message("user", user)
                mem.add_message("assistant", full)
                if sem:
                    mid = sem.remember(user)
                    if mid:
                        print(f"{DIM}[Memoria guardada: {mid}]{RESET}")
                    mid2 = sem.remember(full, source_role="assistant")
                    if mid2:
                        print(f"{DIM}[Memoria del asistente: {mid2}]{RESET}")
        else:
            res = llm.chat(system=full_system, user=user, history=history, stream=False)
            if not res.success:
                log.error("Chat error: %s | user=%r | system_len=%d | prompt_tokens=%d",
                          res.error, user[:100], len(full_system), res.prompt_tokens)
                print(f"{DIM}[ERROR: {res.error}]{RESET}")
            else:
                if not res.content:
                    log.warning("Empty response | user=%r | system_len=%d | prompt=%d | finish=%s",
                                user[:100], len(full_system), res.prompt_tokens, res.finish_reason)
                print(res.content)
                _show_stats(res, msg_ids, mem_count, book_chars)
                if res.content:
                    mem.add_message("user", user)
                    mem.add_message("assistant", res.content)
                    if sem:
                        mid = sem.remember(user)
                        if mid:
                            print(f"{DIM}[Memoria guardada: {mid}]{RESET}")
                        mid2 = sem.remember(res.content, source_role="assistant")
                        if mid2:
                            print(f"{DIM}[Memoria del asistente: {mid2}]{RESET}")


if __name__ == "__main__":
    main()
