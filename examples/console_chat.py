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
_CHROMA_DIR = _ROOT / ".chatdb" / "chroma"
_VLLAMA = _ROOT.parent / "v_llama"

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
from src import MemoryManager, SemanticMemory

DIM = "\033[90m"
CYAN = "\033[36m"
GREEN = "\033[32m"
RESET = "\033[0m"


def _show_stats(r, msg_ids: list[int] | None = None, mem_count: int | None = None) -> None:
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


def _clear_chroma(chroma_dir: Path) -> None:
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir)
        print(f"[ChromaDB limpiada]")
        chroma_dir.mkdir(parents=True, exist_ok=True)


def _ask_continue(db_path: Path, mem: MemoryManager, chroma_dir: Path | None = None) -> None:
    if not db_path.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)
        mem.create_memory_db(str(db_path))
        print(f"[Memoria creada: {db_path.name}]")
        if chroma_dir:
            _clear_chroma(chroma_dir)
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
        if chroma_dir:
            _clear_chroma(chroma_dir)
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
    _ask_continue(Path(args.db), mem, _CHROMA_DIR)
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

    system_prompt = _ask_system_prompt(mem)

    print(f"\nComandos: /clear  /prompt <texto>  /save <nombre>  /load <nombre>  /show_prompt  /remember <texto>  /memories  /search <q>  /forget <id>  /review  /exit\n")

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

            else:
                print(f"[Comando desconocido: {cmd}]")
                continue

        # ── Chat ────────────────────────────────────────────

        extra_context = ""
        mem_count = 0
        if sem:
            results = sem.search(user, n_results=3)
            if results:
                mem_count = len(results)
                lines = "\n".join(f"- {m.content}" for m in results)
                extra_context = lines
                print(f"{DIM}[{mem_count} memorias recuperadas]{RESET}")

        full_system = mem.build_system_prompt(extra_context=extra_context)
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
                _show_stats(stats, msg_ids, mem_count)
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
                _show_stats(res, msg_ids, mem_count)
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
