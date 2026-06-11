"""
Pipeline incremental: traducir y optimizar un libro para ingesta en BookMemory.

Modo interactivo (sin args):
    python examples/translate_book_incremental.py
    Te guia paso a paso.

Modo CLI (con args):
    python examples/translate_book_incremental.py ruta/al/libro.pdf --lang espanol
    python examples/translate_book_incremental.py ruta/al/libro.txt --skip-extract --optimize
    python examples/translate_book_incremental.py ruta/al/libro.pdf --by-page

Al finalizar, pide confirmacion para ingestar en BookMemory.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_EXTRACTED_DIR = _ROOT / ".chatdb" / "extracted"
_VLLAMA = _ROOT.parent / "v_llama"

sys.path.insert(0, str(_ROOT))
from src import BookMemory, MemoryManager

_PAGE_RE = re.compile(r"\[PAGE (\d+)\]")
_CHAPTER_RE = re.compile(r"#CHAPTER#\s*(.+)", re.IGNORECASE)
_SECTION_RE = re.compile(r"#SECTION#\s*(.+)", re.IGNORECASE)


def _pick_model(llm) -> str:
    models = llm.list_models()
    if not models:
        print("No se encontraron modelos GGUF.")
        sys.exit(1)
    print("\nModelos disponibles:")
    for i, m in enumerate(models, 1):
        print(f"  {i}. {m}")
    while True:
        try:
            idx = int(input("Seleccione modelo: ").strip())
            if 1 <= idx <= len(models):
                return models[idx - 1]
        except (ValueError, EOFError):
            pass
        print("  Opcion invalida.")


def _load_llm(config_path: str | None, model_name: str | None, n_ctx: int = 0):
    if not _VLLAMA.exists():
        print(f"ERROR: v_llama no encontrado en {_VLLAMA}")
        sys.exit(1)

    # Guardar referencia a src de v_memory_manager antes de tocar sys.modules
    _our_src = sys.modules.get("src")

    for k in list(sys.modules):
        if k.startswith("src"):
            del sys.modules[k]

    import json as _json, tempfile

    if config_path:
        cfg_path = config_path
    elif n_ctx:
        base_cfg = _json.loads((_VLLAMA / "config.json").read_text(encoding="utf-8"))
        base_cfg["n_ctx"] = n_ctx
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
        _json.dump(base_cfg, tmp)
        tmp.close()
        cfg_path = tmp.name
    else:
        cfg_path = str(_VLLAMA / "config.json")

    sys.path.insert(0, str(_VLLAMA))
    try:
        from src import VLLaMA
    except ImportError as e:
        print(f"ERROR: no se pudo importar VLLaMA: {e}")
        sys.exit(1)

    # Restaurar nuestro src en sys.modules para que los imports locales sigan funcionando
    if _our_src is not None:
        sys.modules["src"] = _our_src

    if n_ctx:
        print(f"Contexto configurado a {n_ctx} tokens")
    llm = VLLaMA(config_path=cfg_path, auto_load=False)
    if model_name:
        llm.load_model(model_name)
    else:
        llm.load_model(_pick_model(llm))
    return llm


# ── Page splitting ─────────────────────────────────────────────


def split_by_pages(text: str) -> list[tuple[str, str]]:
    parts = _PAGE_RE.split(text)
    pages: list[tuple[str, str]] = []
    marker = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if re.match(r"^\d+$", part):
            marker = f"[PAGE {part}]"
        else:
            pages.append((marker, part))
    if not pages:
        pages.append(("[PAGE 1]", text.strip()))
    return pages


def build_page_blocks(pages: list[tuple[str, str]],
                      max_chars: int = 8000) -> list[dict]:
    blocks: list[dict] = []
    buffer = ""
    first_marker = ""

    for marker, content in pages:
        page_text = f"{marker}\n{content}".strip()
        if not first_marker:
            first_marker = marker
        if buffer and len(buffer) + len(page_text) > max_chars:
            blocks.append({"start_page": first_marker, "text": buffer.strip()})
            buffer = ""
            first_marker = marker
        buffer += page_text + "\n\n"

    if buffer.strip():
        blocks.append({"start_page": first_marker, "text": buffer.strip()})
    return blocks


# ── State management ───────────────────────────────────────────


def load_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


# ── Prompt builder ─────────────────────────────────────────────


def build_instructions(lang: str, translate: bool, optimize: bool) -> list[str]:
    lines = []
    if translate:
        lines.append(f"Traduce el siguiente texto al {lang}.")
        lines.append("Conserva nombres propios, personajes, lugares, reglas, "
                      "numeros, dados, DC, tablas y listas.")
    else:
        lines.append("Optimiza el siguiente texto para ingesta RAG sin cambiar el idioma.")
        lines.append("Corrige ortografia y formato sin alterar el contenido.")
    lines.append("No modifiques marcadores [PAGE N].")
    lines.append("No resumas. No omitas texto. No agregues explicaciones.")
    if optimize:
        lines.append("Si detectas inicio de capitulo, antepon: #CHAPTER# Titulo")
        lines.append("Si detectas inicio de seccion, antepon: #SECTION# Titulo")
    lines.append("Manten parrafos separados por linea en blanco.")
    lines.append("Devuelve solo el texto procesado.")
    return lines


# ── Block processing (chunk mode) ──────────────────────────────


def translate_block(llm, block: dict, lang: str,
                    active_chapter: str, active_section: str,
                    block_num: int, total: int,
                    translate: bool, optimize: bool) -> tuple[str | None, str, str]:
    chapter = active_chapter
    section = active_section
    instructions = build_instructions(lang, translate, optimize)

    prompt = "\n".join(instructions) + f"""

Contexto actual:
Capitulo activo: {chapter}
Seccion activa: {section}

Bloque:
{block['text']}"""

    print(f"  Bloque {block_num}/{total} ({block['start_page']})...", end="", flush=True)

    try:
        res = llm.chat(
            system=f"Eres un {'traductor' if translate else 'formateador'} profesional de libros.",
            user=prompt,
            history=[],
        )
        output = res.content.strip() if res and res.content else block["text"]

        ch = _CHAPTER_RE.findall(output)
        if ch:
            chapter = ch[-1].strip()
        sec = _SECTION_RE.findall(output)
        if sec:
            section = sec[-1].strip()

        if len(output) < len(block["text"]) * 0.40:
            print(f" muy corto, usando original")
            output = block["text"]
        else:
            print(" OK")
        return output, chapter, section

    except Exception as e:
        print(f" ERROR: {e}")
        return None, chapter, section


# ── Page-by-page processing ────────────────────────────────────


def translate_page(llm, marker: str, content: str, previous_page: str,
                   lang: str, active_chapter: str, active_section: str,
                   page_num: int, total: int,
                   translate: bool, optimize: bool) -> tuple[str | None, str, str]:
    chapter = active_chapter
    section = active_section
    instructions = build_instructions(lang, translate, optimize)

    prompt_parts = [
        "=" * 50,
        "INSTRUCCIONES",
        "=" * 50,
    ]
    prompt_parts.extend(instructions)
    prompt_parts.append("")

    if previous_page:
        prompt_parts.extend([
            "-" * 50,
            "<CONTEXT_ONLY_DO_NOT_TRANSLATE>",
            "Contenido de la pagina anterior (solo como referencia, NO procesar):",
            previous_page,
            "</CONTEXT_ONLY_DO_NOT_TRANSLATE>",
            "",
        ])

    prompt_parts.extend([
        "-" * 50,
        "<TEXT_TO_PROCESS>",
        f"Pagina actual a procesar:",
        f"{marker}:",
        content,
        "</TEXT_TO_PROCESS>",
    ])

    prompt = "\n".join(prompt_parts)

    print(f"  Pagina {page_num}/{total} ({marker})...", end="", flush=True)

    try:
        res = llm.chat(
            system=(
                f"Eres un {'traductor' if translate else 'formateador'} profesional de libros. "
                f"Capitulo activo: {chapter}. Seccion activa: {section}."
            ),
            user=prompt,
            history=[],
        )
        output = res.content.strip() if res and res.content else f"{marker}\n{content}"

        ch = _CHAPTER_RE.findall(output)
        if ch:
            chapter = ch[-1].strip()
        sec = _SECTION_RE.findall(output)
        if sec:
            section = sec[-1].strip()

        if len(output) < len(content) * 0.40:
            print(f" muy corto, usando original")
            output = f"{marker}\n{content}"
        else:
            print(" OK")
        return output, chapter, section

    except Exception as e:
        print(f" ERROR: {e}")
        return None, chapter, section


# ── Interactive prompts ────────────────────────────────────────


def ask(question: str, default: str = "") -> str:
    if default:
        r = input(f"{question} [{default}]: ").strip()
        return r if r else default
    return input(f"{question}: ").strip()


def ask_choice(question: str, options: list[str], default: int = 0) -> int:
    print(f"\n{question}")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    while True:
        r = input(f"Opcion (1-{len(options)}){' [' + str(default+1) + ']' if default else ''}: ").strip()
        if not r and default is not None:
            return default
        try:
            idx = int(r) - 1
            if 0 <= idx < len(options):
                return idx
        except ValueError:
            pass
        print("  Opcion invalida.")


def ask_yesno(question: str, default: bool = True) -> bool:
    r = input(f"{question} ({'S/n' if default else 's/N'}): ").strip().lower()
    if not r:
        return default
    return r.startswith("s")


def get_interactive_args() -> dict:
    print("\n" + "=" * 50)
    print("  TRADUCTOR Y OPTIMIZADOR DE LIBROS")
    print("=" * 50)
    print()

    modo = ask_choice(
        "Que queres hacer?",
        ["Traducir libro a otro idioma + optimizar",
         "Solo optimizar (agregar #CHAPTER#/#SECTION# en idioma original)",
         "Solo traducir (sin optimizar)",
         "Dos pasos: primero optimizar en idioma original, luego traducir a espanol"],
        default=0,
    )

    translate = modo in (0, 2, 3)
    optimize = modo in (0, 1, 3)
    two_pass = modo == 3
    path = ""
    skip = False
    layout = "auto"
    lang = "espanol"

    if translate:
        while True:
            path = ask("Ruta al libro original (PDF/EPUB/TXT)")
            if not path:
                print("  La ruta es obligatoria.")
                continue
            p = Path(path)
            if not p.exists():
                print(f"  No se encuentra: {path}")
                continue
            break

        txt_candidate = _EXTRACTED_DIR / p.with_suffix(".txt").name
        if txt_candidate.exists():
            skip = ask_yesno(
                f"Ya existe {txt_candidate.name} en extracted/.\n"
                f"  Usar ese TXT (saltar extraccion)?", True)
            if skip:
                path = str(txt_candidate)

        if not skip and p.suffix.lower() == ".pdf":
            layout_idx = ask_choice(
                "Layout del PDF:",
                ["auto", "plain", "blocks", "two_columns"],
                default=0,
            )
            layout = ["auto", "plain", "blocks", "two_columns"][layout_idx]
        lang = ask("A que idioma queres traducir?", "espanol")
    else:
        path = ask("Ruta al TXT para optimizar")
        if not path:
            print("  La ruta es obligatoria.")
            sys.exit(1)

    by_page = ask_yesno(
        "Procesar pagina por pagina (con contexto de pagina anterior)?\n"
        "  Si: cada pagina individualmente\n"
        "  No: agrupar paginas en bloques de N caracteres (recomendado)",
        False,
    )

    n_ctx = 8192
    chunk_size = 8000
    if translate:
        n_ctx = int(ask("Contexto del LLM en tokens", "8192"))
        if not by_page:
            chunk_size = int(ask("Caracteres por bloque", "8000"))
    resume = ask_yesno("Reanudar traduccion anterior (si existe)?", True)

    return {
        "path": path,
        "skip_extract": skip,
        "layout": layout,
        "lang": lang,
        "translate": translate,
        "optimize": optimize,
        "two_pass": two_pass,
        "by_page": by_page,
        "chunk_size": chunk_size,
        "n_ctx": n_ctx,
        "resume": resume,
    }


# ── Main processing function ───────────────────────────────────


def _process_units(llm, units, by_page, unit_type, lang,
                   translate, optimize, out_path, state_path,
                   start_from, total_units) -> int:
    state = load_state(state_path)
    active_chapter = state.get("active_chapter", "")
    active_section = state.get("active_section", "")

    action = "traduciendo" if translate else "optimizando"
    mode_w = "a" if start_from > 0 else "w"
    completed = start_from

    with open(out_path, mode_w, encoding="utf-8") as out:
        for i in range(start_from, total_units):
            unit = units[i]
            if by_page:
                previous = units[i - 1]["content"] if i > 0 else ""
                result, active_chapter, active_section = translate_page(
                    llm, unit["marker"], unit["content"], previous,
                    lang, active_chapter, active_section,
                    i + 1, total_units, translate, optimize,
                )
            else:
                result, active_chapter, active_section = translate_block(
                    llm, {"start_page": unit["marker"], "text": unit["content"]},
                    lang, active_chapter, active_section,
                    i + 1, total_units, translate, optimize,
                )

            if result is None:
                print(f"  [Interrumpido. Usa --resume para continuar]")
                break

            out.write(result + "\n\n")
            out.flush()
            completed = i + 1

            save_state(state_path, {
                "source": "",
                "target": str(out_path),
                "last_completed_block": completed,
                "total_blocks": total_units,
                "model": llm.model_name,
                "lang": lang,
                "active_chapter": active_chapter,
                "active_section": active_section,
            })

    if completed >= total_units and state_path.exists():
        try:
            state_path.unlink()
        except Exception:
            pass

    return completed


def _ingest(file_path: Path, title: str) -> None:
    print()
    if not ask_yesno("Ingestar el libro en BookMemory?", True):
        return
    db_path = str(_ROOT / ".chatdb" / "chat.db")
    mem = MemoryManager()
    if Path(db_path).exists():
        mem.load_memory_db(db_path)
    else:
        mem.create_memory_db(db_path)
    bm = BookMemory(sqlite_conn=mem._conn)
    bid = bm.ingest(str(file_path), title=title, force=True)
    book = bm.get_book(bid)
    print(f"  Libro indexado: {book['title']} ({bid}) - {book['total_chunks']} chunks")
    bm.close()
    mem.close()


# ── Main ───────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Traducir y optimizar un libro para BookMemory"
    )
    parser.add_argument("path", nargs="?", help="Ruta al PDF/EPUB/TXT")
    parser.add_argument("--layout", default="auto",
                        choices=["plain", "blocks", "two_columns", "auto"])
    parser.add_argument("--lang", default="espanol")
    parser.add_argument("--model")
    parser.add_argument("--config")
    parser.add_argument("--n-ctx", type=int, default=8192)
    parser.add_argument("--chunk-size", type=int, default=8000)
    parser.add_argument("--translate", action="store_true", default=True,
                        help="Traducir el texto (default: True)")
    parser.add_argument("--optimize", action="store_true",
                        help="Agregar marcadores #CHAPTER#/#SECTION#")
    parser.add_argument("--by-page", action="store_true",
                        help="Procesar pagina por pagina con contexto anterior")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()

    if not args.path:
        interactive = get_interactive_args()
        path = interactive["path"]
        skip_extract = interactive["skip_extract"]
        layout = interactive["layout"]
        lang = interactive["lang"]
        translate = interactive["translate"]
        optimize = interactive["optimize"]
        two_pass = interactive.get("two_pass", False)
        by_page = interactive["by_page"]
        chunk_size = interactive["chunk_size"]
        n_ctx = interactive["n_ctx"]
        resume = interactive["resume"]
    else:
        path = args.path
        skip_extract = args.skip_extract
        layout = args.layout
        lang = args.lang
        translate = args.translate
        optimize = args.optimize
        by_page = args.by_page
        chunk_size = args.chunk_size
        n_ctx = args.n_ctx
        resume = args.resume

    src = Path(path)
    if not src.exists():
        print(f"ERROR: archivo no encontrado: {src}")
        sys.exit(1)

    if translate and not lang:
        lang = ask("A que idioma queres traducir?", "espanol")
    elif not translate:
        lang = ""

    _EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    bm = BookMemory()

    # ── Extraer o usar TXT existente ────────────────────────────
    if skip_extract:
        txt_path = src
        print(f"\nUsando TXT existente: {txt_path}")
    else:
        txt_path = _EXTRACTED_DIR / src.with_suffix(".txt").name
        print(f"\n--- Extrayendo texto con layout={layout} ---")
        bm.extract_to_txt(str(src), str(txt_path), pdf_layout=layout)
        print(f"Texto extraido: {txt_path} ({txt_path.stat().st_size} bytes)")

    # ── Cargar LLM ──────────────────────────────────────────────
    llm = _load_llm(args.config, args.model, n_ctx=n_ctx)
    print(f"Modelo: {llm.model_name}")

    # ── Preparar paginas/bloques ────────────────────────────────
    with open(txt_path, encoding="utf-8") as f:
        full_text = f.read()

    pages = split_by_pages(full_text)
    total_pages = len(pages)

    if by_page:
        units = [{"marker": m, "content": c} for m, c in pages]
        unit_type = "paginas"
    else:
        raw_blocks = build_page_blocks(pages, max_chars=chunk_size)
        units = [{"marker": b["start_page"], "content": b["text"]} for b in raw_blocks]
        unit_type = "bloques"

    total_units = len(units)
    print(f"Dividido en {total_units} {unit_type} ({total_pages} paginas)")

    # ── Two-pass: primero optimizar en idioma original ─────────
    if two_pass:
        print(f"\n{'='*50}")
        print(f"  PASO 1: Optimizar en idioma original")
        print(f"{'='*50}")
        opt_path = _EXTRACTED_DIR / f"{src.stem}.optimized.txt"
        opt_state_path = _EXTRACTED_DIR / f"{src.stem}.optimized.state.json"
        completed_units = _process_units(
            llm, units, by_page, unit_type, lang="",
            translate=False, optimize=True,
            out_path=opt_path, state_path=opt_state_path,
            start_from=0, total_units=total_units,
        )
        if completed_units < total_units:
            return

        print(f"\n{'='*50}")
        print(f"  PASO 2: Traducir a {lang}")
        print(f"{'='*50}")
        suffix = ".optimized.txt" if optimize else ".txt"
        out_path = _EXTRACTED_DIR / f"{src.stem}.{lang}{suffix}"
        state_path = _EXTRACTED_DIR / f"{src.stem}.{lang}.state.json"
        state = load_state(state_path)
        start_from = state.get("last_completed_block", 0) if resume else 0
        if start_from > 0:
            print(f"Reanudando desde {unit_type} {start_from}/{total_units}")

        _process_units(
            llm, units, by_page, unit_type, lang,
            translate=True, optimize=False,
            out_path=out_path, state_path=state_path,
            start_from=start_from, total_units=total_units,
        )

        # Verificar si el paso 2 completo
        state = load_state(state_path)
        completed = state.get("last_completed_block", 0) >= total_units
        if completed and state_path.exists():
            state_path.unlink()
        out_size = out_path.stat().st_size if out_path.exists() else 0
        print(f"\n--- {'Completado' if completed else 'Incompleto'} ---")
        print(f"Optimizado: {opt_path}")
        print(f"Traducido:  {out_path} ({out_size} bytes)")
        if not completed:
            print(f"  Usa --resume para continuar la traduccion")
            return
        _ingest(out_path, src.stem)
        return

    # ── Procesar (un solo paso) ─────────────────────────────────
    suffix = ".optimized.txt" if optimize else ".txt"
    prefix = f".{lang}" if translate else ""
    out_path = _EXTRACTED_DIR / f"{src.stem}{prefix}{suffix}"
    state_path = _EXTRACTED_DIR / f"{src.stem}{prefix}.state.json"

    state = load_state(state_path)
    start_from = state.get("last_completed_block", 0) if resume else 0

    action = "traduciendo" if translate else "optimizando"
    mode_label = "por pagina" if by_page else "por bloques"
    if start_from > 0:
        print(f"Reanudando desde {unit_type} {start_from}/{total_units}")
    print(f"\n--- {action} {mode_label} ---")

    completed_units = _process_units(
        llm, units, by_page, unit_type, lang,
        translate, optimize, out_path, state_path,
        start_from, total_units,
    )

    completed = completed_units >= total_units
    out_size = out_path.stat().size if out_path.exists() else 0
    print(f"\n--- {'Completado' if completed else 'Incompleto'} ---")
    print(f"Original:  {txt_path}")
    print(f"Procesado: {out_path} ({out_size} bytes)")

    if not completed:
        print(f"  Progreso: {completed_units}/{total_units} {unit_type}")
        print(f"  Usa --resume para continuar")
        return

    _ingest(out_path, src.stem)


if __name__ == "__main__":
    main()
