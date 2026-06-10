"""
Ejemplo: extraer texto de un PDF/EPUB/TXT y traducirlo a otro idioma con un LLM local.

Flujo:
    documento → BookMemory.extract_to_txt() → TXT limpio
    → VLLaMA traduce por bloques → TXT traducido

Uso:
    python examples/document_translate.py ruta/al/libro.pdf
    python examples/document_translate.py ruta/al/libro.epub --layout auto --lang espanol
    python examples/document_translate.py ruta/al/libro.txt --chunk-size 1200 --resume

Este ejemplo esta pensado para documentos propios, material con licencia compatible,
dominio publico, notas personales o traduccion privada.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_EXTRACTED_DIR = _ROOT / ".chatdb" / "extracted"
_VLLAMA = _ROOT.parent / "v_llama"

sys.path.insert(0, str(_ROOT))
from src import BookMemory


def _pick_model(llm) -> str:
    models = llm.list_models()
    if not models:
        print("No se encontraron modelos GGUF.")
        sys.exit(1)
    print("Modelos disponibles:")
    for i, m in enumerate(models, 1):
        print(f"  {i}. {m}")
    while True:
        try:
            idx = int(input("Seleccione modelo (1-{}): ".format(len(models))).strip())
            if 1 <= idx <= len(models):
                return models[idx - 1]
        except (ValueError, EOFError):
            pass


def _load_llm(config_path: str | None, model_name: str | None):
    if not _VLLAMA.exists():
        print(f"ERROR: v_llama no encontrado en {_VLLAMA}")
        sys.exit(1)

    # Limpiar cache de 'src' para evitar conflicto con v_memory_manager
    for k in list(sys.modules):
        if k.startswith("src"):
            del sys.modules[k]

    sys.path.insert(0, str(_VLLAMA))
    try:
        from src import VLLaMA
    except ImportError as e:
        print(f"ERROR: no se pudo importar VLLaMA: {e}")
        sys.exit(1)

    llm = VLLaMA(config_path=config_path, auto_load=False)
    if model_name:
        llm.load_model(model_name)
    else:
        llm.load_model(_pick_model(llm))
    return llm


def build_blocks(paragraphs: list[str], chunk_size: int) -> list[str]:
    blocks = []
    buffer = ""
    for para in paragraphs:
        if len(buffer) + len(para) + 2 <= chunk_size:
            buffer += para + "\n\n"
        else:
            if buffer.strip():
                blocks.append(buffer.strip())
            buffer = para + "\n\n"
    if buffer.strip():
        blocks.append(buffer.strip())
    return blocks


def load_state(state_path: Path) -> dict:
    if state_path.exists():
        return json.loads(state_path.read_text(encoding="utf-8"))
    return {}


def save_state(state_path: Path, state: dict) -> None:
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def translate_block(llm, block: str, lang: str, block_num: int, total: int) -> str | None:
    prompt = (
        f"Traduce el siguiente texto al {lang}.\n"
        "Conserva nombres propios, nombres de lugares, personajes, "
        "reglas, numeros, dados (1d6, DC 19, etc.), tablas, listas y formato.\n"
        "NO traduzcas ni modifiques marcadores como [PAGE N] o [TITLE].\n"
        "NO agregues explicaciones. NO resumas. NO omitas texto.\n"
        "Devuelve solo la traduccion.\n\n"
        f"{block}"
    )

    print(f"  Traduciendo bloque {block_num}/{total}...", end="", flush=True)
    try:
        res = llm.chat(
            system=f"Eres un traductor profesional que traduce al {lang}.",
            user=prompt,
            history=[],
        )
        if res.content and res.content.strip():
            print(" OK")
            return res.content.strip()
        print(" (vacio, usando original)")
        return block
    except Exception as e:
        print(f" ERROR: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Extraer y traducir un documento a otro idioma con LLM local"
    )
    parser.add_argument("path", help="Ruta al PDF/EPUB/TXT")
    parser.add_argument("--layout", default="auto",
                        choices=["plain", "blocks", "two_columns", "auto"])
    parser.add_argument("--lang", default="espanol",
                        help="Idioma de destino (espanol, english, etc.)")
    parser.add_argument("--model", help="Nombre del modelo GGUF")
    parser.add_argument("--config", help="Ruta al config.json de v_llama")
    parser.add_argument("--chunk-size", type=int, default=2000,
                        help="Caracteres por bloque de traduccion (default: 2000)")
    parser.add_argument("--resume", action="store_true",
                        help="Reanudar traduccion desde el ultimo bloque completado")
    args = parser.parse_args()

    src = Path(args.path)
    if not src.exists():
        print(f"ERROR: archivo no encontrado: {src}")
        sys.exit(1)

    _EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

    bm = BookMemory()
    txt_path = _EXTRACTED_DIR / src.with_suffix(".txt").name

    print(f"\n--- Extrayendo texto con layout={args.layout} ---")
    bm.extract_to_txt(str(src), str(txt_path), pdf_layout=args.layout)
    print(f"Texto extraido: {txt_path} ({txt_path.stat().st_size} bytes)")

    # ── Cargar LLM ──────────────────────────────────────────────
    llm = _load_llm(args.config, args.model)
    print(f"Modelo: {llm.model_name}")

    # ── Traducir ────────────────────────────────────────────────
    out_path = _EXTRACTED_DIR / f"{src.stem}.{args.lang}.txt"
    state_path = _EXTRACTED_DIR / f"{src.stem}.{args.lang}.state.json"

    with open(txt_path, encoding="utf-8") as f:
        full_text = f.read()

    paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
    blocks = build_blocks(paragraphs, args.chunk_size)

    state = load_state(state_path)
    start_from = 0

    if args.resume and state.get("target") == str(out_path):
        start_from = state.get("last_completed_block", 0)
        if start_from > 0:
            print(f"Reanudando desde bloque {start_from}/{len(blocks)}")

    print(f"\n--- Traduciendo a {args.lang} ({len(blocks)} bloques) ---")

    mode = "a" if start_from > 0 else "w"
    with open(out_path, mode, encoding="utf-8") as out:
        for i in range(start_from, len(blocks)):
            block = blocks[i]
            result = translate_block(llm, block, args.lang, i + 1, len(blocks))

            if result is None:
                print("  [Traduccion interrumpida. Usa --resume para continuar]")
                break

            out.write(result + "\n\n")
            out.flush()

            save_state(state_path, {
                "source": str(txt_path),
                "target": str(out_path),
                "last_completed_block": i + 1,
                "total_blocks": len(blocks),
                "model": llm.model_name,
                "lang": args.lang,
            })

    completed = state.get("last_completed_block", 0) >= len(blocks) if args.resume else True
    if completed:
        if state_path.exists():
            state_path.unlink()

    print(f"\n--- Traduccion {'completada' if completed else 'incompleta'} ---")
    print(f"Original:  {txt_path}")
    print(f"Traducido: {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
