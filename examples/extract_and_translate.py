"""
Ejemplo: extraer texto de un PDF, guardarlo como TXT, y traducirlo a espanol con un LLM.

Uso:
    python examples/extract_and_translate.py ruta/al/libro.pdf
    python examples/extract_and_translate.py ruta/al/libro.pdf --layout two_columns --model "Gemma-3-1B.gguf"
    python examples/extract_and_translate.py ruta/al/libro.txt --lang english
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_EXTRACTED_DIR = _ROOT / ".chatdb" / "extracted"

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


def main():
    parser = argparse.ArgumentParser(description="Extraer y traducir libro a espanol")
    parser.add_argument("path", help="Ruta al PDF/EPUB/TXT")
    parser.add_argument("--layout", default="auto",
                        choices=["plain", "blocks", "two_columns", "auto"])
    parser.add_argument("--model", help="Nombre del modelo GGUF")
    parser.add_argument("--config", help="Ruta al config.json de v_llama")
    parser.add_argument("--lang", default="espanol",
                        help="Idioma de destino (espanol, english, etc.)")
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
    _VLLAMA = _ROOT.parent / "v_llama"
    if not _VLLAMA.exists():
        print(f"ERROR: v_llama no encontrado en {_VLLAMA}")
        sys.exit(1)

    orig_path = sys.path.copy()
    sys.path.insert(0, str(_VLLAMA))
    try:
        from src import VLLaMA
    except ImportError:
        print("ERROR: no se pudo importar VLLaMA desde v_llama")
        sys.exit(1)
    sys.path[:] = orig_path

    llm = VLLaMA(config_path=args.config, auto_load=False)

    if args.model:
        model_name = args.model
    else:
        model_name = _pick_model(llm)

    llm.load_model(model_name)
    print(f"Modelo cargado: {llm.model_name}")

    # ── Traducir ────────────────────────────────────────────────
    out_path = _EXTRACTED_DIR / f"{src.stem}.{args.lang}.txt"
    chunk_size = 2000

    print(f"\n--- Traduciendo a {args.lang} ---")

    with open(txt_path, encoding="utf-8") as f:
        full_text = f.read()

    paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
    translated: list[str] = []
    buffer = ""
    idx = 0

    for para in paragraphs:
        idx += 1
        if len(buffer) + len(para) < chunk_size:
            buffer += para + "\n\n"
            continue

        prompt = (
            f"Traduce el siguiente texto al {args.lang}. "
            "Conserva el formato, los nombres propios y los numeros. "
            "No agregues explicaciones ni comentarios. Solo devuelve la traduccion.\n\n"
            f"{buffer.strip()}"
        )

        print(f"  Traduciendo bloque {idx}/{len(paragraphs)}...", end="", flush=True)
        try:
            res = llm.chat(
                system=f"Eres un traductor profesional que traduce al {args.lang}.",
                user=prompt,
                history=[],
            )
            if res.content:
                translated.append(res.content.strip())
                print(" OK")
            else:
                translated.append(buffer.strip())
                print(" (vacio, usando original)")
        except Exception as e:
            translated.append(buffer.strip())
            print(f" ERROR: {e}")

        buffer = para + "\n\n"

    if buffer.strip():
        prompt = (
            f"Traduce el siguiente texto al {args.lang}. "
            "Conserva el formato, los nombres propios y los numeros. "
            "No agregues explicaciones ni comentarios.\n\n"
            f"{buffer.strip()}"
        )
        print(f"  Traduciendo bloque final...", end="", flush=True)
        try:
            res = llm.chat(
                system=f"Eres un traductor profesional que traduce al {args.lang}.",
                user=prompt,
                history=[],
            )
            if res.content:
                translated.append(res.content.strip())
                print(" OK")
            else:
                translated.append(buffer.strip())
                print(" (vacio, usando original)")
        except Exception as e:
            translated.append(buffer.strip())
            print(f" ERROR: {e}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(translated))

    print(f"\n--- Traduccion completada ---")
    print(f"Original:  {txt_path}")
    print(f"Traducido: {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
