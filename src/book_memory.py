from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
import struct
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from . import db
from .book_models import BookChunk, ChapterInfo

log = logging.getLogger("book_memory")

_BOOK_TRIGGERS = [
    "libro", "pdf", "documento",
    "capitulo", "pagina", "autor",
    "segun", "segun el", "segun la",
    "que dice", "menciona",
    "texto", "leido", "lei", "leer",
    "indice",
]

_CHAPTER_PATTERNS = [
    re.compile(r"^(?:Cap[ií]tulo|Chapter|Ch\.|Section|Tema)\s+\d+[:\.\s]", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^(?:\d+\.\d+\s+.+)", re.MULTILINE),
    re.compile(r"^(?:#{1,3}\s+.+)", re.MULTILINE),
    re.compile(r"^[A-Z\s]{10,}$", re.MULTILINE),
]

_PAGE_MARKER = re.compile(r"\[PAGE (\d+)\]")

_HYPHENATION = re.compile(r"(\w)-\n(\w)")
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")

_MAX_CHAPTER_WORDS = 4000

CHARS_PER_TOKEN = 3.0
SAFETY_MARGIN = 0.85


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _generate_book_id() -> str:
    return f"book_{uuid.uuid4().hex[:12]}"


def _make_parent_chunk_id(book_id: str, ch_index: int) -> str:
    return f"{book_id}_ch_{ch_index:04d}"


def _make_child_chunk_id(book_id: str, ch_index: int, sec_index: int) -> str:
    return f"{book_id}_ch_{ch_index:04d}_s_{sec_index:04d}"


def _make_cap_chunk_id(book_id: str, ch_index: int, cap_index: int) -> str:
    return f"{book_id}_ch_{ch_index:04d}_c_{cap_index:04d}"


def _make_cap_section_chunk_id(book_id: str, ch_index: int, cap_index: int, sec_index: int) -> str:
    return f"{book_id}_ch_{ch_index:04d}_c_{cap_index:04d}_s_{sec_index:04d}"


def _embedding_to_blob(vector: list[float]) -> bytes:
    arr = np.array(vector, dtype=np.float32)
    return arr.tobytes()


def _blob_to_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def get_max_chars_for_context(model_context_window: int, reserved_tokens: int = 1024) -> int:
    available = (model_context_window - reserved_tokens) * SAFETY_MARGIN
    return int(available * CHARS_PER_TOKEN)


# ── Text cleaning ─────────────────────────────────────────────


def clean_extracted_text(text: str) -> str:
    text = text.replace("\u00ad\n", "")
    text = text.replace("\u00ad", "")
    text = _HYPHENATION.sub(r"\1\2", text)
    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(line)
    text = "\n".join(lines)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


_TOC_ENTRY = re.compile(
    r"(?P<title>(?:Ch\.\s*)?\d*\.?\s*[A-ZÁÉÍÓÚÑa-záéíóúñ][^.\n]{2,80}?)"
    r"\s*\.{2,}\s*(?P<page>\d{1,4})",
)

_MARKDOWN_BOLD = re.compile(r"\*\*(.*?)\*\*")


def clean_markdown_pdf_text(text: str) -> str:
    text = _MARKDOWN_BOLD.sub(r"\1", text)
    text = _TOC_ENTRY.sub(lambda m: f"{m.group('title').strip()}........................{m.group('page')}\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Text extraction ────────────────────────────────────────────


def extract_txt(path: str) -> tuple[str, int]:
    p = Path(path)
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return p.read_text(encoding=enc), 1
        except UnicodeDecodeError:
            continue
    raise ValueError(
        f"No se pudo leer {path} con encodings utf-8, utf-8-sig, cp1252 ni latin-1"
    )


def extract_pdf(path: str, layout: str = "") -> tuple[str, int]:
    if not layout or layout in ("llm", "pymupdf4llm", "markdown", "auto"):
        try:
            import pymupdf4llm
            import pymupdf
            doc = pymupdf.open(path)
            chunks = pymupdf4llm.to_markdown(path, page_chunks=True)
            pages = []
            for chunk in chunks:
                page_num = chunk.get("metadata", {}).get("page_number", 0)
                text = chunk.get("text", "").strip()
                if text:
                    text = clean_markdown_pdf_text(text)
                    pages.append(f"[PAGE {page_num}]\n{text}")
            return "\n\n".join(pages), len(doc)
        except ImportError:
            try:
                import pymupdf
            except ImportError:
                raise ImportError("pymupdf no instalado. Ejecuta: pip install pymupdf")
            return _extract_pdf_columns(path)

    try:
        import pymupdf
    except ImportError:
        raise ImportError("pymupdf no esta instalado. Ejecuta: pip install pymupdf")

    if layout in ("plain", "raw"):
        return _extract_pdf_plain(path)
    if layout in ("blocks", "single"):
        return _extract_pdf_blocks(path)
    if layout in ("two_columns", "2col"):
        return _extract_pdf_two_columns(path)
    if layout in ("columns",):
        return _extract_pdf_columns(path)
    raise ValueError(f"Layout no soportado: {layout}")


def _extract_pdf_plain(path: str) -> tuple[str, int]:
    import pymupdf
    doc = pymupdf.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append(f"[PAGE {i+1}]\n{text}")
    return "\n\n".join(pages), len(doc)


def _extract_pdf_blocks(path: str) -> tuple[str, int]:
    import pymupdf
    doc = pymupdf.open(path)
    pages = []
    for i, page in enumerate(doc):
        blocks = page.get_text("blocks")
        lines = []
        for b in sorted(blocks, key=lambda b: (b[1], b[0])):
            text = b[4].strip()
            if text:
                lines.append(text)
        if lines:
            pages.append(f"[PAGE {i+1}]\n" + "\n\n".join(lines))
    return "\n\n".join(pages), len(doc)


def _cluster_by_x0(normal_blocks: list, page_width: float) -> tuple[list[list[float]], list[float]]:
    """Cluster blocks by x0 position and compute column centers."""
    xs = sorted(set(b[0] for b in normal_blocks))
    if not xs:
        return [], []
    clusters: list[list[float]] = []
    gap = page_width * 0.12
    for x in xs:
        if not clusters:
            clusters.append([x])
        else:
            current_center = sum(clusters[-1]) / len(clusters[-1])
            if abs(x - current_center) > gap:
                clusters.append([x])
            else:
                clusters[-1].append(x)
    clusters = clusters[:4]
    centers = [sum(c) / len(c) for c in clusters]
    return clusters, centers


def _extract_page_blocks(page, page_num: int) -> str:
    """Extract text from a page handling mixed 1/2/3 column layouts."""
    import pymupdf
    blocks = page.get_text("blocks")
    page_width = page.rect.width
    page_height = page.rect.height

    normal_blocks: list[tuple[float, float, float, float, str]] = []
    top_full: list[tuple[float, float, str]] = []
    bottom_full: list[tuple[float, float, str]] = []

    for b in blocks:
        x0, y0, x1, y1, text, *_ = b
        text = text.strip()
        if not text:
            continue
        block_width = x1 - x0
        is_real_full = block_width > page_width * 0.80 and x0 < page_width * 0.15

        if is_real_full and y0 > page_height * 0.55:
            bottom_full.append((y0, x0, text))
            continue
        if is_real_full and y0 < page_height * 0.12:
            top_full.append((y0, x0, text))
            continue

        normal_blocks.append((x0, y0, x1, y1, text))

    if not normal_blocks and not bottom_full and not top_full:
        return ""

    clusters, centers = _cluster_by_x0(normal_blocks, page_width)

    columns: list[list[tuple[float, float, str]]] = [[] for _ in centers]
    for x0, y0, x1, y1, text in normal_blocks:
        col_idx = min(range(len(centers)), key=lambda i: abs(x0 - centers[i]))
        columns[col_idx].append((y0, x0, text))

    page_lines: list[str] = []
    for _, _, text in sorted(top_full, key=lambda x: (x[0], x[1])):
        page_lines.append(text)
    for col in columns:
        col.sort(key=lambda x: (x[0], x[1]))
        page_lines.extend(text for _, _, text in col)
    for _, _, text in sorted(bottom_full, key=lambda x: (x[0], x[1])):
        page_lines.append(text)

    if not page_lines:
        return ""
    return f"[PAGE {page_num}]\n" + "\n\n".join(page_lines)


def _extract_pdf_columns(path: str) -> tuple[str, int]:
    """Extract PDF with per-page multi-column detection (1, 2, or 3 columns)."""
    import pymupdf
    doc = pymupdf.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = _extract_page_blocks(page, i + 1)
        if text:
            pages.append(text)
    return "\n\n".join(pages), len(doc)


def _extract_pdf_two_columns(path: str) -> tuple[str, int]:
    import pymupdf
    doc = pymupdf.open(path)
    pages = []

    for i, page in enumerate(doc):
        blocks = page.get_text("blocks")
        page_width = page.rect.width
        page_height = page.rect.height
        middle = page_width / 2

        top_full = []
        left = []
        right = []
        middle_full = []
        bottom_full = []

        for block in blocks:
            x0, y0, x1, y1, text, *_ = block
            text = text.strip()
            if not text:
                continue
            block_width = x1 - x0
            if block_width > page_width * 0.70:
                if y0 < page_height * 0.20:
                    top_full.append((y0, x0, text))
                elif y0 > page_height * 0.85:
                    bottom_full.append((y0, x0, text))
                else:
                    middle_full.append((y0, x0, text))
            elif x0 < middle:
                left.append((y0, x0, text))
            else:
                right.append((y0, x0, text))

        top_full.sort(key=lambda x: (x[0], x[1]))
        left.sort(key=lambda x: (x[0], x[1]))
        right.sort(key=lambda x: (x[0], x[1]))
        middle_full.sort(key=lambda x: (x[0], x[1]))
        bottom_full.sort(key=lambda x: (x[0], x[1]))

        page_lines = (
            [text for _, _, text in top_full] +
            [text for _, _, text in left] +
            [text for _, _, text in right] +
            [text for _, _, text in middle_full] +
            [text for _, _, text in bottom_full]
        )

        if page_lines:
            pages.append(f"[PAGE {i+1}]\n" + "\n\n".join(page_lines))

    return "\n\n".join(pages), len(doc)


def extract_epub(path: str) -> tuple[str, int]:
    try:
        import ebooklib
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError(
            "Dependencias EPUB no instaladas. Ejecuta: pip install ebooklib beautifulsoup4"
        )

    from ebooklib import epub

    book = epub.read_epub(path)

    book_title = ""
    try:
        book_title = book.get_metadata("DC", "title")[0][0]
    except (IndexError, KeyError):
        pass

    chapters = []
    chapter_num = 0

    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            chapter_num += 1
            soup = BeautifulSoup(item.get_content(), "html.parser")
            for tag in soup(["script", "style", "nav"]):
                tag.decompose()
            text = soup.get_text(separator="\n")
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            if lines:
                chapter_title = lines[0][:80] if lines else f"Chapter {chapter_num}"
                chapters.append(f"[PAGE {chapter_num}]\n{chapter_title}\n\n" + "\n".join(lines[1:]))

    if not chapters:
        raise RuntimeError(f"No se pudo extraer contenido de EPUB: {path}")

    result = "\n\n".join(chapters)
    if book_title:
        result = f"[TITLE]\n{book_title}\n\n" + result
    return result, len(chapters)


def extract_ocr_pdf(path: str, engine: str = "auto") -> tuple[str, int]:
    try:
        import pymupdf
        doc = pymupdf.open(path)
        total_pages = len(doc)
    except ImportError:
        raise ImportError("pymupdf requerido para OCR. Ejecuta: pip install pymupdf")

    if total_pages == 0:
        return "", 0

    if engine == "auto" or engine == "tesseract":
        try:
            return _ocr_tesseract(doc, total_pages)
        except (ImportError, FileNotFoundError) as e:
            if engine == "tesseract":
                raise RuntimeError(f"Tesseract OCR fallo: {e}")
    if engine == "auto" or engine == "paddle":
        try:
            return _ocr_paddle(doc, total_pages)
        except ImportError:
            if engine == "paddle":
                raise ImportError(
                    "PaddleOCR no instalado. Ejecuta: pip install paddleocr"
                )

    raise RuntimeError(
        "OCR no disponible. Instala Tesseract (sistema) o PaddleOCR (pip install paddleocr)"
    )


def _ocr_tesseract(doc, total_pages: int) -> tuple[str, int]:
    import subprocess

    pages = []
    for i in range(total_pages):
        page = doc[i]
        pix = page.get_pixmap(dpi=300)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            pix.save(tmp.name)
            try:
                result = subprocess.run(
                    ["tesseract", tmp.name, "stdout", "-l", "spa+eng"],
                    capture_output=True, text=True, timeout=60,
                )
                text = result.stdout.strip()
                if text:
                    pages.append(f"[PAGE {i+1}]\n{text}")
            finally:
                Path(tmp.name).unlink(missing_ok=True)

    return "\n\n".join(pages), total_pages


def _ocr_paddle(doc, total_pages: int) -> tuple[str, int]:
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        raise ImportError("PaddleOCR no instalado. Ejecuta: pip install paddleocr")

    ocr = PaddleOCR(use_angle_cls=True, lang="es", show_log=False)
    pages = []
    for i in range(total_pages):
        page = doc[i]
        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(img_bytes)
            tmp_path = tmp.name
        try:
            result = ocr.ocr(tmp_path, cls=True)
            if result and result[0]:
                text = "\n".join(line[1][0] for line in result[0])
                pages.append(f"[PAGE {i+1}]\n{text}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    return "\n\n".join(pages), total_pages


# ── Chapter detection ──────────────────────────────────────────


def _find_chapter_number(name: str) -> Optional[int]:
    """Extract chapter number from a marker like 'Chapter 1' or 'Capítulo 3'."""
    m = re.search(r'(?:Chapter|Cap[ií]tulo|Section|Tema|Parte)\s+(\d+)', name, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _detect_chapter_forward(text: str) -> list[tuple[int, int, str]]:
    """Find chapter boundaries using regex patterns.
    Only the FIRST occurrence of each chapter number is a real boundary.
    Duplicates (headers, cross-refs) are ignored.
    Returns [(start_char, end_char, chapter_name)]."""
    raw_markers: list[tuple[int, str]] = []

    for pattern in _CHAPTER_PATTERNS:
        for match in pattern.finditer(text):
            pos = match.start()
            name = match.group(0).strip().rstrip(":")
            if len(name) < 100:
                raw_markers.append((pos, name))

    if not raw_markers:
        return []

    raw_markers.sort(key=lambda x: x[0])

    seen_numbers: set[int] = set()
    unique_markers: list[tuple[int, str]] = []
    for pos, name in raw_markers:
        ch_num = _find_chapter_number(name)
        if ch_num is not None:
            if ch_num in seen_numbers:
                continue
            seen_numbers.add(ch_num)

        if unique_markers and pos - unique_markers[-1][0] < 10:
            continue

        unique_markers.append((pos, name))

    if not unique_markers:
        return []

    MIN_CHAPTER_CHARS = 100
    chapters: list[tuple[int, int, str]] = []
    for i, (pos, name) in enumerate(unique_markers):
        end = unique_markers[i + 1][0] if i + 1 < len(unique_markers) else len(text)
        chapters.append((pos, end, name))

    chapters = [(s, e, n) for s, e, n in chapters if e - s >= MIN_CHAPTER_CHARS]

    return chapters


def _fallback_by_word_count(text: str, max_words: int = _MAX_CHAPTER_WORDS) -> list[tuple[int, int, str]]:
    words = text.split()
    chapters = []
    start = 0
    ch_index = 0

    for i in range(0, len(words), max_words):
        chunk_words = words[i:i + max_words]
        end = len(" ".join(words[:i + len(chunk_words)]))
        if i == 0:
            name = "Inicio"
        else:
            ch_index += 1
            name = f"Parte {ch_index}"
        chapters.append((start, end, name))
        start = end

    return chapters


def detect_chapters(text: str) -> list[tuple[int, int, str]]:
    chapters = _detect_chapter_forward(text)
    if not chapters:
        chapters = _fallback_by_word_count(text)
    return chapters


# ── Chunking ───────────────────────────────────────────────────


def _detect_pages(text: str) -> list[dict]:
    blocks = _PAGE_MARKER.split(text)
    pages = []
    current_page = 1
    for i, block in enumerate(blocks):
        block = block.strip()
        if not block:
            continue
        if block.isdigit():
            current_page = int(block)
        else:
            pages.append({"page": current_page, "text": block})
    if not pages:
        pages.append({"page": 1, "text": text.strip()})
    return pages


def chunk_text(
    text: str,
    chunk_size_chars: int = 800,
    chunk_overlap_chars: int = 150,
) -> list[dict]:
    page_blocks = _detect_pages(text)
    chunks = []
    buffer = ""
    current_page_start = 1
    current_page_end = 1

    for block in page_blocks:
        page_num = block["page"]
        page_text = block["text"]
        paragraphs = re.split(r"\n\s*\n", page_text)

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(buffer) + len(para) > chunk_size_chars and buffer:
                chunks.append({
                    "page_start": current_page_start,
                    "page_end": current_page_end,
                    "text": buffer.strip(),
                })
                overlap = buffer[-chunk_overlap_chars:] if len(buffer) > chunk_overlap_chars else buffer
                buffer = overlap + "\n\n"
                current_page_start = page_num

            buffer += para + "\n\n"
            current_page_end = page_num

    if buffer.strip():
        chunks.append({
            "page_start": current_page_start,
            "page_end": current_page_end,
            "text": buffer.strip(),
        })

    return chunks


# ── Triggers ───────────────────────────────────────────────────


def _normalize_query(text: str) -> str:
    import unicodedata
    text = text.lower().strip()
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def should_search_books(query: str) -> bool:
    q = _normalize_query(query)
    if q.startswith("/book") or q.startswith("/lore"):
        return True
    return any(t in q for t in _BOOK_TRIGGERS)


# ── BookMemory class ───────────────────────────────────────────


class BookMemory:
    def __init__(
        self,
        sqlite_conn: Optional[sqlite3.Connection] = None,
        embed_fn: Optional[Callable[[str], list[float]]] = None,
        user_id: str = "default",
        chunk_size_chars: int = 800,
        chunk_overlap_chars: int = 150,
        search_max_distance: float = 0.75,
        embedding_dim: int = 0,
    ):
        self._conn = sqlite_conn
        self._embed_fn = embed_fn
        self._user_id = user_id
        self._embedding_dim = embedding_dim
        self.chunk_size_chars = chunk_size_chars
        self.chunk_overlap_chars = chunk_overlap_chars
        self.search_max_distance = search_max_distance

    @property
    def conn(self):
        if self._conn is None:
            raise RuntimeError("BookMemory requiere sqlite_conn para esta operacion")
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── Has books / Should search ───────────────────────────────

    def has_books(self) -> bool:
        if self._conn is None:
            return False
        return db.has_books(self.conn, user_id=self._user_id)

    @staticmethod
    def should_search(query: str) -> bool:
        return should_search_books(query)

    # ── Extract to TXT ──────────────────────────────────────────

    def extract_to_txt(self, path: str, output_path: str,
                       pdf_layout: str = "auto") -> str:
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {path}")

        ext = path_obj.suffix.lower()
        if ext == ".txt":
            text, _ = extract_txt(path)
        elif ext == ".epub":
            text, _ = extract_epub(path)
        elif ext == ".pdf":
            text, _ = extract_pdf(path, layout=pdf_layout)
            if len(text.strip()) < 100:
                log.info("PDF sin texto, intentando OCR...")
                try:
                    text, _ = extract_ocr_pdf(path)
                except (ImportError, RuntimeError) as e:
                    raise RuntimeError(f"PDF escaneado sin OCR: {e}")
        else:
            raise ValueError(f"Formato no soportado: {ext}")

        text = clean_extracted_text(text)

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        log.info("Texto extraido guardado en: %s", out)
        return str(out)

    # ── Ingest ──────────────────────────────────────────────────

    def _embed(self, text: str) -> Optional[bytes]:
        if self._embed_fn is None:
            return None
        vector = self._embed_fn(text)
        if not vector:
            return None
        return _embedding_to_blob(vector)

    def ingest(
        self,
        path: str,
        title: str = "",
        author: str = "",
        language: str = "es",
        force: bool = False,
        pdf_layout: str = "auto",
        save_extracted_to: Optional[str] = None,
        index_path: Optional[str] = None,
    ) -> str:
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {path}")

        source_file_hash = _hash_file(path)

        ext = path_obj.suffix.lower()
        if ext == ".txt":
            text, total_pages = extract_txt(path)
        elif ext == ".epub":
            text, total_pages = extract_epub(path)
        elif ext == ".pdf":
            text, total_pages = extract_pdf(path, layout=pdf_layout)
            if len(text.strip()) < 100:
                log.info("PDF sin texto seleccionable, intentando OCR...")
                try:
                    text, total_pages = extract_ocr_pdf(path)
                except (ImportError, RuntimeError) as e:
                    raise RuntimeError(
                        f"PDF escaneado sin OCR disponible: {e}"
                    )
        else:
            raise ValueError(f"Formato no soportado: {ext}")

        text = clean_extracted_text(text)
        source_text_hash = _hash_text(text)

        existing = self.get_book_by_hashes(source_file_hash, source_text_hash)
        if existing and not force:
            log.info("Libro ya indexado: %s (%s)", existing["title"], existing["id"])
            return existing["id"]
        if existing and force:
            log.info("Reindexando libro: %s", existing["id"])
            db.delete_book(self.conn, existing["id"])

        if save_extracted_to:
            out = Path(save_extracted_to)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            log.info("Texto extraido guardado en: %s", out)

        source_type = ext.lstrip(".") if ext else ""
        book_id = _generate_book_id()
        db.insert_book(
            self.conn, book_id, str(path_obj), source_file_hash,
            user_id=self._user_id,
            source_text_hash=source_text_hash,
            source_type=source_type,
            source_layout=pdf_layout if source_type == "pdf" else "",
            source_text_path=save_extracted_to or "",
            language=language,
            title=title or path_obj.stem, author=author,
            status="extracting", total_pages=total_pages,
        )

        try:
            db.update_book(self.conn, book_id, status="chunking")
            if index_path:
                self._index_book_from_file(book_id, text, index_path)
            else:
                self._index_book(book_id, text)
            db.update_book(self.conn, book_id, status="indexed")
        except Exception as e:
            db.update_book(self.conn, book_id, status="error",
                           error_message=str(e))
            raise

        return book_id

    def ingest_text(
        self,
        text: str,
        title: str = "",
        author: str = "",
        language: str = "es",
        force: bool = False,
    ) -> str:
        text = clean_extracted_text(text)
        source_text_hash = _hash_text(text)

        existing = self.get_book_by_hashes("", source_text_hash)
        if existing and not force:
            return existing["id"]
        if existing and force:
            db.delete_book(self.conn, existing["id"])

        book_id = _generate_book_id()
        db.insert_book(
            self.conn, book_id, "", "",
            user_id=self._user_id,
            source_text_hash=source_text_hash,
            source_type="text",
            language=language,
            title=title or "Untitled", author=author,
            status="chunking", total_pages=1,
        )

        try:
            self._index_book(book_id, text)
            db.update_book(self.conn, book_id, status="indexed")
        except Exception as e:
            db.update_book(self.conn, book_id, status="error",
                           error_message=str(e))
            raise

        return book_id

    def extract_and_ingest(
        self,
        path: str,
        cache_dir: str = "./extracted",
        title: str = "",
        author: str = "",
        language: str = "es",
        force_extract: bool = False,
        force_ingest: bool = False,
        pdf_layout: str = "auto",
        ingest: bool = True,
    ) -> str:
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {path}")

        cache_path = Path(cache_dir) / path_obj.with_suffix(".txt").name
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if force_extract or not cache_path.exists():
            self.extract_to_txt(str(path_obj), str(cache_path), pdf_layout=pdf_layout)
            log.info("Texto extraido a cache: %s", cache_path)
        else:
            log.info("Cache TXT existe: %s", cache_path)

        if not ingest:
            return str(cache_path)

        return self.ingest(
            str(cache_path),
            title=title or path_obj.stem,
            author=author,
            language=language,
            force=force_ingest,
        )

    # ── Internal indexing ───────────────────────────────────────

    def _index_book(self, book_id: str, text: str) -> None:
        chapters = detect_chapters(text)

        total_chunks = 0
        for ch_index, (start, end, chapter_name) in enumerate(chapters):
            chapter_text = text[start:end]
            parent_chunk_id = _make_parent_chunk_id(book_id, ch_index)

            parent_hash = _hash_text(chapter_text)
            first_page = 1
            last_page = 1
            page_match = list(re.finditer(r"\[PAGE (\d+)\]", chapter_text))
            if page_match:
                first_page = int(page_match[0].group(1))
                last_page = int(page_match[-1].group(1))

            db.insert_book_chunk(
                self.conn, parent_chunk_id, book_id,
                level="chapter", chapter_index=ch_index, section_index=-1,
                chunk_text=chapter_text, chunk_hash=parent_hash,
                chapter=chapter_name,
                page_start=first_page, page_end=last_page,
                embedding=None,
            )

            child_sections = chunk_text(
                chapter_text,
                chunk_size_chars=self.chunk_size_chars,
                chunk_overlap_chars=self.chunk_overlap_chars,
            )

            for sec_index, section in enumerate(child_sections):
                child_chunk_id = _make_child_chunk_id(book_id, ch_index, sec_index)
                child_hash = _hash_text(section["text"])

                if db.chunk_exists_by_hash(self.conn, book_id, child_hash):
                    continue

                embedding_blob = self._embed(section["text"])

                db.insert_book_chunk(
                    self.conn, child_chunk_id, book_id,
                    level="section", chapter_index=ch_index, section_index=sec_index,
                    chunk_text=section["text"], chunk_hash=child_hash,
                    parent_chunk_id=parent_chunk_id,
                    chapter=chapter_name,
                    page_start=section["page_start"],
                    page_end=section["page_end"],
                    embedding=embedding_blob,
                )
                total_chunks += 1

        db.update_book(self.conn, book_id, status="embedding",
                       total_chapters=len(chapters),
                       total_chunks=total_chunks)
        for ch_index, (_, _, chapter_name) in enumerate(chapters):
            parent_id = _make_parent_chunk_id(book_id, ch_index)
            row = self.conn.execute(
                "SELECT page_start, page_end, char_count FROM book_chunks WHERE chunk_id=?", (parent_id,)
            ).fetchone()
            section_count = self.conn.execute(
                "SELECT COUNT(*) FROM book_chunks WHERE book_id=? AND chapter_index=? AND level='section'",
                (book_id, ch_index),
            ).fetchone()[0]
            if row:
                db.upsert_book_chapter(
                    self.conn, book_id, ch_index,
                    title=chapter_name,
                    page_start=row["page_start"], page_end=row["page_end"],
                    char_count=row["char_count"], chunk_count=section_count,
                )

    # ── Index-based indexing ─────────────────────────────────────

    @staticmethod
    def parse_index(path: str) -> list[tuple[str, int]]:
        entries: list[tuple[str, int]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "|" not in line:
                    continue
                title, page_str = line.rsplit("|", 1)
                title = title.strip()
                try:
                    page = int(page_str.strip())
                    entries.append((title, page))
                except ValueError:
                    continue
        return entries

    @staticmethod
    def group_index_entries(entries: list[tuple[str, int]]) -> list[tuple[str, int, list[tuple[str, int]]]]:
        chapters: list[tuple[str, int, list[tuple[str, int]]]] = []
        current_chapter: Optional[tuple[str, int, list[tuple[str, int]]]] = None
        for title, page in entries:
            is_chapter = bool(re.match(r"(?:Ch\.|Chapter|Cap[ií]tulo|Section|Tema)\s*\d+", title, re.IGNORECASE))
            if is_chapter:
                current_chapter = (title, page, [])
                chapters.append(current_chapter)
            elif current_chapter is not None:
                current_chapter[2].append((title, page))
            else:
                current_chapter = (title, page, [])
                chapters.append(current_chapter)
        return chapters

    def _get_page_text(self, full_text: str, page_num: int) -> str:
        pattern = re.escape(f"[PAGE {page_num}]")
        next_pattern = re.escape(f"[PAGE {page_num + 1}]")
        match = re.search(pattern + r"(.*?)(?:" + next_pattern + "|$)", full_text, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _get_page_range_text(self, full_text: str, from_page: int, to_page: int) -> str:
        parts: list[str] = []
        for p in range(from_page, to_page):
            page_text = self._get_page_text(full_text, p)
            if page_text:
                parts.append(f"[PAGE {p}]\n{page_text}")
        return "\n\n".join(parts)

    def _index_book_from_file(self, book_id: str, text: str, index_path: str) -> None:
        entries = self.parse_index(index_path)
        if not entries:
            raise ValueError(f"Indice vacio o invalido: {index_path}")

        chapters = self.group_index_entries(entries)
        total_chunks = 0

        for ch_index, (ch_title, ch_page, caps) in enumerate(chapters):
            ch_next_page = chapters[ch_index + 1][1] if ch_index + 1 < len(chapters) else ch_page + 1
            chapter_text = self._get_page_range_text(text, ch_page, ch_next_page)
            if not chapter_text.strip():
                chapter_text = f"[PAGE {ch_page}]\n" + self._get_page_text(text, ch_page)

            chapter_chunk_id = _make_parent_chunk_id(book_id, ch_index)
            chapter_hash = _hash_text(chapter_text)

            db.insert_book_chunk(
                self.conn, chapter_chunk_id, book_id,
                level="chapter", chapter_index=ch_index, section_index=-1,
                chunk_text=chapter_text, chunk_hash=chapter_hash,
                chapter=ch_title,
                page_start=ch_page, page_end=ch_next_page - 1,
                embedding=None,
            )

            for cap_index, (cap_title, cap_page) in enumerate(caps):
                cap_next_page = caps[cap_index + 1][1] if cap_index + 1 < len(caps) else ch_next_page
                cap_text = self._get_page_range_text(text, cap_page, cap_next_page)
                if not cap_text.strip():
                    continue

                cap_chunk_id = _make_cap_chunk_id(book_id, ch_index, cap_index)
                cap_hash = _hash_text(cap_text)

                db.insert_book_chunk(
                    self.conn, cap_chunk_id, book_id,
                    level="cap", chapter_index=ch_index, section_index=cap_index,
                    chunk_text=cap_text, chunk_hash=cap_hash,
                    parent_chunk_id=chapter_chunk_id,
                    chapter=cap_title,
                    page_start=cap_page, page_end=cap_next_page - 1,
                    embedding=None,
                )

                cap_sections = chunk_text(
                    cap_text,
                    chunk_size_chars=self.chunk_size_chars,
                    chunk_overlap_chars=self.chunk_overlap_chars,
                )

                for sec_index, section in enumerate(cap_sections):
                    sec_chunk_id = _make_cap_section_chunk_id(book_id, ch_index, cap_index, sec_index)
                    sec_hash = _hash_text(section["text"])

                    if db.chunk_exists_by_hash(self.conn, book_id, sec_hash):
                        continue

                    embedding_blob = self._embed(section["text"])

                    db.insert_book_chunk(
                        self.conn, sec_chunk_id, book_id,
                        level="section", chapter_index=ch_index, section_index=sec_index,
                        chunk_text=section["text"], chunk_hash=sec_hash,
                        parent_chunk_id=cap_chunk_id,
                        chapter=cap_title,
                        page_start=section["page_start"], page_end=section["page_end"],
                        embedding=embedding_blob,
                    )
                    total_chunks += 1

        db.update_book(self.conn, book_id, status="embedding",
                       total_chapters=len(chapters),
                       total_chunks=total_chunks)
        for ch_index, (ch_title, ch_page, caps) in enumerate(chapters):
            ch_next_page = chapters[ch_index + 1][1] if ch_index + 1 < len(chapters) else ch_page + 1
            pid = _make_parent_chunk_id(book_id, ch_index)
            prow = self.conn.execute(
                "SELECT char_count FROM book_chunks WHERE chunk_id=?", (pid,)
            ).fetchone()
            char_count = prow["char_count"] if prow else 0
            section_count = self.conn.execute(
                "SELECT COUNT(*) FROM book_chunks WHERE book_id=? AND chapter_index=? AND level='section'",
                (book_id, ch_index),
            ).fetchone()[0]
            db.upsert_book_chapter(
                self.conn, book_id, ch_index,
                title=ch_title,
                page_start=ch_page, page_end=ch_next_page - 1,
                char_count=char_count, chunk_count=section_count,
            )

    # ── Search ──────────────────────────────────────────────────

    def _load_section_embeddings(self, book_id: str) -> tuple[list[dict], np.ndarray]:
        rows = self.conn.execute("""
            SELECT chunk_id, parent_chunk_id, chapter, chapter_index,
                   chunk_text, page_start, page_end, embedding, char_count
            FROM book_chunks
            WHERE book_id=? AND level='section' AND embedding IS NOT NULL
            ORDER BY chapter_index, section_index
        """, (book_id,)).fetchall()

        if not rows:
            return [], np.array([])

        vectors = []
        infos = []
        for r in rows:
            blob = r["embedding"]
            if blob is None:
                continue
            vec = _blob_to_embedding(blob)
            vectors.append(vec)
            infos.append({
                "chunk_id": r["chunk_id"],
                "parent_chunk_id": r["parent_chunk_id"],
                "chapter": r["chapter"],
                "chapter_index": r["chapter_index"],
                "chunk_text": r["chunk_text"],
                "page_start": r["page_start"],
                "page_end": r["page_end"],
                "char_count": r["char_count"],
            })

        if not vectors:
            return [], np.array([])

        return infos, np.array(vectors)

    def search(
        self,
        query: str,
        n_results: int = 5,
        book_id: Optional[str] = None,
        max_distance: Optional[float] = None,
    ) -> list[BookChunk]:
        if not book_id:
            return []

        query_vector = self._embed(query)
        if query_vector is None:
            return []

        query_vec = _blob_to_embedding(query_vector)

        infos, vectors = self._load_section_embeddings(book_id)
        if not infos:
            return []

        similarities = np.array([
            _cosine_similarity(query_vec, v) for v in vectors
        ])

        threshold = max_distance if max_distance is not None else self.search_max_distance
        valid_idx = np.where(similarities >= threshold)[0]

        if len(valid_idx) == 0:
            return []

        top_k = min(n_results, len(valid_idx))
        top_indices = valid_idx[np.argsort(-similarities[valid_idx])[:top_k]]

        results: list[BookChunk] = []
        for idx in top_indices:
            info = infos[idx]
            results.append(BookChunk(
                chunk_id=info["chunk_id"],
                book_id=book_id,
                parent_chunk_id=info["parent_chunk_id"],
                level="section",
                chapter=info["chapter"],
                chapter_index=info["chapter_index"],
                page_start=info["page_start"],
                page_end=info["page_end"],
                text=info["chunk_text"],
                char_count=info["char_count"],
                distance=1.0 - similarities[idx],
            ))

        return results

    # ── Build context ───────────────────────────────────────────

    def truncate_centered(self, parent: dict, child_rows: list[dict], max_chars: int) -> str:
        parent_text = parent["chunk_text"]
        if len(parent_text) <= max_chars:
            return parent_text

        best_child = next(
            (r for r in child_rows if r["parent_chunk_id"] == parent["chunk_id"]),
            None,
        )
        if not best_child:
            return parent_text[:max_chars]

        child_start = parent_text.find(best_child["chunk_text"][:100])
        if child_start == -1:
            return parent_text[:max_chars]

        half = max_chars // 2
        start = max(0, child_start - half)
        end = min(len(parent_text), child_start + half)

        if start > 0:
            s = parent_text.find(" ", start - 20)
            start = s + 1 if s != -1 else start
        if end < len(parent_text):
            e = parent_text.rfind(" ", 0, end)
            end = e if e != -1 else end

        prefix = "[...]" if start > 0 else ""
        suffix = "[...]" if end < len(parent_text) else ""
        return f"{prefix}{parent_text[start:end]}{suffix}"

    def build_context(
        self,
        query: str,
        n_results: int = 3,
        book_id: Optional[str] = None,
        max_chars: int = 3000,
        max_distance: Optional[float] = None,
    ) -> str:
        chunks = self.search(
            query, n_results=n_results, book_id=book_id,
            max_distance=max_distance,
        )
        if not chunks:
            return ""

        parent_ids = list(set(c.parent_chunk_id for c in chunks if c.parent_chunk_id))
        if not parent_ids:
            return ""

        placeholders = ",".join("?" * len(parent_ids))
        parents = self.conn.execute(f"""
            SELECT chunk_id, chapter, chunk_text, char_count
            FROM book_chunks WHERE chunk_id IN ({placeholders})
        """, parent_ids).fetchall()

        child_dict = [
            {"parent_chunk_id": c.parent_chunk_id, "chunk_text": c.text}
            for c in chunks
        ]

        context_parts = []
        for p in parents:
            parent_dict = dict(p)
            p_max = max_chars // len(parents)
            truncated = self.truncate_centered(parent_dict, child_dict, p_max)
            context_parts.append(f"## {parent_dict['chapter']}\n{truncated}")

        context = "\n\n".join(context_parts)
        return f"[BOOK_CONTEXT]\n{context}\n[/BOOK_CONTEXT]"

    # ── Chapter reading ─────────────────────────────────────────

    def list_chapters(self, book_id: str) -> list[ChapterInfo]:
        rows = db.list_chapters(self.conn, book_id)
        return [
            ChapterInfo(
                chapter_index=r["chapter_index"],
                chapter=r.get("title") or r.get("chapter", ""),
                page_start=r["page_start"],
                page_end=r["page_end"],
                char_count=r["char_count"],
                chunk_count=r.get("chunk_count", 0),
                chunk_id=r.get("chunk_id", ""),
            )
            for r in rows
        ]

    def get_chapter(self, book_id: str, chapter_index: int) -> Optional[dict]:
        row = self.conn.execute("""
            SELECT chunk_id, chapter, chunk_text, page_start, page_end, char_count
            FROM book_chunks
            WHERE book_id=? AND chapter_index=? AND level='chapter'
        """, (book_id, chapter_index)).fetchone()
        return dict(row) if row else None

    def build_chapter_context(self, book_id: str, chapter_index: int, max_chars: int) -> str:
        chapter = self.get_chapter(book_id, chapter_index)
        if not chapter:
            return ""

        text = chapter["chunk_text"]
        if len(text) <= max_chars:
            return f"[BOOK_CONTEXT]\n## {chapter['chapter']}\n{text}\n[/BOOK_CONTEXT]"

        truncated = text[:max_chars]
        truncated = truncated[:truncated.rfind(" ")]
        return f"[BOOK_CONTEXT]\n## {chapter['chapter']}\n{truncated}[...]\n[/BOOK_CONTEXT]"

    # ── Multilanguage handling ──────────────────────────────────

    def _get_book_language(self, book_id: Optional[str]) -> str:
        if not book_id:
            return "en"
        book = db.get_book(self.conn, book_id)
        return book["language"] if book else "en"

    def handle_language_mismatch(self, query: str, book_language: str,
                                  llm=None, strategy: str = "passthrough") -> str:
        if strategy == "passthrough" or not llm:
            return query

        try:
            from lingua import LanguageDetectorBuilder
            detector = LanguageDetectorBuilder.from_all_languages().build()
            query_lang = detector.detect_language_of(query[:3000])
            ql = query_lang.iso_code_639_1.name.lower() if query_lang else "en"
        except Exception:
            ql = "es"

        if book_language == ql:
            return query

        if strategy == "translate":
            prompt = f"Translate to {book_language}, output only the translation:\n{query}"
            try:
                res = llm.complete(prompt, max_tokens=200)
                return res.strip()
            except Exception:
                return query

        if strategy == "expand":
            prompt = f"""From: "{query}"
Extract 3-5 key concepts and translate them to {book_language}.
Output only the keywords separated by spaces."""
            try:
                keywords = llm.complete(prompt, max_tokens=50).strip()
                return f"{query} {keywords}"
            except Exception:
                return query

        return query

    # ── CRUD ────────────────────────────────────────────────────

    def get_book(self, book_id: str) -> Optional[dict]:
        return db.get_book(self.conn, book_id)

    def get_book_by_hashes(self, source_file_hash: str,
                           source_text_hash: str) -> Optional[dict]:
        return db.get_book_by_hashes(self.conn, source_file_hash, source_text_hash)

    def get_book_by_file_hash(self, source_file_hash: str) -> Optional[dict]:
        return db.get_book_by_file_hash(self.conn, source_file_hash)

    def list_books(self) -> list[dict]:
        return db.list_books(self.conn, user_id=self._user_id)

    def get_chunks(self, book_id: str, limit: int = 50) -> list[BookChunk]:
        rows = db.get_book_chunks(self.conn, book_id, limit=limit)
        return [
            BookChunk(
                chunk_id=r["chunk_id"],
                book_id=r["book_id"],
                parent_chunk_id=r.get("parent_chunk_id"),
                level=r.get("level", "section"),
                chapter=r.get("chapter", ""),
                chapter_index=r.get("chapter_index", 0),
                section_index=r.get("section_index", 0),
                page_start=r.get("page_start", 0),
                page_end=r.get("page_end", 0),
                text=r["chunk_text"],
                char_count=r.get("char_count", 0),
            )
            for r in rows
        ]

    def delete_book(self, book_id: str) -> None:
        db.delete_book(self.conn, book_id)
        log.info("Libro eliminado: %s", book_id)

    def reindex_book(self, book_id: str) -> None:
        book = db.get_book(self.conn, book_id)
        if not book:
            raise ValueError(f"Libro no encontrado: {book_id}")

        if book.get("source_path") and Path(book["source_path"]).exists():
            self.ingest(
                book["source_path"],
                title=book["title"],
                author=book["author"],
                language=book["language"],
                force=True,
            )
        else:
            raise RuntimeError(
                f"No se puede reindexar {book_id}: source_path no disponible"
            )

    def validate_index(self, book_id: str) -> dict:
        book = db.get_book(self.conn, book_id)
        if not book:
            return {"exists": False, "error": "Libro no encontrado"}

        sql_chunks = db.count_book_chunks(self.conn, book_id)
        sql_sections = self.conn.execute(
            "SELECT COUNT(*) FROM book_chunks WHERE book_id=? AND level='section'",
            (book_id,),
        ).fetchone()[0]

        chunk_ids = set()
        for r in self.conn.execute(
            "SELECT chunk_id FROM book_chunks WHERE book_id=? AND level='section'",
            (book_id,),
        ).fetchall():
            chunk_ids.add(r[0])

        return {
            "exists": True,
            "book_id": book_id,
            "title": book["title"],
            "status": book["status"],
            "total_chunks": sql_chunks,
            "section_chunks": sql_sections,
            "chunk_ids_sorted": sorted(chunk_ids),
        }

    def get_stats(self) -> dict:
        total_books = db.count_books(self.conn, user_id=self._user_id)
        total_chunks = 0
        books = db.list_books(self.conn, user_id=self._user_id)
        for b in books:
            total_chunks += db.count_book_chunks(self.conn, b["id"])
        return {
            "total_books": total_books,
            "total_chunks": total_chunks,
        }
