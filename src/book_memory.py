from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
import tempfile
import uuid
from pathlib import Path
from typing import Any, Optional

from . import db
from .book_models import BookChunk

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
    re.compile(r"^(?:Cap[ií]tulo|Chapter|Section|Tema)\s+\d+[:\.\s]", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^(?:\d+\.\d+\s+.+)", re.MULTILINE),
    re.compile(r"^(?:#{1,3}\s+.+)", re.MULTILINE),
    re.compile(r"^[A-Z\s]{10,}$", re.MULTILINE),
]

_PAGE_MARKER = re.compile(r"\[PAGE (\d+)\]")

_HYPHENATION = re.compile(r"(\w)-\n(\w)")
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


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


def _make_chunk_id(book_id: str, chunk_index: int) -> str:
    return f"{book_id}_chunk_{chunk_index:06d}"


# ── Text cleaning ─────────────────────────────────────────────


def clean_extracted_text(text: str) -> str:
    text = text.replace("\u00ad\n", "")
    text = text.replace("\u00ad", "")
    text = _HYPHENATION.sub(r"\1\2", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
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


def extract_pdf(path: str, layout: str = "plain") -> tuple[str, int]:
    try:
        import pymupdf
    except ImportError:
        raise ImportError(
            "pymupdf no esta instalado. Ejecuta: pip install pymupdf"
        )

    if layout == "plain":
        return _extract_pdf_plain(path)
    if layout == "blocks":
        return _extract_pdf_blocks(path)
    if layout == "two_columns":
        return _extract_pdf_two_columns(path)
    if layout == "auto":
        return _extract_pdf_auto(path)
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


def _extract_pdf_auto(path: str) -> tuple[str, int]:
    import pymupdf
    doc = pymupdf.open(path)
    two_column_pages = 0
    total_pages = len(doc)

    for page in doc:
        if _page_looks_two_columns(page):
            two_column_pages += 1

    if total_pages > 0 and (two_column_pages / total_pages) >= 0.30:
        return _extract_pdf_two_columns(path)
    return _extract_pdf_blocks(path)


def _page_looks_two_columns(page) -> bool:
    blocks = page.get_text("blocks")
    xs = [b[0] for b in blocks if b[4].strip()]
    if len(xs) < 6:
        return False
    page_width = page.rect.width
    middle = page_width / 2
    left_count = sum(1 for x in xs if x < middle)
    right_count = sum(1 for x in xs if x >= middle)
    return left_count >= 3 and right_count >= 3


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


def _detect_chapter(text: str) -> str:
    for pattern in _CHAPTER_PATTERNS:
        match = pattern.search(text)
        if match:
            candidate = match.group(0).strip().rstrip(":")
            if len(candidate) < 100:
                return candidate
    return ""


def chunk_text(
    text: str,
    chunk_size_chars: int = 1000,
    chunk_overlap_chars: int = 200,
) -> list[dict]:
    page_blocks = _detect_pages(text)
    chunks = []
    buffer = ""
    current_chapter = ""
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

            detected_chapter = _detect_chapter(para)

            if detected_chapter and buffer.strip():
                chunks.append({
                    "chapter": current_chapter,
                    "page_start": current_page_start,
                    "page_end": current_page_end,
                    "text": buffer.strip(),
                })
                buffer = ""
                current_page_start = page_num

            if detected_chapter:
                current_chapter = detected_chapter

            if len(buffer) + len(para) > chunk_size_chars and buffer:
                chunks.append({
                    "chapter": current_chapter,
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
            "chapter": current_chapter,
            "page_start": current_page_start,
            "page_end": current_page_end,
            "text": buffer.strip(),
        })

    merged = []
    for c in chunks:
        if len(c["text"]) < 50 and merged:
            merged[-1]["text"] += "\n\n" + c["text"]
            merged[-1]["page_end"] = c["page_end"]
        else:
            merged.append(c)

    return merged


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
        persist_dir: str = "./chroma_db",
        sqlite_conn: Optional[sqlite3.Connection] = None,
        collection_name: str = "book_memory",
        chunk_size_chars: int = 1000,
        chunk_overlap_chars: int = 200,
        search_max_distance: float = 0.75,
    ):
        self._persist_dir = persist_dir
        self._conn: Optional[sqlite3.Connection] = sqlite_conn
        self._collection_name = collection_name
        self._collection: Optional[Any] = None
        self._reranker: Any = None
        self.chunk_size_chars = chunk_size_chars
        self.chunk_overlap_chars = chunk_overlap_chars
        self.search_max_distance = search_max_distance

    # ── Lifecycle ───────────────────────────────────────────────

    def _ensure_collection(self):
        if self._collection is not None:
            return
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb no esta instalado. Ejecuta: pip install chromadb"
            )
        client = chromadb.PersistentClient(path=self._persist_dir)
        self._collection = client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def collection(self):
        self._ensure_collection()
        return self._collection

    @property
    def conn(self):
        if self._conn is None:
            raise RuntimeError("BookMemory requiere sqlite_conn")
        return self._conn

    def close(self):
        self._collection = None

    # ── Has books / Should search ───────────────────────────────

    def has_books(self) -> bool:
        return db.has_books(self.conn)

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

    def ingest(
        self,
        path: str,
        title: str = "",
        author: str = "",
        force: bool = False,
        pdf_layout: str = "auto",
        save_extracted_to: Optional[str] = None,
    ) -> str:
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {path}")

        source_hash = _hash_file(path)
        existing = self.get_book_by_hash(source_hash)
        if existing and not force:
            log.info("Libro ya indexado: %s (%s)", existing["title"], existing["id"])
            return existing["id"]
        if existing and force:
            log.info("Reindexando libro: %s", existing["id"])
            self._delete_from_chroma(existing["id"])
            db.delete_book(self.conn, existing["id"])

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

        if save_extracted_to:
            out = Path(save_extracted_to)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            log.info("Texto extraido guardado en: %s", out)

        book_id = _generate_book_id()
        db.insert_book(
            self.conn, book_id, str(path_obj), source_hash,
            title=title or path_obj.stem, author=author,
            status="extracting", total_pages=total_pages,
        )

        try:
            db.update_book(self.conn, book_id, status="chunking")
            raw_chunks = chunk_text(
                text,
                chunk_size_chars=self.chunk_size_chars,
                chunk_overlap_chars=self.chunk_overlap_chars,
            )
            if not raw_chunks:
                raise RuntimeError("No se generaron chunks del libro")

            db.update_book(self.conn, book_id, status="embedding")
            for i, c in enumerate(raw_chunks):
                chunk_id = _make_chunk_id(book_id, i)
                chunk_hash = _hash_text(c["text"])
                db.insert_book_chunk(
                    self.conn, chunk_id, book_id, i, c["text"], chunk_hash,
                    chapter=c["chapter"],
                    page_start=c["page_start"],
                    page_end=c["page_end"],
                )

            db.update_book(self.conn, book_id, status="embedding",
                           total_chunks=len(raw_chunks))
            self._index_chunks(book_id, raw_chunks)

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
        force: bool = False,
    ) -> str:
        source_hash = _hash_text(text)
        existing = self.get_book_by_hash(source_hash)
        if existing and not force:
            return existing["id"]
        if existing and force:
            self._delete_from_chroma(existing["id"])
            db.delete_book(self.conn, existing["id"])

        text = clean_extracted_text(text)

        book_id = _generate_book_id()
        db.insert_book(
            self.conn, book_id, "", source_hash,
            title=title or "Untitled", author=author,
            status="chunking", total_pages=1,
        )

        try:
            raw_chunks = chunk_text(
                text,
                chunk_size_chars=self.chunk_size_chars,
                chunk_overlap_chars=self.chunk_overlap_chars,
            )
            if not raw_chunks:
                raise RuntimeError("No se generaron chunks del libro")

            db.update_book(self.conn, book_id, status="embedding")
            for i, c in enumerate(raw_chunks):
                chunk_id = _make_chunk_id(book_id, i)
                chunk_hash = _hash_text(c["text"])
                db.insert_book_chunk(
                    self.conn, chunk_id, book_id, i, c["text"], chunk_hash,
                    chapter=c["chapter"],
                    page_start=c["page_start"],
                    page_end=c["page_end"],
                )

            db.update_book(self.conn, book_id, status="embedding",
                           total_chunks=len(raw_chunks))
            self._index_chunks(book_id, raw_chunks)

            db.update_book(self.conn, book_id, status="indexed")
        except Exception as e:
            db.update_book(self.conn, book_id, status="error",
                           error_message=str(e))
            raise

        return book_id

    # ── Internal ChromaDB + FTS5 helpers ────────────────────────

    def _index_chunks(self, book_id: str, raw_chunks: list[dict]) -> None:
        coll = self.collection
        ids = []
        documents = []
        metadatas = []

        book = db.get_book(self.conn, book_id)
        book_title = book["title"] if book else ""

        for i, c in enumerate(raw_chunks):
            chunk_id = _make_chunk_id(book_id, i)
            ids.append(chunk_id)
            documents.append(c["text"])
            metadatas.append({
                "book_id": book_id,
                "book_title": book_title,
                "chunk_index": i,
                "chunk_id": chunk_id,
                "language": "es",
            })
            db.insert_book_chunk_fts(self.conn, chunk_id, book_id, c["text"], c.get("chapter", ""))

        coll.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def _delete_from_chroma(self, book_id: str) -> None:
        coll = self.collection
        results = coll.get(where={"book_id": {"$eq": book_id}})
        if results["ids"]:
            coll.delete(ids=results["ids"])

    # ── Reranker ────────────────────────────────────────────────

    def load_reranker(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        try:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(model_name)
            log.info("Reranker cargado: %s", model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers no instalado. Ejecuta: pip install sentence-transformers"
            )

    def _rerank(self, query: str, chunks: list[BookChunk], top_n: int = 5) -> list[BookChunk]:
        if self._reranker is None or not chunks:
            return chunks[:top_n]

        pairs = [(query, c.text) for c in chunks]
        scores = self._reranker.predict(pairs)
        ranked = sorted(zip(chunks, scores), key=lambda x: float(x[1]), reverse=True)
        return [c for c, _ in ranked][:top_n]

    # ── Search ──────────────────────────────────────────────────

    def search(
        self,
        query: str,
        n_results: int = 5,
        book_id: Optional[str] = None,
        max_distance: Optional[float] = None,
        rerank: bool = False,
    ) -> list[BookChunk]:
        coll = self.collection
        where: dict = {}
        if book_id:
            where["book_id"] = {"$eq": book_id}

        results = coll.query(
            query_texts=[query],
            n_results=n_results,
            where=where if where else None,
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        chunks: list[BookChunk] = []
        threshold = max_distance if max_distance is not None else self.search_max_distance

        for i, chroma_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i] if results["distances"] else 0.0
            if distance > threshold:
                continue

            row = db.get_book_chunk_by_chunk_id(self.conn, chroma_id)
            if row is None:
                continue

            book_row = db.get_book(self.conn, row["book_id"])
            book_title = book_row["title"] if book_row else ""

            chunks.append(BookChunk(
                chunk_id=chroma_id,
                book_id=row["book_id"],
                book_title=book_title,
                chapter=row["chapter"],
                page_start=row["page_start"],
                page_end=row["page_end"],
                chunk_index=row["chunk_index"],
                text=row["chunk_text"],
                char_count=row["char_count"],
                distance=distance,
            ))

        if rerank and chunks:
            chunks = self._rerank(query, chunks, top_n=n_results)

        return chunks

    def search_keyword(self, query: str, n_results: int = 5,
                       book_id: Optional[str] = None) -> list[BookChunk]:
        rows = db.search_book_chunks_fts(self.conn, query, n_results, book_id)
        if not rows:
            return []

        results: list[BookChunk] = []
        for r in rows:
            book_row = db.get_book(self.conn, r["book_id"])
            results.append(BookChunk(
                chunk_id=r["chunk_id"],
                book_id=r["book_id"],
                book_title=book_row["title"] if book_row else "",
                chapter=r["chapter"],
                page_start=r["page_start"],
                page_end=r["page_end"],
                chunk_index=r["chunk_index"],
                text=r["chunk_text"],
                char_count=r["char_count"],
                distance=0.0,
            ))
        return results

    def search_hybrid(self, query: str, n_results: int = 5,
                      book_id: Optional[str] = None,
                      sem_weight: float = 0.5) -> list[BookChunk]:
        sem_weight = max(0.0, min(1.0, sem_weight))
        sem_results = self.search(query, n_results=n_results * 2, book_id=book_id)
        kw_results = self.search_keyword(query, n_results=n_results * 2, book_id=book_id)

        scores: dict[str, tuple[BookChunk, float]] = {}

        for rank, c in enumerate(sem_results):
            score = sem_weight * (1.0 / (rank + 1))
            prev = scores.get(c.chunk_id, (c, 0.0))[1]
            scores[c.chunk_id] = (c, prev + score)

        for rank, c in enumerate(kw_results):
            score = (1.0 - sem_weight) * (1.0 / (rank + 1))
            prev = scores.get(c.chunk_id, (c, 0.0))[1]
            scores[c.chunk_id] = (c, prev + score)

        ranked = sorted(scores.values(), key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in ranked[:n_results]]

    # ── Build context ───────────────────────────────────────────

    def build_context(
        self,
        query: str,
        n_results: int = 5,
        book_id: Optional[str] = None,
        max_chars: int = 5000,
        max_distance: Optional[float] = None,
    ) -> str:
        chunks = self.search(
            query, n_results=n_results, book_id=book_id,
            max_distance=max_distance,
        )
        if not chunks:
            return ""

        lines = ["[BOOK_CONTEXT]"]
        total_chars = 0
        added = False

        for c in chunks:
            header = f"Fuente: {c.book_title}"
            if c.chapter:
                header += f" ({c.chapter}"
                if c.page_start:
                    header += f", pagina {c.page_start}"
                header += ")"
            entry = f"{header}:\n  {c.text.strip()}"
            if total_chars + len(entry) > max_chars:
                if not added:
                    entry = entry[:max_chars].rstrip() + "..."
                else:
                    break
            lines.append("")
            lines.append(entry)
            total_chars += len(entry)
            added = True

        return "\n".join(lines) if added else ""

    # ── CRUD ────────────────────────────────────────────────────

    def get_book(self, book_id: str) -> Optional[dict]:
        return db.get_book(self.conn, book_id)

    def get_book_by_hash(self, source_hash: str) -> Optional[dict]:
        return db.get_book_by_hash(self.conn, source_hash)

    def list_books(self) -> list[dict]:
        return db.list_books(self.conn)

    def get_chunks(self, book_id: str, limit: int = 50) -> list[BookChunk]:
        rows = db.get_book_chunks(self.conn, book_id, limit=limit)
        return [
            BookChunk(
                chunk_id=r["chunk_id"],
                book_id=r["book_id"],
                chapter=r["chapter"],
                page_start=r["page_start"],
                page_end=r["page_end"],
                chunk_index=r["chunk_index"],
                text=r["chunk_text"],
                char_count=r["char_count"],
            )
            for r in rows
        ]

    def delete_book(self, book_id: str) -> None:
        self._delete_from_chroma(book_id)
        db.delete_book(self.conn, book_id)
        log.info("Libro eliminado: %s", book_id)

    def reindex_book(self, book_id: str) -> None:
        book = db.get_book(self.conn, book_id)
        if not book:
            raise ValueError(f"Libro no encontrado: {book_id}")

        self._delete_from_chroma(book_id)
        rows = db.get_book_chunks(self.conn, book_id)
        if not rows:
            raise RuntimeError(f"El libro {book_id} no tiene chunks para reindexar")
        raw_chunks = [
            {"text": r["chunk_text"], "chapter": r["chapter"],
             "page_start": r["page_start"], "page_end": r["page_end"]}
            for r in rows
        ]
        self._index_chunks(book_id, raw_chunks)
        db.update_book(self.conn, book_id, status="indexed")
        log.info("Libro reindexado: %s", book_id)

    def validate_index(self, book_id: str) -> dict:
        book = db.get_book(self.conn, book_id)
        if not book:
            return {"exists": False, "error": "Libro no encontrado"}

        sql_chunks = db.count_book_chunks(self.conn, book_id)

        coll = self.collection
        chroma_results = coll.get(where={"book_id": {"$eq": book_id}})
        chroma_count = len(chroma_results["ids"]) if chroma_results["ids"] else 0

        return {
            "exists": True,
            "book_id": book_id,
            "title": book["title"],
            "status": book["status"],
            "sql_chunks": sql_chunks,
            "chroma_chunks": chroma_count,
            "match": sql_chunks == chroma_count,
            "sql_chunk_ids": {r["chunk_id"] for r in db.get_book_chunks(self.conn, book_id)},
            "chroma_chunk_ids": set(chroma_results["ids"]) if chroma_results["ids"] else set(),
        }

    def get_stats(self) -> dict:
        total_books = db.count_books(self.conn)
        total_chunks = 0
        books = db.list_books(self.conn)
        for b in books:
            total_chunks += db.count_book_chunks(self.conn, b["id"])
        return {
            "total_books": total_books,
            "total_chunks": total_chunks,
        }
