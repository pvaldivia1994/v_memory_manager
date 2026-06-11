# Plan: Parent-Child para BookMemory

## Problema

La estructura actual de `books` usa 3 tablas SQLite con sincronización manual:

- `books` — metadatos
- `book_chunks` — chunks planos de ~1000 chars
- `book_chunks_fts` — virtual table FTS5 sincronizada a mano con `insert_book_chunk_fts()` y `delete_book_chunks_fts()`

No hay relaciones declarativas reales entre `book_chunks` y `book_chunks_fts`. Si una operación olvida actualizar la FTS5, quedan inconsistencias. Los chunks son uniformes y no respetan la jerarquía natural del libro (capítulos, secciones). El LLM recibe fragmentos aislados sin el contexto del capítulo completo, lo que produce respuestas imprecisas o referencias rotas.

---

## Solución propuesta

Eliminar FTS5 y ChromaDB como store externo. Pasar a una sola fuente de verdad: **SQLite con sqlite-vec** para embeddings embebidos como `BLOB`. La búsqueda se hace con `sqlite-vec` directamente, sin sincronización entre stores.

---

## Estructura de datos

### `books`

```sql
CREATE TABLE books (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL DEFAULT 'default',
    title           TEXT NOT NULL DEFAULT '',
    author          TEXT NOT NULL DEFAULT '',
    source_path     TEXT NOT NULL,
    source_file_hash TEXT NOT NULL,
    source_text_hash TEXT NOT NULL DEFAULT '',
    source_type     TEXT NOT NULL DEFAULT '',
    source_layout   TEXT NOT NULL DEFAULT '',
    language        TEXT NOT NULL DEFAULT 'es',
    total_chapters  INTEGER NOT NULL DEFAULT 0,
    total_chunks    INTEGER NOT NULL DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'pending',
    error_message   TEXT NOT NULL DEFAULT '',
    embedding_model TEXT NOT NULL DEFAULT '',
    embedding_dim   INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(source_file_hash, source_text_hash)
);
```

Se agrega `user_id` para aislamiento multi-usuario (mismo patrón que `SemanticMemory`). Se agrega `total_chapters` y se elimina `chunker_version`, `schema_version`. `status` conserva el pipeline: `extracting → chunking → embedding → indexed`.

### `book_chunks` (jerárquica)

```sql
CREATE TABLE book_chunks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id        TEXT NOT NULL UNIQUE,
    book_id         TEXT NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    parent_chunk_id TEXT REFERENCES book_chunks(chunk_id),
    level           TEXT NOT NULL CHECK(level IN ('chapter', 'section')),
    chapter         TEXT NOT NULL DEFAULT '',
    chapter_index   INTEGER NOT NULL DEFAULT 0,
    section_index   INTEGER NOT NULL DEFAULT 0,
    page_start      INTEGER NOT NULL DEFAULT 0,
    page_end        INTEGER NOT NULL DEFAULT 0,
    chunk_text      TEXT NOT NULL,
    char_count      INTEGER NOT NULL DEFAULT 0,
    chunk_hash      TEXT NOT NULL DEFAULT '',
    embedding       BLOB,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_book_chunks_book_id ON book_chunks(book_id);
CREATE INDEX idx_book_chunks_parent ON book_chunks(parent_chunk_id);
CREATE INDEX idx_book_chunks_book_chapter ON book_chunks(book_id, chapter_index);
CREATE UNIQUE INDEX idx_book_chunks_book_section ON book_chunks(book_id, chapter_index, section_index);
CREATE INDEX idx_book_chunks_hash ON book_chunks(chunk_hash);
```

El `UNIQUE INDEX` sobre `(book_id, chapter_index, section_index)` puede colisionar si un `ingest()` falla a mitad y se reintenta. El pipeline maneja esto con `ON CONFLICT REPLACE` o chequeo previo de existencia.

| Campo | Uso |
|---|---|
| `chunk_id` | Formato: `{book_id}_ch_{ch_index:04d}` (padre) o `{book_id}_ch_{ch_index:04d}_s_{sec_index:04d}` (hijo) |
| `parent_chunk_id` | `NULL` para nivel `chapter`, apunta al padre para nivel `section` |
| `level` | `"chapter"` (padre) o `"section"` (hijo) |
| `chapter_index` | Orden del capítulo dentro del libro |
| `section_index` | Orden de la sección dentro del capítulo (0 para padres) |
| `chunk_hash` | SHA256 del `chunk_text`. Sirve como guardia de idempotencia en re-indexación |
| `embedding` | Vector F32 como `BLOB` — solo para nivel `section`; `NULL` para `chapter` |

### ¿Por qué sqlite-vec y no ChromaDB?

ChromaDB ya usa SQLite internamente (DuckDB + su propio store). Tener ChromaDB + SQLite es pagar overhead de dos bases de datos para lo mismo. sqlite-vec permite:

- Una sola base de datos `.db` como fuente de verdad
- Embeddings como `BLOB(F32)` en la misma fila del chunk hijo
- Búsqueda vectorial directa sin stores externos
- Sin sincronización, sin procesos separados

sqlite-vec se carga como extensión de SQLite. No requiere servidor, no requiere `pip install chromadb`, no requiere construir colecciones separadas.

---

## Pipeline de chunking

### Paso 1: Extracción de texto

Se mantiene igual que hoy — `extract_txt()`, `extract_epub()`, `extract_pdf()` con sus variantes de layout y OCR. Devuelven `(text, total_pages)`.

### Paso 2: Limpieza

Se mantiene `clean_extracted_text()`.

### Paso 3: Detección de capítulos

`_detect_chapter()` actual + **fallback estructural**: si la detección por texto no encuentra capítulos (PDF mal parseado, sin marcadores), se dividen forzadamente bloques de ~4000 palabras como pseudo-capítulos. Esto evita que un libro sin estructura detectable caiga en un solo chunk padre de 100k tokens.

```python
MAX_CHAPTER_WORDS = 4000

def detect_chapters(text: str) -> list[tuple[int, int, str]]:
    chapters = _detect_chapter_regex(text)
    if not chapters:
        chapters = _fallback_by_word_count(text, MAX_CHAPTER_WORDS)
    return chapters
```

### Paso 4: Creación de chunks padre (capítulos)

Por cada capítulo detectado:

```python
for ch_index, (start, end, chapter_name) in enumerate(chapters):
    chapter_text = text[start:end]
    parent_chunk_id = f"{book_id}_ch_{ch_index:04d}"
    
    insert_chunk(
        chunk_id=parent_chunk_id, book_id=book_id,
        parent_chunk_id=NULL, level="chapter",
        chapter=chapter_name, chapter_index=ch_index, section_index=0,
        chunk_text=chapter_text, chunk_hash=sha256(chapter_text),
        embedding=NULL
    )
```

### Paso 5: Creación de chunks hijo (secciones)

Cada capítulo se subdivide con `chunk_text()`, ~800 chars con ~150 overlap:

```python
child_sections = chunk_text(chapter_text, chunk_size_chars=800, chunk_overlap_chars=150)

for sec_index, section in enumerate(child_sections):
    child_chunk_id = f"{book_id}_ch_{ch_index:04d}_s_{sec_index:04d}"
    child_hash = sha256(section["text"])
    
    # Idempotencia: si este hash ya existe, skip
    if chunk_exists_by_hash(conn, book_id, child_hash):
        continue
    
    embed_vector = embedding_model.embed(section["text"])
    
    insert_chunk(
        chunk_id=child_chunk_id, book_id=book_id,
        parent_chunk_id=parent_chunk_id, level="section",
        chapter=chapter_name, chapter_index=ch_index,
        section_index=sec_index,
        chunk_text=section["text"], chunk_hash=child_hash,
        embedding=embed_vector
    )
```

No hay merge de chunks < 50 chars — los hijos pequeños son válidos como puntos de entrada para búsqueda.

---

## Flujo de búsqueda y construcción de contexto

### Advertencia sobre filtro `book_id` en sqlite-vec

sqlite-vec aplica filtros SQL (`WHERE`) **después** de la búsqueda ANN (Approximate Nearest Neighbors). Si filtras por `book_id` dentro de la query, los top-N pueden venir todos de otro libro y el filtro descarta todo, devolviendo 0 resultados.

**Solución**: buscar sin filtro con un `LIMIT` amplio y filtrar en Python:

```python
def search(query: str, book_id: str, user_id: str, n_results: int = 3, max_chars: int = 3000):
    query_vector = embedding_model.embed(query)
    
    rows = conn.execute("""
        SELECT bc.chunk_id, bc.book_id, b.user_id, bc.parent_chunk_id,
               bc.chunk_text, bc.distance
        FROM book_chunks bc
        JOIN books b ON b.id = bc.book_id
        WHERE bc.level = 'section' AND bc.embedding MATCH ?
        ORDER BY bc.distance
        LIMIT ?
    """, (query_vector, n_results * 10)).fetchall()  # margen 10x porque ANN filtra después
    
    rows = [r for r in rows if r["book_id"] == book_id and r["user_id"] == user_id][:n_results]
    
    if not rows:
        return ""
    
    parent_ids = set(r["parent_chunk_id"] for r in rows)
    
    parents = conn.execute("""
        SELECT chunk_id, chapter, chunk_text, char_count
        FROM book_chunks WHERE chunk_id IN ({})
    """.format(",".join("?" * len(parent_ids))), list(parent_ids)).fetchall()
    
    context_parts = []
    for parent in parents:
        parent_text = truncate_centered(parent, rows, max_chars // len(parents))
        context_parts.append(f"## {parent['chapter']}\n{parent_text}")
    
    return "\n\n".join(context_parts)
```

### Truncado centrado

```python
def truncate_centered(parent: dict, child_rows: list, max_chars: int):
    parent_text = parent["chunk_text"]
    if len(parent_text) <= max_chars:
        return parent_text
    
    best_child = next((r for r in child_rows if r["parent_chunk_id"] == parent["chunk_id"]), None)
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
```

### Límite de contexto con llama.cpp

El context window de llama.cpp es fijo (4k–32k tokens). La estimación chars/token debe ser conservadora para español con tokenización BPE:

```python
CHARS_PER_TOKEN = 3.0   # conservador para español BPE
SAFETY_MARGIN = 0.85    # 15% de buffer

def get_max_chars_for_context(model_context_window: int, reserved_tokens: int = 1024) -> int:
    available = (model_context_window - reserved_tokens) * SAFETY_MARGIN
    return int(available * CHARS_PER_TOKEN)
```

Si el capítulo padre excede `max_chars`, se trunca centrado (como arriba). Si hay múltiples padres relevantes, se divide `max_chars` equitativamente.

---

## Integración con MemoryManager

### `build_system_prompt()` (en `memory.py`)

```python
extra_context = ""
if book_mem and book_mem.has_books(user_id):
    book_context = book_mem.build_context(user_input, user_id=user_id,
                                          max_chars=get_max_chars_for_context(model_window))
    if book_context:
        extra_context = f"[BOOK_CONTEXT]\n{book_context}\n[/BOOK_CONTEXT]"
```

`has_books(user_id)`: `SELECT COUNT(*) FROM books WHERE status = 'indexed' AND user_id = ?`

---

## Tablas y código a eliminar

### SQLite

- `book_chunks_fts` (virtual table FTS5)
- Shadow tables de FTS5: `book_chunks_fts_data`, `book_chunks_fts_idx`, `book_chunks_fts_content`, `book_chunks_fts_docsize`, `book_chunks_fts_config`

### Código en `db.py`

- `_ensure_fts5()`
- `insert_book_chunk_fts()`
- `delete_book_chunk_fts()`
- `delete_book_chunks_fts()`
- `search_book_chunks_fts()`

### Código en `book_memory.py`

- `search_keyword()`
- `search_hybrid()`
- `search_smart()` (depende de FTS5 + ChromaDB)
- `build_context_smart()`
- Merge de chunks < 50 chars en `chunk_text()`
- `_index_chunks()` actual (apunta a ChromaDB)
- Toda referencia a ChromaDB: `chromadb`, `collection`, `collection.upsert()`

### Dependencias

- `chromadb` → eliminar
- `sqlite-vec` → agregar

---

## Cambios en `db.py`

| Función | Acción |
|---|---|
| `init_db()` | Eliminar `_ensure_fts5()`, cargar extensión sqlite-vec |
| `SCHEMA_SQL` | Reemplazar `book_chunks` con la versión jerárquica + `user_id` en `books` |
| `insert_book()` | Agregar `user_id`, `total_chapters`. Eliminar `chunker_version`, `schema_version` |
| `insert_book_chunk()` | Agregar `parent_chunk_id`, `level`, `chapter_index`, `section_index`, `embedding`, `chunk_hash` |
| `update_book()` | Sin cambios |
| `get_book()` | Sin cambios |
| `list_books()` | Agregar filtro por `user_id` |
| `delete_book()` | Sin cambios (CASCADE se encarga de chunks) |
| `has_books()` | **NUEVA** — `SELECT COUNT(*) FROM books WHERE status='indexed' AND user_id=?` |
| `chunk_exists_by_hash()` | **NUEVA** — idempotencia en re-indexación |
| `search_book_chunks_fts()` | **ELIMINAR** |
| `insert_book_chunk_fts()` | **ELIMINAR** |
| `delete_book_chunks_fts()` | **ELIMINAR** |
| `delete_book_chunk_fts()` | **ELIMINAR** |
| `_ensure_fts5()` | **ELIMINAR** |
| `get_parent_by_chunk_id()` | **NUEVA** — obtener padre desde chunk_id |
| `search_by_vector()` | **NUEVA** — búsqueda ANN con sqlite-vec |

---

## Cambios en `book_memory.py`

| Función | Acción |
|---|---|
| `__init__()` | Eliminar inicialización de ChromaDB. Aceptar `user_id` |
| `ingest()` | Simplificar: eliminar paso de ChromaDB. Usar `user_id` |
| `ingest_text()` | Simplificar igual |
| `extract_and_ingest()` | Sin cambios (solo extracción) |
| `chunk_text()` | Eliminar merge de < 50 chars |
| `_detect_chapter()` | Agregar fallback `_fallback_by_word_count()` |
| `_index_chunks()` | Reemplazar: insertar en SQLite con embedding vía sqlite-vec, chunk_hash como guardia de idempotencia, `ON CONFLICT` handling |
| `search()` | Reemplazar: consulta ANN con margen 10x, filtro `book_id` en Python, recuperación de padres, truncado centrado |
| `search_keyword()` | **ELIMINAR** |
| `search_hybrid()` | **ELIMINAR** |
| `search_smart()` | **ELIMINAR** |
| `build_context()` | Reemplazar: wrapper de `search()` con `get_max_chars_for_context()` |
| `build_context_smart()` | **ELIMINAR** |
| `has_books()` | Sin cambios, pero ahora chequea `user_id` |
| `get_book()` | Sin cambios |
| `list_books()` | Sin cambios |
| `delete_book()` | Sin cambios |

---

## Migración

La data existente en `books` y `book_chunks` **no es compatible** con el nuevo esquema jerárquico.

Opción recomendada: **reset total**. Un solo `DROP TABLE IF EXISTS book_chunks, book_chunks_fts` + recrear con nuevo schema. El usuario pierde los índices vectoriales, pero los libros se reindexan automáticamente al próximo `ingest()`.

Si se necesita migrar datos existentes: script `v1_to_v2.py` que lee la tabla `book_chunks` vieja, reagrupa por capítulo detectado en el texto, y reinserta respetando el nuevo schema.

---

---

## Lectura de capítulos completos

El diseño padre-hijo da gratis la lectura de capítulos enteros, porque el texto completo de cada capítulo ya está en el `chunk_text` del padre.

### `list_chapters()` — navegación

```python
def list_chapters(self, book_id: str) -> list[dict]:
    """Devuelve tabla de contenido del libro."""
    rows = conn.execute("""
        SELECT chapter_index, chapter, page_start, page_end, char_count
        FROM book_chunks
        WHERE book_id = ? AND level = 'chapter'
        ORDER BY chapter_index ASC
    """, (book_id,)).fetchall()
    return [dict(r) for r in rows]
```

### `get_chapter()` — leer capítulo completo

```python
def get_chapter(self, book_id: str, chapter_index: int) -> dict | None:
    """Devuelve el texto completo de un capítulo."""
    row = conn.execute("""
        SELECT chunk_id, chapter, chunk_text, page_start, page_end, char_count
        FROM book_chunks
        WHERE book_id = ? AND chapter_index = ? AND level = 'chapter'
    """, (book_id, chapter_index)).fetchone()
    return dict(row) if row else None
```

### `build_chapter_context()` — inyección en prompt

La diferencia con `truncate_centered()` (que corta alrededor de un resultado de búsqueda) es que acá se quiere lectura lineal desde el inicio:

```python
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
```

### Integración en consola

En `console_chat.py`, comandos propuestos:

```
/book chapters         → lista capítulos disponibles
/book chapter 0        → inyecta capítulo completo en el contexto
/book chapter "Introducción"  → match por nombre
```

---

---

## Multilenguaje: libro en idioma X, query en idioma Y

### Problema

Los embeddings son sensibles al idioma. Si el libro está en inglés y el usuario pregunta en español, la similitud coseno entre ambos vectores será baja aunque semánticamente sean equivalentes. El impacto depende del modelo:

| Modelo | Comportamiento cross-lingual |
|---|---|
| Monolingüal (e.g. `paraphrase-es-*`) | Muy malo — casi no encuentra nada |
| Multilingüal (e.g. `multilingual-e5`, `paraphrase-multilingual-mpnet`) | Bueno — espacio vectorial compartido |
| Con instrucción (e.g. `mE5-large-instruct`) | Muy bueno — especialmente con prefijos |

### Detección de idioma al indexar

Se agrega dependencia `lingua` (o `langdetect`) para poblar `language` automáticamente al hacer `ingest()`:

```python
from lingua import Language, LanguageDetectorBuilder

detector = LanguageDetectorBuilder.from_all_languages().build()

def detect_language(text: str) -> str:
    sample = text[:3000]
    lang = detector.detect_language_of(sample)
    return lang.iso_code_639_1.name.lower() if lang else "en"
```

Esto se ejecuta post-extracción, pre-chunking, y se guarda en `books.language`.

### Estrategias de búsqueda cross-lingual

En `search()`, si `book["language"] != detect_language(query)`, se elige una estrategia según config:

```python
# En configurations:
# book_query_mismatch_strategy = "passthrough" (default) | "translate" | "expand"

def _handle_language_mismatch(query: str, book_lang: str, strategy: str) -> str:
    if strategy == "passthrough":
        return query  # confiar en modelo multilingüal
    
    query_lang = detect_language(query)
    if book_lang == query_lang:
        return query
    
    if strategy == "translate":
        # Traducir query completa al idioma del libro
        prompt = f"Translate to {book_lang}, output only the translation:\n{query}"
        return llm.complete(prompt, max_tokens=200).strip()
    
    if strategy == "expand":
        # Extraer conceptos clave y traducirlos, concatenar
        prompt = f"""From: "{query}"
Extract 3-5 key concepts and translate them to {book_lang}.
Output only the keywords separated by spaces."""
        keywords = llm.complete(prompt, max_tokens=50).strip()
        return f"{query} {keywords}"
    
    return query
```

La estrategia default es `"passthrough"` — el usuario que carga un modelo multilingüal no paga overhead. `"translate"` y `"expand"` son opt-in desde `configurations`.

### Integración en `search()`

```python
def search(self, query: str, book_id: str, user_id: str, n_results: int = 3, max_chars: int = 3000):
    book = self.get_book(book_id)
    strategy = self.get_config("book_query_mismatch_strategy", default="passthrough")
    
    query = self._handle_language_mismatch(query, book["language"], strategy)
    query_vector = embedding_model.embed(query)
    # ... resto igual
```

### Resumen de cambios

| Dónde | Cambio |
|---|---|
| `ingest()` | Detectar y guardar `language` automáticamente (librería `lingua`) |
| `books.language` | Ya existe, ahora se puebla con detector automático |
| `search()` | Comparar idiomas antes de embeddear, aplicar estrategia |
| `configurations` | Nueva key `book_query_mismatch_strategy` |
| Dependencias | Agregar `lingua` o `langdetect` |

---

## Flujo completo (visión integrada)

```
1. INGESTA (indexar un libro)
   ──────────────────────────
   
   ingest("libro.pdf")
     │
     ├─ 1. Hash SHA256 del archivo → dedup
     │
     ├─ 2. Extraer texto según extensión:
     │      .txt → extract_txt()
     │      .pdf → extract_pdf() con layout auto
     │      .epub → extract_epub()
     │      Si PDF sin texto → extract_ocr_pdf()
     │
     ├─ 3. clean_extracted_text() → elimina soft hyphens, hyphenation
     │
     ├─ 4. INSERT INTO books (status='extracting')
     │
     ├─ 5. detect_chapters(texto)
     │      ├─ _detect_chapter_regex()
     │      └─ si no encuentra → _fallback_by_word_count(4000)
     │
     ├─ 6. Por cada capítulo:
     │      ├─ INSERT chunk PADRE (level='chapter', embedding=NULL)
     │      └─ chunk_text(capítulo, 800 chars, 150 overlap)
     │           └─ Por cada sección:
     │                ├─ sha256(section_text) → si existe, SKIP
     │                ├─ embedding_model.embed(section_text)
     │                └─ INSERT chunk HIJO (level='section',
     │                     parent_chunk_id=id_del_padre, embedding=BLOB)
     │
     └─ 7. UPDATE books SET status='indexed'


2. BÚSQUEDA RAG (chat → contexto)
   ────────────────────────────────
   
   build_context("¿quién es el dios del inframundo?")
     │
     ├─ 1. embedding_model.embed(query) → vector
     │
     ├─ 2. sqlite-vec ANN sin filtro:
     │      SELECT ... FROM book_chunks
     │      WHERE level='section' AND embedding MATCH ?
     │      LIMIT n_results * 10       ← margen porque ANN filtra después
     │
     ├─ 3. Filtrar en Python por book_id + user_id
     │      rows = [r for r in rows if r["book_id"] == book_id][:n_results]
     │
     ├─ 4. Extraer parent_chunk_id únicos de los hijos encontrados
     │
     ├─ 5. SELECT padres completos desde SQLite
     │
     ├─ 6. Por cada padre: truncate_centered()
     │      └─ Ubica al hijo dentro del texto del padre
     │      └─ Corta ventana de max_chars // len(padres) alrededor del hijo
     │      └─ Agrega [...] si hay texto cortado
     │
     └─ 7. Inyecta [BOOK_CONTEXT] en el prompt del LLM


3. LECTURA DIRECTA (opcional, por comando)
   ────────────────────────────────────────
   
   /book chapters → SELECT level='chapter' ORDER BY chapter_index
   /book chapter 0 → SELECT chunk_text WHERE chapter_index=0 AND level='chapter'
                     └─ build_chapter_context() → inyecta desde el inicio
                        (truncado lineal, NO centrado)


4. ARQUITECTURA DE DATOS
   ──────────────────────
   
   SQLite (única fuente de verdad)
   ├─ books          → metadatos + user_id + status
   └─ book_chunks    → padres (chapter) e hijos (section)
                        ├─ chunk_hash → idempotencia
                        └─ embedding BLOB → solo hijos
   
   ChromaDB          → ELIMINADO
   book_chunks_fts   → ELIMINADO
   sqlite-vec        → NUEVO (extensión SQLite para MATCH)


5. FLUJO DE DATOS POR REGISTRO
   ────────────────────────────
   
   books:
     id="book_abc123", user_id="default", title="Mitología", status="indexed"
   
   book_chunks:
     id=1  | chunk_id="book_abc123_ch_0000"           | level=chapter | parent=NULL    | embedding=NULL
     id=2  | chunk_id="book_abc123_ch_0000_s_0000"     | level=section | parent=ch_0000 | embedding=BLOB
     id=3  | chunk_id="book_abc123_ch_0000_s_0001"     | level=section | parent=ch_0000 | embedding=BLOB
     id=4  | chunk_id="book_abc123_ch_0001"           | level=chapter | parent=NULL    | embedding=NULL
     id=5  | chunk_id="book_abc123_ch_0001_s_0000"     | level=section | parent=ch_0001 | embedding=BLOB
     ...
```

---

## Por definir (para próximos ciclos)

- **Reranking**: después de encontrar candidatos ANN, rerankear con cross-encoder
- **Query expansion**: generar variaciones de la pregunta para mejorar recall
- **Tamaño de embedding variable**: si el modelo de embedding cambia de dimensiones, cómo adaptar el `BLOB`
- **Índices separados por libro en sqlite-vec**: evaluar si conviene una virtual table por libro para evitar el workaround del filtro `book_id` post-ANN
