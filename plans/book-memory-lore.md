# Plan: Book Memory (BookMemory)

## Problema

El agente no tiene acceso a conocimiento de libros. Si el usuario quiere preguntar sobre "ese libro de arquitectura limpia que procese", no hay mecanismo para inyectar ese conocimiento en el contexto. `SemanticMemory` guarda hechos cortos de conversacion, no libros enteros.

No hay chunking, extraccion de PDF, ni bloque `[BOOK_CONTEXT]`. El conocimiento estructurado de libros no existe en el sistema.

## Solucion propuesta

Agregar `BookMemory` -- un sistema que:
1. Extrae texto de un PDF/TXT
2. Lo divide en chunks semanticos (capitulo -> seccion -> parrafos con overlap)
3. Genera embeddings y los guarda en ChromaDB (coleccion `book_memory`)
4. Persiste metadatos en SQLite (tablas `books` + `book_chunks`)
5. Expone `search()` para recuperar chunks via ChromaDB (IDs + distances) + SQLite (texto real)
6. Inyecta un bloque `[BOOK_CONTEXT]` en el prompt solo cuando hay resultados relevantes

### Piramide de memoria actualizada

```
Ultimos mensajes exactos (sliding window)
|
Resumen conversacional (ConversationSummaryMemory)
|
Memorias semanticas del usuario/asistente (SemanticMemory)
|
[BOOK_CONTEXT] Conocimiento de libros (BookMemory)
|
DB completa como fuente historica
```

### Que NO hace BookMemory

- No reemplaza `SemanticMemory`. Son ortogonales y separados.
- No edita libros. Solo ingesta y consulta.
- No depende de LLM para chunking. Usa reglas estructurales.
- No soporta PDF escaneado en v1 (sin OCR). Se deja hook para futuro.

## Arquitectura clave

```
SQLite = verdad / control / metadata / chunks reales
ChromaDB = indice semantico (solo para obtener IDs)
```

**Flujo de search():**
```
1. Query a ChromaDB -> IDs + distances
2. Filtrar por max_distance
3. Leer chunks reales desde SQLite
4. Devolver BookChunk completo
```

Esto evita inconsistencias entre ChromaDB y SQLite, y permite reconstruir el indice desde SQLite si es necesario.

## Esquema de datos

### SQLite -- `books`

```sql
CREATE TABLE IF NOT EXISTS books (
    id              TEXT PRIMARY KEY,          -- book_{uuid_abreviado}
    title           TEXT NOT NULL DEFAULT '',
    author          TEXT NOT NULL DEFAULT '',
    source_path     TEXT NOT NULL,
    source_hash     TEXT NOT NULL UNIQUE,      -- SHA256 del archivo original
    total_pages     INTEGER NOT NULL DEFAULT 0,
    total_chunks    INTEGER NOT NULL DEFAULT 0,
    language        TEXT NOT NULL DEFAULT 'es',
    status          TEXT NOT NULL DEFAULT 'pending',  -- pending, extracting, chunking, embedding, indexed, partial, error, deleted
    error_message   TEXT NOT NULL DEFAULT '',

    embedding_model TEXT NOT NULL DEFAULT '',  -- ej: all-MiniLM-L6-v2
    embedding_dim   INTEGER NOT NULL DEFAULT 0,
    chunker_version TEXT NOT NULL DEFAULT 'v1',
    schema_version  TEXT NOT NULL DEFAULT 'v1',

    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
```

### SQLite -- `book_chunks`

Con `chunk_id` textual estable para referenciar desde ChromaDB:

```sql
CREATE TABLE IF NOT EXISTS book_chunks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id        TEXT NOT NULL UNIQUE,      -- {book_id}_chunk_{chunk_index:06d}
    book_id         TEXT NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    chapter         TEXT NOT NULL DEFAULT '',
    page_start      INTEGER NOT NULL DEFAULT 0,
    page_end        INTEGER NOT NULL DEFAULT 0,
    chunk_index     INTEGER NOT NULL,
    chunk_text      TEXT NOT NULL,
    token_count     INTEGER NOT NULL DEFAULT 0,
    char_count      INTEGER NOT NULL DEFAULT 0,
    chunk_hash      TEXT NOT NULL,             -- SHA256 del texto (dedup interno)
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_book_chunks_book_id ON book_chunks(book_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_book_chunks_book_index ON book_chunks(book_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_book_chunks_hash ON book_chunks(chunk_hash);
```

Nota: `chunk_id = f"{book_id}_chunk_{chunk_index:06d}"` -- ej: `book_abc123_chunk_000042`.

### SQLite FTS5 (futuro, P4)

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS book_chunks_fts
USING fts5(
    chunk_text,
    book_id UNINDEXED,
    chunk_id UNINDEXED,
    chapter UNINDEXED
);
```

### ChromaDB -- coleccion `book_memory`

Cada documento en ChromaDB solo contiene metadatos minimos. El texto real se lee desde SQLite:

| Campo | Descripcion |
|-------|-------------|
| `id` | `{book_id}_chunk_{chunk_index:06d}` (mismo que chunk_id en SQLite) |
| `document` | Texto del chunk (para busqueda, pero NO es fuente de verdad) |
| `embedding` | Vector de embeddings |
| `metadata.book_id` | ID del libro |
| `metadata.chunk_index` | Orden dentro del libro |
| `metadata.chunk_id` | Mismo que id |
| `metadata.language` | Idioma |

## Clase `BookMemory`

```python
class BookMemory:
    """
    Sistema de memoria documental para libros completos.

    Ingesta: PDF/TXT -> texto limpio -> chunks -> embeddings -> ChromaDB + SQLite
    Consulta: pregunta -> ChromaDB (IDs) -> SQLite (texto real) -> [BOOK_CONTEXT]
    """
```

### Constructor

```python
BookMemory(
    persist_dir="./chroma_db",
    sqlite_conn=sqlite3.Connection,
    collection_name="book_memory",
    chunk_size_chars: int = 1000,
    chunk_overlap_chars: int = 200,
    search_max_distance: float = 0.75,
)
```

`chunk_size_chars` y `chunk_overlap_chars` en caracteres (no tokens) para v1.

### Tamanos recomendados de chunk

| Tipo de libro | Chunk size | Overlap |
|---------------|------------|---------|
| Narrativo / Novela | 800 - 1500 chars | 150 - 250 chars |
| Tecnico / Documentacion | 300 - 800 chars | 80 - 150 chars |
| General | 500 - 1200 chars | 100 - 200 chars |

## API

| Metodo | Descripcion |
|--------|-------------|
| `has_books() -> bool` | True si hay al menos un libro indexado |
| `ingest(path, title="", author="", force=False) -> str` | Ingesta PDF/TXT. Retorna `book_id` |
| `ingest_text(text, title="", author="", force=False) -> str` | Ingesta texto directo (tests, EPUB externo) |
| `search(query, n_results=5, book_id=None, max_distance=None) -> list[BookChunk]` | Busqueda semantica via ChromaDB + SQLite |
| `search_keyword(query, n_results=5) -> list[BookChunk]` | Busqueda exacta via SQLite FTS5 (P4+) |
| `search_hybrid(query, n_results=5) -> list[BookChunk]` | Combinacion semantico + keyword (P4+) |
| `should_search(query) -> bool` | Detecta si la query es sobre libros/documentos |
| `build_context(query, n_results=5, book_id=None, max_chars=5000) -> str` | Bloque `[BOOK_CONTEXT]` solo si hay resultados |
| `get_book(book_id) -> dict` | Metadatos del libro |
| `get_book_by_hash(source_hash) -> dict or None` | Buscar libro por hash (dedup) |
| `list_books() -> list[dict]` | Todos los libros indexados |
| `get_chunks(book_id, limit=50) -> list[BookChunk]` | Chunks desde SQLite |
| `delete_book(book_id) -> None` | Elimina libro de SQLite + ChromaDB |
| `reindex_book(book_id) -> None` | Regenera embeddings manteniendo chunks |
| `validate_index(book_id) -> dict` | Verifica consistencia SQLite vs ChromaDB |
| `get_stats() -> dict` | Stats globales (total libros, chunks) |

## Data classes (P0)

```python
@dataclass
class BookChunk:
    chunk_id: str = ""        # {book_id}_chunk_{chunk_index:06d}
    book_id: str = ""
    book_title: str = ""
    chapter: str = ""
    page_start: int = 0
    page_end: int = 0
    chunk_index: int = 0
    text: str = ""
    char_count: int = 0
    distance: float = 0.0     # solo en search()

@dataclass
class BookInfo:
    book_id: str = ""
    title: str = ""
    author: str = ""
    source_path: str = ""
    source_hash: str = ""
    total_pages: int = 0
    total_chunks: int = 0
    status: str = ""
    embedding_model: str = ""
    created_at: str = ""
```

## Extraccion de texto

### PDF con texto seleccionable (pymupdf)

```python
def extract_pdf(path: str) -> tuple[str, int]:
    import pymupdf
    doc = pymupdf.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append(f"[PAGE {i+1}]\n{text}")
    return "\n\n".join(pages), len(doc)
```

**Dependencia opcional:** `pymupdf`. Sin ella, `ingest()` solo soporta TXT.

### TXT

Carga directa. Se asume pagina 1.

### Hook para OCR (futuro)

```python
if len(text.strip()) < 100:
    raise RuntimeError(
        "PDF parece escaneado. OCR no implementado en v1."
    )
```

## Chunking

### Estrategia: por limite con boundaries semanticas

```python
def chunk_text(
    text: str,
    chunk_size_chars: int = 1000,
    chunk_overlap_chars: int = 200,
) -> list[dict]:
    """
    Divide en chunks respetando parrafos y paginas.
    Cada dict: {chapter, page_start, page_end, text}
    """
```

Algoritmo:

1. Dividir texto por `[PAGE N]` -> bloques por pagina
2. Dentro de cada pagina, detectar capitulos/secciones por:
   - Lineas en mayusculas sostenidas
   - Patrones `Capitulo \d+`, `Chapter \d+`, `# `, `## `
   - Lineas cortas que parecen titulos
3. Acumular parrafos hasta `chunk_size_chars`
4. Si un parrafo excede `chunk_size_chars`, partir por oraciones
5. Al iniciar nuevo chunk, incluir `chunk_overlap_chars` del chunk anterior
6. Asignar `page_start`, `page_end` segun los `[PAGE N]` markers

### Validacion de chunks

- Si `chunk_text` vacio -> skip
- Si `char_count < 50` -> merge con siguiente chunk
- Si `char_count > chunk_size_chars * 2` -> split forzado

## Control de reingesta

```python
def ingest(self, path: str, title: str = "", author: str = "", force: bool = False) -> str:
    source_hash = self._hash_file(path)
    existing = self.get_book_by_hash(source_hash)
    if existing and not force:
        return existing["id"]  # skip silencioso
    if existing and force:
        self.delete_book(existing["id"])  # reemplazar completo
    # continuar con ingesta normal
```

## Cuando buscar en libros (trigger logic)

BookMemory NO debe buscar en cada turno. Solo cuando tenga sentido:

```python
_BOOK_TRIGGERS = [
    "libro", "pdf", "documento", "capitulo", "capitulo",
    "pagina", "pagina", "autor",
    "segun", "segun", "segun el", "segun la",
    "que dice", "que dice", "menciona",
    "texto", "leido", "lei", "leer",
    "indice", "indice",
]

def should_search(self, query: str) -> bool:
    if not self.has_books():
        return False
    q = query.lower().strip()
    # Comando explicito
    if q.startswith("/book") or q.startswith("/lore"):
        return True
    # Triggers de busqueda documental
    return any(t in q for t in _BOOK_TRIGGERS)
```

Esto evita busquedas innecesarias y reduce ruido en el prompt.

## Integracion en el prompt

### Orden final

```
[core prompt]
[USO DE MEMORIA]
[USER_MEMORY]
[ASSISTANT_MEMORY]
[CONVERSATION_SUMMARY]
[prompts guardados]
[BOOK_CONTEXT]          <- solo si hay resultados relevantes
[sliding window]
```

`[BOOK_CONTEXT]` se inyecta SOLO cuando `build_context()` encuentra resultados que pasan `max_distance`. Si no hay resultados, no se inyecta nada (excepto si el usuario pregunto explicitamente por un libro).

### Reglas en `[USO DE MEMORIA]`

Agregar:

- `[BOOK_CONTEXT]` contiene extractos recuperados de libros/documentos.
- `[BOOK_CONTEXT]` NO debe tratarse como preferencia del usuario ni del asistente.
- No inventes citas, paginas ni capitulos que no esten en `[BOOK_CONTEXT]`.
- Si los fragmentos recuperados NO contienen la respuesta, dimelo. No inventes.
- Si hay contradiccion entre el libro y la memoria conversacional, aclara la diferencia.

### `build_context()` output -- sin scores en prompt

```python
def build_context(self, query: str, n_results: int = 5,
                  book_id: str | None = None,
                  max_chars: int = 5000) -> str:
    chunks = self.search(query, n_results=n_results, book_id=book_id)
    if not chunks:
        return ""  # no inyectar nada

    lines = ["[BOOK_CONTEXT]"]
    total_chars = 0
    for c in chunks:
        header = f"Fuente: {c.book_title}"
        if c.chapter:
            header += f" ({c.chapter}"
            if c.page_start:
                header += f", pagina {c.page_start}"
            header += ")"
        entry = f"{header}:\n  {c.text.strip()}"
        if total_chars + len(entry) > max_chars:
            break
        lines.append("")
        lines.append(entry)
        total_chars += len(entry)

    return "\n".join(lines)
```

Output:

```
[BOOK_CONTEXT]

Fuente: Arquitectura Limpia (Capitulo 3, pagina 45):
  El principio de inversion de dependencias establece que las entidades...

Fuente: Arquitectura Limpia (Capitulo 4):
  Los casos de uso deben ser independientes de los detalles de infraestructura...
```

Sin resultados: retorna `""` (no se inyecta nada en el prompt).

## Integracion en el flujo

```
turno N:
usuario pregunta
|
guardar mensaje usuario
|
semantic_memory.remember_user()
|
summary_manager.maybe_update()
|
book_context = ""
if book_memory.has_books() and book_memory.should_search(user_query):
    book_context = book_memory.build_context(user_query, max_chars=5000)
|
build_system_prompt(book_context=book_context)
|
get_history(max_messages=10)
|
generar respuesta
|
guardar respuesta
|
semantic_memory.remember_assistant()
```

### Cambios en `MemoryManager.build_system_prompt()`

```python
def build_system_prompt(self, ..., book_context: str = "") -> str:
    ...
    if book_context:
        parts.append(book_context)
    ...
```

### Comando explicito en consola

```python
elif cmd in ("/book", "/lore"):
    if not book_memory or not book_memory.has_books():
        print("[No hay libros indexados]")
        continue
    if len(parts) < 2:
        print("[Uso: /book <consulta>]")
        continue
    ctx = book_memory.build_context(parts[1])
    print(ctx if ctx else "[Sin resultados en libros]")
    continue
```

## Tests

| # | Caso |
|---|------|
| 1 | `ingest_text()` con texto directo -> retorna book_id, chunks en SQLite |
| 2 | `ingest("test.pdf")` con PDF real -> retorna book_id, ChromaDB tiene N chunks |
| 3 | `ingest` con PDF escaneado (sin texto) -> lanza RuntimeError |
| 4 | `search("principios SOLID")` con libro indexado -> retorna BookChunk con texto real |
| 5 | `search` con consulta sin relacion -> retorna lista vacia |
| 6 | `search` con `max_distance=0.10` -> filtra chunks de baja relevancia |
| 7 | `build_context("herencia")` -> string con `[BOOK_CONTEXT]` y chunks |
| 8 | `build_context` con `max_chars=500` -> trunca correctamente |
| 9 | `build_context` con consulta sin match -> retorna `""` |
| 10 | `delete_book(book_id)` -> SQLite + ChromaDB limpios |
| 11 | `get_book_by_hash(hash)` con hash existente -> retorna book_id |
| 12 | `ingest` mismo archivo dos veces -> skip, retorna mismo book_id |
| 13 | `ingest` mismo archivo con `force=True` -> reemplaza |
| 14 | `list_books()` con 3 libros -> retorna 3 dicts |
| 15 | `get_chunks(book_id)` -> retorna chunks ordenados por chunk_index |
| 16 | `validate_index(book_id)` -> reporta match SQLite vs ChromaDB |
| 17 | `has_books()` sin libros -> False, con libros -> True |
| 18 | `should_search()` con trigger "que dice el libro" -> True |
| 19 | `should_search()` con "hola como estas" -> False |
| 20 | Chunking: texto de prueba se divide correctamente con overlap |
| 21 | Chunking: capitulos detectados correctamente |
| 22 | `reindex_book(book_id)` -> embeddings regenerados |

## Orden de implementacion

| Prioridad | Item |
|-----------|------|
| **P0** | Data classes `BookChunk`, `BookInfo` |
| **P0** | Tablas `books` + `book_chunks` en SCHEMA_SQL + migracion |
| **P0** | Clase `BookMemory` con `__init__`, `_ensure_collection()`, `close()` |
| **P0** | `extract_txt()` + `chunk_text()` |
| **P0** | `ingest_text()` -- testear pipeline sin PDF |
| **P0** | Guardar en SQLite (books + book_chunks) |
| **P1** | Conexion ChromaDB, coleccion `book_memory`, embeddings |
| **P1** | `extract_pdf()` via pymupdf |
| **P1** | `ingest(path)` para TXT/PDF |
| **P1** | `search(query, n_results, book_id, max_distance)` |
| **P1** | `has_books()`, `should_search()` |
| **P2** | `build_context(query, max_chars)` -> bloque `[BOOK_CONTEXT]` |
| **P2** | Integracion en `MemoryManager.build_system_prompt(book_context=...)` |
| **P2** | Control de reingesta (`force=True`, `get_book_by_hash()`) |
| **P2** | No inyectar BOOK_CONTEXT vacio |
| **P3** | `delete_book()`, `list_books()`, `get_chunks()`, `get_stats()` |
| **P3** | `reindex_book()`, `validate_index()` |
| **P3** | Comando `/book` o `/lore` en `console_chat.py` |
| **P4** | SQLite FTS5 + `search_keyword()` |
| **P4** | `search_hybrid()` (semantico + keyword combinado) |
| **P5** | OCR para PDF escaneado, EPUB directo, reranker |

## Dependencias

### Nuevas dependencias opcionales

```toml
[project.optional-dependencies]
lore = ["pymupdf"]
```

```bash
pip install .[lore]
```

Sin `pymupdf`, `ingest()` soporta solo TXT. `search()`, `build_context()` y `should_search()` funcionan siempre si hay datos.

### Dependencias existentes reutilizadas

- `chromadb` -- ya en el proyecto (usada por `SemanticMemory`)
- `sqlite3` -- stdlib
- `hashlib` -- stdlib

## Archivos a crear/modificar

### Nuevos archivos

| Archivo | Contenido |
|---------|-----------|
| `src/book_memory.py` | Clase `BookMemory`, extractores, chunking, triggers |
| `src/book_models.py` | Data classes `BookChunk`, `BookInfo` |

### Archivos a modificar

| Archivo | Cambio |
|---------|--------|
| `src/db.py` | Agregar `books` + `book_chunks` al SCHEMA_SQL y migracion |
| `src/db.py` | Funciones CRUD para `books` y `book_chunks` |
| `src/__init__.py` | Exportar `BookMemory`, `BookChunk`, `BookInfo` |
| `src/memory.py` | `build_system_prompt()` acepta `book_context` |
| `pyproject.toml` | Agregar optional-dependency `lore` |
| `README.md` | Documentar `BookMemory` API |
