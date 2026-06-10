# Plan: Book Memory (LORE)

## Problema

El agente no tiene acceso a conocimiento de libros. Si el usuario quiere preguntar sobre "ese libro de arquitectura limpia que procesé", no hay mecanismo para inyectar ese conocimiento en el contexto. `SemanticMemory` guarda hechos cortos de conversación, no libros enteros.

No hay chunking, extracción de PDF, ni bloque `[LORE]`. El conocimiento estructurado de libros no existe en el sistema.

## Solución propuesta

Agregar `BookMemory` — un sistema que:
1. Extrae texto de un PDF/TXT
2. Lo divide en chunks semánticos (capítulo -> seccion -> parrafos con overlap)
3. Genera embeddings y los guarda en ChromaDB (coleccion `book_memory`)
4. Persiste metadatos en SQLite (tablas `books` + `book_chunks`)
5. Expone `search()` para recuperar chunks por similitud semantica
6. Inyecta un bloque `[LORE]` en el prompt cuando hay resultados relevantes

### Piramide de memoria actualizada

```
Ultimos mensajes exactos (sliding window)
|
Resumen conversacional (ConversationSummaryMemory)
|
Memorias semanticas del usuario/asistente (SemanticMemory)
|
[LORE] Conocimiento de libros (BookMemory)
|
DB completa como fuente historica
```

### Que NO hace BookMemory

- No reemplaza `SemanticMemory`. Son ortogonales.
- No edita libros. Solo ingesta y consulta.
- No depende de LLM para chunking. Usa reglas estructurales.
- No soporta PDF escaneado en v1 (sin OCR). Se deja hook para futuro.

## Esquema de datos

### SQLite -- `books`

```sql
CREATE TABLE IF NOT EXISTS books (
    id              TEXT PRIMARY KEY,          -- book_{uuid}
    title           TEXT NOT NULL DEFAULT '',
    author          TEXT NOT NULL DEFAULT '',
    source_path     TEXT NOT NULL,
    source_hash     TEXT NOT NULL,             -- SHA256 del archivo original
    total_pages     INTEGER NOT NULL DEFAULT 0,
    total_chunks    INTEGER NOT NULL DEFAULT 0,
    language        TEXT NOT NULL DEFAULT 'es',
    status          TEXT NOT NULL DEFAULT 'indexed',  -- indexed, partial, error
    error_message   TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
```

### SQLite -- `book_chunks`

```sql
CREATE TABLE IF NOT EXISTS book_chunks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    book_id         TEXT NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    chapter         TEXT NOT NULL DEFAULT '',
    page_start      INTEGER NOT NULL DEFAULT 0,
    page_end        INTEGER NOT NULL DEFAULT 0,
    chunk_index     INTEGER NOT NULL,
    chunk_text      TEXT NOT NULL,
    token_count     INTEGER NOT NULL DEFAULT 0,
    char_count      INTEGER NOT NULL DEFAULT 0,
    chunk_hash      TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_book_chunks_book_id ON book_chunks(book_id);
CREATE INDEX IF NOT EXISTS idx_book_chunks_hash ON book_chunks(chunk_hash);
```

### ChromaDB -- coleccion `book_memory`

Cada documento en ChromaDB:

| Campo | Descripcion |
|-------|-------------|
| `id` | `book_{book_id}_chunk_{chunk_index}` |
| `document` | Texto del chunk |
| `embedding` | Vector de embeddings |
| `metadata.book_id` | ID del libro |
| `metadata.book_title` | Titulo del libro |
| `metadata.chapter` | Capitulo o seccion detectada |
| `metadata.chunk_index` | Orden dentro del libro |
| `metadata.total_chunks` | Total de chunks del libro |
| `metadata.page_start` | Pagina inicial |
| `metadata.page_end` | Pagina final |
| `metadata.char_count` | Caracteres del chunk |
| `metadata.language` | Idioma |
| `metadata.created_at` | ISO timestamp |

## Clase `BookMemory`

```python
class BookMemory:
    """
    Sistema de memoria para libros completos.

    Ingesta: PDF/TXT -> texto limpio -> chunks -> embeddings -> ChromaDB + SQLite
    Consulta: pregunta -> embedding -> ChromaDB -> contexto en [LORE]
    """
```

### Constructor

```python
BookMemory(
    persist_dir="./chroma_db",
    sqlite_conn=sqlite3.Connection,
    collection_name="book_memory",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
)
```

### Tamanos recomendados de chunk

| Tipo de libro | Chunk size | Overlap |
|---------------|------------|---------|
| Narrativo / Novela | 800 - 1500 chars | 150 - 250 chars |
| Tecnico / Documentacion | 300 - 800 chars | 80 - 150 chars |
| General | 500 - 1200 chars | 100 - 200 chars |

## API

| Metodo | Descripcion |
|--------|-------------|
| `ingest(path) -> str` | Ingesta PDF/TXT. Retorna `book_id` |
| `ingest_text(text, title, ...) -> str` | Ingesta texto directo (EPUB externo) |
| `search(query, n_results=5, book_id=None) -> list[BookChunk]` | Busqueda semantica |
| `build_context(query, n_results=5) -> str` | Bloque `[LORE]` para inyectar |
| `get_book(book_id) -> dict` | Metadatos del libro |
| `list_books() -> list[dict]` | Todos los libros indexados |
| `get_chunks(book_id, limit=50) -> list[BookChunk]` | Chunks desde SQLite |
| `delete_book(book_id) -> None` | Elimina libro de SQLite + ChromaDB |
| `get_stats() -> dict` | Stats globales |

## Data classes

```python
@dataclass
class BookChunk:
    chunk_id: str = ""
    book_id: str = ""
    book_title: str = ""
    chapter: str = ""
    page_start: int = 0
    page_end: int = 0
    chunk_index: int = 0
    text: str = ""
    char_count: int = 0
    distance: float = 0.0    # solo en search()

@dataclass
class BookInfo:
    book_id: str = ""
    title: str = ""
    author: str = ""
    source_path: str = ""
    total_pages: int = 0
    total_chunks: int = 0
    status: str = ""
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

**Dependencia opcional:** `pymupdf`. Si no esta instalado, `ingest()` lanza `ImportError`.

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
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
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
3. Acumular parrafos hasta `chunk_size`
4. Si un parrafo excede `chunk_size`, partir por oraciones
5. Al iniciar nuevo chunk, incluir overlap del chunk anterior
6. Asignar `page_start`, `page_end` segun los `[PAGE N]` markers

### Validacion de chunks

- Si `chunk_text` esta vacio -> skip
- Si `char_count < 50` -> merge con siguiente chunk
- Si `char_count > chunk_size * 2` -> split forzado

## Integracion en el prompt

### Orden actualizado

```
[core prompt]
[USO DE MEMORIA]
[USER_MEMORY]
[ASSISTANT_MEMORY]
[extra_context]
[CONVERSATION_SUMMARY]
[LORE]              <- NUEVO
[prompts guardados]
[sliding window]
```

### Reglas en `[USO DE MEMORIA]`

Agregar:

- `[LORE]` contiene extractos de libros que has leido.
- `[LORE]` tiene prioridad sobre `CONVERSATION_SUMMARY` si hay contradiccion.
- `[LORE]` NO debe tratarse como preferencia del usuario ni del asistente.
- Si el usuario pregunta por "el libro que procese", busca en `[LORE]`.

### `build_context()` output

```
[LORE]
Del libro "Arquitectura Limpia" (Capitulo 3, pagina 45):
  El principio de inversion de dependencias establece que...

Del libro "Arquitectura Limpia" (Capitulo 4, pagina 62):
  Los casos de uso deben ser independientes de los detalles...
```

Sin resultados:

```
[LORE]
- Sin conocimiento de libros relevante.
```

## Integracion en el flujo

```
turno N:
usuario pregunta sobre un libro
|
guardar mensaje usuario
|
semantic_memory.remember_user()
|
summary_manager.maybe_update()
|
book_memory.build_context(user_query)   <- NUEVO
|
build_system_prompt(book_context=...)   <- NUEVO parametro
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

Agregar parametro opcional `book_context: str = ""`:

```python
def build_system_prompt(self, ..., book_context: str = "") -> str:
    ...
    if book_context:
        parts.append(book_context)
    ...
```

## Tests

| # | Caso |
|---|------|
| 1 | `ingest("test.pdf")` con PDF real -> retorna book_id, ChromaDB tiene N chunks |
| 2 | `ingest` con PDF escaneado (sin texto) -> lanza RuntimeError |
| 3 | `search("principios SOLID")` con libro indexado -> retorna chunks relevantes |
| 4 | `search` con consulta sin relacion -> retorna lista vacia |
| 5 | `build_context("herencia")` -> string con `[LORE]` y chunks |
| 6 | `build_context` sin `book_id` -> busca en todos los libros |
| 7 | `delete_book(book_id)` -> SQLite + ChromaDB limpios |
| 8 | `list_books()` con 3 libros -> retorna 3 dicts |
| 9 | `get_chunks(book_id)` -> retorna chunks ordenados por chunk_index |
| 10 | Chunking: texto de prueba se divide correctamente con overlap |
| 11 | Chunking: capitulos detectados correctamente |
| 12 | Re-ingesta del mismo PDF (mismo hash) -> skip o actualiza (decidir) |

## Orden de implementacion

| Prioridad | Item |
|-----------|------|
| **P0** | Tablas `books` + `book_chunks` en SCHEMA_SQL + migracion |
| **P0** | Clase `BookMemory` con `__init__`, `_ensure_collection()`, `close()` |
| **P0** | `extract_pdf()` + `extract_txt()` helpers |
| **P0** | `chunk_text()` con deteccion de capitulos y overlap |
| **P0** | `ingest(path)` -> extraer + chunquear + embedding + guardar en ambas DBs |
| **P1** | `search(query, n_results, book_id)` en ChromaDB |
| **P1** | `build_context(query)` -> bloque `[LORE]` |
| **P2** | `get_book()`, `list_books()`, `get_chunks()`, `delete_book()`, `get_stats()` |
| **P2** | Data classes `BookChunk`, `BookInfo` |
| **P3** | Integracion en `MemoryManager.build_system_prompt(book_context=...)` |
| **P3** | Integracion en `console_chat.py` (comando `/lore`) |
| **P4** | `ingest_text()` para texto directo |
| **P5** | (futuro) OCR para PDF escaneado, re-ingesta con diff, EPUB directo |

## Dependencias

### Nuevas dependencias opcionales

```toml
[project.optional-dependencies]
lore = ["pymupdf"]
```

```bash
pip install .[lore]
```

Sin `pymupdf`, `ingest()` soporta solo TXT. `search()` y `build_context()` funcionan siempre si hay datos.

### Dependencias existentes reutilizadas

- `chromadb` -- ya esta en el proyecto (usada por `SemanticMemory`)
- `sqlite3` -- stdlib
- `hashlib` -- stdlib

## Archivos a crear/modificar

### Nuevos archivos

| Archivo | Contenido |
|---------|-----------|
| `src/book_memory.py` | Clase `BookMemory`, extractores, chunking |
| `src/book_models.py` | Data classes `BookChunk`, `BookInfo` |

### Archivos a modificar

| Archivo | Cambio |
|---------|--------|
| `src/db.py` | Agregar `books` + `book_chunks` al SCHEMA_SQL y migracion |
| `src/db.py` | Funciones CRUD para `books` y `book_chunks` |
| `src/__init__.py` | Exportar `BookMemory` y `BookChunk` |
| `src/memory.py` | `build_system_prompt()` acepta `book_context` |
| `pyproject.toml` | Agregar optional-dependency `lore` |
| `README.md` | Documentar `BookMemory` API |
