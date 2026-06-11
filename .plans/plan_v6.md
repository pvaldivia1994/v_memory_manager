# v_memory_manager — Plan v6: BookMemory v2 — Dual Hashing, Detección Robusta, TXT Cache, Smart Search

## 1. Diagnóstico del Estado Actual

BookMemory v1 está funcional con extracción configurable, chunking con overlap, SQLite + FTS5 + ChromaDB, y búsqueda semántica/híbrida/keyword. La review identificó problemas reales en la experiencia de indexar, re-indexar y recuperar contenido.

### 1.1. Problemas encontrados

#### P1 — El hash único no distingue entre extracciones con distinto layout

`source_hash` es el SHA256 del archivo original (PDF/EPUB). Si indexas un PDF con `layout="blocks"` y luego con `layout="two_columns"`, el hash es idéntico → el sistema bloquea el re-index sin `force=True`.

```python
# book_memory.py:564-568
source_hash = _hash_file(path)
existing = self.get_book_by_hash(source_hash)
if existing and not force:
    return existing["id"]
```

No tiene trazabilidad de qué layout produjo qué chunks ni dónde está el TXT usado.

#### P2 — `_page_looks_two_columns()` puede detectar falso positivo

Usa coordenadas X de todos los bloques sin filtrar por ancho ni tamaño. Tablas, listas anchas, headers y captions pueden activar `two_columns` incorrectamente.

#### P3 — No hay flujo "extraer → revisar → ingerir automáticamente"

`extract_to_txt()` e `ingest()` existen, pero no hay un método que los encadene.

#### P4 — Búsqueda pobre cuando libro y query están en distinto idioma

Embeddings de `all-MiniLM-L6-v2` no entienden "dioses" ↔ "deities/gods". El sistema no hace traducción ni expansión de queries.

---

## 2. Mejoras Propuestas

### Mejora A — Dual Hashing + Metadata de Extracción

**Problema que resuelve:** P1

**Mentalidad correcta:**
```
source_file_hash  = trazabilidad (qué archivo original)
source_text_hash  = identidad indexable real (qué contenido entró)
```

Para BookMemory, mismo PDF con distinto layout produce dos versiones distintas de contenido. El lookup primario es por `source_text_hash`.

**Schema books table:**
```sql
source_file_hash    TEXT NOT NULL,       -- SHA256 del archivo original
source_text_hash    TEXT NOT NULL,       -- SHA256 del texto limpio extraído
source_type         TEXT NOT NULL DEFAULT '',  -- pdf, epub, txt
source_layout       TEXT NOT NULL DEFAULT '',  -- plain, blocks, two_columns, auto
source_text_path    TEXT NOT NULL DEFAULT '',  -- ruta al TXT usado/source_file_hash, source_text_hash,

UNIQUE(source_file_hash, source_text_hash)
```

DB se recrea desde cero (proyecto en etapa inicial), schema se aplica limpio con `init_db()`.

**Lookup principal:**
```python
def ingest(self, path, ...):
    source_file_hash = _hash_file(path)
    text = clean_extracted_text(extracted)
    source_text_hash = _hash_text(text)
    existing = self.get_book_by_hashes(source_file_hash, source_text_hash)
    if existing and not force:
        return existing["id"]
    if existing and force:
        self._delete_book(existing["id"])
    db.insert_book(..., source_file_hash, source_text_hash,
                   source_type=ext, source_layout=pdf_layout,
                   source_text_path=save_extracted_to or "")
```

---

### Mejora B — Filtro de ancho + longitud mínima en `_page_looks_two_columns()`

**Problema que resuelve:** P2

```python
def _page_looks_two_columns(page) -> bool:
    blocks = page.get_text("blocks")
    page_width = page.rect.width
    max_col_width = page_width * 0.65
    text_blocks = [
        b for b in blocks
        if b[4].strip()
        and len(b[4].strip()) >= 20
        and (b[2] - b[0]) < max_col_width
    ]
    xs = [b[0] for b in text_blocks]
    if len(xs) < 6:
        return False
    middle = page_width / 2
    left_count = sum(1 for x in xs if x < middle)
    right_count = sum(1 for x in xs if x >= middle)
    return left_count >= 3 and right_count >= 3
```

Tres filtros: (1) `len >= 20` elimina números de página, headers cortos, etc. (2) ancho < 65% elimina bloques de ancho completo. (3) conteo >= 3 a cada lado.

---

### Mejora C — `extract_and_ingest()` con flag `ingest=False`

**Problema que resuelve:** P3

```python
def extract_and_ingest(
    self,
    path: str,
    cache_dir: str = "./extracted",
    title: str = "",
    author: str = "",
    force_extract: bool = False,
    force_ingest: bool = False,
    pdf_layout: str = "auto",
    ingest: bool = True,
) -> str:
```

**Flujo:**
1. Si el TXT cache no existe o `force_extract=True`: extraer a `cache_dir/{filename}.clean.txt`.
2. Si `ingest=True`: llamar a `ingest()` apuntando al TXT cache.
3. Retornar el `book_id` (o `""` si `ingest=False`).

**Dos casos de uso:**
```python
# Solo extraer para revisar
txt_path = bm.extract_and_ingest("libro.pdf", pdf_layout="two_columns", ingest=False)

# Extraer e ingerir directo
book_id = bm.extract_and_ingest("libro.pdf", pdf_layout="two_columns", ingest=True)
```

---

### Mejora D — Smart Search Multilingüe

**Problema que resuelve:** P4

La causa raíz de búsqueda pobre no es el chunking ni el layout, sino que el embedding `all-MiniLM-L6-v2` no cruza idiomas bien:

```
query:  "dioses"             → embedding español
chunk:  "deities of Grimnir" → embedding inglés
```

**Qué hacer — 3 capas:**

#### D1 — `translate_query(query, target_language)`

Traducir la query al idioma del libro antes de embedding:

```python
def translate_query(self, query: str, book_language: str = "en",
                    llm=None) -> str:
    if not llm or query_language == book_language:
        return query
    prompt = f"Traduce esta pregunta a {book_language}. Devuelve solo la traduccion:\n{query}"
    res = llm.chat(system="Eres un traductor.", user=prompt)
    return res.content.strip() if res and res.content else query
```

#### D2 — `expand_query(query, llm=None, rules=None)`

Expandir la query para capturar sinónimos relevantes:

```python
def expand_query(self, query: str, llm=None) -> list[str]:
    if not llm:
        return [query]
    prompt = f"Genera 3 formas alternativas de buscar esta informacion:\n{query}"
    res = llm.chat(system="Eres un experto en recuperacion de informacion.", user=prompt)
    variants = [query] + [v.strip() for v in res.content.strip().split("\n") if v.strip()] if res and res.content else [query]
    return variants[:5]
```

#### D3 — `search_smart()` — multi-query + dedup + rerank

```python
def search_smart(
    self,
    query: str,
    book_id: Optional[str] = None,
    n_results: int = 5,
    query_language: str = "es",
    book_language: str = "en",
    llm=None,
    expand: bool = True,
    rerank: bool = False,
) -> list[BookChunk]:
```

**Algoritmo:**
1. Generar queries: `[query_original, query_traducida] + expanded_queries`
2. Para cada query, ejecutar `search_hybrid()`
3. Acumular resultados en `dict[chunk_id, (BookChunk, score_sum, match_count)]`
4. Score final = `sem_weight * f(raw_score) + keyword_bonus + match_count_bonus`
5. Deduplicar por `chunk_id`
6. Opcional: rerank final con CrossEncoder
7. Retornar top N

**Estructura de resultado:**
```python
class SmartSearchResult:
    chunk: BookChunk
    score: float
    matched_queries: list[str]
    query_language: str
```

Esto no requiere embeddings multilingües caros. Usa el LLM que ya tenés para traducir la query.

---

## 3. Bugs en el Pipeline de Traducción (`translate_book_incremental.py`)

#### Bug 1 — `.stat().size` en lugar de `.stat().st_size`

Línea 661:
```python
out_size = out_path.stat().size if out_path.exists() else 0
#             ^^^^^^^^^^^^^^^^ no existe, deberia ser .st_size
```

#### Bug 2 — Two-pass no traduce el texto optimizado

Paso 2 (líneas 617-622) reusa `units` del texto original. Debe releer `optimized.txt` y regenerar unidades:

```python
optimized_text = opt_path.read_text(encoding="utf-8")
optimized_pages = split_by_pages(optimized_text)
if by_page:
    pass2_units = [{"marker": m, "content": c} for m, c in optimized_pages]
else:
    raw_blocks = build_page_blocks(optimized_pages, max_chars=chunk_size)
    pass2_units = [{"marker": b["start_page"], "content": b["text"]} for b in raw_blocks]
_process_units(llm, pass2_units, ...)
```

#### Bug 3 — `--translate` CLI arg siempre True

Línea 510-511:
```python
parser.add_argument("--translate", action="store_true", default=True)
```

Con `action="store_true"` + `default=True`, siempre es True. Reemplazar:

```python
parser.add_argument("--no-translate", action="store_true",
                    help="No traducir; solo optimizar")
...
translate = not args.no_translate
```

#### Bug 4 — `sys.modules["src"]` frágil

La manipulación (líneas 61-88) para cargar VLLaMA desde `v_llama` puede romperse. Soluciones:
- A corto plazo: usar `importlib` en lugar de borrar `sys.modules`.
- A largo plazo: que `v_llama` use un nombre de paquete propio (`vllama`, `vtool_llama`) en vez de `src`.

---

## 4. Mejoras al Pipeline de Traducción

### Mejora E — `clean_llm_output()` con eliminación de prefijos y fences

```python
_LLM_PREFIXES = [
    "Aquí tienes la traducción:", "Aqui tienes la traduccion:",
    "Texto procesado:", "Traducción:", "Traduccion:",
    "Aquí está:", "Aquí esta:",
]

def clean_llm_output(output: str) -> str:
    output = output.strip()
    for p in _LLM_PREFIXES:
        if output.lower().startswith(p.lower()):
            output = output[len(p):].strip()
    if output.startswith("```"):
        output = re.sub(r"^```(?:text|markdown)?\s*", "", output, flags=re.I)
        output = re.sub(r"\s*```$", "", output)
    return output.strip()
```

### Mejora F — `validate_processed_text()` con `strict_pages` opcional

```python
def validate_processed_text(original: str, output: str,
                            strict_pages: bool = True) -> bool:
    if not output.strip():
        return False
    if len(output) < len(original) * 0.35:
        return False
    if len(output) > len(original) * 2.5:
        return False  # LLM está divagando
    if strict_pages:
        original_pages = set(_PAGE_RE.findall(original))
        output_pages = set(_PAGE_RE.findall(output))
        if original_pages and not original_pages.issubset(output_pages):
            return False
    return True
```

### Mejora G — `_ingest()` debe compartir conexión

Mover la ingesta al flujo principal, reusando el `BookMemory` de la extracción (o al menos el `sqlite_conn`). Así se evitan duplicados cuando el usuario ya tenía el libro indexado.

---

## 5. Marcadores — Mantener `[PAGE N]` consistente

BookMemory ya entiende `[PAGE N]`. El pipeline de traducción debe:
- Usar SIEMPRE `[PAGE N]`, no `#PAGE N#` ni otros formatos.
- En `build_instructions()` y prompts, reforzar: "No modifiques marcadores [PAGE N]".
- `clean_llm_output()` no debe limpiar `[PAGE N]`.

---

## 6. Mejoras Descartadas

| Idea | Por qué no |
|---|---|
| **OCR como layout option** | `extract_ocr_pdf()` ya existe como fallback automático. |
| **Parser semántico con ML** | Sobreingeniería. PyMuPDF blocks + column detection alcanza. |
| **Chunking recursivo con LLM** | Latencia + costo innecesario para RAG. |
| **Separar BookExtractor/BookTranslator ahora** | Correcto conceptualmente, pero prematuro. Se separa cuando haya un segundo consumidor. |
| **Parent-child chunks** | Útil, pero depende de tener buena recuperación primero. Se hace después de Smart Search. |

---

## 7. Plan de Implementación (Orden de prioridad)

### Fase 1 — Bugs del Pipeline de Traducción (impacto alto, esfuerzo bajo)
- [ ] **1a:** Arreglar `.stat().size` → `.stat().st_size`
- [ ] **1b:** Arreglar two-pass: paso 2 debe leer el archivo optimizado
- [ ] **1c:** Reemplazar `--translate` por `--no-translate`
- [ ] **1d:** Agregar `clean_llm_output()` con fences
- [ ] **1e:** Agregar `validate_processed_text()` con `strict_pages`
- [ ] **1f:** Reforzar `[PAGE N]` en prompts de traducción

### Fase 2 — Dual Hashing + Metadata (impacto alto, esfuerzo bajo)
- [ ] **2a:** Schema: `source_file_hash` + `source_text_hash` + `source_type`/`layout`/`text_path`
- [ ] **2b:** Renombrar `source_hash` → `source_file_hash` en todo el código
- [ ] **2c:** Computar `source_text_hash` en `ingest()` e `ingest_text()`
- [ ] **2d:** `get_book_by_hashes(file_hash, text_hash)` como lookup primario
- [ ] **2e:** Actualizar `reindex_book()`, `validate_index()`, `get_book()`

### Fase 3 — Detección Robusta de Columnas (impacto medio, esfuerzo bajo)
- [ ] **3a:** Agregar filtro `len >= 20` y ancho < 65% en `_page_looks_two_columns()`
- [ ] **3b:** Verificar con PDFs reales de tabla/lista que no active falsos positivos

### Fase 4 — TXT Cache / `extract_and_ingest()` (impacto medio, esfuerzo bajo)
- [ ] **4a:** Implementar `extract_and_ingest()` con flag `ingest=False`
- [ ] **4b:** Actualizar `translate_book_incremental.py` para usarlo
- [ ] **4c:** Tests del flujo completo

### Fase 5 — Smart Search Multilingüe (impacto alto, esfuerzo medio)
- [ ] **5a:** Agregar campo `language` a `books` y `book_chunks`
- [ ] **5b:** Implementar `translate_query(query, target_language, llm)`
- [ ] **5c:** Implementar `expand_query(query, llm)`
- [ ] **5d:** Implementar `search_smart()` con multi-query + dedup
- [ ] **5e:** Integrar `search_smart()` en `build_context()` como opción

### Fase 6 — Pruebas de Integración (impacto alto, esfuerzo medio)
- [ ] **6a:** Dual hash: mismo PDF con distinto layout produce dos libros
- [ ] **6b:** Columnas: PDF de tabla no se detecta como two_columns
- [ ] **6c:** TXT cache: `extract_and_ingest(ingest=False)` solo extrae
- [ ] **6d:** Smart search: query en español contra chunk en inglés
- [ ] **6e:** Pipeline: extraer → optimizar → traducir → ingestar
- [ ] **6f:** Búsquedas: `search()`, `search_hybrid()`, `search_smart()`, `build_context()`

---

## 8. Verificación

| Mejora | Test |
|---|---|
| A | Indexar mismo PDF con `layout="blocks"` y `layout="two_columns"` → dos libros distintos |
| A | `get_book_by_hashes(file_hash, text_hash)` retorna libro exacto |
| A | Metadata `source_layout` se guarda correctamente |
| B | Página con tabla ancha → no se detecta como two_columns |
| B | Página real de dos columnas → se detecta correctamente |
| C | `extract_and_ingest(ingest=False)` → no crea libro, solo TXT |
| C | Llamada repetida sin `force` → retorna mismo book_id |
| D | `search_smart("dioses", book_language="en")` → encuentra "deities" / "gods" |
| D | Query traducida produce score más alto que sin traducir |
| Pipeline | `translate_book_incremental.py` → two-pass produce TXT traducido del optimizado |
