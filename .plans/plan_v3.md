# v_memory_manager — Plan v3: Semantic memory with ChromaDB

## 1. Propósito

Agregar un módulo de memoria semántica que detecte automáticamente información
recordable de las conversaciones (sin LLM) y la almacene en ChromaDB para
búsqueda semántica y deduplicación.

## 2. Stack técnico

| Componente | Tecnología |
|---|---|
| Vector DB | ChromaDB 1.5.9 |
| Embeddings | `chromadb` default (all-MiniLM-L6-v2) |
| Clasificación | Reglas + scoring (sin LLM) |
| Persistencia | ChromaDB persistente en disco |

## 3. Clase `SemanticMemory`

Módulo independiente en `src/semantic_memory.py`. No depende de `MemoryManager`.

```python
class SemanticMemory:
    def __init__(self, persist_dir: str = "./chroma_db")
```

### API pública

| Método | Descripción |
|---|---|
| `analyze(text) -> AnalysisResult` | Analiza texto sin guardar. Decide si es recordable |
| `remember(text, source="auto") -> str \| None` | Analiza y guarda si corresponde. Devuelve ID o None |
| `forget(memory_id)` | Elimina una memoria por ID |
| `search(query, n_results=5)` | Busca memorias similares por texto |
| `search_by_tags(tags)` | Lista memorias por tags |
| `list_memories(limit=50)` | Lista todas las memorias |
| `count()` | Total de memorias |
| `get_memory(memory_id)` | Obtiene una memoria por ID |

### `AnalysisResult`

```python
@dataclass
class AnalysisResult:
    should_remember: bool
    reason: str               # noise, explicit, rule_score, possible, low_score
    confidence: float         # 0.0 - 1.0
    content: str              # Texto normalizado para guardar
    tags: list[str]
```

### `MemoryRecord`

```python
@dataclass
class MemoryRecord:
    id: str
    content: str
    tags: list[str]
    confidence: float
    memory_type: str          # explicit, preference, project_fact, environment, long_term_instruction, pending
    status: str               # active, pending_review, archived
    created_at: str
    source: str               # auto, explicit
```

## 4. Detección de memoria (sin LLM)

### 4.1. Filtro de ruido
- Mensajes < 8 caracteres
- Saludos, confirmaciones, risas ("hola", "ok", "gracias", "jajaja")

### 4.2. Comandos explícitos
- `/remember <texto>` — guarda exactamente `<texto>`
- `/forget <id>` — elimina por ID

### 4.3. Patrones por frases
- `prefiero`, `me gusta`, `no me gusta` → `preference`
- `estoy creando`, `mi proyecto`, `se llama` → `project_fact`
- `mi pc tiene`, `uso windows`, `trabajo con` → `environment`
- `de ahora en adelante`, `siempre que` → `long_term_instruction`

### 4.4. Sistema de puntuación
- 0.0 - 1.0 basado en: tipo detectado, longitud, marcadores personales, términos técnicos
- >= 0.75 → active
- 0.40 - 0.74 → pending_review
- < 0.40 → ignorar

### 4.5. Extracción de tags técnicos
- `python`, `windows`, `llama.cpp`, `chromadb`, `sqlite`, `unity`, `ollama`, `gguf`, etc.

## 5. Integración con console_chat.py

### Nuevos comandos
| Comando | Descripción |
|---|---|
| `/remember <texto>` | Guarda memoria explícita |
| `/forget <id>` | Elimina memoria |
| `/memories` | Lista memorias guardadas |
| `/search <query>` | Busca en memorias por similitud |

### Flujo automático
Después de cada mensaje, si fue exitoso, se pasa el texto del usuario por
`SemanticMemory.analyze()`. Si `should_remember`, se guarda automáticamente.

## 6. Edge cases

- **ChromaDB no instalado**: `ImportError` con mensaje claro
- **Duplicados**: buscar similitud antes de insertar, ignorar si > 0.95相似
- **Persistencia**: ChromaDB guarda en disco automáticamente con `PersistentClient`
- **Tags vacíos**: se guarda sin tags, no block
- **Confianza baja**: se guarda como `pending_review`, no se inyecta en system prompt
