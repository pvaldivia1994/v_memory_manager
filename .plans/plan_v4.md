# v_memory_manager — Plan v4: SQLite-backed semantic memories + Roleplay

## 1. Propósito

Unificar el almacenamiento de TODAS las memorias semánticas (normales y roleplay) en
una sola tabla SQLite como source of truth. ChromaDB queda como índice de búsqueda
semántica. El borrado/archivado se hace por ID de SQLite, que contiene el `chroma_id`
para eliminar también de ChromaDB.

## 2. Stack técnico

| Componente | Tecnología |
|---|---|
| Source of truth | SQLite (tabla `semantic_memories` en la DB de MemoryManager) |
| Search index | ChromaDB (colecciones separadas por namespace) |
| Detección normal | Reglas + scoring (existente en `semantic_memory.py`) |
| Detección roleplay | Patrones regex (nuevo en `roleplay_memory.py`) |

## 3. Arquitectura

```
memory/
├── semantic_memory.py      # existente — memoria real del usuario
├── roleplay_memory.py      # nueva — memoria ficticia narrativa
├── memory_models.py        # nueva — modelos compartidos
└── memory_router.py        # nueva — router normal/roleplay
```

## 4. Tabla SQLite unificada — `semantic_memories`

Una sola tabla para TODO: memorias normales + roleplay. Se diferencian por `namespace`.

```sql
CREATE TABLE IF NOT EXISTS semantic_memories (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    chroma_id       TEXT NOT NULL UNIQUE,
    namespace       TEXT NOT NULL DEFAULT 'user',  -- 'user' | 'roleplay'
    content         TEXT NOT NULL,
    tags            TEXT NOT NULL DEFAULT '',
    confidence      REAL NOT NULL DEFAULT 1.0,
    memory_type     TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'active', -- active | pending_review | archived | deleted
    source          TEXT NOT NULL DEFAULT 'auto',
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now')),

    -- Roleplay-specific (vacíos para namespace='user')
    owner_type      TEXT NOT NULL DEFAULT '',       -- user_character | assistant_character | shared_world
    character_id    TEXT NOT NULL DEFAULT '',
    source_role     TEXT NOT NULL DEFAULT '',       -- user | assistant
    canon_status    TEXT NOT NULL DEFAULT 'canon',
    fact_key        TEXT NOT NULL DEFAULT '',
    fact_value      TEXT NOT NULL DEFAULT '',
    scene_id        TEXT NOT NULL DEFAULT '',
    world_id        TEXT NOT NULL DEFAULT ''
);

CREATE INDEX idx_semantic_memories_chroma ON semantic_memories(chroma_id);
CREATE INDEX idx_semantic_memories_namespace ON semantic_memories(namespace, status);
```

### Beneficios de la tabla unificada

| Operación | Antes (solo ChromaDB) | Después (SQLite + ChromaDB) |
|---|---|---|
| Listar memorias | `collection.get()` sin filtros potentes | `SELECT * FROM semantic_memories WHERE namespace='user'` |
| Buscar por ID | No tenía ID estable | `chroma_id` propio, búsqueda por PK |
| Borrar | `collection.delete(ids=[...])` | `DELETE FROM semantic_memories WHERE id=?` + `collection.delete(ids=[chroma_id])` |
| Editar metadata | Update complejo en Chroma | `UPDATE semantic_memories SET ... WHERE id=?` + sync a Chroma |
| Migrar namespace | No aplica | Cambiar `namespace` en SQLite |
| Auditoría | No había historial | `updated_at`, `status` con `deleted` (soft-delete opcional) |

## 5. Cambios en `SemanticMemory` (existente)

### 5.1. Constructor ahora acepta `sqlite_conn`

```python
class SemanticMemory:
    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        sqlite_conn: sqlite3.Connection | None = None,
        user_id: str = "default",
        namespace: str = "user",
    )
```

### 5.2. `_store()` ahora escribe en SQLite + ChromaDB

```python
def _store(self, result, source, original_text=""):
    # 1. ChromaDB (búsqueda)
    chroma_id = self._insert_chroma(result)

    # 2. SQLite (source of truth)
    self._insert_sqlite(chroma_id, result, source, original_text)

    return chroma_id
```

### 5.3. CRUD contra SQLite

| Método | Cambio |
|---|---|
| `get_memory(memory_id)` | Ahora busca por `chroma_id` en SQLite |
| `forget(memory_id)` | `DELETE FROM semantic_memories WHERE chroma_id=?` + `collection.delete(ids=[chroma_id])` |
| `archive(memory_id)` | `UPDATE semantic_memories SET status='archived' WHERE chroma_id=?` |
| `list_memories(limit)` | `SELECT * FROM semantic_memories WHERE namespace=? AND status != 'deleted'` |
| `search(query, n_results)` | ChromaDB igual, pero filtra por `namespace` + `status='active'` |

### 5.4. Namespace en ChromaDB metadata

Todas las colecciones de ChromaDB incluyen `namespace` en metadata para poder filtrar:

```python
metadatas=[{
    "namespace": self._namespace,   # "user" o "roleplay"
    "user_id": self._user_id,
    ...
}]
```

## 6. Clase `RoleplaySemanticMemory`

### 6.1. Constructor

```python
class RoleplaySemanticMemory:
    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        sqlite_conn: sqlite3.Connection | None = None,
        world_id: str = "default_world",
        user_character_id: str = "user_character",
        assistant_character_id: str = "assistant_character",
    )
```

- Usa `namespace = "roleplay"` en metadata de ChromaDB
- Dos collections: `roleplay_user_memories` (owner_type=user_character) y `roleplay_character_memories` (owner_type=assistant_character)

### 6.2. API

| Método | Descripción |
|---|---|
| `remember(text, source_role, scene_id, source)` | Analiza con patrones, guarda en SQLite + ChromaDB |
| `search_user(query, n_results)` | ChromaDB query filtrando por `namespace='roleplay'` + `owner_type='user_character'` |
| `search_character(query, n_results)` | ChromaDB query filtrando por `namespace='roleplay'` + `owner_type='assistant_character'` |
| `build_context(query, n_results)` | Construye bloque `[ROLEPLAY_MEMORY]` para el prompt |
| `get_memory(memory_id)` | Busca por `chroma_id` en SQLite |
| `forget(memory_id)` | Elimina de SQLite + ChromaDB |
| `list_memories(limit)` | `SELECT * FROM semantic_memories WHERE namespace='roleplay'` |

### 6.3. Tipos de memoria roleplay

```
character_identity, character_preference, character_trait,
character_backstory, character_goal, character_fear,
relationship_state, promise, inventory, injury,
world_lore, story_event, scene_state
```

### 6.4. Dataclasses

```python
@dataclass
class RoleplayAnalysisResult:
    should_remember: bool
    reason: str
    confidence: float
    content: str
    memory_type: str
    tags: list[str]
    owner_type: str          # user_character | assistant_character | shared_world
    character_id: str
    source_role: str         # user | assistant
    canon_status: str        # canon | soft_canon | temporary | contradicted
    fact_key: str
    fact_value: str
    scene_id: str
    world_id: str
```

### 6.5. Detección de patrones (sin LLM)

```python
ROLEPLAY_PATTERNS = {
    "favorite_color": {
        "memory_type": "character_preference",
        "patterns": [r"mi color favorito es\s+(.+)", r"mi color preferido es\s+(.+)"],
    },
    "name": {
        "memory_type": "character_identity",
        "patterns": [r"me llamo\s+(.+)", r"mi nombre es\s+(.+)"],
    },
    "origin": {
        "memory_type": "character_backstory",
        "patterns": [r"vengo de\s+(.+)", r"nac(i|í) en\s+(.+)"],
    },
    "fear": {
        "memory_type": "character_fear",
        "patterns": [r"tengo miedo de\s+(.+)", r"tengo miedo a\s+(.+)", r"me asustan\s+(.+)"],
    },
    "promise": {
        "memory_type": "promise",
        "patterns": [r"prometo\s+(.+)", r"te prometo\s+(.+)", r"juro\s+(.+)"],
    },
    "age": {
        "memory_type": "character_identity",
        "patterns": [r"tengo\s+(\d+)\s+años"],
    },
}
```

### 6.6. Contradicciones

- Misma `character_id + fact_key + status=active` → conflicto
- Si `fact_value` difiere: marcar anterior como `contradicted` (SQLite + ChromaDB)
- Nueva guarda como `canon`

## 7. `MemoryRouter`

```python
class MemoryRouter:
    def __init__(self, semantic: SemanticMemory, roleplay: RoleplaySemanticMemory)
```

| Método | Descripción |
|---|---|
| `remember_user(text)` | Roleplay → `roleplay.remember(role="user")`. Normal → `semantic.remember(text)` |
| `remember_assistant(text)` | Roleplay → `roleplay.remember(role="assistant")`. Normal → no guarda |
| `build_context(query)` | Roleplay → `roleplay.build_context(query)`. Normal → `semantic.search(query)` |
| `forget(memory_id)` | Busca `chroma_id` en SQLite, borra de ambas DBs |
| `list_memories(limit)` | Unificado de ambas según namespace |

## 8. Flujo `forget(memory_id)`

```python
def forget(self, memory_id: str) -> None:
    row = self._conn.execute(
        "SELECT chroma_id FROM semantic_memories WHERE chroma_id = ?", (memory_id,)
    ).fetchone()
    if not row:
        return
    chroma_id = row[0]
    self._collection.delete(ids=[chroma_id])
    self._conn.execute("DELETE FROM semantic_memories WHERE chroma_id = ?", (chroma_id,))
    self._conn.commit()
```

## 9. Inyección en system prompt

### Modo normal

```
[USER_MEMORY]
- Preferencia del usuario: Le gustan las galletas de chocolate

[USO DE MEMORIA]
- USER_MEMORY describe al usuario que está conversando.
- Si USER_MEMORY contiene la respuesta, responde directamente.
```

### Modo roleplay

```
[ROLEPLAY MODE]
Estás participando en una historia de roleplay.
Las memorias siguientes pertenecen al mundo ficticio.
No las confundas con datos reales del usuario.

[ROLEPLAY_USER_CHARACTER_MEMORY]
- Mikaela tiene miedo de los espejos.

[ROLEPLAY_ASSISTANT_CHARACTER_MEMORY]
- Juan tiene como color favorito el rojo.

[ROLEPLAY MEMORY RULES]
- Usa estas memorias como canon de la historia.
- No digas "como modelo de lenguaje" dentro del roleplay.
```

## 10. Tareas de implementación

1. Agregar tabla `semantic_memories` a `db.py` (schema v3)
2. Migración v2→v3: `CREATE TABLE IF NOT EXISTS semantic_memories`
3. Crear `memory_models.py` — dataclasses compartidas (RoleplayAnalysisResult, etc.)
4. Refactor `SemanticMemory._store()` para escribir también en SQLite
5. Refactor `SemanticMemory.forget()` para borrar de SQLite + ChromaDB
6. Refactor `SemanticMemory.get_memory()` para leer de SQLite
7. Agregar `namespace` a metadata de ChromaDB en `SemanticMemory`
8. Crear `roleplay_memory.py` — clase RoleplaySemanticMemory
9. Crear `memory_router.py` — router normal/roleplay
10. Actualizar `__init__.py`
11. Agregar comandos roleplay a `console_chat.py`
12. Actualizar README
13. Tests

## 11. Edge cases

- **Sin SQLite**: `SemanticMemory` funciona solo con ChromaDB (modo legacy), `RoleplaySemanticMemory` requiere SQLite
- **Namespace no coincide**: ChromaDB busca por `namespace` + `status`, SQLite filtra igual
- **Contradicción no resuelta**: nueva como `pending_review`, conservar la canon
- **Roleplay desactivado**: router no llama a RoleplaySemanticMemory
- **Migración v2→v3**: solo `CREATE TABLE IF NOT EXISTS semantic_memories`, no rompe nada existente
