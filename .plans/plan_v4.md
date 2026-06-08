# v_memory_manager — Plan v4: SQLite-backed semantic memories + Roleplay

## 1. Propósito

Unificar TODAS las memorias semánticas (normales + roleplay) en SQLite como source of truth.
ChromaDB queda solo como índice de búsqueda semántica. Borrar/archivar se hace por memory_id
desde SQLite, que contiene el chroma_id para sincronizar ChromaDB.

## 2. Principios

```
SQLite manda.
Chroma solo busca.
```

## 3. Stack técnico

| Componente | Tecnología |
|---|---|
| Source of truth | SQLite (tabla `semantic_memories`) |
| Search index | ChromaDB (colecciones separadas por namespace) |
| Detección normal | Reglas + scoring (`semantic_memory.py`) |
| Detección roleplay | Patrones regex (`roleplay_memory.py`) |

## 4. Tabla SQLite — `semantic_memories`

```sql
CREATE TABLE IF NOT EXISTS semantic_memories (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id       TEXT NOT NULL UNIQUE,
    chroma_id       TEXT UNIQUE,

    namespace       TEXT NOT NULL DEFAULT 'normal',
    scope           TEXT NOT NULL DEFAULT 'user',

    content         TEXT NOT NULL,
    original_text   TEXT NOT NULL DEFAULT '',
    tags            TEXT NOT NULL DEFAULT '',

    confidence      REAL NOT NULL DEFAULT 1.0,
    importance      REAL NOT NULL DEFAULT 0.5,

    memory_type     TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'active',
    source          TEXT NOT NULL DEFAULT 'auto',

    source_message_ids TEXT NOT NULL DEFAULT '',

    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now')),

    -- Roleplay-specific
    owner_type      TEXT NOT NULL DEFAULT '',
    character_id    TEXT NOT NULL DEFAULT '',
    source_role     TEXT NOT NULL DEFAULT '',
    canon_status    TEXT NOT NULL DEFAULT 'canon',
    fact_key        TEXT NOT NULL DEFAULT '',
    fact_value      TEXT NOT NULL DEFAULT '',
    scene_id        TEXT NOT NULL DEFAULT '',
    world_id        TEXT NOT NULL DEFAULT '',
    expires_scope   TEXT NOT NULL DEFAULT 'never'
);

CREATE INDEX IF NOT EXISTS idx_semantic_memories_memory_id
ON semantic_memories(memory_id);

CREATE INDEX IF NOT EXISTS idx_semantic_memories_chroma_id
ON semantic_memories(chroma_id);

CREATE INDEX IF NOT EXISTS idx_semantic_memories_ns_scope_status
ON semantic_memories(namespace, scope, status);

CREATE INDEX IF NOT EXISTS idx_semantic_memories_rp_fact
ON semantic_memories(world_id, character_id, fact_key, status, canon_status);
```

### Namespaces

```
normal                 → memorias reales del usuario
roleplay               → memorias ficticias
```

### Scopes

```
user_profile
user_preference
project
user_character         → personaje del usuario en roleplay
assistant_character    → personaje IA en roleplay
shared_world           → lore compartido
relationship           → relación entre personajes
scene_state            → estado temporal de escena
```

### Status flow

```
active ↔ archived ↘
pending_review → active (tras revisión)
deleted (soft, se borra de Chroma pero queda en SQLite)
```

### Campos clave

| Campo | Descripción |
|---|---|
| `memory_id` | ID estable externo (`mem_abc123`). No depende de Chroma. |
| `chroma_id` | ID en ChromaDB. Puede ser igual a `memory_id` o diferente tras reindex. |
| `namespace` | `normal` \| `roleplay` |
| `scope` | Agrupa memorias por contexto semántico |
| `importance` | 0.0–1.0. Distinto de `confidence`. `confidence` = qué tan seguro estoy de la detección. `importance` = qué tan relevante es para respuestas futuras. |
| `canon_status` | `canon` \| `soft_canon` \| `temporary` \| `contradicted` |
| `expires_scope` | `never` \| `end_of_scene` \| `end_of_day` \| `end_of_chapter` |
| `source_message_ids` | Trazabilidad a mensajes originales |

## 5. Operaciones CRUD

| Operación | SQLite | ChromaDB |
|---|---|---|
| `remember` | INSERT | add |
| `archive` | UPDATE status='archived' | update status |
| `forget` | UPDATE status='deleted' | delete (libera espacio) |
| `purge` | DELETE físico | delete |
| `get_memory` | SELECT por memory_id | — |
| `list_memories` | SELECT por namespace/scope/status | — |
| `search` | — | query (filtra por namespace + status) |

### forget vs purge

```
forget(memory_id) → soft: status='deleted', borra de Chroma pero SQLite conserva el registro
purge(memory_id)  → hard: DELETE físico de SQLite + Chroma
```

## 6. Arquitectura de archivos

```
src/
├── semantic_memory.py      # existente — refactorizada para usar SQLite
├── roleplay_memory.py      # nueva — RoleplaySemanticMemory
├── memory_models.py        # nueva — dataclasses compartidas
├── memory_router.py        # nueva — router normal/roleplay
├── db.py                   # existente — schema v3 + migración
├── memory.py               # existente — MemoryManager
└── __init__.py             # existente — exports
```

## 7. `SemanticMemory` — cambios

### 7.1. Constructor

```python
class SemanticMemory:
    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        sqlite_conn: sqlite3.Connection | None = None,
        user_id: str = "default",
        namespace: str = "normal",
        scope: str = "user",
    )
```

### 7.2. `_store()` escribe en SQLite + ChromaDB

```python
def _store(self, result, source, original_text="", msg_ids=""):
    memory_id = f"mem_{uuid.hex[:16]}"
    chroma_id = memory_id

    # ChromaDB
    self.collection.add(
        ids=[chroma_id],
        documents=[result.content],
        metadatas=[{
            "namespace": self._namespace,
            "scope": self._scope,
            "status": status,
            "confidence": result.confidence,
            "user_id": self._user_id,
            "tags": tags_str,
        }],
    )

    # SQLite
    self._conn.execute("""
        INSERT INTO semantic_memories
        (memory_id, chroma_id, namespace, scope, content, original_text,
         tags, confidence, importance, memory_type, status, source,
         source_message_ids)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (...))

    return memory_id
```

### 7.3. `forget()` → soft delete

```python
def forget(self, memory_id: str) -> None:
    row = self._conn.execute(
        "SELECT chroma_id FROM semantic_memories WHERE memory_id = ?",
        (memory_id,),
    ).fetchone()
    if not row:
        return
    self.collection.delete(ids=[row[0]])
    self._conn.execute(
        "UPDATE semantic_memories SET status='deleted', updated_at=datetime('now') WHERE memory_id=?",
        (memory_id,),
    )
    self._conn.commit()
```

### 7.4. `purge()` — borrado físico

```python
def purge(self, memory_id: str) -> None:
    row = self._conn.execute(
        "SELECT chroma_id FROM semantic_memories WHERE memory_id = ?", (memory_id,)
    ).fetchone()
    if row and row[0]:
        self.collection.delete(ids=[row[0]])
    self._conn.execute("DELETE FROM semantic_memories WHERE memory_id = ?", (memory_id,))
    self._conn.commit()
```

## 8. `RoleplaySemanticMemory`

### 8.1. Constructor

```python
class RoleplaySemanticMemory:
    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        sqlite_conn: sqlite3.Connection,
        world_id: str = "default_world",
        user_character_id: str = "user_character",
        assistant_character_id: str = "assistant_character",
    )
```

- `namespace = "roleplay"` en metadata de ChromaDB
- Dos collections: `roleplay_user_character_memories` y `roleplay_assistant_character_memories`
- La collection `roleplay_assistant_character_memories` también almacena `scope=shared_world`, `scope=relationship`, `scope=scene_state`

### 8.2. API

| Método | Descripción |
|---|---|
| `remember(text, source_role, scene_id, source)` | Analiza con regex, guarda en SQLite + ChromaDB |
| `remember_force(content, owner_type, character_id, ...)` | Guarda manualmente sin analizar |
| `search_user(query, n_results)` | ChromaDB: `namespace=roleplay`, `scope=user_character`, `status=active` |
| `search_character(query, n_results)` | ChromaDB: `namespace=roleplay`, `scope=assistant_character`, `status=active` |
| `search_world(query, n_results)` | ChromaDB: `namespace=roleplay`, `scope=shared_world`, `status=active` |
| `build_context(query, n_results)` | Construye bloques `[ROLEPLAY_*_MEMORY]` para el prompt |
| `get_memory(memory_id)` | SELECT por memory_id en SQLite |
| `forget(memory_id)` | Soft delete en SQLite + ChromaDB |
| `purge(memory_id)` | DELETE físico |
| `archive(memory_id)` | status = archived |
| `list_memories(limit)` | SELECT por namespace |
| `count()` | COUNT por namespace |

### 8.3. Patrones de detección

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

### 8.4. Contradicciones según tipo

| memory_type | Política |
|---|---|
| `character_identity` | Nueva → `pending_review`. Anterior conserva canon. |
| `character_backstory` | Nueva → `pending_review`. Anterior conserva canon. |
| `character_preference` | Nueva reemplaza a anterior (archiva). |
| `relationship_state` | Nueva reemplaza a anterior. |
| `scene_state` | Nueva reemplaza a anterior. |
| `world_lore` | Nueva → `pending_review`. Anterior conserva canon. |

```python
CONFLICT_POLICY = {
    "character_identity": "pending_review",
    "character_backstory": "pending_review",
    "character_preference": "replace",
    "relationship_state": "replace",
    "scene_state": "replace",
    "world_lore": "pending_review",
}
```

### 8.5. `canon_status`

- `canon`: verdad establecida, usar siempre
- `soft_canon`: puede cambiar, usar pero con flexibilidad
- `temporary`: estado de escena, no usar fuera de ella
- `contradicted`: reemplazada por una versión más reciente

### 8.6. `expires_scope`

| Valor | Significado |
|---|---|
| `never` | Memoria permanente |
| `end_of_scene` | Se archiva al cambiar de escena |
| `end_of_day` | Se archiva al terminar la sesión |
| `end_of_chapter` | Se archiva al cambiar de capítulo |

## 9. `MemoryRouter`

```python
class MemoryRouter:
    def __init__(
        self,
        semantic: SemanticMemory,
        roleplay: RoleplaySemanticMemory,
        roleplay_enabled: bool = False,
    )

    def set_roleplay_enabled(self, enabled: bool)
    def remember_user(self, text: str, msg_id: str = "")
    def remember_assistant(self, text: str, msg_id: str = "")
    def build_context(self, query: str) -> str
    def forget(self, memory_id: str)
    def get_memory(self, memory_id: str)
    def list_memories(self, limit: int = 50)
```

| Método | Roleplay ON | Roleplay OFF |
|---|---|---|
| `remember_user(text)` | `roleplay.remember(role="user")` | `semantic.remember(text)` |
| `remember_assistant(text)` | `roleplay.remember(role="assistant")` | No guarda |
| `build_context(query)` | `roleplay.build_context(query)` | `semantic.search(query)` |

## 10. Inyección en system prompt

### Modo normal

```
[USER_MEMORY]
- Al usuario le gustan las galletas de chocolate

[USO DE MEMORIA]
- USER_MEMORY describe al usuario real.
- Si USER_MEMORY responde la pregunta, responde directamente.
```

### Modo roleplay

```
[ROLEPLAY MODE]
Estás participando en una historia de roleplay.
Las memorias siguientes pertenecen al mundo ficticio.
No las confundas con datos reales del usuario.

[REALITY BOUNDARY]
- USER_MEMORY describe al usuario real (fuera del roleplay).
- ROLEPLAY_USER_CHARACTER_MEMORY describe al personaje del usuario dentro de la ficción.
- ROLEPLAY_ASSISTANT_CHARACTER_MEMORY describe al personaje que interpretas.
- Nunca mezcles estas memorias.

[ROLEPLAY_USER_CHARACTER_MEMORY]
- Mikaela tiene miedo de los espejos.

[ROLEPLAY_ASSISTANT_CHARACTER_MEMORY]
- Juan tiene como color favorito el rojo.

[ROLEPLAY_WORLD_MEMORY]
- La capital está bajo control militar.

[ROLEPLAY MEMORY RULES]
- Usa estas memorias como canon.
- No digas "como modelo de lenguaje" dentro del roleplay.
```

## 11. Tareas de implementación

1. **`db.py`**: schema v3 con `semantic_memories`, migración v2→v3
2. **`memory_models.py`**: dataclasses (`RoleplayAnalysisResult`, `MemoryRecord` unificado)
3. **Refactor `SemanticMemory`**: `_store()` escribe SQLite + ChromaDB, `forget()` soft delete, `purge()`, `get_memory()` vía SQLite
4. **`roleplay_memory.py`**: clase `RoleplaySemanticMemory` con patrones regex, contradicciones, collections separadas
5. **`memory_router.py`**: router normal/roleplay con flag
6. **`__init__.py`**: exports actualizados
7. **`console_chat.py`**: comandos `/remember`, `/memories`, `/forget`, `/roleplay on/off`
8. **README**: documentación
9. **Tests**: smoke tests

## 12. Edge cases

- **Sin SQLite**: SemanticMemory funciona solo con ChromaDB (modo legacy). RoleplaySemanticMemory requiere SQLite.
- **ChromaDB no instalado**: error claro "pip install chromadb"
- **Reindex de Chroma**: se puede reconstruir desde SQLite porque el `chroma_id` puede regenerarse.
- **Roleplay sin personajes**: usar character_ids por defecto.
- **Escena sin ID**: usar `scene_id = ""`, no se trackea por escena.
- **Contradicción no resuelta**: nueva como `pending_review`, anterior conserva canon.
