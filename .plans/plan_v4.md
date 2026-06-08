# v_memory_manager — Plan v4: RoleplaySemanticMemory

## 1. Propósito

Crear un sistema de memoria semántica para roleplay/narrativa, separado de `SemanticMemory`
(memoria real del usuario). Guarda hechos ficticios de personajes (usuario e IA) en ChromaDB
+ SQLite, con detección de contradicciones, canon status, y contexto inyectable al prompt.

## 2. Stack técnico

| Componente | Tecnología |
|---|---|
| Vector DB | ChromaDB (2 collections: `roleplay_user`, `roleplay_character`) |
| SQLite | Tabla `roleplay_memories` en la DB existente de MemoryManager |
| Detección | Patrones regex (sin LLM) |

## 3. Arquitectura

```
memory/
├── semantic_memory.py      # existente — memoria real del usuario
├── roleplay_memory.py      # nueva — memoria ficticia narrativa
├── memory_models.py        # nueva — modelos compartidos
└── memory_router.py        # nueva — router normal/roleplay
```

## 4. Clase `RoleplaySemanticMemory`

### 4.1. Constructor

```python
class RoleplaySemanticMemory:
    def __init__(
        self,
        persist_dir: str = "./chroma_db/roleplay",
        sqlite_conn: sqlite3.Connection | None = None,
        world_id: str = "default_world",
        user_character_id: str = "user_character",
        assistant_character_id: str = "assistant_character",
    )
```

### 4.2. Dos collections ChromaDB

| Collection | Guarda |
|---|---|
| `roleplay_user_memories` | Hechos del personaje del usuario (Mikaela) |
| `roleplay_character_memories` | Hechos del personaje IA (Juan) + lore compartido |

### 4.3. Tipos de memoria

```
character_identity, character_preference, character_trait,
character_backstory, character_goal, character_fear,
relationship_state, promise, inventory, injury,
world_lore, story_event, scene_state
```

### 4.4. Dataclasses

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
    fact_key: str            # favorite_color, fear, promise, origin, etc.
    fact_value: str
    scene_id: str
    world_id: str

@dataclass
class RoleplayMemoryRecord:
    id: str
    content: str
    tags: list[str]
    confidence: float
    memory_type: str
    status: str               # active | pending_review | archived
    created_at: str
    source: str
    owner_type: str
    character_id: str
    source_role: str
    canon_status: str
    fact_key: str
    fact_value: str
    scene_id: str
    world_id: str
    chroma_id: str            # ID en ChromaDB para búsqueda inversa
```

## 5. API pública

| Método | Descripción |
|---|---|
| `remember(text, source_role, scene_id, source)` | Analiza y guarda hechos roleplay. Devuelve IDs |
| `search_user(query, n_results)` | Busca en memorias del personaje del usuario |
| `search_character(query, n_results)` | Busca en memorias del personaje IA |
| `build_context(query, n_results)` | Construye bloque `[ROLEPLAY_MEMORY]` para el prompt |
| `get_memory(memory_id)` | Obtiene una memoria por ID de SQLite |
| `forget(memory_id)` | Elimina de SQLite + ChromaDB |
| `archive(memory_id)` | Soft-delete en SQLite + ChromaDB |
| `list_memories(limit)` | Lista todas las memorias roleplay |
| `count()` | Total de memorias |
| `resolve_contradiction(memory_id, resolution)` | Resuelve contradicción manualmente |

## 6. SQLite — tabla `roleplay_memories`

```sql
CREATE TABLE IF NOT EXISTS roleplay_memories (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    chroma_id       TEXT NOT NULL UNIQUE,
    content         TEXT NOT NULL,
    tags            TEXT NOT NULL DEFAULT '',
    confidence      REAL NOT NULL DEFAULT 1.0,
    memory_type     TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'active',
    source          TEXT NOT NULL DEFAULT 'auto',
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    owner_type      TEXT NOT NULL DEFAULT '',
    character_id    TEXT NOT NULL DEFAULT '',
    source_role     TEXT NOT NULL DEFAULT '',
    canon_status    TEXT NOT NULL DEFAULT 'canon',
    fact_key        TEXT NOT NULL DEFAULT '',
    fact_value      TEXT NOT NULL DEFAULT '',
    scene_id        TEXT NOT NULL DEFAULT '',
    world_id        TEXT NOT NULL DEFAULT ''
);
```

## 7. Detección de patrones (sin LLM)

```python
ROLEPLAY_PATTERNS = {
    "favorite_color": {
        "memory_type": "character_preference",
        "patterns": [
            r"mi color favorito es\s+(.+)",
            r"mi color preferido es\s+(.+)",
        ],
    },
    "name": {
        "memory_type": "character_identity",
        "patterns": [
            r"me llamo\s+(.+)",
            r"mi nombre es\s+(.+)",
        ],
    },
    "origin": {
        "memory_type": "character_backstory",
        "patterns": [
            r"vengo de\s+(.+)",
            r"nac(i|í) en\s+(.+)",
        ],
    },
    "fear": {
        "memory_type": "character_fear",
        "patterns": [
            r"(le|tengo) tengo miedo de\s+(.+)",
            r"tengo miedo a\s+(.+)",
            r"me asustan\s+(.+)",
        ],
    },
    "promise": {
        "memory_type": "promise",
        "patterns": [
            r"prometo\s+(.+)",
            r"te prometo\s+(.+)",
            r"juro\s+(.+)",
        ],
    },
    "age": {
        "memory_type": "character_identity",
        "patterns": [
            r"tengo\s+(\d+)\s+años",
        ],
    },
}
```

## 8. Contradicciones y canon

- Misma `character_id + fact_key + status=active` → detectar conflicto
- Si `fact_value` difiere: marcar memoria anterior como `contradicted`
- Guardar nueva como `canon`
- Opción de resolución manual por ID

## 9. `MemoryRouter`

```python
class MemoryRouter:
    def __init__(self, semantic: SemanticMemory, roleplay: RoleplaySemanticMemory)
```

| Método | Descripción |
|---|---|
| `remember_user(text)` | Roleplay → `roleplay.remember(role="user")`. Normal → `semantic.remember(text)` |
| `remember_assistant(text)` | Roleplay → `roleplay.remember(role="assistant")`. Normal → no guarda |
| `build_context(query)` | Roleplay → `roleplay.build_context(query)`. Normal → `semantic.search(query)` |

## 10. Inyección en system prompt

Con roleplay activo, `build_system_prompt()` incluye:

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
- ROLEPLAY_USER_CHARACTER_MEMORY describe al personaje del usuario.
- ROLEPLAY_ASSISTANT_CHARACTER_MEMORY describe a tu personaje.
- Usa estas memorias como canon.
- No digas "como modelo de lenguaje" dentro del roleplay.
```

## 11. Tareas de implementación

1. Crear `memory_models.py` — dataclasses compartidas
2. Crear `roleplay_memory.py` — clase RoleplaySemanticMemory
3. Crear `memory_router.py` — router normal/roleplay
4. Agregar tabla `roleplay_memories` a `db.py` (schema v3)
5. Agregar migración v2→v3
6. Actualizar `__init__.py`
7. Agregar comandos roleplay a `console_chat.py`
8. Actualizar README
9. Tests

## 12. Edge cases

- **Roleplay desactivado**: no se guarda nada del asistente, router usa SemanticMemory
- **Sin personaje definido**: usar character_ids por defecto
- **Contradicción no resuelta**: marcar nueva como `pending_review`, conservar la canon
- **Búsqueda sin resultados**: devolver `- Sin memorias relevantes.`
- **SQLite sin ChromaDB**: ChromaDB es fuente de búsqueda, SQLite es source of truth. Ambos necesarios.
- **Migración v2→v3**: `CREATE TABLE IF NOT EXISTS roleplay_memories`
