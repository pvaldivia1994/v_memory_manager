# v_memory_manager — Plan v2: Long-term memory + `build_system_prompt`

## 1. Propósito

Agregar memorias a largo plazo y un método `build_system_prompt()` que construya el
system prompt dinámicamente a partir de:

1. **core_prompt** — el system prompt base (actual `_active` en prompts)
2. **Prompts guardados** — todos los prompts de la tabla `prompts` (excluyendo `_active`), ordenados por un nuevo campo `orden`
3. **Memorias a largo plazo** — entradas de una nueva tabla `long_term_memories`

Estructura final de `get_history()`:

```
[build_system_prompt(), msg_N-4, msg_N-3, msg_N-2, msg_N-1, msg_N]
```

El primer elemento es el system prompt construido; le siguen los últimos N-1 mensajes
del sliding-window (donde N = `max_messages`).

---

## 2. Stack técnico

| Componente | Tecnología |
|---|---|
| Base de datos | SQLite (`sqlite3` stdlib) |
| Tipado | `dataclasses` |
| Tests | `unittest` / `pytest` (por definir) |

---

## 3. Cambios en el schema

### 3.1. `prompts` — nuevo campo `orden`

```sql
CREATE TABLE prompts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    content     TEXT NOT NULL,
    orden       INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
```

- `orden` define la posición en la construcción del system prompt
- `_active` (core_prompt) tiene `orden=0` y siempre va primero
- Los prompts guardados por el usuario tienen `orden > 0`

### 3.2. Nueva tabla `long_term_memories`

```sql
CREATE TABLE long_term_memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    content     TEXT NOT NULL,
    tags        TEXT NOT NULL DEFAULT '',
    weight      REAL NOT NULL DEFAULT 1.0,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
```

| Campo | Tipo | Descripción |
|---|---|---|
| `content` | TEXT | Contenido de la memoria |
| `tags` | TEXT | Tags separados por coma (ej: `"personal,gustos,importante"`) |
| `weight` | REAL | Peso/prioridad (mayor = más relevante, default 1.0) |

Las memorias se ordenan por `weight DESC, created_at DESC` al construir el system prompt.

---

## 4. API pública — nuevos métodos en `MemoryManager`

### 4.1. `build_system_prompt() -> str`

Construye el system prompt completo:

```
core_prompt (name='_active')
--- si existen prompts guardados ---
[orden 1] prompt_name_1
[orden 2] prompt_name_2
...
--- si existen long-term memories ---
[Long-term memories]
- memoria_1  (tags: ...)
- memoria_2  (tags: ...)
...
```

Flujo:
1. Obtener `_active` de `prompts` (core_prompt)
2. Obtener todos los prompts con `name != '_active'` ordenados por `orden ASC`
3. Obtener long-term memories ordenadas por `weight DESC, created_at DESC`
4. Concatenar todo con separadores legibles

### 4.2. Long-term memories CRUD

| Método | Descripción |
|---|---|
| `add_long_term_memory(content, tags="", weight=1.0)` | Inserta una nueva memoria |
| `get_long_term_memories(tag=None, min_weight=None)` | Lista memorias, filtro opcional por tag y peso mínimo |
| `delete_long_term_memory(memory_id)` | Elimina una memoria por ID |
| `count_long_term_memories()` | Total de memorias almacenadas |

### 4.3. `save_prompt` — agregar parámetro `orden`

```python
def save_prompt(self, name: str, content: str, orden: int = 0) -> None
```

---

## 5. Cambios en `get_history()`

Se mantiene el sliding-window paramétrico actual:

```python
def get_history(self, max_messages: int = 10) -> list[Message]:
    # system = build_system_prompt()  ← reemplaza get_system_prompt()
    # window = últimos max_messages-1 mensajes de messages
    # retorna [system, msg_N-4, msg_N-3, ..., msg_N]
```

Sin cambios en la semántica de `max_messages`: define N = system + últimos N-1 mensajes.

---

## 6. Migración de DB existente

Para DBs creadas con schema v1 (sin `orden` en `prompts`, sin `long_term_memories`):

- `load_memory_db()` debe detectar schema v1 y ejecutar `ALTER TABLE` para agregar `orden`
- Crear `long_term_memories` si no existe
- Los prompts existentes reciben `orden=0` por defecto

Estrategia: en `verify_schema()`, verificar columnas de `prompts` y crear tabla
nueva si falta. No romper DBs existentes.

---

## 7. Tareas de implementación (orden sugerido)

1. **Escribir plan v2** ← estamos acá
2. **Actualizar `db.py`**:
   - Schema v2 con `orden` en `prompts` + `long_term_memories`
   - CRUD para long_term_memories
   - Actualizar `upsert_prompt` para aceptar `orden`
   - Migración automática en `verify_schema()`
3. **Actualizar `models.py`** — agregar dataclass `LongTermMemory`
4. **Implementar `build_system_prompt()` en `memory.py`**
5. **Ajustar `get_history()`** para usar `build_system_prompt()`
6. **Actualizar `deque.py`** si es necesario
7. **Actualizar `__init__.py`** — exportar nuevos tipos
8. **Actualizar `console_chat.py`** — comandos para long-term memory
9. **Actualizar `README.md`** — documentar nueva API
10. **Tests**

---

## 8. Edge cases

- **Sin prompts guardados**: `build_system_prompt()` devuelve solo el core_prompt + memorias
- **Sin long-term memories**: se omite la sección de memorias
- **Sin core_prompt**: `build_system_prompt()` devuelve `""` (igual que antes)
- **Tags vacíos**: se muestran sin tags
- **Weight muy alto/bajo**: se ordena naturalmente, sin límites
- **DB v1 migrando a v2**: `verify_schema()` debe ser tolerante
- **Prompts con mismo orden**: se ordenan alfabéticamente por name como tiebreaker
