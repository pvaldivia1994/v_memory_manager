# v_memory_manager — Plan de implementación

## 1. Propósito

Librería hermana de `v_llama` que funciona como módulo de memoria persistente.
Gestiona el historial de conversaciones, system prompts y configuraciones usando
SQLite con un comportamiento de **ventana deslizante** (sliding-window deque).

---

## 2. Stack técnico

| Componente | Tecnología | Razón |
|---|---|---|
| Base de datos | SQLite (`sqlite3` stdlib) | Zero dependencias, embebido, suficiente para este alcance |
| Tipado | `dataclasses` + `TypedDict` | Consistente con v_llama |
| Tests | `unittest` / `pytest` | Coincidir con lo que use v_llama (verificar) |

---

## 3. Estructura de directorios

```
v_memory_manager/
├── src/
│   ├── __init__.py
│   ├── db.py             # CRUD SQLite + init schema
│   ├── memory.py         # MemoryManager (fachada pública)
│   ├── models.py         # Dataclasses: Message, Prompt
│   ├── deque.py          # Lógica de sliding-window
│   └── def_system.md     # System prompt por defecto
├── examples/
│   └── console_chat.py   # Chat con memoria persistente
├── tests/
│   └── test_memory.py
├── .plans/
│   └── plan_v1.md        # Este documento
└── pyproject.toml
```

---

## 4. Esquema de base de datos (SQLite)

```sql
CREATE TABLE messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    role        TEXT NOT NULL CHECK(role IN ('user','assistant')),
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE prompts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE configurations (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL
);
```

### Notas

- `messages.role` solo acepta `'user'` y `'assistant'`. El system prompt NO va acá.
- El system prompt activo se guarda en `prompts` con el nombre reservado `'_active'`.
- `prompts.name` es UNIQUE: guardar con el mismo nombre hace UPSERT.

---

## 5. `def_system.md`

Archivo Markdown ubicado en `src/def_system.md`. Contenido por defecto:

```markdown
Eres un asistente útil y natural.
```

Durante `create_memory_db()`, la librería lee este archivo y lo inserta en `prompts`
con `name='_active'`. Si el usuario pasa un `default_system_path` alternativo, se usa ese.

---

## 6. API pública (`MemoryManager`)

### 6.1. Ciclo de vida

| Método | Descripción |
|---|---|
| `create_memory_db(path, default_system_path=None)` | Crea DB nueva en `path`, corre schema, lee `def_system.md` (o `default_system_path`) y lo inserta como `_active` |
| `load_memory_db(path)` | Carga DB existente, verifica schema |
| `drop_memory_db()` | Elimina el archivo .db del disco |
| `clear_memory_db()` | Trunca solo `messages`; `prompts` y `configurations` se conservan. El system prompt activo sobrevive |

### 6.2. Mensajes (sliding-window)

| Método | Descripción |
|---|---|
| `add_message(role, content)` | Si role es `'user'`/`'assistant'` → inserta en `messages`. Si role es `'system'` → guarda/actualiza en `prompts` como `_active` |
| `get_history(max_messages=10) -> list[Message]` | Combina el system prompt (`_active` de `prompts`) con los últimos N-1 mensajes de `messages`. Sliding-window: system siempre primero, luego los más recientes |
| `get_system_prompt() -> str` | Devuelve el contenido de `_active` en `prompts`, o `""` si no existe |
| `count_messages() -> int` | Total de filas en `messages` |

### 6.3. Prompts guardados

| Método | Descripción |
|---|---|
| `save_prompt(name, content)` | Guarda o actualiza un prompt por nombre en `prompts` |
| `load_prompt(name) -> str \| None` | Carga un prompt por nombre. `None` si no existe |
| `list_prompts() -> list[str]` | Lista nombres de prompts guardados (excluye `_active`) |
| `delete_prompt(name)` | Elimina un prompt de `prompts` |

### 6.4. Configuraciones

| Método | Descripción |
|---|---|
| `get_config(key, default=None)` | Obtiene un valor de `configurations` |
| `set_config(key, value)` | Establece un valor en `configurations` |
| `all_configs() -> dict` | Devuelve todas las configuraciones |

---

## 7. Sliding-window (deque)

### Comportamiento

```
Estado inicial (limit=5, _active="Eres un asistente..."):
  → [system: "Eres un asistente...", u1, a1, u2, a2]

Agrega u3:    → [system, a1, u2, a2, u3]
Agrega a3:    → [system, u2, a2, u3, a3]
```

- El system prompt (`_active`) **siempre** se mantiene en índice 0
- Los mensajes restantes son los últimos N-1 de `messages`
- El `max_messages` de `get_history()` define N

### Implementación

```python
def get_history(self, max_messages: int = 10) -> list[Message]:
    system = self._get_active_prompt()
    msgs = self._get_all_messages_sorted()
    window = msgs[-(max_messages - 1):] if max_messages > 1 else []
    result = []
    if system:
        result.append(Message(role="system", content=system.content))
    result.extend(window)
    return result
```

---

## 8. Dependencia con v_llama

`v_memory_manager` es **independiente** de `v_llama` — no importa nada de v_llama.
La integración se hace desde afuera (el ejemplo y el usuario deciden cómo usarlo).

---

## 9. Ejemplo de integración (`examples/console_chat.py`)

Flujo:
1. `create_memory_db("./chat.db")` al iniciar si no existe → inserta `_active` desde `def_system.md`
2. `system_prompt = mem.get_system_prompt()` → obtiene el `_active`
3. Por cada mensaje: `add_message("user", texto)`, `add_message("assistant", respuesta)`
4. Antes de llamar a `v_llama.chat()`: arma `history` desde `get_history()` filtrando role=system
5. `/prompt <texto>` → `add_message("system", texto)` (guarda como `_active`) + `clear_memory_db()` (limpia solo messages)
6. `/clear` → `clear_memory_db()` (solo messages, el system prompt se conserva)
7. `/save <nombre>` → `save_prompt(nombre, system_prompt)`
8. `/load <nombre>` → `load_prompt(nombre)` → `add_message("system", loaded)` + `clear_memory_db()`

---

## 10. Tareas de implementación (orden sugerido)

1. **Crear estructura del package** — `src/`, `pyproject.toml`, `__init__.py`, `def_system.md`
2. **Implementar `models.py`** — dataclasses `Message`, `Prompt`
3. **Implementar `db.py`** — Schema SQLite, CRUD en `messages`, `prompts`, `configurations`
4. **Implementar `deque.py`** — Lógica de sliding-window
5. **Implementar `memory.py`** — `MemoryManager` que compone db + deque y expone la API pública
6. **Ajustar `examples/console_chat.py`** — Reflejar el nuevo diseño (clear conserva system prompt, etc.)
7. **Tests** — Cobertura básica de cada método
8. **Verificación** — Correr el ejemplo, asegurar que persiste y recarga correctamente

---

## 11. No funcional / Edge cases

- **DB corrupta**: `load_memory_db` debe detectar schema inválido y levantar error claro
- **Archivo en uso**: capturar `sqlite3.OperationalError` y relanzar como `DatabaseLockedError`
- **Ruta inválida**: validar directorio padre existe antes de crear
- **`def_system.md` faltante**: si no existe el archivo, `create_memory_db()` usa un string hardcodeado como fallback
- **Concurrencia**: SQLite soporta lecturas concurrentes; escrituras se serializan naturalmente. No se agrega locking extra en esta versión
- **`_active` sin prompts**: si no hay fila `_active` en `prompts`, `get_system_prompt()` devuelve `""` y `get_history()` omite el system prompt
