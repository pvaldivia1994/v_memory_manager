# Plan: Mover contexto volátil a user message

## Problema

Actualmente `[BOOK_CONTEXT]`, `[USER_MEMORY]`, `[ASSISTANT_MEMORY]`, `[CONVERSATION_SUMMARY]` y `extra_context` se inyectan en el **system prompt** vía `build_system_prompt()`.

Esto:
- Invalida el KV cache de llama.cpp en cada turno (system prompt cambia)
- Duplica búsquedas semánticas (console_chat busca + build_system_prompt busca de nuevo)
- Duplica construcción del system prompt (console_chat lo construye + get_history() lo reconstruye)

## Solución

Separar en dos:

### System prompt (estático, cacheable toda la sesión)

Solo lo que NO cambia entre turnos:
- Core system prompt (`def_system.md`)
- Reglas de uso de memoria (`[USO DE MEMORIA]`)
- Prompts guardados por el usuario (`/save`)

### User message (dinámico, por turno)

Lo que cambia por cada mensaje:
- `[BOOK_CONTEXT]` — contexto de libros
- `[USER_MEMORY]` — memorias semánticas del usuario
- `[ASSISTANT_MEMORY]` — memorias semánticas del asistente
- `[CONVERSATION_SUMMARY]` — resumen conversacional
- `extra_context` — contexto adicional

Se inyecta al final del mensaje del usuario, separado por `--- Contexto adicional ---`.

## Cambios detallados

### 1. `memory.py` — `build_system_prompt()`

**Eliminar** los bloques dinámicos:
- Búsqueda semántica (`semantic_memory.search`)
- `[ASSISTANT_MEMORY]` (long_term_memories legacy)
- `[CONVERSATION_SUMMARY]`
- `book_context`
- `extra_context`

Eliminar parámetros: `extra_context`, `semantic_memory`, `user_query`, `conv_summary_memory`, `book_context`.

Queda solo: core prompt + reglas + prompts guardados.

### 2. `memory.py` — nuevo método `build_user_message()`

```python
def build_user_message(self, user_input: str, book_context: str = "",
                       semantic_memory=None, conv_summary=None,
                       extra_context: str = "") -> str:
    parts = [user_input]
    context_lines = []

    if book_context:
        context_lines.append(book_context)

    if semantic_memory:
        user_mem = semantic_memory.search(user_input, n_results=3, scope="user")
        if user_mem:
            lines = "\n".join(f"- {m.content}" for m in user_mem)
            context_lines.append(f"[USER_MEMORY]\n{lines}")

        asst_mem = semantic_memory.search(user_input, n_results=3, scope="assistant")
        if asst_mem:
            lines = "\n".join(f"- {m.content}" for m in asst_mem)
            context_lines.append(f"[ASSISTANT_MEMORY]\n{lines}")

    if conv_summary:
        block = conv_summary.build_context_block()
        if block:
            context_lines.append(block)

    if extra_context:
        context_lines.append(f"[USER_MEMORY]\n{extra_context}")

    if context_lines:
        parts.append("--- Contexto adicional ---")
        parts.extend(context_lines)

    return "\n\n".join(parts)
```

Nota: `semantic_memory.search()` ya acepta `scope=` — no necesita cambios.

### 3. `memory.py` — `get_history()`

Eliminar parámetros: `extra_context`, `semantic_memory`, `user_query`, `conv_summary_memory`, `book_context`.

Simplificar a:

```python
def get_history(self, max_messages: int = 10) -> list[Message]:
    self._require_conn()
    system_prompt = self.build_system_prompt()
    return deque.build_history(self._conn, max_messages, system_prompt)
```

### 4. `console_chat.py` — chat loop

```python
# Una sola búsqueda semántica (n_results=3)
memories_user = sem.search(user, n_results=3, scope="user")
memories_asst = sem.search(user, n_results=3, scope="assistant")

# Book context
book_context = book_mem.build_context(user, n_results=3, max_chars=3000) if book_mem else ""

# System prompt estático + user message con contexto
full_system = mem.build_system_prompt()
user_message = mem.build_user_message(
    user,
    book_context=book_context,
    conv_summary=conv_summary,
    extra_context=extra_context_from_memories,
)

# Al LLM
llm.chat(system=full_system, user=user_message, history=history)

# A DB (solo el mensaje crudo)
mem.add_message("user", user)
sem.remember(user)
sem.remember(response, source_role="assistant")
```

Eliminar la llamada a `sem.search()` duplicada dentro de `build_system_prompt()`. Las búsquedas se hacen UNA vez en console_chat.

### 5. `console_chat.py` — `/show_prompt`

Debe mostrar solo el system prompt estático (sin contexto dinámico).

### 6. `deque.py`

No necesita cambios. `deque.build_history()` recibe un `system_prompt` string ya construido. Como `get_history()` ya no incluye contexto dinámico, el history del sliding window se mantiene limpio.

## Orden en el user message

```
{mensaje del usuario}

--- Contexto adicional ---
[BOOK_CONTEXT: Raiders_of_the_Serpent_Sea]
## History of Grimnir (pag 4-20)
...

[USER_MEMORY]
- Le gusta la ciencia ficción

[ASSISTANT_MEMORY]
- El asistente explicó conceptos de arquitectura limpia

[CONVERSATION_SUMMARY]
- El usuario preguntó sobre diseño de software
```

## Regla: el contexto adicional nunca alimenta memorias

```
Usuario escribe: "Cuales son los dioses de Grimnir"
  ↓
Al LLM: "Cuales son los dioses de Grimnir\n\n--- Contexto adicional ---\n[BOOK_CONTEXT]..."
  ↓
sem.remember() evalúa SOLO: "Cuales son los dioses de Grimnir"
  ↓
add_message guarda SOLO: "Cuales son los dioses de Grimnir"
```

## Archivos a modificar

| Archivo | Cambio |
|---|---|
| `src/memory.py` | `build_system_prompt()` — eliminar parámetros dinámicos y bloques |
| `src/memory.py` | Nuevo `build_user_message()` |
| `src/memory.py` | `get_history()` — eliminar parámetros, simplificar |
| `examples/console_chat.py` | Reemplazar armado de system prompt por `build_user_message()` |
| `examples/console_chat.py` | Eliminar búsqueda semántica duplicada en extra_context |
| `examples/console_chat.py` | Agregar búsqueda separada para `scope="assistant"` |
| `examples/console_chat.py` | `/show_prompt` solo muestra system prompt estático |

## No cambia

- `BookMemory.build_context()` — misma firma
- `SemanticMemory.search()` — ya tiene `scope=`
- `ConversationSummaryMemory` — misma API
- `deque.py` — no tocar
- `db.py` — no tocar
- Tabla `messages` — sigue guardando solo mensajes crudos
