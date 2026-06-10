# Plan: ConversationSummaryMemory (v3)

## Problema

Actualmente con `max_messages=10` y una conversación de 50 mensajes:

```
[system prompt]
[USER_MEMORY]
[extra_context]
[prompts guardados]
[ASSISTANT_MEMORY]
[USO DE MEMORIA]
[sliding window: msg_42–50]
```

Los mensajes **1–41** existen en DB pero el modelo **no los ve**. `SemanticMemory` captura hechos estables ("al usuario no le gustan los emojis"), pero **no hay continuidad conversacional** — qué se estaba discutiendo, qué decisiones se tomaron, qué problemas se encontraron.

## Solución propuesta

Agregar una **memoria episódica resumida** que compacte lo que quedó fuera del sliding window en un bloque estructurado.

### Responsabilidad

| Módulo | Qué guarda | Persistencia |
|--------|------------|--------------|
| `SemanticMemory` | Hechos estables del usuario/asistente/proyecto | Alta (vive mucho tiempo) |
| `ConversationSummaryMemory` | Resumen temporal de la conversación previa | Media (se reescribe/compacta) |
| Sliding window | Últimos mensajes exactos | Baja (se pierde al salir de la ventana) |

Ejemplo concreto:

```
SemanticMemory:
- El usuario usa ChromaDB.
- El usuario no quiere emojis.
- El asistente se llama Juan.

ConversationSummary:
- Se estaba diseñando una clase ConversationSummaryMemory.
- Se decidió usar una tabla conversation_summary_state.
- Quedó pendiente integrar maybe_update().

Sliding window:
- Últimos 9 mensajes exactos.
```

Pirámide de memoria:

```
Últimos mensajes exactos (sliding window)
↓
Resumen conversacional (ConversationSummaryMemory)
↓
Memorias semánticas persistentes (SemanticMemory)
↓
DB completa como fuente histórica
```

## Esquema de datos

```sql
CREATE TABLE IF NOT EXISTS conversation_summary_state (
    conversation_id TEXT NOT NULL,
    user_id TEXT NOT NULL DEFAULT 'default',

    summary TEXT NOT NULL DEFAULT '',
    last_summarized_message_id INTEGER NOT NULL DEFAULT 0,
    last_summarized_created_at TEXT NOT NULL DEFAULT '',

    summary_version INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'active',

    summary_error_count INTEGER NOT NULL DEFAULT 0,
    last_error TEXT NOT NULL DEFAULT '',

    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,

    PRIMARY KEY (conversation_id, user_id)
);
```

**Columnas:**
- `conversation_id`, `user_id` — PK compuesta para multi-usuario sin colisiones
- `summary` — texto del resumen vivo
- `last_summarized_message_id`, `last_summarized_created_at` — hasta dónde cubre el resumen
- `summary_version` — permite saber si se generó con reglas viejas al migrar formato
- `status` — `active` / `disabled` / `archived`
- `summary_error_count`, `last_error` — visibilidad de fallos silenciosos del summarizer

## Clase

```python
class ConversationSummaryMemory:
    """
    Mantiene un resumen vivo de la parte de la conversación
    que ya quedó fuera del sliding window.

    No reemplaza SemanticMemory.
    No guarda hechos permanentes.
    No borra mensajes.
    Solo compacta continuidad conversacional.
    """

    def __init__(
        self,
        sqlite_conn,
        conversation_id: str,
        user_id: str = "default",
        message_table: str = "messages",
        max_messages: int = 10,
        reserved_system_messages: int = 1,
        summarize_margin: int = 4,
        max_summary_chars: int = 3000,
        summarizer=None,
    ):
        self.window_size = max_messages - reserved_system_messages
        ...
```

**Nota:** `window_size = max_messages - reserved_system_messages` para que quede claro que el system prompt ocupa 1 slot.

## API

| Método | Descripción |
|--------|-------------|
| `get_summary() -> str` | Devuelve el resumen actual |
| `get_last_summarized_message_id() -> int` | Último ID de mensaje ya resumido |
| `get_state() -> dict` | Estado completo para debug (largo, fechas, versión, status, errores) |
| `build_context_block() -> str` | Bloque `[CONVERSATION_SUMMARY]` listo para inyectar |
| `maybe_update() -> bool` | Decide si hay que resumir y lo hace |
| `update_summary(messages: list[dict]) -> None` | Ejecuta el resumen contra un lote de mensajes |
| `reset() -> None` | Limpia TODO: summary, errores, fechas, last_summarized |
| `flush() -> bool` | (P5) Resumir al cerrar sesión sin esperar próximo turno |

## Política de `maybe_update()`

```python
def maybe_update(self) -> bool:
    state = self.get_state()
    if state.get("status") != "active":
        return False

    cutoff_id = self._first_sliding_window_message_id() - 1
    from_id = self.get_last_summarized_message_id() + 1

    if cutoff_id < from_id:
        return False

    messages = self._get_messages_range(from_id, cutoff_id)

    # excluir mensajes system del lote
    messages = [m for m in messages if m.get("role") != "system"]

    if len(messages) < self.summarize_margin:
        return False

    self.update_summary(messages)
    return True
```

### `_first_sliding_window_message_id()` debe replicar `get_history()`

Debe usar la **misma lógica** que `deque.build_history()`:

1. Tomar últimos `window_size` mensajes
2. Descartar assistant inicial huérfano
3. Descartar user final incompleto
4. Devolver el primer ID del resultado

No duplicar lógica manualmente. Ideal: extraer a un helper compartido o usar el mismo `deque` module.

### `status='disabled'` hace que `maybe_update()` no haga nada

Si `status != 'active'`, `maybe_update()` retorna `False` sin tocar nada. `build_context_block()` devuelve:

```
[CONVERSATION_SUMMARY]
- Resumen desactivado.
```

## `update_summary()` con manejo de errores

```python
def update_summary(self, messages: list[dict]) -> None:
    old_summary = self.get_summary()

    try:
        new_summary = self.summarizer(
            old_summary=old_summary,
            messages=messages,
            max_chars=self.max_summary_chars,
        )
    except Exception as e:
        self._increment_error(str(e))
        return  # no romper el chat

    if not new_summary or not new_summary.strip():
        self._increment_error("summarizer returned empty")
        return

    if len(new_summary) > self.max_summary_chars:
        new_summary = new_summary[:self.max_summary_chars].rstrip() + "\n- [Resumen truncado por límite]"

    self._save_summary(new_summary, last_id=messages[-1]["id"])
```

### Callable summarizer

```python
def my_summarizer(
    old_summary: str,
    messages: list[dict],
    max_chars: int,
) -> str:
    ...
```

`messages` debe incluir `id`, `role`, `content`, `created_at` — no solo el texto. El summarizer necesita distinguir quién dijo qué.

La clase no depende de ningún modelo. Quien la usa elige qué pasarle.

## Formato del resumen

Estructurado, no narrativo:

```
Tema actual:
- ...

Estado actual:
- ...

Decisiones tomadas:
- ...

Detalles técnicos importantes:
- ...

Pendientes:
- ...
```

**Cambio respecto a v2:** se agregó "Estado actual" para capturar "en qué punto estamos", no solo decisiones y pendientes.

## Orden del prompt

```
[core prompt]
[USO DE MEMORIA]
[USER_MEMORY]
[ASSISTANT_MEMORY]
[extra_context]
[CONVERSATION_SUMMARY]
[prompts guardados]
[sliding window]
```

`extra_context` va ANTES que `CONVERSATION_SUMMARY` porque es contexto explícito e intencional del usuario para este turno.

## Reglas para `[USO DE MEMORIA]`

Agregar al bloque existente:

- `CONVERSATION_SUMMARY` resume partes anteriores de esta conversación que ya no están en el sliding window.
- El sliding window tiene más detalle reciente que `CONVERSATION_SUMMARY`.
- Si `CONVERSATION_SUMMARY` contradice los últimos mensajes, usar los últimos mensajes.
- Si `USER_MEMORY` contradice `CONVERSATION_SUMMARY`, usar `USER_MEMORY`.
- `extra_context` tiene prioridad sobre `CONVERSATION_SUMMARY` si contradicen.
- No tratar `CONVERSATION_SUMMARY` como una cita exacta; es una compresión.

## Integración en el flujo

`maybe_update()` corre **1 vez por turno**, solo antes de construir el prompt.

```
turno N:
usuario msg_50
↓
guardar mensaje usuario
↓
semantic_memory.remember_user()
↓
summary_manager.maybe_update()
↓
build_system_prompt()
↓
get_history(max_messages=10)
↓
generar respuesta (msg_51)
↓
guardar respuesta
↓
semantic_memory.remember_assistant()
```

El msg_51 no se resume hasta el próximo turno. Para MVP está bien. En P5 se puede agregar `flush()` para resumir al cerrar sesión.

## Tests

| # | Caso |
|---|------|
| 1 | Con 8 mensajes y `max_messages=10` → no resume |
| 2 | Con 15 mensajes → resume solo los que quedan fuera |
| 3 | Si `last_summarized_message_id` ya cubre cutoff → no resume |
| 4 | Si summarizer devuelve vacío → no actualizar, incrementa error |
| 5 | Si summarizer lanza excepción → no romper el chat, incrementa error |
| 6 | `reset()` → summary vacío, errores 0, last_summarized=0, fechas limpias |
| 7 | `build_context_block()` sin resumen → devuelve "Sin resumen previo" |
| 8 | Mensajes system se excluyen del lote a resumir |
| 9 | Resumen truncado por `max_summary_chars` agrega marca |
| 10 | **No huecos:** `last_summarized_message_id == first_window_id - 1` después de `maybe_update()` |
| 11 | **No resumir visibles:** mensajes en la ventana actual no se incluyen |
| 12 | `status='disabled'` → `maybe_update()` no hace nada, `build_context_block()` devuelve "desactivado" |

## Orden de implementación

| Prioridad | Item |
|-----------|------|
| **P0** | Tabla `conversation_summary_state` con PK compuesta + `summary_version` + `status` + `summary_error_count` + `last_error` |
| **P0** | Clase `ConversationSummaryMemory` con `get_summary()`, `get_last_summarized_message_id()`, `get_state()`, `build_context_block()`, `reset()` |
| **P0** | `_first_sliding_window_message_id()` con la misma lógica que `deque.build_history()` (extraer helper compartido) |
| **P1** | `maybe_update()` con cálculo de cutoff + respeto a `status` |
| **P2** | `update_summary()` con summarizer callable + manejo de errores + `max_summary_chars` |
| **P3** | Integración en `build_system_prompt()` + reglas en `[USO DE MEMORIA]` |
| **P4** | Integración en `console_chat.py` + tests de regresión (12 casos) |
| **P5** | (futuro) `flush()`, summary blocks por rangos, búsqueda semántica en Chroma, compactación de resúmenes |

## Cambios v1→v2→v3

| Aspecto | v1 | v2 | v3 |
|---------|----|----|----|
| PK | `conversation_id` solo | `(conversation_id, user_id)` | igual |
| `summary_version` + `status` | ❌ | ✅ | ✅ |
| `summary_error_count` + `last_error` | ❌ | ❌ | ✅ |
| `maybe_update()` | 2 veces por turno | 1 vez antes del prompt | igual + respeta `status` |
| Cálculo ventana | `sliding_window_size` ambiguo | `max_messages - reserved` | igual |
| `_first_window_id()` lógica | ❌ | ❌ | ✅ debe replicar `build_history()` |
| Manejo errores summarizer | ❌ | ✅ no rompe el chat | ✅ + contador de errores |
| Excluir system de resumen | ❌ | ✅ | igual |
| `get_state()` | ❌ | ✅ | ✅ + errores |
| `reset()` | genérico | status + summary + id | ✅ + errores + fechas |
| Formato resumen | 4 secciones | 4 secciones | ✅ + "Estado actual" |
| `extra_context` vs summary | sin definir | sin definir | ✅ extra_context va ANTES |
| Tests | genéricos | 9 casos | 12 casos + no huecos |
