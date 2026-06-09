# v_memory_manager — Plan v5: Evolución de la Memoria Semántica

## 1. Diagnóstico del Estado Actual

Tras analizar todo el código fuente (`semantic_memory.py`, `memory.py`, `db.py`, `deque.py`, `console_chat.py`, `memory_models.py`), la infraestructura base del plan v4 está implementada y funcional. Pero hay **problemas reales en el flujo** que impactan directamente la calidad de las memorias que se guardan y recuperan.

### 1.1. Problemas encontrados

#### P1 — Las memorias del asistente contaminan el contexto
En `console_chat.py:347-353`, se llama `sem.remember(full)` sobre la respuesta del asistente. Pero `analyze_text()` fue diseñada para detectar frases del **usuario** ("prefiero", "me gusta", "mi proyecto"). El resultado es que:
- Respuestas genéricas del LLM como "Claro, puedo ayudarte con eso" se intentan analizar innecesariamente.
- Si el LLM dice "Te sugiero usar Python", el sistema lo detecta como `project_fact` y lo guarda como si fuera una preferencia del usuario.
- No hay distinción `source_role` al guardar, así que al buscar no se sabe si es un hecho del usuario o un consejo del asistente.

#### P2 — La deduplicación es binaria y bloquea actualizaciones
En `_store()` (línea 265-277), si ChromaDB encuentra una memoria con distancia `< 0.1`, devuelve el ID existente sin hacer nada. Esto significa:
- "Mi color favorito es el rojo" → se guarda.
- "Mi color favorito ahora es el azul" → distancia semántica ~0.08 → **se ignora silenciosamente**.
- El usuario queda con información desactualizada para siempre.

#### P3 — `search()` no filtra por `user_id` en la práctica
El filtro `where` en `search()` (línea 356-362) incluye `user_id`, pero en `_store()` se guarda como metadata. El problema es que ChromaDB con el embedding `all-MiniLM-L6-v2` busca por similitud de documento, y el `user_id` no siempre se verifica con rigor si hay colisiones de namespace entre instancias.

#### P4 — `search_by_tags()` carga TODAS las memorias
En línea 366-369, `search_by_tags()` llama `list_memories(limit=1000)` y filtra en Python. Esto no escala. Debería hacer un `SELECT` con `WHERE tags LIKE` en SQLite.

#### P5 — `importance` nunca se actualiza
El campo `importance` se inicializa en `0.5` y nunca cambia. Cada vez que se recupera una memoria relevante, debería subir ligeramente (decay inverso). Es un campo muerto.

#### P6 — `pending_review` no tiene flujo de revisión
Las memorias con confianza 0.40–0.74 se guardan como `pending_review`, pero:
- Se retornan en `search()` igual que las `active`.
- No hay ningún comando ni mecanismo para revisarlas y aprobarlas/rechazarlas.
- En la práctica, `pending_review` ≡ `active`.

#### P7 — `long_term_memories` y `semantic_memories` son redundantes
`long_term_memories` (plan v2) guarda facts estáticos en SQLite. `semantic_memories` (plan v3/v4) hace lo mismo pero con ChromaDB de búsqueda. No hay migración ni unificación. Son dos sistemas paralelos que generan confusión sobre dónde guardar qué.

---

## 2. Mejoras Propuestas (Solo Valor Real)

### Mejora A — Separar análisis por rol (user vs assistant)

**Problema que resuelve:** P1

**Qué hacer:**
- Añadir parámetro `source_role: str = "user"` a `remember()`.
- Para mensajes del asistente, usar un analizador distinto que solo busque hechos explícitos sobre el usuario (ej. "Tu nombre es Pablo", "Te gusta Python"), no patrones vagos.
- Guardar `source_role` en la metadata de ChromaDB y en SQLite (la columna ya existe pero no se usa).
- En `search()`, poder filtrar por `source_role` opcionalmente.

**Cambios concretos:**
```python
# semantic_memory.py
def remember(self, text: str, source: str = "auto", 
             msg_ids: str = "", source_role: str = "user") -> Optional[str]:
    if source_role == "assistant":
        result = analyze_assistant_text(text)  # análisis más estricto
    else:
        result = self.analyze(text)
    ...
```

```python
def analyze_assistant_text(text: str) -> AnalysisResult:
    """Solo extrae hechos explícitos que el LLM afirma sobre el usuario."""
    # Solo detectar: "tu nombre es X", "te gusta X", "usas X"
    # NO detectar: "te sugiero", "puedes probar", "es una buena idea"
    ...
```

---

### Mejora B — Resolución de conflictos con archivado automático

**Problema que resuelve:** P2

**Qué hacer:**
Cuando `_store()` encuentra una memoria similar (distancia < 0.3), en lugar de ignorar la nueva, evaluar si es una **actualización**:

1. Si distancia < 0.05 → realmente duplicado → ignorar (retornar ID existente).
2. Si distancia 0.05–0.30 y mismo `memory_type` → posible actualización:
   - Archivar la memoria vieja (`status='archived'`).
   - Guardar la nueva como `active`.
   - Log: `"Memoria actualizada: {old_id} → {new_id}"`.
3. Si distancia > 0.30 → es nueva → guardar normalmente.

**Cambios concretos:**
```python
# semantic_memory.py → _store()
existing = coll.query(query_texts=[result.content], n_results=1, where={...})
if existing["distances"] and existing["distances"][0]:
    dist = existing["distances"][0][0]
    if dist < 0.05:
        return existing["ids"][0][0]  # duplicado real
    if dist < 0.30:
        old_id = existing["ids"][0][0]
        old_meta = existing["metadatas"][0][0]
        if old_meta.get("memory_type") == result.memory_type:
            self.archive(old_id)  # archivar la vieja
            # continuar guardando la nueva
```

---

### Mejora C — `search_by_tags()` eficiente en SQLite

**Problema que resuelve:** P4

**Qué hacer:**
Reemplazar la implementación actual que carga 1000 memorias por un query SQL directo.

```python
def search_by_tags(self, tags: list[str]) -> list[MemoryRecord]:
    placeholders = " OR ".join("tags LIKE ?" for _ in tags)
    params = [f"%{t}%" for t in tags]
    rows = self.conn.execute(
        f"SELECT * FROM semantic_memories WHERE namespace=? AND status='active' AND ({placeholders})",
        [self._namespace] + params,
    ).fetchall()
    return [self._row_to_record(r) for r in rows]
```

---

### Mejora D — Flujo real de `pending_review`

**Problema que resuelve:** P6

**Qué hacer:**
- `search()` solo retorna memorias `active` por defecto.
- Añadir método `review_pending(limit=10) -> list[MemoryRecord]` que lista memorias pendientes.
- Añadir método `approve(memory_id)` y `reject(memory_id)`.
- Añadir comando `/review` en `console_chat.py` que muestra las pendientes y permite aprobar/rechazar interactivamente.

```python
def review_pending(self, limit: int = 10) -> list[MemoryRecord]:
    rows = self.conn.execute(
        "SELECT * FROM semantic_memories WHERE namespace=? AND status='pending_review' ORDER BY created_at DESC LIMIT ?",
        (self._namespace, limit),
    ).fetchall()
    return [self._row_to_record(r) for r in rows]

def approve(self, memory_id: str) -> None:
    self.conn.execute(
        "UPDATE semantic_memories SET status='active', updated_at=datetime('now') WHERE memory_id=?",
        (memory_id,),
    )
    self.conn.commit()
    # actualizar chromadb metadata
    row = self.conn.execute("SELECT chroma_id FROM semantic_memories WHERE memory_id=?", (memory_id,)).fetchone()
    if row and row[0]:
        self.collection.update(ids=[row[0]], metadatas=[{"status": "active"}])

def reject(self, memory_id: str) -> None:
    self.forget(memory_id)  # reutilizar soft delete
```

**Cambio en `search()`:**
```python
# Cambiar el filtro de status:
{"status": {"$eq": "active"}}  # antes: {"$in": ["active", "pending_review"]}
```

---

### Mejora E — Boost de `importance` al recuperar

**Problema que resuelve:** P5

**Qué hacer:**
Cada vez que `search()` retorna una memoria y esa memoria se inyecta en el prompt, incrementar `importance` ligeramente. Esto crea un ranking orgánico: las memorias que el usuario consulta frecuentemente suben de prioridad.

```python
def search(self, query: str, n_results: int = 5) -> list[MemoryRecord]:
    ...
    records = self._to_records(results)
    # boost importance de las memorias recuperadas
    for r in records:
        if r.memory_id:
            self._boost_importance(r.memory_id, increment=0.02, cap=1.0)
    return records

def _boost_importance(self, memory_id: str, increment: float = 0.02, cap: float = 1.0) -> None:
    self.conn.execute(
        "UPDATE semantic_memories SET importance = MIN(importance + ?, ?) WHERE memory_id = ?",
        (increment, cap, memory_id),
    )
    # commit en batch, no individual, para no saturar I/O
```

**Uso futuro:** cuando haya demasiadas memorias, ordenar por `importance DESC` para priorizar las más consultadas en el prompt.

---

### Mejora F — Unificar `long_term_memories` → `semantic_memories`

**Problema que resuelve:** P7

**Qué hacer:**
Las `long_term_memories` son básicamente memorias manuales con tags y peso. La tabla `semantic_memories` ya tiene `tags`, `importance` (equivale a `weight`), y `source` ("manual"). Se puede migrar.

1. Crear función `migrate_long_term_to_semantic(conn, semantic_memory)` que:
   - Lee todas las `long_term_memories`.
   - Las inserta en `semantic_memories` con `source="legacy"`, `memory_type="explicit"`, `status="active"`.
   - Las indexa en ChromaDB para búsqueda semántica.
2. En `build_system_prompt()`, reemplazar la lectura de `long_term_memories` por `semantic_memory.search()`.
3. Marcar `long_term_memories` como **deprecated** (mantener CRUD por compatibilidad, pero las nuevas van a `semantic_memories`).

**Esto NO elimina la tabla.** Solo redirige el flujo para que todo pase por un solo sistema.

---

## 3. Mejoras Descartadas (No aportan valor real ahora)

| Idea | Por qué no |
|---|---|
| **Extracción con LLM** | Agrega una dependencia fuerte (requiere modelo cargado + latencia). Las reglas heurísticas funcionan razonablemente bien para el caso de uso actual. Se puede agregar después como capa opcional. |
| **Consolidación asíncrona** | Compleja de implementar correctamente (merge de memorias con LLM, control de pérdida de información). Prematura: primero hay que tener suficientes memorias acumuladas para que sea un problema real. |
| **MemoryRouter** | Sobreingeniería para el estado actual. El roleplay ni siquiera está implementado. Cuando exista `roleplay_memory.py`, se agrega el router. |

---

## 4. Plan de Implementación (Orden de prioridad)

### Fase 1 — Fixes de integridad (impacto alto, esfuerzo bajo)
- [ ] **B:** Resolución de conflictos en `_store()` — evita datos desactualizados
- [ ] **A:** Separar análisis user/assistant — evita contaminar memorias
- [ ] **C:** `search_by_tags()` eficiente — fix de rendimiento

### Fase 2 — Flujo completo de revisión (impacto medio, esfuerzo medio)
- [ ] **D:** `pending_review` funcional con `/review`, `approve()`, `reject()`
- [ ] **D:** Cambiar `search()` para solo retornar `active`

### Fase 3 — Ranking orgánico (impacto medio, esfuerzo bajo)
- [ ] **E:** Boost de `importance` al recuperar memorias
- [ ] Ordenar resultados de búsqueda por `importance` como factor secundario

### Fase 4 — Unificación (impacto alto, esfuerzo medio)
- [ ] **F:** Migración de `long_term_memories` → `semantic_memories`
- [ ] **F:** `build_system_prompt()` usa `SemanticMemory` en lugar de leer `long_term_memories` directamente

---

## 5. Verificación

| Mejora | Test |
|---|---|
| A | Enviar respuesta genérica del LLM → no debe guardarse como memoria |
| B | Guardar "favorito es rojo", luego "favorito es azul" → solo "azul" queda activa |
| C | `search_by_tags(["python"])` con 500+ memorias → respuesta < 50ms |
| D | `/review` muestra pendientes, `/approve` las pasa a active |
| E | Buscar "python" 5 veces → importance de esa memoria sube |
| F | Memorias legacy aparecen en búsqueda semántica |
