# v_memory_manager

Persistent memory manager for LLM conversations. SQLite-backed, zero external dependencies.

v0.5.0 — Optional spaCy NLP engine, semantic fact extraction, lemmatization fallbacks, smart conflict resolution, review flow.

## Quick start

```python
from v_memory_manager import MemoryManager

mem = MemoryManager()
mem.create_memory_db("chat.db")

mem.add_message("user", "Hola")
mem.add_message("assistant", "Hola!")

history = mem.get_history()
for m in history:
    print(f"{m.role}: {m.content}")
```

## API

### Lifecycle

| Method | Description |
|--------|-------------|
| `create_memory_db(path, default_system_path=None)` | Creates new DB, loads default system prompt |
| `load_memory_db(path)` | Loads existing DB, auto-migrates schema if needed |
| `drop_memory_db()` | Deletes the .db file |
| `clear_memory_db()` | Truncates messages only (prompts + memories survive) |
| `close()` | Closes the connection |

### Messages (sliding-window)

| Method | Description |
|--------|-------------|
| `add_message(role, content)` | `user`/`assistant` → messages table. `system` → updates core prompt |
| `get_history(max_messages=10, extra_context="")` | Returns `build_system_prompt()` + last N-1 messages |
| `build_system_prompt(extra_context="", semantic_memory=None, user_query="")` | Core prompt + semantic memories (RAG) + saved prompts + long-term memories + memory rules |
| `get_system_prompt()` | Returns only the core prompt (`_active`) |
| `count_messages()` | Total messages stored |

`build_system_prompt()` supports optional RAG-style injection: pass a `SemanticMemory` instance and the current `user_query` to automatically retrieve and inject the top 5 relevant memories.

### Prompts

| Method | Description |
|--------|-------------|
| `save_prompt(name, content, orden=0)` | Save/update a named prompt with sort order |
| `load_prompt(name)` | Load a prompt by name (returns `None` if missing) |
| `list_prompts()` | List saved prompt names (excludes `_active`) |
| `delete_prompt(name)` | Delete a saved prompt |

### Long-term memories (SQLite) — deprecated

> **Note:** Use `SemanticMemory` for new memories. `long_term_memories` is kept for backward compatibility. Use `migrate_long_term_to_semantic()` to migrate existing data.

| Method | Description |
|--------|-------------|
| `add_long_term_memory(content, tags="", weight=1.0)` | Insert a memory |
| `get_long_term_memories(tag=None, min_weight=None)` | List memories, optional filter |
| `delete_long_term_memory(memory_id)` | Delete a memory |
| `count_long_term_memories()` | Total memories stored |

### Configurations

| Method | Description |
|--------|-------------|
| `get_config(key, default=None)` | Get a config value |
| `set_config(key, value)` | Set a config value |
| `all_configs()` | All configs as dict |

## Semantic Memory (ChromaDB + SQLite)

Detects rememberable information from user and assistant messages using rules (no LLM). Stores in ChromaDB (search index) + SQLite (source of truth).

### Optional spaCy NLP Engine

The package includes an optional advanced NLP layer using **spaCy**. If spaCy is not installed or the model is not found, the system transparently falls back to regular expression parsing.

When spaCy is enabled (using the `es_core_news_sm` model), it adds:
- **Lemmatization**: Verbs are normalized to their base dictionary form. For example, "me gustaban", "me gustaría", and "me gusta" all match the lemma `gustar`.
- **Named Entity Recognition (NER)**: Automatically extracts names of people, organizations, and locations, turning them into search tags (e.g., `persona:Juan`, `ubicacion:Mexico`).
- **Semantic Fact Extraction**: Uses dependency grammar parsing to extract subject-verb-object triples, storing them in the `fact_key` and `fact_value` database columns (e.g., "Mi color favorito es el azul" -> key: `color`, value: `azul`).

### Requirements

```bash
pip install chromadb
```

To enable the optional advanced NLP/spaCy engine:

```bash
pip install spacy
python -m spacy download es_core_news_sm
```

Or install with the `nlp` package option:

```bash
pip install .[nlp]
```

### Quick start

```python
from v_memory_manager import SemanticMemory, MemoryManager

mem = MemoryManager()
mem.create_memory_db("chat.db")

sem = SemanticMemory(
    persist_dir="./chroma_db",
    sqlite_conn=mem._conn,
)

# Analyze without saving
result = sem.analyze("Me gustan las galletas de chocolate")
print(result.should_remember, result.confidence)

# Save user message (auto-detected)
mid = sem.remember("Prefiero Python")

# Save assistant message (strict analysis, detects self-facts & user-facts)
mid = sem.remember("Mi color favorito es el rojo", source_role="assistant")

# Force save
sem.remember_force("Al usuario le gusta Python", tags=["python"])

# Search by similarity (only returns active memories)
results = sem.search("lenguaje de programacion")
for r in results:
    print(f"  {r.content}")

# Search by tags (efficient SQL query)
results = sem.search_by_tags(["python", "comida"])

# Soft delete (SQLite status='deleted', ChromaDB removed)
sem.forget(memory_id)

# Hard delete (removed from both DBs)
sem.purge(memory_id)

# Archive (keeps in SQLite, hidden from search)
sem.archive(memory_id)

# List from SQLite
sem.list_memories()

# Get from SQLite by memory_id
sem.get_memory(memory_id)
```

### Role-aware analysis

User and assistant messages are analyzed differently:

| Source | Analysis | What it detects |
|---|---|---|
| `source_role="user"` | Full heuristic scoring | Preferences, project facts, environment, instructions |
| `source_role="assistant"` | Strict pattern matching | Facts about the user ("tu nombre es...") + assistant self-facts ("mi color favorito es...") |

Assistant messages are filtered strictly to avoid saving generic responses ("Claro, puedo ayudarte") or suggestions ("Te sugiero usar Python"). Only explicit facts are remembered.

**Assistant memory types:**

| Type | Example | Stored as |
|---|---|---|
| `preference` | "Tu nombre es Pablo" | `Preferencia del usuario: ...` |
| `assistant_preference` | "Mi color favorito es el rojo" | `Memoria del asistente: ...` |

### Smart conflict resolution

When storing a new memory, the system checks for existing similar memories:

| Distance | Action |
|---|---|
| < 0.05 | **Duplicate** — returns existing ID, no change |
| 0.05–0.30 (same type) | **Update** — archives old memory, saves new one |
| > 0.30 | **New** — saves normally |

This prevents stale data: if the user says "Mi color favorito es el rojo" and later "Mi color favorito ahora es el azul", the old memory is archived and the new one becomes active.

### Detection rules (no LLM)

| Step | Description |
|---|---|
| Noise filter | Ignores greetings, short messages (< 8 chars) |
| Explicit commands | `/remember`, `recuerda que`, `guarda esto` |
| Pattern hints | `prefiero` → preference, `mi proyecto` → project_fact |
| Scoring | 0.0–1.0 based on type, length, personal markers, tech terms |
| Tag extraction | Technical (`python`, `wsl`) + general (`comida`, `gustos`) |

### Status flow

| Confidence | Status | Retrieved in `search()` |
|---|---|---|
| >= 0.75 | `active` | Yes |
| 0.40–0.74 | `pending_review` | No (must be approved first) |
| < 0.40 | — | Not saved |

### Review flow

Memories with `pending_review` status are NOT returned in `search()`. They must be explicitly reviewed:

```python
# List pending memories
pending = sem.review_pending(limit=10)

# Approve (status → active, now searchable)
sem.approve(memory_id)

# Reject (soft delete)
sem.reject(memory_id)
```

### Importance ranking

Every time a memory is retrieved via `search()`, its `importance` score is boosted by +0.02 (capped at 1.0). This creates an organic ranking: frequently accessed memories rise in priority.

### Operations

| Operation | SQLite | ChromaDB |
|---|---|---|
| `remember` | INSERT | add |
| `archive` | UPDATE status | update status |
| `forget` | UPDATE status='deleted' | delete (soft) |
| `purge` | DELETE | delete (hard) |
| `approve` | UPDATE status='active' | update status |
| `reject` | UPDATE status='deleted' | delete (soft) |
| `get_memory` | SELECT by memory_id | — |
| `list_memories` | SELECT by namespace | — |
| `review_pending` | SELECT status='pending_review' | — |
| `search` | — | query (active only) |
| `search_by_tags` | SELECT WHERE tags LIKE | — |

## Migration

Migrate legacy `long_term_memories` to the unified `semantic_memories` system:

```python
from v_memory_manager import migrate_long_term_to_semantic

count = migrate_long_term_to_semantic(conn, semantic_memory)
print(f"Migrated {count} memories")

# Safe to run multiple times — skips already migrated entries
```

## System prompt injection

When relevant memories are found, they are injected into the system prompt:

```
[USER_MEMORY]
- Preferencia del usuario: Le gustan las galletas de chocolate
- Memoria del asistente: Mi color favorito es el rojo

[ASSISTANT_MEMORY]
- (legacy long-term memories, if any)

[USO DE MEMORIA]
- USER_MEMORY describe al usuario que está conversando.
- ASSISTANT_MEMORY describe al asistente.
- Si el usuario pregunta por "mi", "me", "yo", "mis gustos", "mi nombre" o "mi favorito", revisa USER_MEMORY primero.
- Si USER_MEMORY contiene la respuesta, responde directamente usando esa memoria.
- No digas "como modelo de lenguaje no tengo preferencias" cuando el usuario pregunta por sus propias preferencias.
```

## Data model (SQLite)

```sql
messages             -- id, role (user/assistant), content, created_at
prompts              -- id, name (unique), content, orden, created_at
long_term_memories   -- id, content, tags, weight, created_at (deprecated)
configurations       -- key (PK), value
semantic_memories    -- unified table for ALL semantic memories
```

### `semantic_memories` columns

| Column | Type | Description |
|---|---|---|
| `memory_id` | TEXT UNIQUE | Stable ID (`mem_abc123`) |
| `chroma_id` | TEXT UNIQUE | ChromaDB document ID |
| `namespace` | TEXT | `normal` |
| `scope` | TEXT | `user`, `project`, etc. |
| `content` | TEXT | Memory content |
| `original_text` | TEXT | Raw source message |
| `tags` | TEXT | Comma-separated tags |
| `confidence` | REAL | Detection confidence 0–1 |
| `importance` | REAL | Relevance 0–1 (boosted on retrieval) |
| `memory_type` | TEXT | `explicit`, `preference`, `assistant_preference`, etc. |
| `status` | TEXT | `active`, `pending_review`, `archived`, `deleted` |
| `source` | TEXT | `auto`, `manual`, or `legacy` |
| `source_role` | TEXT | `user` or `assistant` |
| `source_message_ids` | TEXT | Traceability to source messages |
| `created_at` | TEXT | ISO 8601 |
| `updated_at` | TEXT | ISO 8601 |
| `owner_type` | TEXT | (roleplay, reserved) |
| `character_id` | TEXT | (roleplay, reserved) |
| `canon_status` | TEXT | (roleplay, reserved) |
| `fact_key` | TEXT | Semantic key/subject extracted from memory (spaCy) |
| `fact_value` | TEXT | Semantic value/object extracted from memory (spaCy) |
| `scene_id` | TEXT | (roleplay, reserved) |
| `world_id` | TEXT | (roleplay, reserved) |
| `expires_scope` | TEXT | (roleplay, reserved) |

## Schema migration

| Version | Changes |
|---|---|
| v1 (0.1.0) | Initial: messages, prompts, configurations |
| v2 (0.2.0) | Added `orden` to prompts, `long_term_memories` table |
| v3 (0.3.0) | Added `semantic_memories` table |
| v4 (0.4.0) | Smart dedup, role-aware analysis, review flow, importance ranking, migration |
| v5 (0.5.0) | Optional spaCy NLP engine, lemmatization fallbacks, NER tags, semantic fact extraction |

Auto-migrated on `load_memory_db()`.

## Console chat commands

| Command | Description |
|---|---|
| `/remember <text>` | Force save a memory |
| `/memories` | List all memories |
| `/search <query>` | Semantic search |
| `/forget <id>` | Delete a memory |
| `/review` | Review pending memories (approve/reject) |
| `/clear` | Clear message history |
| `/prompt <text>` | Change system prompt |
| `/save <name>` | Save prompt |
| `/load <name>` | Load prompt |
| `/show_prompt` | Show built system prompt |
| `/exit` | Exit |

Memories are saved automatically from both user and assistant messages. User messages use full heuristic analysis. Assistant messages use strict pattern matching to avoid saving generic responses.

## Integration with v_llama

```python
from v_memory_manager import MemoryManager, SemanticMemory
from v_llama import VLLaMA

mem = MemoryManager()
mem.create_memory_db("chat.db")

sem = SemanticMemory(
    persist_dir="./chroma_db",
    sqlite_conn=mem._conn,
)

llm = VLLaMA()
llm.load_model("model.gguf")

user = "Hola"

# RAG-style: pass semantic_memory + user_query for dynamic retrieval
system_prompt = mem.build_system_prompt(semantic_memory=sem, user_query=user)

history = [
    {"role": m.role, "content": m.content}
    for m in mem.get_history() if m.role != "system"
]
res = llm.chat(system=system_prompt, user=user, history=history)

mem.add_message("user", user)
mem.add_message("assistant", res.content)

# Remember with role distinction
sem.remember(user)                                    # user analysis
sem.remember(res.content, source_role="assistant")    # strict assistant analysis
```

See `examples/console_chat.py` for a complete interactive chat with model selection, session management, streaming, and semantic memory.
