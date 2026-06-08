# v_memory_manager

Persistent memory manager for LLM conversations. SQLite-backed, zero external dependencies.

v0.3.0 — SQLite-backed semantic memories with ChromaDB search.

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
| `build_system_prompt(extra_context="")` | Core prompt + saved prompts + long-term memories + memory rules |
| `get_system_prompt()` | Returns only the core prompt (`_active`) |
| `count_messages()` | Total messages stored |

Each `Message` includes an `id` field matching the DB row ID.

### Prompts

| Method | Description |
|--------|-------------|
| `save_prompt(name, content, orden=0)` | Save/update a named prompt with sort order |
| `load_prompt(name)` | Load a prompt by name (returns `None` if missing) |
| `list_prompts()` | List saved prompt names (excludes `_active`) |
| `delete_prompt(name)` | Delete a saved prompt |

### Long-term memories (SQLite)

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

Detects rememberable information from both user and assistant messages using rules (no LLM). Stores in ChromaDB (search index) + SQLite (source of truth).

### Requirements

```bash
pip install chromadb
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

# Save automatically (returns None if not rememberable)
mid = sem.remember("Prefiero Python")
if mid:
    print(f"Saved: {mid}")

# Force save
sem.remember_force("Al usuario le gusta Python", tags=["python"])

# Search by similarity
results = sem.search("lenguaje de programacion")
for r in results:
    print(f"  {r.content}")

# Soft delete (SQLite status='deleted', ChromaDB removed)
sem.forget(memory_id)

# Hard delete (removed from both DBs)
sem.purge(memory_id)

# List from SQLite
sem.list_memories()

# Get from SQLite by memory_id
sem.get_memory(memory_id)
```

### Rule-based detection (no LLM)

| Step | Description |
|---|---|
| Noise filter | Ignores greetings, short messages (< 8 chars) |
| Explicit commands | `/remember`, `recuerda que`, `guarda esto` |
| Pattern hints | `prefiero` → preference, `mi proyecto` → project_fact |
| Scoring | 0.0–1.0 based on type, length, personal markers, tech terms |
| Tag extraction | Technical (`python`, `wsl`) + general (`comida`, `gustos`) |

### Status flow

| Confidence | Status | Retrieved in search |
|---|---|---|
| >= 0.75 | `active` | Yes |
| 0.40–0.74 | `pending_review` | Yes |
| < 0.40 | — | Not saved |

Both user and assistant messages are analyzed and saved when they contain rememberable information (preferences, project facts, environment details, etc.).

### Operations

| Operation | SQLite | ChromaDB |
|---|---|---|
| `remember` | INSERT | add |
| `archive` | UPDATE status | update status |
| `forget` | UPDATE status='deleted' | delete (soft) |
| `purge` | DELETE | delete (hard) |
| `get_memory` | SELECT by memory_id | — |
| `list_memories` | SELECT by namespace | — |
| `search` | — | query |

## System prompt injection

When relevant memories are found, they are injected into the system prompt:

```
[USER_MEMORY]
- Preferencia del usuario: Le gustan las galletas de chocolate

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
long_term_memories   -- id, content, tags, weight, created_at
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
| `importance` | REAL | Relevance 0–1 |
| `memory_type` | TEXT | `explicit`, `preference`, etc. |
| `status` | TEXT | `active`, `pending_review`, `archived`, `deleted` |
| `source` | TEXT | `auto` or `manual` |
| `source_message_ids` | TEXT | Traceability to source messages |
| `created_at` | TEXT | ISO 8601 |
| `updated_at` | TEXT | ISO 8601 |
| `owner_type` | TEXT | (roleplay, unused) |
| `character_id` | TEXT | (roleplay, unused) |
| `source_role` | TEXT | (roleplay, unused) |
| `canon_status` | TEXT | (roleplay, unused) |
| `fact_key` | TEXT | (roleplay, unused) |
| `fact_value` | TEXT | (roleplay, unused) |
| `scene_id` | TEXT | (roleplay, unused) |
| `world_id` | TEXT | (roleplay, unused) |
| `expires_scope` | TEXT | (roleplay, unused) |

## Schema migration

| Version | Changes |
|---|---|
| v1 (0.1.0) | Initial: messages, prompts, configurations |
| v2 (0.2.0) | Added `orden` to prompts, `long_term_memories` table |
| v3 (0.3.0) | Added `semantic_memories` table |

Auto-migrated on `load_memory_db()`.

## Console chat commands

| Command | Description |
|---|---|
| `/remember <text>` | Force save a memory |
| `/memories` | List all memories |
| `/search <query>` | Semantic search |
| `/forget <id>` | Delete a memory |
| `/clear` | Clear message history |
| `/prompt <text>` | Change system prompt |
| `/save <name>` | Save prompt |
| `/load <name>` | Load prompt |
| `/show_prompt` | Show built system prompt |
| `/exit` | Exit |

Memories are saved automatically from both user and assistant messages when they contain rememberable information.

## Integration with v_llama

```python
from v_memory_manager import MemoryManager
from v_llama import VLLaMA

mem = MemoryManager()
mem.create_memory_db("chat.db")

llm = VLLaMA()
llm.load_model("model.gguf")

system_prompt = mem.build_system_prompt()
user = "Hola"

history = [
    {"role": m.role, "content": m.content}
    for m in mem.get_history() if m.role != "system"
]
res = llm.chat(system=system_prompt, user=user, history=history)

mem.add_message("user", user)
mem.add_message("assistant", res.content)
```

See `examples/console_chat.py` for a complete interactive chat with model selection, session management, streaming, and semantic memory.
