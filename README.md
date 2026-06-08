# v_memory_manager

Persistent memory manager for LLM conversations. SQLite-backed, zero external dependencies.

v0.3.0 — added `SemanticMemory` module with ChromaDB for semantic search, rule-based memory detection, and auto-injection into system prompts.

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
| `get_history(max_messages=10)` | Returns `build_system_prompt()` + last N-1 messages |
| `build_system_prompt()` | Core prompt + saved prompts (ordered) + long-term memories |
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

Prompts are ordered by `orden ASC, name ASC` when constructing the system prompt.

### Long-term memories

| Method | Description |
|--------|-------------|
| `add_long_term_memory(content, tags="", weight=1.0)` | Insert a memory |
| `get_long_term_memories(tag=None, min_weight=None)` | List memories, optional filter by tag/weight |
| `get_long_term_memory(memory_id)` | Get a single memory by ID |
| `delete_long_term_memory(memory_id)` | Delete a memory |
| `count_long_term_memories()` | Total memories stored |

Memories are ordered by `weight ASC, created_at ASC` in the system prompt construction.

### Configurations

| Method | Description |
|--------|-------------|
| `get_config(key, default=None)` | Get a config value |
| `set_config(key, value)` | Set a config value |
| `all_configs()` | All configs as dict |

## Data model

```sql
messages           -- id, role (user/assistant), content, created_at
prompts            -- id, name (unique), content, orden, created_at
long_term_memories -- id, content, tags, weight, created_at
configurations     -- key (PK), value
```

### Schema migration (v1 → v2)

DBs created with v0.1.0 are auto-migrated on `load_memory_db()`:
- `orden INTEGER DEFAULT 0` is added to `prompts`
- `long_term_memories` table is created if missing

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

See `examples/console_chat.py` for a complete interactive chat with model selection, session management, streaming, and memory commands.

## Semantic Memory (ChromaDB)

The `SemanticMemory` module detects rememberable information from conversations using rules (no LLM) and stores it in ChromaDB for semantic search and deduplication.

### Requirements

```bash
pip install chromadb
```

### Quick start

```python
from v_memory_manager import SemanticMemory

sem = SemanticMemory(persist_dir="./chroma_db", user_id="default")

# Analyze without saving
result = sem.analyze("Me gustan las galletas de chocolate")
print(result.should_remember, result.confidence, result.memory_type)

# Analyze + save automatically
mid = sem.remember("Prefiero ejemplos en Python")
if mid:
    print(f"Saved: {mid}")

# Force save
sem.remember_force("Al usuario le gusta Python", tags=["python"])

# Search by similarity
results = sem.search("lenguaje de programacion", n_results=5)
for r in results:
    print(f"  [{r.confidence:.2f}] {r.content}")

# List / archive / delete
sem.list_memories()
sem.archive(memory_id)
sem.forget(memory_id)
```

### Rule-based detection (no LLM)

| Step | Description |
|---|---|
| Noise filter | Ignores greetings, short messages (< 8 chars) |
| Explicit commands | `/remember`, `recuerda que`, `guarda esto`, etc. |
| Pattern hints | `prefiero` → preference, `mi proyecto` → project_fact, etc. |
| Scoring | 0.0–1.0 based on type, length, personal markers, tech terms |
| Tag extraction | Technical (`python`, `wsl`) + general (`comida`, `gustos`) |

### Status flow

| Confidence | Status | Behavior |
|---|---|---|
| >= 0.75 | `active` | Retrieved on search, injected into system prompt |
| 0.40–0.74 | `pending_review` | Saved but not retrieved by default |
| < 0.40 | — | Ignored |

### Schema (ChromaDB metadata)

| Field | Type | Description |
|---|---|---|
| `user_id` | str | Owner of the memory |
| `tags` | str | Comma-separated tags (e.g. `python,gustos`) |
| `confidence` | float | 0.0–1.0 |
| `memory_type` | str | `explicit`, `preference`, `project_fact`, etc. |
| `status` | str | `active`, `pending_review`, `archived` |
| `source` | str | `auto`, `manual` |
| `original_text` | str | Raw user message (first 500 chars) |
| `created_at` | str | ISO 8601 timestamp |

### Integration with `build_system_prompt()`

When semantic memories are found, they are injected into the system prompt as:

```
[USER_MEMORY]
- Preferencia del usuario: Al usuario le gustan las galletas de chocolate

[USO DE MEMORIA]
- USER_MEMORY describe al usuario que está conversando.
- ASSISTANT_MEMORY describe al asistente.
- Si USER_MEMORY contiene la respuesta, responde directamente usando esa memoria.
- No digas "como modelo de lenguaje no tengo preferencias" cuando el usuario pregunta por sus propias preferencias.
```

### Console chat commands

| Command | Description |
|---|---|
| `/remember <text>` | Save explicit memory |
| `/memories` | List all saved memories |
| `/search <query>` | Semantic search |
| `/forget <id>` | Delete a memory |
