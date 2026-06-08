# v_memory_manager

Persistent memory manager for LLM conversations. SQLite-backed, zero external dependencies.

v0.3.0 — SQLite-backed semantic memories + roleplay memory system.

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
| `build_system_prompt(extra_context="")` | Core prompt + [USER_MEMORY] + saved prompts + [ASSISTANT_MEMORY] + memory rules |
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

The `SemanticMemory` module detects rememberable information using rules (no LLM) and stores it in both ChromaDB (search index) and SQLite (source of truth).

### Requirements

```bash
pip install chromadb
```

### Quick start

```python
import sqlite3
from v_memory_manager import SemanticMemory, MemoryManager

mem = MemoryManager()
mem.create_memory_db("chat.db")

sem = SemanticMemory(
    persist_dir="./chroma_db",
    sqlite_conn=mem._conn,
    namespace="normal",
    scope="user",
)

# Analyze without saving
result = sem.analyze("Me gustan las galletas de chocolate")
print(result.should_remember, result.confidence)

# Save automatically
mid = sem.remember("Prefiero Python")
if mid:
    print(f"Saved: {mid}")

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

### Rule-based detection

| Step | Description |
|---|---|
| Noise filter | Ignores greetings, short messages (< 8 chars) |
| Explicit commands | `/remember`, `recuerda que`, `guarda esto` |
| Pattern hints | `prefiero` → preference, `mi proyecto` → project_fact |
| Scoring | 0.0–1.0 based on type, length, personal markers, tech terms |
| Tag extraction | Technical (`python`, `wsl`) + general (`comida`, `gustos`) |

### Status flow

| Confidence | Status | Retrieved |
|---|---|---|
| >= 0.75 | `active` | Yes |
| 0.40–0.74 | `pending_review` | No |
| < 0.40 | — | Ignored |

### Operations

| Operation | SQLite | ChromaDB |
|---|---|---|
| `remember` | INSERT | add |
| `archive` | UPDATE status | update status |
| `forget` | UPDATE status='deleted' | delete (soft) |
| `purge` | DELETE | delete (hard) |
| `get_memory` | SELECT | — |
| `list_memories` | SELECT | — |
| `search` | — | query |

## Roleplay Memory (ChromaDB + SQLite)

Separate system for narrative/fictional memories. Detects character facts from both user and assistant messages using regex patterns.

### Quick start

```python
from v_memory_manager import RoleplaySemanticMemory

rp = RoleplaySemanticMemory(
    persist_dir="./chroma_db",
    sqlite_conn=mem._conn,
    user_character_id="mikaela",
    assistant_character_id="juan",
)

# Auto-detect from message
ids = rp.remember("Mi color favorito es rojo", source_role="assistant")

# Force save
rp.remember_force(
    content="Juan fue exiliado de la capital.",
    owner_type="assistant_character",
    character_id="juan",
    memory_type="character_backstory",
    fact_key="exile_origin",
    fact_value="exiliado de la capital",
)

# Search by character
rp.search_user("miedo")
rp.search_character("color")
rp.search_world("capital")

# Build prompt context
ctx = rp.build_context("miedo", n_results=3)
```

### Detected patterns

| Pattern | memory_type | Conflict policy |
|---|---|---|
| `mi color favorito es...` | `character_preference` | replace |
| `me llamo...`, `mi nombre es...` | `character_identity` | pending_review |
| `vengo de...`, `nací en...` | `character_backstory` | pending_review |
| `tengo miedo de...` | `character_fear` | pending_review |
| `prometo...`, `juro...` | `promise` | pending_review |
| `tengo X años` | `character_identity` | pending_review |

### ChromaDB collections

| Collection | Stores |
|---|---|
| `roleplay_user_character_memories` | Facts about the user's character |
| `roleplay_assistant_character_memories` | Facts about the AI character + world lore |
| `roleplay_world_memories` | Shared world lore |

### Contradictions

Matching `character_id + fact_key` with a different value triggers a policy:

| memory_type | Action |
|---|---|
| `character_preference` | Archive old, save new as canon |
| `relationship_state` | Archive old, save new as canon |
| `scene_state` | Archive old, save new as canon |
| `character_identity` | New saved as `pending_review` |
| `character_backstory` | New saved as `pending_review` |
| `world_lore` | New saved as `pending_review` |

## MemoryRouter

Decides between normal and roleplay memory based on a flag.

```python
from v_memory_manager import MemoryRouter, SemanticMemory, RoleplaySemanticMemory

router = MemoryRouter(semantic=sem, roleplay=rp, roleplay_enabled=False)
router.set_roleplay_enabled(True)

# Auto-routes based on mode
router.remember_user("Me gusta Python")
router.remember_assistant("Mi color favorito es rojo")

# Build context for prompt injection
ctx = router.build_context("color favorito")
```

| Mode | `remember_user` | `remember_assistant` | `build_context` returns |
|---|---|---|---|
| Normal | `SemanticMemory.remember()` | ignored | `[USER_MEMORY]` block |
| Roleplay | `RoleplaySemanticMemory.remember(role="user")` | `RoleplaySemanticMemory.remember(role="assistant")` | `[ROLEPLAY_*]` blocks |

## System prompt injection

### Normal mode

```
[USER_MEMORY]
- Preferencia del usuario: Le gustan las galletas de chocolate

[USO DE MEMORIA]
- USER_MEMORY describe al usuario.
- Si USER_MEMORY contiene la respuesta, responde directamente.
```

### Roleplay mode

```
[ROLEPLAY_USER_CHARACTER_MEMORY]
- mikaela tiene miedo de los espejos.

[ROLEPLAY_ASSISTANT_CHARACTER_MEMORY]
- juan tiene como color favorito el rojo.

[ROLEPLAY MEMORY RULES]
- Las memorias pertenecen al mundo ficticio.
- No las confundas con datos reales del usuario.
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
| `id` | INTEGER | PK |
| `memory_id` | TEXT UNIQUE | Stable external ID (`mem_abc123`) |
| `chroma_id` | TEXT UNIQUE | ChromaDB document ID |
| `namespace` | TEXT | `normal` or `roleplay` |
| `scope` | TEXT | `user`, `project`, `user_character`, etc. |
| `content` | TEXT | Memory content |
| `original_text` | TEXT | Raw source message |
| `tags` | TEXT | Comma-separated tags |
| `confidence` | REAL | Detection confidence 0–1 |
| `importance` | REAL | Relevance 0–1 |
| `memory_type` | TEXT | `explicit`, `preference`, `character_identity`, etc. |
| `status` | TEXT | `active`, `pending_review`, `archived`, `deleted` |
| `source` | TEXT | `auto` or `manual` |
| `source_message_ids` | TEXT | Traceability to source messages |
| `owner_type` | TEXT | `user_character`, `assistant_character` (roleplay) |
| `character_id` | TEXT | Character name (roleplay) |
| `source_role` | TEXT | `user` or `assistant` (roleplay) |
| `canon_status` | TEXT | `canon`, `soft_canon`, `temporary`, `contradicted` |
| `fact_key` | TEXT | `favorite_color`, `name`, `fear`, etc. (roleplay) |
| `fact_value` | TEXT | Extracted value (roleplay) |
| `scene_id` | TEXT | Scene tracking (roleplay) |
| `world_id` | TEXT | World tracking (roleplay) |
| `expires_scope` | TEXT | `never`, `end_of_scene`, `end_of_day` |

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
| `/remember <text>` | Save explicit memory |
| `/memories` | List all memories |
| `/search <query>` | Semantic search |
| `/forget <id>` | Delete a memory |
| `/roleplay` | Toggle roleplay mode ON/OFF |
| `/clear` | Clear message history |
| `/prompt <text>` | Change system prompt |
| `/save <name>` | Save prompt |
| `/load <name>` | Load prompt |
| `/show_prompt` | Show built system prompt |
| `/exit` | Exit |

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

See `examples/console_chat.py` for a complete interactive chat with model selection, session management, streaming, roleplay, and semantic memory.
