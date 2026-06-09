# v_memory_manager

Persistent memory manager for LLM conversations. SQLite-backed, zero external dependencies.

v0.5.0 — Negative preference detection, assistant/user scope separation, optional spaCy NLP engine, rule-based context builder.

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
| `clear_memory_db()` | Truncates messages **and semantic memories** (prompts survive) |
| `close()` | Closes the connection |

### Messages (sliding-window)

| Method | Description |
|--------|-------------|
| `add_message(role, content)` | `user`/`assistant` → messages table. `system` → updates core prompt |
| `get_history(max_messages=10, extra_context="")` | Returns `build_system_prompt()` + last N-1 messages |
| `build_system_prompt(extra_context="", semantic_memory=None, user_query="")` | Core prompt + semantic memories (RAG) + saved prompts + memory rules |
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

### Constructor

```python
SemanticMemory(
    persist_dir="./chroma_db",
    sqlite_conn=conn,
    user_id="default",
    namespace="normal",
    scope="user",
    allow_assistant_memory=False,
)
```

- `allow_assistant_memory` — when `True`, `remember(source_role="assistant")` will analyze and store assistant self-facts. Default `False` for library safety.

### Optional spaCy NLP Engine

The package includes an optional advanced NLP layer using **spaCy**. If spaCy is not installed or the model is not found, the system transparently falls back to regular expression parsing.

When spaCy is enabled (using the `es_core_news_sm` model), it adds:
- **Lemmatization**: Verbs are normalized to their base dictionary form. For example, "me gustaban", "me gustaría", and "me gusta" all match the lemma `gustar`.
- **Named Entity Recognition (NER)**: Automatically extracts names of people, organizations, and locations, turning them into search tags (e.g., `persona:Juan`, `ubicacion:Mexico`).
- **Semantic Fact Extraction**: Uses dependency grammar parsing to extract subject-verb-object triples, storing them in the `fact_key` and `fact_value` database columns (e.g., "Mi color favorito es el azul" → key: `color`, value: `azul`).

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

# Save assistant message (requires allow_assistant_memory=True)
sem = SemanticMemory(sqlite_conn=conn, allow_assistant_memory=True)
mid = sem.remember("Mi color favorito es el rojo", source_role="assistant")

# Wrappers
sem.remember_user("Me gusta Python")
sem.remember_assistant("Mi nombre es Juan")

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

User and assistant messages are analyzed differently and stored in separate scopes:

| Source | Scope | Analysis |
|---|---|---|
| `source_role="user"` | `user` | Full heuristic scoring: preferences, project facts, environment, instructions |
| `source_role="assistant"` | `assistant` or `assistant_claims` | Regex + lemma extraction for self-facts; claims about user stored separately |

Assistant messages are filtered strictly — questions, answer markers (recipes, tutorials), and emotional expressions are blocked.

**Memory types:**

| Type | Scope | Example |
|---|---|---|
| `positive_preference` | `user` | "Me gustan las galletas" |
| `negative_preference` | `user` | "No me gustan las respuestas largas" |
| `negative_instruction` | `user` | "No quiero que uses emojis" |
| `assistant_instruction` | `user` | "Quiero que expliques paso a paso" |
| `project_fact` | `user` | "Estoy creando un juego con Unity" |
| `environment` | `user` | "Uso Windows 11" |
| `assistant_preference` | `assistant` | "Al asistente le gusta Python" |
| `assistant_negative_preference` | `assistant` | "Al asistente no le gusta el ruido" |
| `assistant_identity` | `assistant` | "El asistente dice que se llama Juan" |
| `assistant_claim_about_user` | `assistant_claims` | "Afirmación del asistente sobre el usuario: te gusta Python" |

### build_context()

Generates a structured prompt-ready string separating user, assistant, and claim memories with rules:

```python
context = sem.build_context("galletas")
print(context)
# [MEMORY_RULES]
# - USER_MEMORY describe al usuario.
# - ASSISTANT_MEMORY describe al asistente.
# ...
#
# [USER_MEMORY]
# - Preferencia positiva del usuario: le gustan las galletas de chocolate.
#
# [ASSISTANT_MEMORY]
# - El asistente dice que se llama Juan.
#
# [ASSISTANT_CLAIMS_ABOUT_USER]
# - Sin memorias relevantes.
```

### Smart conflict resolution

When storing a new memory, the system checks for existing similar memories:

| Distance | Action |
|---|---|
| < 0.05 | **Duplicate** — returns existing ID, no change |
| >= 0.05 | **New** — saves normally (no auto-archive) |

### Detection rules (no LLM)

| Step | Description |
|---|---|
| Noise filter | Ignores greetings, short messages (< 8 chars), short-but-meaningful exceptions ("no emojis") |
| Explicit commands | `/remember`, `recuerda que`, `guarda esto` |
| Regex detection | Word-boundary patterns for negative preferences, instructions |
| Pattern hints | `prefiero` → positive_preference, `mi proyecto` → project_fact |
| Scoring | 0.0–1.0 based on type, length, personal markers, tech terms |
| Tag extraction | Technical (`python`, `wsl`) + general (`comida`, `gustos`, `disgustos`, `restricciones`) |

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
| `get_memory` | SELECT by memory_id + user_id | — |
| `list_memories` | SELECT by namespace + scope + user_id | — |
| `review_pending` | SELECT status='pending_review' + scope | — |
| `search` | — | query (active only, by scope) |
| `search_by_tags` | SELECT WHERE tags LIKE + scope | — |

### Queries with scope

All query methods accept an optional `scope` parameter:

```python
sem.list_memories(scope="assistant")
sem.count(scope="user")
sem.review_pending(scope="assistant")
sem.search_by_tags(["dislikes"], scope="assistant")
sem.search("color favorito", scope="assistant")
```

## Negative memory patterns

The system detects and categorizes user preferences, restrictions, and updated preferences using regex patterns:

| Pattern | Type | Example match |
|---|---|---|
| `\bno me gusta\s+(.+)` | `negative_preference` | "No me gusta el chocolate" |
| `\bodio\s+(.+)` | `negative_preference` | "Odio las respuestas largas" |
| `\bno quiero que\s+(.+)` | `negative_instruction` | "No quiero que uses emojis" |
| `\bevita\s+(.+)` | `negative_instruction` | "Evita el markdown" |
| `\bsin\s+(emojis\|markdown\|...)` | `negative_instruction` | "Sin tablas por favor" |
| `\bya no me gusta\s+(.+)` | `negative_preference` | "Ya no me gusta Python" |
| `\bya no quiero\s+(.+)` | `negative_instruction` | "Ya no quiero respuestas cortas" |

## Migration

Migrate legacy `long_term_memories` to the unified `semantic_memories` system:

```python
from v_memory_manager import migrate_long_term_to_semantic

count = migrate_long_term_to_semantic(conn, semantic_memory)
print(f"Migrated {count} memories")

# Safe to run multiple times — skips already migrated entries
```

## System prompt injection

When relevant memories are found, they are injected into the system prompt via `build_context()`:

```
[MEMORY_RULES]
- USER_MEMORY describe al usuario.
- ASSISTANT_MEMORY describe al asistente.
- ASSISTANT_CLAIMS_ABOUT_USER son afirmaciones del asistente sobre el usuario.
- USER_MEMORY tiene más autoridad que ASSISTANT_CLAIMS_ABOUT_USER.
- Si el usuario pregunta por "mi", usa USER_MEMORY.
- Si el usuario pregunta por "tu", usa ASSISTANT_MEMORY.
- No confundas memorias del usuario con memorias del asistente.

[USER_MEMORY]
- Preferencia positiva del usuario: Le gustan las galletas de chocolate.
- Restricción persistente del usuario: No quiere respuestas con emojis.

[ASSISTANT_MEMORY]
- El asistente dice que se llama Juan.
- Al asistente le gusta Python.

[ASSISTANT_CLAIMS_ABOUT_USER]
- Afirmación del asistente sobre el usuario: te gusta Python.
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
| `scope` | TEXT | `user`, `assistant`, or `assistant_claims` |
| `content` | TEXT | Memory content |
| `original_text` | TEXT | Raw source message |
| `tags` | TEXT | Comma-delimited tags (`,tag1,tag2,`) |
| `confidence` | REAL | Detection confidence 0–1 |
| `importance` | REAL | Relevance 0–1 (boosted on retrieval) |
| `memory_type` | TEXT | `positive_preference`, `negative_preference`, `assistant_preference`, etc. |
| `status` | TEXT | `active`, `pending_review`, `archived`, `deleted` |
| `source` | TEXT | `auto`, `manual`, or `legacy` |
| `source_role` | TEXT | `user` or `assistant` |
| `user_id` | TEXT | Multi-user isolation |
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
| v5 (0.5.0) | Negative preference/instruction detection, user/assistant scope separation, `user_id` column, optional spaCy NLP engine, patterns module, regression tests |

Auto-migrated on `load_memory_db()`.

## Console chat commands

| Command | Description |
|---|---|
| `/remember <text>` | Force save a memory |
| `/memories` | List all memories |
| `/search <query>` | Semantic search |
| `/forget <id>` | Delete a memory |
| `/review` | Review pending memories (approve/reject) |
| `/clear` | Clear message history + semantic memories |
| `/prompt <text>` | Change system prompt |
| `/save <name>` | Save prompt |
| `/load <name>` | Load prompt |
| `/show_prompt` | Show built system prompt |
| `/exit` | Exit |

Memories are saved automatically from both user and assistant messages. User messages use full heuristic analysis with negative detection. Assistant messages (when enabled) use regex extraction with word-boundary patterns and spaCy lemma fallback.

## Integration with v_llama

```python
from v_memory_manager import MemoryManager, SemanticMemory
from v_llama import VLLaMA

mem = MemoryManager()
mem.create_memory_db("chat.db")

sem = SemanticMemory(
    persist_dir="./chroma_db",
    sqlite_conn=mem._conn,
    allow_assistant_memory=True,
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
sem.remember_user(user)
sem.remember_assistant(res.content)
```

See `examples/console_chat.py` for a complete interactive chat with model selection, session management, streaming, semantic memory, and ChromaDB cleanup.

## Source layout

```
src/
  patterns.py         — All pattern constants (noise, hints, regex, lemmas, NER mappings)
  semantic_memory.py  — Core memory analysis, CRUD, context builder, detection functions
  memory.py           — MemoryManager (messages, prompts, configs)
  memory_models.py    — Data classes (AnalysisResult, MemoryRecord)
  models.py           — Data classes (Message, LongTermMemory)
  db.py               — SQLite schema, migrations, CRUD helpers
  deque.py            — Sliding-window history builder
  nlp_engine.py       — Optional spaCy NLP layer (lemmatization, NER, dependency parsing)
tests/
  test_regression.py  — 45+ regression tests covering user + assistant memory edge cases
  test_plan_v5.py     — Integration tests (ChromaDB conflict resolution, review flow, migration)
```
