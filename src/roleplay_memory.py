from __future__ import annotations

import re
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from .memory_models import MemoryRecord, RoleplayAnalysisResult

ROLEPLAY_PATTERNS: dict[str, dict] = {
    "favorite_color": {
        "memory_type": "character_preference",
        "patterns": [r"mi color favorito es\s+(.+)", r"mi color preferido es\s+(.+)"],
    },
    "name": {
        "memory_type": "character_identity",
        "patterns": [r"me llamo\s+(.+)", r"mi nombre es\s+(.+)"],
    },
    "origin": {
        "memory_type": "character_backstory",
        "patterns": [r"vengo de\s+(.+)", r"nac(i|í) en\s+(.+)"],
    },
    "fear": {
        "memory_type": "character_fear",
        "patterns": [
            r"tengo miedo de\s+(.+)",
            r"tengo miedo a\s+(.+)",
            r"me asustan\s+(.+)",
            r"le temo a\s+(.+)",
        ],
    },
    "promise": {
        "memory_type": "promise",
        "patterns": [
            r"prometo\s+(.+)",
            r"te prometo\s+(.+)",
            r"juro\s+(.+)",
        ],
    },
    "age": {
        "memory_type": "character_identity",
        "patterns": [r"tengo\s+(\d+)\s+años"],
    },
}

CONFLICT_POLICY: dict[str, str] = {
    "character_identity": "pending_review",
    "character_backstory": "pending_review",
    "character_preference": "replace",
    "relationship_state": "replace",
    "scene_state": "replace",
    "world_lore": "pending_review",
    "promise": "pending_review",
    "character_fear": "pending_review",
    "character_trait": "replace",
    "character_goal": "replace",
}


def clean_roleplay_text(text: str) -> str:
    text = re.sub(r"\*[^*]+\*", " ", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_memory_content(
    character_id: str, fact_key: str, fact_value: str,
    memory_type: str, original_text: str,
) -> str:
    templates = {
        "favorite_color": f"{character_id} tiene como color favorito el {fact_value}.",
        "name": f"El personaje se llama {fact_value}.",
        "origin": f"{character_id} viene de {fact_value}.",
        "fear": f"{character_id} tiene miedo de {fact_value}.",
        "promise": f"{character_id} hizo una promesa: {fact_value}.",
        "age": f"{character_id} tiene {fact_value} años.",
    }
    if fact_key in templates:
        return templates[fact_key]
    return f"{character_id} estableció en roleplay: {original_text}"


def analyze_roleplay_text(
    text: str,
    source_role: str,
    user_character_id: str,
    assistant_character_id: str,
    world_id: str = "default_world",
    scene_id: str = "",
) -> list[RoleplayAnalysisResult]:
    clean = clean_roleplay_text(text)
    lowered = clean.lower()

    if source_role == "user":
        owner_type = "user_character"
        character_id = user_character_id
    else:
        owner_type = "assistant_character"
        character_id = assistant_character_id

    results: list[RoleplayAnalysisResult] = []

    for fact_key, config in ROLEPLAY_PATTERNS.items():
        for pattern in config["patterns"]:
            m = re.search(pattern, lowered, flags=re.IGNORECASE)
            if not m:
                continue
            value = m.group(1).strip() if m.lastindex and m.group(1) else ""
            if not value:
                continue
            content = build_memory_content(
                character_id=character_id, fact_key=fact_key,
                fact_value=value, memory_type=config["memory_type"],
                original_text=clean,
            )
            conflict_policy = CONFLICT_POLICY.get(config["memory_type"], "pending_review")
            initial_status = "canon" if conflict_policy == "replace" else "canon"  # store as canon, conflict detection handles archiving
            results.append(RoleplayAnalysisResult(
                should_remember=True,
                reason="roleplay_pattern",
                confidence=0.8,
                content=content,
                memory_type=config["memory_type"],
                tags=[character_id, config["memory_type"], fact_key, value],
                owner_type=owner_type,
                character_id=character_id,
                source_role=source_role,
                canon_status=initial_status,
                fact_key=fact_key,
                fact_value=value,
                scene_id=scene_id,
                world_id=world_id,
            ))

    return results


COLLECTION_USER = "roleplay_user_character_memories"
COLLECTION_CHARACTER = "roleplay_assistant_character_memories"
COLLECTION_WORLD = "roleplay_world_memories"


class RoleplaySemanticMemory:
    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        sqlite_conn: sqlite3.Connection | None = None,
        world_id: str = "default_world",
        user_character_id: str = "user_character",
        assistant_character_id: str = "assistant_character",
    ):
        self._persist_dir = persist_dir
        self._conn: sqlite3.Connection | None = sqlite_conn
        self._world_id = world_id
        self._user_char_id = user_character_id
        self._assistant_char_id = assistant_character_id
        self._collections: dict[str, Any] = {}

    # ── Lifecycle ──────────────────────────────────────────────

    def _ensure_collections(self):
        if self._collections:
            return
        try:
            import chromadb
        except ImportError:
            raise ImportError("chromadb no está instalado. Ejecuta: pip install chromadb")
        client = chromadb.PersistentClient(path=self._persist_dir)
        for name in (COLLECTION_USER, COLLECTION_CHARACTER, COLLECTION_WORLD):
            self._collections[name] = client.get_or_create_collection(
                name=name, metadata={"hnsw:space": "cosine"},
            )

    def _coll(self, name: str):
        self._ensure_collections()
        return self._collections[name]

    @property
    def conn(self):
        if self._conn is None:
            raise RuntimeError("RoleplaySemanticMemory requiere sqlite_conn")
        return self._conn

    def close(self):
        self._collections.clear()

    # ── CRUD ───────────────────────────────────────────────────

    def remember(
        self, text: str, source_role: str,
        scene_id: str = "", source: str = "auto",
    ) -> list[str]:
        results = analyze_roleplay_text(
            text=text, source_role=source_role,
            user_character_id=self._user_char_id,
            assistant_character_id=self._assistant_char_id,
            world_id=self._world_id, scene_id=scene_id,
        )
        saved = []
        for r in results:
            mid = self._store(r, source, text)
            if mid:
                saved.append(mid)
        return saved

    def remember_force(
        self, content: str, owner_type: str = "assistant_character",
        character_id: str = "", memory_type: str = "character_backstory",
        fact_key: str = "", fact_value: str = "",
        tags: Optional[list[str]] = None,
    ) -> str:
        if not character_id:
            character_id = self._assistant_char_id
        result = RoleplayAnalysisResult(
            should_remember=True, reason="manual",
            confidence=1.0, content=content,
            memory_type=memory_type,
            tags=tags or [],
            owner_type=owner_type,
            character_id=character_id,
            source_role="assistant",
            canon_status="canon",
            fact_key=fact_key,
            fact_value=fact_value,
            world_id=self._world_id,
        )
        return self._store(result, source="manual", original_text=content) or ""

    def _select_collection(self, result: RoleplayAnalysisResult) -> str:
        if result.owner_type == "user_character":
            return COLLECTION_USER
        return COLLECTION_CHARACTER

    def _store(
        self, result: RoleplayAnalysisResult,
        source: str, original_text: str = "",
    ) -> Optional[str]:
        coll_name = self._select_collection(result)
        coll = self._coll(coll_name)
        memory_id = f"rpmem_{uuid.uuid4().hex[:16]}"
        chroma_id = memory_id
        now = datetime.now(timezone.utc).isoformat()
        tags_str = ",".join(result.tags) if result.tags else ""

        status = "active" if result.confidence >= 0.75 else "pending_review"

        existing_id = self._find_existing_fact(coll, result)
        if existing_id:
            policy = CONFLICT_POLICY.get(result.memory_type, "pending_review")
            if policy == "replace":
                self._archive_by_chroma_id(coll, existing_id)
            else:
                result.canon_status = "soft_canon"
                status = "pending_review"

        coll.add(
            ids=[chroma_id],
            documents=[result.content],
            metadatas=[{
                "namespace": "roleplay",
                "memory_type": result.memory_type,
                "status": status,
                "tags": tags_str,
                "confidence": result.confidence,
                "owner_type": result.owner_type,
                "character_id": result.character_id,
                "source_role": result.source_role,
                "canon_status": result.canon_status,
                "fact_key": result.fact_key,
                "fact_value": result.fact_value,
                "scene_id": result.scene_id,
                "world_id": result.world_id,
                "source": source,
                "original_text": original_text[:500] if original_text else "",
                "created_at": now,
            }],
        )

        self.conn.execute("""
            INSERT INTO semantic_memories
            (memory_id, chroma_id, namespace, scope, content, original_text,
             tags, confidence, importance, memory_type, status, source,
             source_message_ids, created_at, updated_at,
             owner_type, character_id, source_role, canon_status,
             fact_key, fact_value, scene_id, world_id, expires_scope)
            VALUES (?, ?, 'roleplay', ?, ?, ?, ?, ?, ?, ?, ?, ?, '', ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, 'never')
        """, (
            memory_id, chroma_id, result.owner_type,
            result.content, original_text[:500] if original_text else "",
            tags_str, result.confidence, result.importance,
            result.memory_type, status, source, now, now,
            result.owner_type, result.character_id, result.source_role,
            result.canon_status, result.fact_key, result.fact_value,
            result.scene_id, result.world_id,
        ))
        self.conn.commit()
        return memory_id

    def _find_existing_fact(self, coll, result: RoleplayAnalysisResult) -> Optional[str]:
        if not result.fact_key:
            return None
        existing = coll.get(where={
            "$and": [
                {"character_id": {"$eq": result.character_id}},
                {"fact_key": {"$eq": result.fact_key}},
                {"status": {"$eq": "active"}},
                {"canon_status": {"$eq": "canon"}},
            ],
        })
        if not existing["ids"]:
            return None
        return existing["ids"][0]

    def _archive_by_chroma_id(self, coll, chroma_id: str) -> None:
        item = coll.get(ids=[chroma_id])
        if not item["ids"]:
            return
        meta = item["metadatas"][0]
        if isinstance(meta, dict):
            meta["canon_status"] = "contradicted"
            meta["status"] = "archived"
            coll.update(ids=[chroma_id], metadatas=[meta])
        self.conn.execute(
            "UPDATE semantic_memories SET status='archived', canon_status='contradicted', updated_at=datetime('now') WHERE chroma_id=?",
            (chroma_id,),
        )
        self.conn.commit()

    def archive(self, memory_id: str) -> None:
        row = self.conn.execute(
            "SELECT chroma_id FROM semantic_memories WHERE memory_id=?", (memory_id,)
        ).fetchone()
        if not row:
            return
        if row[0]:
            coll = self._pick_coll_by_memory_id(memory_id)
            if coll:
                coll.update(ids=[row[0]], metadatas=[{"status": "archived"}])
        self.conn.execute(
            "UPDATE semantic_memories SET status='archived', updated_at=datetime('now') WHERE memory_id=?",
            (memory_id,),
        )
        self.conn.commit()

    def forget(self, memory_id: str) -> None:
        row = self.conn.execute(
            "SELECT chroma_id FROM semantic_memories WHERE memory_id=?", (memory_id,)
        ).fetchone()
        if not row:
            return
        if row[0]:
            coll = self._pick_coll_by_memory_id(memory_id)
            if coll:
                coll.delete(ids=[row[0]])
        self.conn.execute(
            "UPDATE semantic_memories SET status='deleted', updated_at=datetime('now') WHERE memory_id=?",
            (memory_id,),
        )
        self.conn.commit()

    def purge(self, memory_id: str) -> None:
        row = self.conn.execute(
            "SELECT chroma_id FROM semantic_memories WHERE memory_id=?", (memory_id,)
        ).fetchone()
        if row and row[0]:
            coll = self._pick_coll_by_memory_id(memory_id)
            if coll:
                coll.delete(ids=[row[0]])
        self.conn.execute("DELETE FROM semantic_memories WHERE memory_id=?", (memory_id,))
        self.conn.commit()

    def _pick_coll_by_memory_id(self, memory_id: str) -> Any:
        row = self.conn.execute(
            "SELECT scope FROM semantic_memories WHERE memory_id=?", (memory_id,)
        ).fetchone()
        if not row:
            return None
        scope = row[0]
        if scope == "user_character":
            return self._coll(COLLECTION_USER)
        if scope in ("assistant_character", "shared_world", "relationship", "scene_state"):
            return self._coll(COLLECTION_CHARACTER)
        return self._coll(COLLECTION_WORLD)

    # ── Query ──────────────────────────────────────────────────

    def _query_collection(self, coll_name: str, query: str, n_results: int) -> list[MemoryRecord]:
        coll = self._coll(coll_name)
        results = coll.query(
            query_texts=[query],
            n_results=n_results,
            where={
                "$and": [
                    {"world_id": {"$eq": self._world_id}},
                    {"status": {"$eq": "active"}},
                    {"canon_status": {"$eq": "canon"}},
                ],
            },
        )
        return self._to_records(results)

    def search_user(self, query: str, n_results: int = 5) -> list[MemoryRecord]:
        return self._query_collection(COLLECTION_USER, query, n_results)

    def search_character(self, query: str, n_results: int = 5) -> list[MemoryRecord]:
        return self._query_collection(COLLECTION_CHARACTER, query, n_results)

    def search_world(self, query: str, n_results: int = 5) -> list[MemoryRecord]:
        return self._query_collection(COLLECTION_WORLD, query, n_results)

    def build_context(self, query: str, n_results: int = 5) -> str:
        user_mems = self.search_user(query, n_results)
        char_mems = self.search_character(query, n_results)
        world_mems = self.search_world(query, n_results)
        lines: list[str] = []
        lines.append("[ROLEPLAY_USER_CHARACTER_MEMORY]")
        if user_mems:
            for m in user_mems:
                lines.append(f"- {m.content}")
        else:
            lines.append("- Sin memorias relevantes.")
        lines.append("")
        lines.append("[ROLEPLAY_ASSISTANT_CHARACTER_MEMORY]")
        if char_mems:
            for m in char_mems:
                lines.append(f"- {m.content}")
        else:
            lines.append("- Sin memorias relevantes.")
        lines.append("")
        lines.append("[ROLEPLAY_WORLD_MEMORY]")
        if world_mems:
            for m in world_mems:
                lines.append(f"- {m.content}")
        else:
            lines.append("- Sin memorias relevantes.")
        return "\n".join(lines)

    def get_memory(self, memory_id: str) -> Optional[MemoryRecord]:
        row = self.conn.execute(
            "SELECT * FROM semantic_memories WHERE memory_id=?", (memory_id,)
        ).fetchone()
        return self._row_to_record(row) if row else None

    def list_memories(self, limit: int = 50) -> list[MemoryRecord]:
        rows = self.conn.execute(
            "SELECT * FROM semantic_memories WHERE namespace='roleplay' AND status != 'deleted' ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def count(self) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM semantic_memories WHERE namespace='roleplay' AND status NOT IN ('deleted','archived')",
        ).fetchone()[0]

    # ── Internal ───────────────────────────────────────────────

    @staticmethod
    def _to_records(raw: dict) -> list[MemoryRecord]:
        ids = raw.get("ids") or [[]]
        docs = raw.get("documents") or [[]]
        metas = raw.get("metadatas") or [[]]
        flat_ids = ids[0] if ids and isinstance(ids[0], list) else ids
        flat_docs = docs[0] if docs and isinstance(docs[0], list) else docs
        flat_metas = metas[0] if metas and isinstance(metas[0], list) else metas
        records = []
        for i in range(len(flat_ids)):
            mid = flat_ids[i] if i < len(flat_ids) else ""
            doc = flat_docs[i] if i < len(flat_docs) else ""
            meta = flat_metas[i] if i < len(flat_metas) else {}
            records.append(RoleplaySemanticMemory._from_meta(mid, doc, meta))
        return records

    @staticmethod
    def _from_meta(mid: str, doc: str, meta: dict) -> MemoryRecord:
        tags_str = meta.get("tags", "") or ""
        return MemoryRecord(
            chroma_id=mid,
            content=doc,
            tags=tags_str.split(",") if tags_str else [],
            confidence=float(meta.get("confidence", 0)),
            memory_type=meta.get("memory_type", ""),
            status=meta.get("status", "active"),
            namespace="roleplay",
            created_at=meta.get("created_at", ""),
            source=meta.get("source", "auto"),
            owner_type=meta.get("owner_type", ""),
            character_id=meta.get("character_id", ""),
            source_role=meta.get("source_role", ""),
            canon_status=meta.get("canon_status", "canon"),
            fact_key=meta.get("fact_key", ""),
            fact_value=meta.get("fact_value", ""),
            scene_id=meta.get("scene_id", ""),
            world_id=meta.get("world_id", ""),
        )

    @staticmethod
    def _row_to_record(row) -> MemoryRecord:
        g = lambda k, d="": row[k] if k in row.keys() else d
        tags_str = g("tags", "")
        return MemoryRecord(
            memory_id=g("memory_id"),
            chroma_id=g("chroma_id"),
            content=g("content"),
            tags=tags_str.split(",") if tags_str else [],
            confidence=float(g("confidence", 0) or 0),
            importance=float(g("importance", 0.5) or 0.5),
            memory_type=g("memory_type"),
            namespace=g("namespace", "roleplay"),
            scope=g("scope", ""),
            status=g("status", "active"),
            source=g("source", "auto"),
            original_text=g("original_text"),
            source_message_ids=g("source_message_ids"),
            created_at=g("created_at"),
            updated_at=g("updated_at"),
            owner_type=g("owner_type"),
            character_id=g("character_id"),
            source_role=g("source_role"),
            canon_status=g("canon_status", "canon"),
            fact_key=g("fact_key"),
            fact_value=g("fact_value"),
            scene_id=g("scene_id"),
            world_id=g("world_id"),
            expires_scope=g("expires_scope", "never"),
        )
