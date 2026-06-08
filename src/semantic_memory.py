from __future__ import annotations

import re
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from .memory_models import AnalysisResult, MemoryRecord

_NOISE = {
    "hola", "buenas", "hey", "ok", "okay", "dale", "perfecto",
    "gracias", "sí", "si", "no", "jajaja", "jeje", "lol",
    "continua", "continúa", "sigue", "dale", "listo", "claro",
    "excelente", "genial", "sale", "bueno",
}

_EXPLICIT_PREFIXES = [
    "/remember", "/mem",
    "recuerda que", "acuérdate de que",
    "guarda esto", "guarda que", "ten en cuenta que",
]

_MEMORY_HINTS: dict[str, list[str]] = {
    "preference": [
        "prefiero", "me gusta", "no me gusta", "quiero que",
        "no quiero que", "me gustaría", "me encanta",
        "odio", "detesto", "favorito", "favorita",
        "mi favorito", "mi favorita", "suelo usar",
    ],
    "project_fact": [
        "estoy creando", "estoy desarrollando", "mi proyecto",
        "mi app", "mi juego", "mi librería", "se llama",
    ],
    "environment": [
        "mi pc tiene", "tengo una", "uso windows", "uso linux",
        "uso wsl", "trabajo con",
    ],
    "long_term_instruction": [
        "de ahora en adelante", "a partir de ahora",
        "siempre que", "para futuras conversaciones",
    ],
}

_TECH_KEYWORDS: dict[str, list[str]] = {
    "python": ["python"],
    "windows": ["windows", "windows 11"],
    "linux": ["linux"],
    "wsl": ["wsl"],
    "llama.cpp": ["llama.cpp", "llama cpp", "llamacpp"],
    "chromadb": ["chroma", "chromadb", "chroma db"],
    "sqlite": ["sqlite", "sqlite3"],
    "unity": ["unity"],
    "comfyui": ["comfyui", "comfy ui"],
    "ollama": ["ollama"],
    "gguf": ["gguf"],
    "cuda": ["cuda"],
    "docker": ["docker"],
    "node": ["node", "nodejs"],
}

_GENERAL_KEYWORDS: dict[str, list[str]] = {
    "comida": ["galleta", "galletas", "comida", "pizza", "chocolate", "café", "helado"],
    "nombre": ["me llamo", "mi nombre es", "llámame"],
    "gustos": ["me gusta", "prefiero", "favorito", "favorita", "me encanta"],
}


def is_noise(text: str) -> bool:
    t = text.lower().strip()
    return t in _NOISE or len(t) < 8


def extract_explicit(text: str) -> Optional[str]:
    lowered = text.lower()
    for prefix in _EXPLICIT_PREFIXES:
        idx = lowered.find(prefix)
        if idx >= 0:
            return text[idx + len(prefix):].strip(" :.-")
    return None


def detect_type(text: str) -> Optional[str]:
    t = text.lower()
    for mtype, hints in _MEMORY_HINTS.items():
        if any(h in t for h in hints):
            return mtype
    return None


def extract_tags(text: str) -> list[str]:
    t = text.lower()
    tags = []
    all_keywords = {**_TECH_KEYWORDS, **_GENERAL_KEYWORDS}
    for tag, variants in all_keywords.items():
        if any(v in t for v in variants):
            tags.append(tag)
    return list(dict.fromkeys(tags))


def memory_score(text: str) -> float:
    t = text.lower().strip()
    if is_noise(t):
        return 0.0
    if extract_explicit(text):
        return 1.0

    score = 0.0

    mtype = detect_type(text)
    if mtype:
        score += 0.45
    if len(t.split()) >= 12:
        score += 0.15
    if "?" in t:
        score -= 0.15

    tags = extract_tags(text)
    score += min(len(tags) * 0.08, 0.24)

    personal = ["mi ", "mis ", "uso ", "tengo ", "prefiero ", "quiero "]
    if any(m in t for m in personal):
        score += 0.2

    return max(0.0, min(score, 1.0))


def normalize_text(text: str, memory_type: str) -> str:
    prefixes = {
        "explicit": "Memoria explícita del usuario",
        "preference": "Preferencia del usuario",
        "project_fact": "Información de proyecto del usuario",
        "environment": "Entorno técnico del usuario",
        "long_term_instruction": "Instrucción persistente del usuario",
        "pending": "Posible memoria del usuario",
    }
    return f"{prefixes.get(memory_type, 'Memoria del usuario')}: {text.strip()}"


def analyze_text(text: str) -> AnalysisResult:
    if is_noise(text):
        return AnalysisResult(should_remember=False, reason="noise")

    explicit = extract_explicit(text)
    if explicit:
        return AnalysisResult(
            should_remember=True,
            reason="explicit",
            confidence=1.0,
            content=normalize_text(explicit, "explicit"),
            memory_type="explicit",
            tags=extract_tags(explicit),
        )

    score = memory_score(text)
    mtype = detect_type(text)

    if score >= 0.75 and mtype:
        return AnalysisResult(
            should_remember=True,
            reason="rule_score",
            confidence=min(score, 0.95),
            content=normalize_text(text, mtype),
            memory_type=mtype,
            tags=extract_tags(text),
        )

    if score >= 0.40:
        return AnalysisResult(
            should_remember=True,
            reason="possible",
            confidence=score,
            content=normalize_text(text, "pending"),
            memory_type="pending",
            tags=extract_tags(text),
        )

    return AnalysisResult(should_remember=False, reason="low_score", confidence=score)


class SemanticMemory:
    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        sqlite_conn: Optional[sqlite3.Connection] = None,
        user_id: str = "default",
        namespace: str = "normal",
        scope: str = "user",
    ):
        self._collection: Optional[Any] = None
        self._persist_dir = persist_dir
        self._conn: Optional[sqlite3.Connection] = sqlite_conn
        self._user_id = user_id
        self._namespace = namespace
        self._scope = scope

    # ── Lifecycle ──────────────────────────────────────────────

    def _ensure_collection(self):
        if self._collection is not None:
            return
        try:
            import chromadb
        except ImportError:
            raise ImportError("chromadb no está instalado. Ejecuta: pip install chromadb")
        client = chromadb.PersistentClient(path=self._persist_dir)
        self._collection = client.get_or_create_collection(
            name="semantic_memories",
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def collection(self):
        self._ensure_collection()
        return self._collection

    @property
    def conn(self):
        if self._conn is None:
            raise RuntimeError("SemanticMemory requiere sqlite_conn")
        return self._conn

    def close(self):
        self._collection = None

    # ── Analyze ────────────────────────────────────────────────

    @staticmethod
    def analyze(text: str) -> AnalysisResult:
        return analyze_text(text)

    # ── CRUD ───────────────────────────────────────────────────

    def remember(
        self, text: str, source: str = "auto", msg_ids: str = ""
    ) -> Optional[str]:
        result = self.analyze(text)
        if not result.should_remember:
            return None
        return self._store(result, source, text, msg_ids)

    def remember_force(
        self, content: str, tags: Optional[list[str]] = None
    ) -> str:
        return self._store(AnalysisResult(
            should_remember=True,
            reason="explicit",
            confidence=1.0,
            content=content,
            memory_type="explicit",
            tags=tags or [],
        ), source="manual", original_text=content)

    def _store(
        self, result: AnalysisResult, source: str,
        original_text: str = "", msg_ids: str = "",
    ) -> str:
        memory_id = f"mem_{uuid.uuid4().hex[:16]}"
        chroma_id = memory_id
        now = datetime.now(timezone.utc).isoformat()
        tags_str = ",".join(result.tags) if result.tags else ""
        status = "active" if result.confidence >= 0.75 else "pending_review"

        coll = self.collection
        existing = coll.query(
            query_texts=[result.content],
            n_results=1,
            where={
                "$and": [
                    {"namespace": {"$eq": self._namespace}},
                    {"status": {"$eq": "active"}},
                ],
            },
        )
        if existing["distances"] and existing["distances"][0]:
            if existing["distances"][0][0] < 0.1:
                return existing["ids"][0][0]

        coll.add(
            ids=[chroma_id],
            documents=[result.content],
            metadatas=[{
                "namespace": self._namespace,
                "scope": self._scope,
                "user_id": self._user_id,
                "tags": tags_str,
                "confidence": result.confidence,
                "memory_type": result.memory_type,
                "status": status,
                "source": source,
                "original_text": original_text[:500] if original_text else "",
                "created_at": now,
            }],
        )

        self.conn.execute("""
            INSERT INTO semantic_memories
            (memory_id, chroma_id, namespace, scope, content, original_text,
             tags, confidence, importance, memory_type, status, source,
             source_message_ids, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            memory_id, chroma_id, self._namespace, self._scope,
            result.content, original_text[:500] if original_text else "",
            tags_str, result.confidence, result.importance,
            result.memory_type, status, source, msg_ids, now, now,
        ))
        self.conn.commit()
        return memory_id

    def archive(self, memory_id: str) -> None:
        self.conn.execute(
            "UPDATE semantic_memories SET status='archived', updated_at=datetime('now') WHERE memory_id=?",
            (memory_id,),
        )
        self.conn.commit()
        row = self.conn.execute(
            "SELECT chroma_id FROM semantic_memories WHERE memory_id=?", (memory_id,)
        ).fetchone()
        if row and row[0]:
            self.collection.update(
                ids=[row[0]],
                metadatas=[{"status": "archived"}],
            )

    def forget(self, memory_id: str) -> None:
        row = self.conn.execute(
            "SELECT chroma_id FROM semantic_memories WHERE memory_id = ?", (memory_id,)
        ).fetchone()
        if not row:
            return
        if row[0]:
            self.collection.delete(ids=[row[0]])
        self.conn.execute(
            "UPDATE semantic_memories SET status='deleted', updated_at=datetime('now') WHERE memory_id=?",
            (memory_id,),
        )
        self.conn.commit()

    def purge(self, memory_id: str) -> None:
        row = self.conn.execute(
            "SELECT chroma_id FROM semantic_memories WHERE memory_id = ?", (memory_id,)
        ).fetchone()
        if row and row[0]:
            self.collection.delete(ids=[row[0]])
        self.conn.execute("DELETE FROM semantic_memories WHERE memory_id = ?", (memory_id,))
        self.conn.commit()

    # ── Query ──────────────────────────────────────────────────

    def search(self, query: str, n_results: int = 5) -> list[MemoryRecord]:
        coll = self.collection
        results = coll.query(
            query_texts=[query],
            n_results=n_results,
            where={
                "$and": [
                    {"namespace": {"$eq": self._namespace}},
                    {"user_id": {"$eq": self._user_id}},
                    {"status": {"$eq": "active"}},
                ],
            },
        )
        return self._to_records(results)

    def search_by_tags(self, tags: list[str]) -> list[MemoryRecord]:
        records = self.list_memories(limit=1000)
        wanted = set(tags)
        return [r for r in records if wanted.intersection(set(r.tags)) and r.status == "active"]

    def list_memories(self, limit: int = 50) -> list[MemoryRecord]:
        rows = self.conn.execute(
            "SELECT * FROM semantic_memories WHERE namespace=? AND status != 'deleted' ORDER BY created_at DESC LIMIT ?",
            (self._namespace, limit),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_memory(self, memory_id: str) -> Optional[MemoryRecord]:
        row = self.conn.execute(
            "SELECT * FROM semantic_memories WHERE memory_id = ?", (memory_id,)
        ).fetchone()
        return self._row_to_record(row) if row else None

    def count(self) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM semantic_memories WHERE namespace=? AND status NOT IN ('deleted','archived')",
            (self._namespace,),
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
            records.append(SemanticMemory._from_meta(mid, doc, meta))
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
            namespace=meta.get("namespace", "normal"),
            scope=meta.get("scope", "user"),
            created_at=meta.get("created_at", ""),
            source=meta.get("source", "auto"),
            original_text=meta.get("original_text", ""),
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
            namespace=g("namespace", "normal"),
            scope=g("scope", "user"),
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
