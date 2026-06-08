from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

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


@dataclass
class AnalysisResult:
    should_remember: bool = False
    reason: str = ""
    confidence: float = 0.0
    content: str = ""
    memory_type: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class MemoryRecord:
    id: str = ""
    content: str = ""
    tags: list[str] = field(default_factory=list)
    confidence: float = 0.0
    memory_type: str = ""
    status: str = "active"
    created_at: str = ""
    source: str = "auto"


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
    def __init__(self, persist_dir: str = "./chroma_db", user_id: str = "default"):
        self._collection: Optional["Collection"] = None
        self._persist_dir = persist_dir
        self._user_id = user_id

    # ── Lifecycle ──────────────────────────────────────────────

    def _ensure_collection(self):
        if self._collection is not None:
            return
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb no está instalado. Ejecuta: pip install chromadb"
            )
        client = chromadb.PersistentClient(path=self._persist_dir)
        self._collection = client.get_or_create_collection(
            name="semantic_memories",
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def collection(self):
        self._ensure_collection()
        return self._collection

    def close(self):
        self._collection = None

    # ── Analyze ────────────────────────────────────────────────

    @staticmethod
    def analyze(text: str) -> AnalysisResult:
        return analyze_text(text)

    # ── CRUD ───────────────────────────────────────────────────

    def remember(self, text: str, source: str = "auto") -> Optional[str]:
        result = self.analyze(text)
        if not result.should_remember:
            return None
        return self._store(result, source, text)

    def remember_force(self, content: str, tags: Optional[list[str]] = None) -> str:
        return self._store(AnalysisResult(
            should_remember=True,
            reason="explicit",
            confidence=1.0,
            content=content,
            memory_type="explicit",
            tags=tags or [],
        ), source="manual", original_text=content)

    def _store(self, result: AnalysisResult, source: str, original_text: str = "") -> str:
        coll = self.collection
        mem_id = f"mem_{uuid.uuid4().hex[:16]}"
        now = datetime.now(timezone.utc).isoformat()

        status = "active" if result.confidence >= 0.75 else "pending_review"
        tags_str = ",".join(result.tags) if result.tags else ""

        existing = coll.query(
            query_texts=[result.content],
            n_results=1,
            where={"status": {"$eq": "active"}},
        )
        if existing["distances"] and existing["distances"][0]:
            if existing["distances"][0][0] < 0.1:
                return existing["ids"][0][0]

        coll.add(
            ids=[mem_id],
            documents=[result.content],
            metadatas=[{
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
        return mem_id

    def archive(self, memory_id: str) -> None:
        record = self.get_memory(memory_id)
        if not record:
            return
        now = datetime.now(timezone.utc).isoformat()
        self.collection.update(
            ids=[memory_id],
            metadatas=[{
                "user_id": self._user_id,
                "tags": ",".join(record.tags),
                "confidence": record.confidence,
                "memory_type": record.memory_type,
                "status": "archived",
                "source": record.source,
                "created_at": record.created_at,
                "updated_at": now,
            }],
        )

    def forget(self, memory_id: str) -> None:
        self.collection.delete(ids=[memory_id])

    # ── Query ──────────────────────────────────────────────────

    def search(self, query: str, n_results: int = 5) -> list[MemoryRecord]:
        coll = self.collection
        results = coll.query(
            query_texts=[query],
            n_results=n_results,
            where={
                "$and": [
                    {"user_id": {"$eq": self._user_id}},
                    {"status": {"$eq": "active"}},
                ],
            },
        )
        return self._to_records(results)

    def search_by_tags(self, tags: list[str]) -> list[MemoryRecord]:
        records = self.list_memories(limit=1000)
        wanted = set(tags)
        return [
            r for r in records
            if wanted.intersection(set(r.tags)) and r.status == "active"
        ]

    def list_memories(self, limit: int = 50) -> list[MemoryRecord]:
        coll = self.collection
        results = coll.get(
            limit=limit,
            where={"user_id": {"$eq": self._user_id}},
        )
        return self._to_records({
            "ids": [results["ids"]] if results["ids"] else [],
            "documents": [results["documents"]] if results["documents"] else [],
            "metadatas": [results["metadatas"]] if results["metadatas"] else [],
            "distances": None,
        })

    def get_memory(self, memory_id: str) -> Optional[MemoryRecord]:
        coll = self.collection
        results = coll.get(ids=[memory_id])
        if not results["ids"]:
            return None
        return self._to_record(
            results["ids"][0],
            results["documents"][0] if results["documents"] else "",
            results["metadatas"][0] if results["metadatas"] else {},
        )

    def count(self) -> int:
        return self.collection.count()

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
            records.append(SemanticMemory._to_record(mid, doc, meta))
        return records

    @staticmethod
    def _to_record(mid: str, doc: str, meta: dict) -> MemoryRecord:
        tags_str = meta.get("tags", "") or ""
        return MemoryRecord(
            id=mid,
            content=doc,
            tags=tags_str.split(",") if tags_str else [],
            confidence=float(meta.get("confidence", 0)),
            memory_type=meta.get("memory_type", ""),
            status=meta.get("status", "active"),
            created_at=meta.get("created_at", ""),
            source=meta.get("source", "auto"),
        )
