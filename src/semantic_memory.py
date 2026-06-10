from __future__ import annotations

import re
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from .memory_models import AnalysisResult, MemoryRecord
from .nlp_engine import (
    SpacyResult,
    extract_entities_as_tags,
    extract_fact_triple,
    get_analyzer,
    is_spacy_available,
    lemmatized_match,
)
from .patterns import (
    _ASSISTANT_ANSWER_MARKERS,
    _ASSISTANT_SELF_PATTERNS,
    _ASSISTANT_USER_FACT_PATTERNS,
    _CONDITIONAL_MARKERS,
    _CORRECTION_PATTERNS,
    _CORRECTION_SIGNALS,
    _DETECT_TYPE_PRIORITY,
    _EXPLICIT_PREFIXES,
    _GENERAL_KEYWORDS,
    _LEMMA_MEMORY_HINTS,
    _LEMMA_PERSONAL_MARKERS,
    _MEMORY_HINTS,
    _NEGATIVE_PATTERNS,
    _NOISE,
    _QUESTION_LEMMAS,
    _SHORT_MEANINGFUL,
    _TECH_KEYWORDS,
    normalize_accents,
)


def spacy_analyze(text: str) -> Optional[SpacyResult]:
    """Ejecuta análisis spaCy si está disponible, o None."""
    if not is_spacy_available():
        return None
    return get_analyzer().analyze(text)


def is_noise(text: str) -> bool:

    t = text.lower().strip()

    if t in _NOISE:
        return True

    if t in _SHORT_MEANINGFUL:
        return False

    return len(t) < 8


def extract_explicit(text: str) -> Optional[str]:
    lowered = text.lower()
    for prefix in _EXPLICIT_PREFIXES:
        idx = lowered.find(prefix)
        if idx >= 0:
            return text[idx + len(prefix):].strip(" :.-")
    return None


def detect_type(text: str, nlp_result: Optional[SpacyResult] = None) -> Optional[str]:
    """Detecta el tipo de memoria con regex + lemma (si spaCy disponible)."""
    t = text.lower()

    # 1. Regex tradicional (prioridad)
    for mtype in _DETECT_TYPE_PRIORITY:
        hints = _MEMORY_HINTS.get(mtype, [])
        if any(h in t for h in hints):
            return mtype

    # 2. Fallback por lemma (spaCy)
    if nlp_result and nlp_result.lemmas:
        lemma_set = set(nlp_result.lemmas)
        for mtype in _DETECT_TYPE_PRIORITY:
            lemma_hints = _LEMMA_MEMORY_HINTS.get(mtype, [])
            if lemma_set & set(lemma_hints):
                # Si el tipo es negative_preference, verificar negación
                if mtype == "negative_preference" and not nlp_result.has_negation:
                    # "gustar" sin negación → positive, no negative
                    if lemma_set & set(_LEMMA_MEMORY_HINTS.get("positive_preference", [])):
                        return "positive_preference"
                return mtype

    return None


def extract_tags(text: str, nlp_result: Optional[SpacyResult] = None) -> list[str]:
    """Extrae tags por keywords + entidades NER (si spaCy disponible)."""
    t = text.lower()
    tags = []

    # 1. Keywords por substring (sistema original)
    all_keywords = {**_TECH_KEYWORDS, **_GENERAL_KEYWORDS}
    for tag, variants in all_keywords.items():
        if any(v in t for v in variants):
            tags.append(tag)

    # 2. Tags NER de spaCy
    if nlp_result:
        ner_tags = extract_entities_as_tags(nlp_result)
        tags.extend(ner_tags)

    return list(dict.fromkeys(tags))


def memory_score(text: str, nlp_result: Optional[SpacyResult] = None) -> float:
    """Calcula score de memorabilidad con regex + spaCy."""
    t = text.lower().strip()
    if is_noise(t):
        return 0.0
    if extract_explicit(text):
        return 1.0

    score = 0.0

    mtype = detect_type(text, nlp_result)
    if mtype:
        score += 0.50
    if len(t.split()) >= 12:
        score += 0.15
    if "?" in t:
        score -= 0.15

    tags = extract_tags(text, nlp_result)
    score += min(len(tags) * 0.08, 0.24)

    # Detección personal por regex (original)
    personal = [
        "mi ", "mis ", "uso ", "tengo ",
        "prefiero ", "quiero ", "me gusta",
        "me encanta", "me llamo", "estoy ",
        "no quiero ", "no me gusta ",
        "odio ", "detesto ",
        "evita ", "sin ",
    ]
    if any(m in t for m in personal):
        score += 0.25

    # Bonus spaCy: primera persona confirmada + entidades
    if nlp_result:
        if nlp_result.is_first_person and lemmatized_match(nlp_result, _LEMMA_PERSONAL_MARKERS):
            score += 0.10
        if nlp_result.entities:
            score += min(len(nlp_result.entities) * 0.05, 0.15)

    return max(0.0, min(score, 1.0))


def normalize_text(text: str, memory_type: str) -> str:
    prefixes = {
        "explicit": "Memoria expl\u00edcita del usuario",
        "positive_preference": "Preferencia positiva del usuario",
        "negative_preference": "Preferencia negativa del usuario",
        "assistant_instruction": "Instrucci\u00f3n persistente del usuario",
        "negative_instruction": "Restricci\u00f3n persistente del usuario",
        "project_fact": "Informaci\u00f3n de proyecto del usuario",
        "environment": "Entorno t\u00e9cnico del usuario",
        "pending": "Posible memoria del usuario",
        "preference": "Preferencia del usuario",
        "assistant_claim_about_user": "Afirmaci\u00f3n del asistente sobre el usuario",
        "assistant_preference": "Preferencia del asistente",
        "assistant_negative_preference": "Preferencia negativa del asistente",
        "assistant_identity": "Identidad del asistente",
    }
    return f"{prefixes.get(memory_type, 'Memoria del usuario')}: {text.strip()}"


def is_question_like(text: str, nlp_result: Optional[SpacyResult] = None) -> bool:
    """Detecta preguntas con regex + sentence_type de spaCy."""
    # spaCy: detección por estructura
    if nlp_result and nlp_result.sentence_type == "interrogative":
        return True

    # Regex fallback
    t = text.lower().strip()
    return (
        "?" in t
        or t.startswith("cómo ")
        or t.startswith("como ")
        or t.startswith("qué ")
        or t.startswith("que ")
        or t.startswith("por qué")
        or t.startswith("porque ")
    )


def detect_negative_memory(text: str, nlp_result: Optional[SpacyResult] = None) -> Optional[AnalysisResult]:
    t = text.lower().strip()

    for pattern, memory_type, template in _NEGATIVE_PATTERNS:
        match = re.search(pattern, t)
        if not match:
            continue

        value = match.group(1).strip(" .,:;")
        if not value:
            continue

        tags = extract_tags(text, nlp_result)
        if "negative" not in tags:
            tags = tags + ["negative"]

        importance = 0.85 if memory_type == "negative_instruction" else 0.75

        return AnalysisResult(
            should_remember=True,
            reason="negative_pattern",
            confidence=0.90,
            importance=importance,
            content=template.format(value),
            memory_type=memory_type,
            tags=tags,
        )

    # Fallback por lemma (spaCy)
    if nlp_result and nlp_result.root_verb:
        is_negated_positive = nlp_result.has_negation and lemmatized_match(nlp_result, _LEMMA_MEMORY_HINTS.get("positive_preference", []))
        is_direct_negative = not nlp_result.has_negation and lemmatized_match(nlp_result, _LEMMA_MEMORY_HINTS.get("negative_preference", []))
        is_negative_instruction = not nlp_result.has_negation and lemmatized_match(nlp_result, _LEMMA_MEMORY_HINTS.get("negative_instruction", []))

        if is_negated_positive or is_direct_negative or is_negative_instruction:
            verb_text = nlp_result.root_verb
            idx = text.lower().find(verb_text)
            value = ""
            if idx >= 0:
                value = text[idx + len(verb_text):].strip(" .,:;!?")

            if not value and nlp_result.noun_chunks:
                value = nlp_result.noun_chunks[0]
            if not value and nlp_result.objects:
                value = nlp_result.objects[0]

            if value:
                if is_negative_instruction:
                    memory_type = "negative_instruction"
                    template = "Restricción persistente del usuario: debe evitar {}."
                    importance = 0.85
                else:
                    memory_type = "negative_preference"
                    template = "Preferencia negativa del usuario: no le gusta {}."
                    importance = 0.75

                tags = extract_tags(text, nlp_result)
                if "negative" not in tags:
                    tags = tags + ["negative"]

                return AnalysisResult(
                    should_remember=True,
                    reason="negative_lemma",
                    confidence=0.85,
                    importance=importance,
                    content=template.format(value),
                    memory_type=memory_type,
                    tags=tags,
                )

    return None


def detect_correction_memory(text: str, nlp_result: Optional[SpacyResult] = None) -> Optional[AnalysisResult]:
    """Detecta correcciones/actualizaciones de preferencias.

    Señales como 'en realidad', 'cambié de opinión', 'ahora prefiero'
    indican que el usuario está actualizando un hecho previo.
    Estas memorias se guardan con alta confianza y tag 'correction'
    para que _store() pueda archivar la versión anterior.
    """
    t = text.lower().strip()
    t_normalized = normalize_accents(t)

    # Check regex patterns first (most specific)
    for pattern, memory_type, template in _CORRECTION_PATTERNS:
        match = re.search(pattern, t_normalized)
        if not match:
            match = re.search(pattern, t)
        if match:
            groups = match.groups()
            value = template.format(*[g.strip(" .,:;") for g in groups])
            tags = extract_tags(text, nlp_result)
            if "correction" not in tags:
                tags = tags + ["correction"]
            return AnalysisResult(
                should_remember=True,
                reason="correction_pattern",
                confidence=0.95,
                importance=0.90,
                content=f"Actualización del usuario: {value}",
                memory_type=memory_type if memory_type != "preference_update" else (detect_type(text, nlp_result) or "positive_preference"),
                tags=tags,
            )

    # Check signal words (less specific, but still high confidence)
    for signal in _CORRECTION_SIGNALS:
        if signal in t or signal in t_normalized:
            tags = extract_tags(text, nlp_result)
            if "correction" not in tags:
                tags = tags + ["correction"]
            mtype = detect_type(text, nlp_result) or "positive_preference"
            return AnalysisResult(
                should_remember=True,
                reason="correction_signal",
                confidence=0.90,
                importance=0.85,
                content=normalize_text(text, mtype),
                memory_type=mtype,
                tags=tags,
            )

    return None


def detect_assistant_self_memory(text: str, nlp_result: Optional[SpacyResult] = None) -> Optional[AnalysisResult]:
    t = text.lower().strip()

    patterns = [
        # Negativos primero para evitar que "no me gusta" matchee como "me gusta"
        (
            r"\bno me gusta[n]?\s+([^.!?\n]+)",
            "assistant_negative_preference",
            "Al asistente no le gusta {}.",
            ["assistant", "dislikes"],
        ),
        (
            r"\bodio\s+([^.!?\n]+)",
            "assistant_negative_preference",
            "El asistente odia {}.",
            ["assistant", "dislikes"],
        ),
        (
            r"\bdetesto\s+([^.!?\n]+)",
            "assistant_negative_preference",
            "El asistente detesta {}.",
            ["assistant", "dislikes"],
        ),
        (
            r"\bmi color favorito es\s+([^.!?\n]+)",
            "assistant_preference",
            "El asistente tiene como color favorito {}.",
            ["assistant", "favorite_color"],
        ),
        (
            r"\bmi nombre es\s+([^.!?\n]+)",
            "assistant_identity",
            "El asistente dice que su nombre es {}.",
            ["assistant", "name"],
        ),
        (
            r"\bme llamo\s+([^.!?\n]+)",
            "assistant_identity",
            "El asistente dice que se llama {}.",
            ["assistant", "name"],
        ),
        (
            r"\bme gusta[n]?\s+([^.!?\n]+)",
            "assistant_preference",
            "Al asistente le gusta {}.",
            ["assistant", "likes"],
        ),
        (
            r"\bprefiero\s+([^.!?\n]+)",
            "assistant_preference",
            "El asistente prefiere {}.",
            ["assistant", "prefers"],
        ),
        (
            r"\bme encanta\s+([^.!?\n]+)",
            "assistant_preference",
            "Al asistente le encanta {}.",
            ["assistant", "likes"],
        ),
        (
            r"\bme encantaria\s+([^.!?\n]+)",
            "assistant_preference",
            "Al asistente le encantaria {}.",
            ["assistant", "likes", "desires"],
        ),
        (
            r"\bme gustaria\s+([^.!?\n]+)",
            "assistant_preference",
            "Al asistente le gustaria {}.",
            ["assistant", "likes", "desires"],
        ),
    ]

    for pattern, memory_type, template, tags in patterns:
        match = re.search(pattern, t)
        if not match:
            continue

        value = text[match.start(1):match.end(1)].strip(" .,:;")
        if not value:
            continue

        return AnalysisResult(
            should_remember=True,
            reason="assistant_self_pattern",
            confidence=0.90,
            importance=0.70,
            content=template.format(value),
            memory_type=memory_type,
            tags=tags,
        )

    # Fallback por lemma (spaCy) — solo matchea root_lemma, no todos los lemas
    if nlp_result and nlp_result.is_first_person and nlp_result.root_verb:
        root_lemma = nlp_result.root_lemma
        memory_type = None
        template = None
        tags = ["assistant"]

        if root_lemma in _LEMMA_MEMORY_HINTS.get("negative_preference", []):
            memory_type = "assistant_negative_preference"
            template = "Al asistente no le gusta {}."
            tags.append("dislikes")
        elif root_lemma in _LEMMA_MEMORY_HINTS.get("positive_preference", []):
            memory_type = "assistant_preference"
            template = "Al asistente le gusta {}."
            tags.append("likes")
        elif root_lemma in _LEMMA_MEMORY_HINTS.get("personal_identity", []):
            memory_type = "assistant_identity"
            template = "El asistente dice que {}."
            tags.append("identity")

        if memory_type and template:
            verb_text = nlp_result.root_verb
            idx = text.lower().find(verb_text)
            value = ""
            if idx >= 0:
                value = text[idx + len(verb_text):].strip(" .,:;!?")

            if not value and nlp_result.noun_chunks:
                value = nlp_result.noun_chunks[0]
            if not value and nlp_result.objects:
                value = nlp_result.objects[0]

            if value:
                if root_lemma in ("llamar", "llamarse"):
                    template = "El asistente dice que se llama {}."
                elif root_lemma == "vivir":
                    template = "El asistente dice que vive en {}."
                elif root_lemma == "gustar" and nlp_result.has_negation:
                    memory_type = "assistant_negative_preference"
                    template = "Al asistente no le gusta {}."
                    if "likes" in tags:
                        tags.remove("likes")
                    if "dislikes" not in tags:
                        tags.append("dislikes")

                return AnalysisResult(
                    should_remember=True,
                    reason="assistant_self_lemma",
                    confidence=0.85,
                    importance=0.70,
                    content=template.format(value),
                    memory_type=memory_type,
                    tags=tags,
                )

    return None


def _strip_question_tail(text: str) -> str:
    return re.split(r"\?", text, maxsplit=1)[0].strip()


def analyze_assistant_text(text: str) -> AnalysisResult:
    if is_noise(text):
        return AnalysisResult(should_remember=False, reason="noise")

    nlp_result = spacy_analyze(text)

    has_question = "?" in text or "\u00bf" in text
    t = text.lower()

    if has_question:
        candidate = _strip_question_tail(text)
        if candidate and candidate != text:
            candidate_nlp = spacy_analyze(candidate)
            self_memory = detect_assistant_self_memory(candidate, candidate_nlp)
            if self_memory:
                return self_memory
        return AnalysisResult(should_remember=False, reason="assistant_question")

    for marker in _ASSISTANT_ANSWER_MARKERS:
        if marker in t:
            return AnalysisResult(should_remember=False, reason="assistant_answer")

    self_memory = detect_assistant_self_memory(text, nlp_result)
    if self_memory:
        if nlp_result:
            triple = extract_fact_triple(nlp_result)
            if triple:
                self_memory.fact_key, _, self_memory.fact_value = triple
        return self_memory

    for pattern in _ASSISTANT_USER_FACT_PATTERNS:
        if re.search(pattern, t):
            for cond in _CONDITIONAL_MARKERS:
                if cond in t:
                    return AnalysisResult(should_remember=False, reason="assistant_conditional")
            tags = extract_tags(text, nlp_result)
            
            fact_key = ""
            fact_value = ""
            if nlp_result:
                triple = extract_fact_triple(nlp_result)
                if triple:
                    fact_key, _, fact_value = triple

            return AnalysisResult(
                should_remember=True,
                reason="assistant_user_fact",
                confidence=0.70,
                content=f"Afirmación del asistente sobre el usuario: {text.strip()}",
                memory_type="assistant_claim_about_user",
                tags=tags + ["assistant_claim"],
                fact_key=fact_key,
                fact_value=fact_value,
            )

    for pattern in _ASSISTANT_SELF_PATTERNS:
        if re.search(pattern, t):
            for cond in _CONDITIONAL_MARKERS:
                if cond in t:
                    return AnalysisResult(should_remember=False, reason="assistant_conditional")
            tags = extract_tags(text, nlp_result)
            
            fact_key = ""
            fact_value = ""
            if nlp_result:
                triple = extract_fact_triple(nlp_result)
                if triple:
                    fact_key, _, fact_value = triple

            return AnalysisResult(
                should_remember=True,
                reason="assistant_self_fact",
                confidence=0.70,
                content=f"Memoria del asistente: {text.strip()}",
                memory_type="assistant_preference",
                tags=tags,
                fact_key=fact_key,
                fact_value=fact_value,
            )

    return AnalysisResult(should_remember=False, reason="assistant_skip")


def analyze_text(text: str) -> AnalysisResult:
    if is_noise(text):
        return AnalysisResult(should_remember=False, reason="noise")

    nlp_result = spacy_analyze(text)

    explicit = extract_explicit(text)
    if explicit:
        explicit_nlp = spacy_analyze(explicit)
        return AnalysisResult(
            should_remember=True,
            reason="explicit",
            confidence=1.0,
            content=normalize_text(explicit, "explicit"),
            memory_type="explicit",
            tags=extract_tags(explicit, explicit_nlp),
        )

    if is_question_like(text, nlp_result):
        candidate = _strip_question_tail(text)
        if candidate and candidate != text.strip():
            candidate_score = memory_score(candidate)
            candidate_mtype = detect_type(candidate)
            candidate_explicit = extract_explicit(candidate)
            if not (candidate_explicit or candidate_score >= 0.40 or candidate_mtype):
                return AnalysisResult(should_remember=False, reason="question_like")
        else:
            return AnalysisResult(should_remember=False, reason="question_like")

    negative = detect_negative_memory(text, nlp_result)
    if negative:
        if nlp_result:
            triple = extract_fact_triple(nlp_result)
            if triple:
                negative.fact_key, _, negative.fact_value = triple
        return negative

    correction = detect_correction_memory(text, nlp_result)
    if correction:
        if nlp_result:
            triple = extract_fact_triple(nlp_result)
            if triple:
                correction.fact_key, _, correction.fact_value = triple
        return correction

    score = memory_score(text, nlp_result)
    mtype = detect_type(text, nlp_result)

    fact_key = ""
    fact_value = ""
    if nlp_result:
        triple = extract_fact_triple(nlp_result)
        if triple:
            fact_key, _, fact_value = triple

    if score >= 0.75 and mtype:
        return AnalysisResult(
            should_remember=True,
            reason="rule_score",
            confidence=min(score, 0.95),
            content=normalize_text(text, mtype),
            memory_type=mtype,
            tags=extract_tags(text, nlp_result),
            fact_key=fact_key,
            fact_value=fact_value,
        )

    if score >= 0.40:
        return AnalysisResult(
            should_remember=True,
            reason="possible",
            confidence=score,
            content=normalize_text(text, "pending"),
            memory_type="pending",
            tags=extract_tags(text, nlp_result),
            fact_key=fact_key,
            fact_value=fact_value,
        )

    return AnalysisResult(
        should_remember=False,
        reason="low_score",
        confidence=score,
        fact_key=fact_key,
        fact_value=fact_value,
    )


class SemanticMemory:
    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        sqlite_conn: Optional[sqlite3.Connection] = None,
        user_id: str = "default",
        namespace: str = "normal",
        scope: str = "user",
        allow_assistant_memory: bool = False,
    ):
        self._collection: Optional[Any] = None
        self._persist_dir = persist_dir
        self._conn: Optional[sqlite3.Connection] = sqlite_conn
        self._user_id = user_id
        self._namespace = namespace
        self._scope = scope
        self._allow_assistant_memory = allow_assistant_memory

    # --- Lifecycle -------------------------------------------------

    def _ensure_collection(self):
        if self._collection is not None:
            return
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb no est\u00e1 instalado. Ejecuta: pip install chromadb"
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

    @property
    def conn(self):
        if self._conn is None:
            raise RuntimeError("SemanticMemory requiere sqlite_conn")
        return self._conn

    def close(self):
        self._collection = None

    # --- Analyze ---------------------------------------------------

    @staticmethod
    def analyze(text: str) -> AnalysisResult:
        return analyze_text(text)

    # --- Chroma metadata helpers -----------------------------------

    def _update_chroma_metadata(self, chroma_id: str, updates: dict[str, Any]) -> None:
        item = self.collection.get(ids=[chroma_id])
        if not item["ids"]:
            return
        metadata = item["metadatas"][0] or {}
        metadata.update(updates)
        metadata["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.collection.update(
            ids=[chroma_id],
            metadatas=[metadata],
        )

    # --- CRUD ------------------------------------------------------

    def remember(
        self, text: str, source: str = "auto", msg_ids: str = "",
        source_role: str = "user",
    ) -> Optional[str]:
        if source_role == "assistant":
            if not self._allow_assistant_memory:
                return None
            result = analyze_assistant_text(text)
        else:
            result = self.analyze(text)
        if not result.should_remember:
            return None
        return self._store(result, source, text, msg_ids, source_role=source_role)

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

    def remember_user(self, text: str, source: str = "auto", msg_ids: str = "") -> Optional[str]:
        return self.remember(text, source=source, msg_ids=msg_ids, source_role="user")

    def remember_assistant(self, text: str, source: str = "auto", msg_ids: str = "") -> Optional[str]:
        return self.remember(text, source=source, msg_ids=msg_ids, source_role="assistant")

    def _store(
        self, result: AnalysisResult, source: str,
        original_text: str = "", msg_ids: str = "",
        source_role: str = "user",
    ) -> str:
        memory_id = f"mem_{uuid.uuid4().hex[:16]}"
        chroma_id = memory_id
        now = datetime.now(timezone.utc).isoformat()
        tags_str = "," + ",".join(result.tags) + "," if result.tags else ""
        status = "active" if result.confidence >= 0.75 else "pending_review"
        effective_scope = (
            "assistant_claims"
            if result.memory_type == "assistant_claim_about_user"
            else "assistant"
            if source_role == "assistant"
            else self._scope
        )

        coll = self.collection

        # --- Conflict resolution -----------------------------------
        existing = coll.query(
            query_texts=[result.content],
            n_results=1,
            where={
                "$and": [
                    {"namespace": {"$eq": self._namespace}},
                    {"scope": {"$eq": effective_scope}},
                    {"user_id": {"$eq": self._user_id}},
                    {"status": {"$in": ["active", "pending_review"]}},
                ],
            },
        )
        if existing["distances"] and existing["distances"][0]:
            dist = existing["distances"][0][0]
            if dist < 0.05:
                return existing["ids"][0][0]

        try:
            coll.add(
                ids=[chroma_id],
                documents=[result.content],
                metadatas=[{
                    "memory_id": memory_id,
                    "chroma_id": chroma_id,
                    "namespace": self._namespace,
                    "scope": effective_scope,
                    "user_id": self._user_id,
                    "tags": tags_str,
                    "confidence": result.confidence,
                    "memory_type": result.memory_type,
                    "status": status,
                    "source": source,
                    "source_role": source_role,
                    "original_text": original_text[:500] if original_text else "",
                    "created_at": now,
                }],
            )

            self.conn.execute("""
                INSERT INTO semantic_memories
                (memory_id, chroma_id, namespace, scope, content, original_text,
                 tags, confidence, importance, memory_type, status, source,
                 source_message_ids, source_role, user_id, created_at, updated_at,
                 fact_key, fact_value)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory_id, chroma_id, self._namespace, effective_scope,
                result.content, original_text[:500] if original_text else "",
                tags_str, result.confidence, result.importance,
                result.memory_type, status, source, msg_ids, source_role,
                self._user_id, now, now,
                getattr(result, "fact_key", ""), getattr(result, "fact_value", "")
            ))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            try:
                coll.delete(ids=[chroma_id])
            except Exception:
                pass
            raise

        return memory_id

    def archive(self, memory_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "UPDATE semantic_memories SET status='archived', updated_at=? WHERE memory_id=? AND namespace=? AND user_id=?",
            (now, memory_id, self._namespace, self._user_id),
        )
        self.conn.commit()
        row = self.conn.execute(
            "SELECT chroma_id FROM semantic_memories WHERE memory_id=? AND namespace=? AND user_id=?",
            (memory_id, self._namespace, self._user_id),
        ).fetchone()
        if row and row[0]:
            self._update_chroma_metadata(row[0], {"status": "archived"})

    def forget(self, memory_id: str) -> None:
        row = self.conn.execute(
            "SELECT chroma_id FROM semantic_memories WHERE memory_id=? AND namespace=? AND user_id=?",
            (memory_id, self._namespace, self._user_id),
        ).fetchone()
        if not row:
            return
        if row[0]:
            self.collection.delete(ids=[row[0]])
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "UPDATE semantic_memories SET status='deleted', updated_at=? WHERE memory_id=? AND namespace=? AND user_id=?",
            (now, memory_id, self._namespace, self._user_id),
        )
        self.conn.commit()

    def purge(self, memory_id: str) -> None:
        row = self.conn.execute(
            "SELECT chroma_id FROM semantic_memories WHERE memory_id=? AND namespace=? AND user_id=?",
            (memory_id, self._namespace, self._user_id),
        ).fetchone()
        if row and row[0]:
            self.collection.delete(ids=[row[0]])
        self.conn.execute(
            "DELETE FROM semantic_memories WHERE memory_id=? AND namespace=? AND user_id=?",
            (memory_id, self._namespace, self._user_id),
        )
        self.conn.commit()

    # --- Query -----------------------------------------------------

    def search(self, query: str, n_results: int = 5, scope: Optional[str] = None) -> list[MemoryRecord]:
        coll = self.collection
        target_scope = scope or self._scope
        results = coll.query(
            query_texts=[query],
            n_results=n_results,
            where={
                "$and": [
                    {"namespace": {"$eq": self._namespace}},
                    {"scope": {"$eq": target_scope}},
                    {"user_id": {"$eq": self._user_id}},
                    {"status": {"$eq": "active"}},
                ],
            },
        )
        records = self._to_records(results)

        boosted = False
        for r in records:
            mid = r.memory_id or r.chroma_id
            if mid:
                self._boost_importance(mid)
                boosted = True
        if boosted:
            self.conn.commit()

        return records

    def search_by_tags(self, tags: list[str], scope: Optional[str] = None) -> list[MemoryRecord]:
        if not tags:
            return []
        target_scope = scope or self._scope
        placeholders = " OR ".join("tags LIKE ?" for _ in tags)
        params: list[Any] = [self._namespace, target_scope, self._user_id] + [f"%,{t},%" for t in tags]
        rows = self.conn.execute(
            "SELECT * FROM semantic_memories WHERE namespace=? AND scope=? AND user_id=? AND status='active' AND ({}) ORDER BY importance DESC, created_at DESC".format(placeholders),
            params,
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def list_memories(self, limit: int = 50, scope: Optional[str] = None) -> list[MemoryRecord]:
        target_scope = scope or self._scope
        rows = self.conn.execute(
            "SELECT * FROM semantic_memories WHERE namespace=? AND scope=? AND user_id=? AND status != 'deleted' ORDER BY created_at DESC LIMIT ?",
            (self._namespace, target_scope, self._user_id, limit),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_memory(self, memory_id: str) -> Optional[MemoryRecord]:
        row = self.conn.execute(
            "SELECT * FROM semantic_memories WHERE memory_id=? AND namespace=? AND user_id=?",
            (memory_id, self._namespace, self._user_id),
        ).fetchone()
        return self._row_to_record(row) if row else None

    def count(self, scope: Optional[str] = None) -> int:
        target_scope = scope or self._scope
        return self.conn.execute(
            "SELECT COUNT(*) FROM semantic_memories WHERE namespace=? AND scope=? AND user_id=? AND status NOT IN ('deleted','archived')",
            (self._namespace, target_scope, self._user_id),
        ).fetchone()[0]

    # --- Review flow -----------------------------------------------

    def review_pending(self, limit: int = 10, scope: Optional[str] = None) -> list[MemoryRecord]:
        target_scope = scope or self._scope
        rows = self.conn.execute(
            "SELECT * FROM semantic_memories WHERE namespace=? AND scope=? AND user_id=? AND status='pending_review' ORDER BY created_at DESC LIMIT ?",
            (self._namespace, target_scope, self._user_id, limit),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def approve(self, memory_id: str) -> None:
        self.conn.execute(
            "UPDATE semantic_memories SET status='active', updated_at=? WHERE memory_id=?",
            (datetime.now(timezone.utc).isoformat(), memory_id),
        )
        self.conn.commit()
        row = self.conn.execute(
            "SELECT chroma_id FROM semantic_memories WHERE memory_id=?", (memory_id,)
        ).fetchone()
        if row and row[0]:
            self._update_chroma_metadata(row[0], {"status": "active"})

    def reject(self, memory_id: str) -> None:
        self.forget(memory_id)

    # --- Context builder -------------------------------------------

    def build_context(self, query: str, n_results: int = 5) -> str:
        user_mems = self.search(query, n_results=n_results, scope="user")
        assistant_mems = self.search(query, n_results=n_results, scope="assistant")
        assistant_claims = self.search(query, n_results=n_results, scope="assistant_claims")

        lines: list[str] = []

        lines.append("[MEMORY_RULES]")
        lines.append("- USER_MEMORY describe al usuario.")
        lines.append("- ASSISTANT_MEMORY describe al asistente.")
        lines.append("- ASSISTANT_CLAIMS_ABOUT_USER son afirmaciones del asistente sobre el usuario.")
        lines.append("- USER_MEMORY tiene m\u00e1s autoridad que ASSISTANT_CLAIMS_ABOUT_USER.")
        lines.append("- Si el usuario pregunta por \"mi\", usa USER_MEMORY.")
        lines.append("- Si el usuario pregunta por \"tu\", usa ASSISTANT_MEMORY.")
        lines.append("- No confundas memorias del usuario con memorias del asistente.")
        lines.append("")

        lines.append("[USER_MEMORY]")
        lines.extend(f"- {m.content}" for m in user_mems)
        if not user_mems:
            lines.append("- Sin memorias relevantes.")

        lines.append("")
        lines.append("[ASSISTANT_MEMORY]")
        lines.extend(f"- {m.content}" for m in assistant_mems)
        if not assistant_mems:
            lines.append("- Sin memorias relevantes.")

        lines.append("")
        lines.append("[ASSISTANT_CLAIMS_ABOUT_USER]")
        lines.extend(f"- {m.content}" for m in assistant_claims)
        if not assistant_claims:
            lines.append("- Sin memorias relevantes.")

        return "\n".join(lines)

    # --- Importance ------------------------------------------------

    def _boost_importance(self, memory_id: str, increment: float = 0.02, cap: float = 1.0) -> None:
        self.conn.execute(
            "UPDATE semantic_memories SET importance = MIN(importance + ?, ?) WHERE memory_id = ?",
            (increment, cap, memory_id),
        )

    # --- Internal --------------------------------------------------

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
            memory_id=meta.get("memory_id", mid),
            chroma_id=mid,
            content=doc,
            tags=[t for t in tags_str.split(",") if t],
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
            tags=[t for t in tags_str.split(",") if t],
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


# --- Migration ----------------------------------------------------

def migrate_long_term_to_semantic(
    conn: sqlite3.Connection, semantic: "SemanticMemory"
) -> int:
    rows = conn.execute(
        "SELECT id, content, tags, weight FROM long_term_memories"
    ).fetchall()
    migrated = 0
    for row in rows:
        content = row[1] if isinstance(row, (list, tuple)) else row["content"]
        tags_str = row[2] if isinstance(row, (list, tuple)) else row["tags"]
        weight = row[3] if isinstance(row, (list, tuple)) else row["weight"]

        existing = conn.execute(
            "SELECT 1 FROM semantic_memories WHERE content=? AND source='legacy'",
            (content,),
        ).fetchone()
        if existing:
            continue

        tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []
        result = AnalysisResult(
            should_remember=True,
            reason="legacy_migration",
            confidence=1.0,
            content=content,
            memory_type="explicit",
            tags=tags,
            importance=float(weight) if weight else 0.5,
        )
        semantic._store(result, source="legacy", original_text=content)
        migrated += 1

    return migrated
