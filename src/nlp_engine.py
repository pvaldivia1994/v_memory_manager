"""
Motor NLP basado en spaCy para análisis lingüístico avanzado.

Proporciona lemmatización, NER, dependency parsing y pattern matching
por lemma/POS como capa opcional sobre el sistema de regex existente.

Si spaCy no está instalado, todas las funciones retornan None/valores
vacíos para permitir fallback transparente al sistema regex.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Estado global del módulo ──────────────────────────────────

_spacy_available: Optional[bool] = None
_nlp_instance: Any = None
_model_name: str = "es_core_news_sm"


def _check_spacy() -> bool:
    """Verifica si spaCy está disponible (cacheado)."""
    global _spacy_available
    if _spacy_available is None:
        try:
            import spacy  # noqa: F401
            _spacy_available = True
        except ImportError:
            _spacy_available = False
            logger.info("spaCy no está instalado. Usando solo regex.")
    return _spacy_available


def is_spacy_available() -> bool:
    """Retorna True si spaCy está disponible."""
    return _check_spacy()


# ── Dataclasses ───────────────────────────────────────────────

@dataclass
class Entity:
    """Entidad detectada por NER."""
    text: str
    label: str  # PER, LOC, ORG, MISC
    start: int = 0
    end: int = 0


@dataclass
class SpacyResult:
    """Resultado del análisis spaCy de un texto."""
    lemmas: list[str] = field(default_factory=list)
    lemma_text: str = ""
    entities: list[Entity] = field(default_factory=list)
    root_verb: str = ""
    root_lemma: str = ""
    subject: str = ""
    objects: list[str] = field(default_factory=list)
    is_first_person: bool = False
    is_second_person: bool = False
    pos_tags: list[tuple[str, str]] = field(default_factory=list)
    sentence_type: str = "declarative"  # declarative, interrogative, imperative
    has_negation: bool = False
    noun_chunks: list[str] = field(default_factory=list)


_EMPTY_RESULT = SpacyResult()


# ── SpacyAnalyzer ─────────────────────────────────────────────

class SpacyAnalyzer:
    """Analizador NLP usando spaCy con carga lazy del modelo.

    Uso:
        analyzer = SpacyAnalyzer()
        result = analyzer.analyze("Me gusta el chocolate")
        if result:
            print(result.root_lemma)   # "gustar"
            print(result.entities)     # []
            print(result.is_first_person)  # True
    """

    def __init__(self, model_name: str = "es_core_news_sm"):
        self._model_name = model_name
        self._nlp: Any = None
        self._available: Optional[bool] = None

    @property
    def available(self) -> bool:
        """Retorna si spaCy y el modelo están disponibles."""
        if self._available is None:
            self._available = self._try_load()
        return self._available

    @property
    def nlp(self) -> Any:
        """Retorna la instancia del modelo spaCy (lazy load)."""
        if self._nlp is None:
            self._try_load()
        return self._nlp

    def _try_load(self) -> bool:
        """Intenta cargar el modelo de spaCy."""
        if not _check_spacy():
            self._available = False
            return False
        try:
            import spacy
            self._nlp = spacy.load(self._model_name)
            self._available = True
            logger.info("Modelo spaCy '%s' cargado.", self._model_name)
            return True
        except OSError:
            logger.warning(
                "Modelo spaCy '%s' no encontrado. "
                "Instálalo con: python -m spacy download %s",
                self._model_name, self._model_name,
            )
            self._available = False
            return False
        except Exception as e:
            logger.warning("Error cargando spaCy: %s", e)
            self._available = False
            return False

    def analyze(self, text: str) -> Optional[SpacyResult]:
        """Analiza un texto y retorna un SpacyResult, o None si no disponible."""
        if not self.available or self._nlp is None:
            return None

        doc = self._nlp(text)
        result = SpacyResult()

        # Lemmas
        result.lemmas = [token.lemma_.lower() for token in doc]
        result.lemma_text = " ".join(result.lemmas)

        # POS tags
        result.pos_tags = [(token.text, token.pos_) for token in doc]

        # NER
        result.entities = [
            Entity(text=ent.text, label=ent.label_, start=ent.start_char, end=ent.end_char)
            for ent in doc.ents
        ]

        # Noun chunks
        try:
            result.noun_chunks = [chunk.text for chunk in doc.noun_chunks]
        except Exception:
            pass

        # Root verb + subject + objects via dependency parsing
        root_token = None
        for token in doc:
            if token.dep_ == "ROOT":
                root_token = token
                result.root_verb = token.text.lower()
                result.root_lemma = token.lemma_.lower()
                break

        # Check for copula
        if root_token:
            for child in root_token.children:
                if child.dep_ == "cop":
                    # Treat the root token as an object, and copula as root verb
                    result.objects.append(root_token.text.lower())
                    result.root_verb = child.text.lower()
                    result.root_lemma = child.lemma_.lower()
                    break

        # Sujeto y objetos
        for token in doc:
            if token.dep_ in ("nsubj", "nsubj:pass"):
                result.subject = token.text.lower()
            elif token.dep_ in ("obj", "obl", "iobj", "dobj"):
                result.objects.append(token.text.lower())

        # Persona (primera / segunda)
        first_person_pronouns = {"yo", "me", "mi", "mí", "nos", "nuestro", "nuestra"}
        second_person_pronouns = {"tú", "tu", "te", "ti", "usted", "ustedes"}
        tokens_lower = {t.text.lower() for t in doc}
        result.is_first_person = bool(tokens_lower & first_person_pronouns)
        result.is_second_person = bool(tokens_lower & second_person_pronouns)

        # Negación
        result.has_negation = any(
            token.dep_ == "advmod" and token.lemma_.lower() == "no"
            for token in doc
        )

        # Tipo de oración
        if any(t.text in ("?", "¿") for t in doc):
            result.sentence_type = "interrogative"
        elif doc[-1].text == "!" if len(doc) > 0 else False:
            result.sentence_type = "exclamative"
        else:
            # Detectar imperativo por POS del root
            for token in doc:
                if token.dep_ == "ROOT" and token.morph.get("Mood") == ["Imp"]:
                    result.sentence_type = "imperative"
                    break

        return result


# ── Funciones de utilidad ─────────────────────────────────────

def extract_entities_as_tags(result: Optional[SpacyResult]) -> list[str]:
    """Convierte las entidades NER de un SpacyResult en tags.

    "Vivo en México" → ["ubicacion:México"]
    "Me llamo Pablo" → ["persona:Pablo"]
    """
    if not result or not result.entities:
        return []

    from .patterns import _NER_TAG_MAP

    tags: list[str] = []
    for ent in result.entities:
        tag_prefix = _NER_TAG_MAP.get(ent.label)
        if tag_prefix:
            tags.append(f"{tag_prefix}:{ent.text}")
    return tags


def lemmatized_match(result: Optional[SpacyResult], lemma_patterns: list[str]) -> bool:
    """Verifica si alguno de los lemmas del texto coincide con los patterns.

    result contiene lemmas como ["gustar", "chocolate"]
    lemma_patterns como ["gustar", "encantar"]
    → True si hay intersección.
    """
    if not result or not result.lemmas:
        return False
    lemma_set = set(result.lemmas)
    return bool(lemma_set & set(lemma_patterns))


def extract_fact_triple(
    result: Optional[SpacyResult],
) -> Optional[tuple[str, str, str]]:
    """Extrae un triple (sujeto, relación, objeto) del análisis.

    "Mi color favorito es el azul" → ("mi color favorito", "ser", "azul")
    "Me gusta Python" → ("me", "gustar", "python")

    Retorna None si no puede extraer un triple válido.
    """
    if not result or not result.root_lemma:
        return None

    subject = result.subject or ""
    verb = result.root_lemma
    objects = result.objects or []

    # Psych-verbs y verbos de preferencia comunes en español
    psych_verbs = {
        "gustar", "encantar", "molestar", "aburrir", "incomodar", 
        "desagradar", "disgustar", "atraer", "molar", "fascinar", "apasionar"
    }

    if (result.is_first_person or "me" in objects or "yo" in objects) and verb != "ser":
        if verb in psych_verbs:
            experiencer = "me"
            target = ""
            if subject and subject != "me":
                target = subject
            else:
                other_objs = [o for o in objects if o != "me" and o != "yo"]
                if other_objs:
                    target = other_objs[0]
            if target:
                return (experiencer, verb, target)
        else:
            # Transitive verb: experiencer is "me", target is first non-personal object
            experiencer = "me"
            target = ""
            other_objs = [o for o in objects if o != "me" and o != "yo"]
            if other_objs:
                target = other_objs[0]
            elif subject and subject != "me" and subject != "yo":
                target = subject
            if target:
                return (experiencer, verb, target)

    obj = objects[0] if objects else ""
    if not subject and not obj:
        return None

    return (subject, verb, obj)


# ── Singleton global ──────────────────────────────────────────

_global_analyzer: Optional[SpacyAnalyzer] = None


def get_analyzer(model_name: str = "es_core_news_sm") -> SpacyAnalyzer:
    """Retorna el singleton global del analizador spaCy."""
    global _global_analyzer
    if _global_analyzer is None:
        _global_analyzer = SpacyAnalyzer(model_name=model_name)
    return _global_analyzer
