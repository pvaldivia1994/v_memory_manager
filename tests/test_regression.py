"""Regression tests — verify core analysis rules haven't regressed."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.semantic_memory import (
    analyze_text,
    analyze_assistant_text,
    is_noise,
)


def _check(label: str, ok: bool) -> None:
    status = "\u2713" if ok else "FAIL"
    print(f"  {status} {label}" if ok else f"  {status} {label}")
    assert ok, f"FAIL: {label}"


print("=== User memory ===")

r = analyze_text("Me gustan las galletas de chocolate")
_check("positive_preference -> positive_preference", r.memory_type == "positive_preference")
_check("should_remember", r.should_remember)

r = analyze_text("No me gustan las respuestas largas")
_check("negative_preference -> negative_preference", r.memory_type == "negative_preference")
_check("should_remember", r.should_remember)

r = analyze_text("No quiero que uses emojis")
_check("negative_instruction -> negative_instruction", r.memory_type == "negative_instruction")
_check("should_remember", r.should_remember)

r = analyze_text("\u00bfC\u00f3mo hago una memoria sem\u00e1ntica?")
_check("question -> should_remember False", not r.should_remember)
_check("reason == question_like", r.reason == "question_like")

r = analyze_text("\u00bfPuedes recordar que me gusta el chocolate?")
_check("explicit question -> should_remember False", not r.should_remember)

r = analyze_text("/remember que me gusta el chocolate")
_check("explicit prefix -> should_remember", r.should_remember)

r = analyze_text("ok")
_check("noise -> should_remember False", not r.should_remember)

r = analyze_text("Sin tablas por favor")
_check("negative_instruction via sin", r.memory_type == "negative_instruction")
_check("should_remember", r.should_remember)

r = analyze_text("Ya no me gusta Python")
_check("ya no -> negative_preference", r.memory_type == "negative_preference")
_check("content includes ya no", "ya no" in r.content.lower())

r = analyze_text("Odio el color amarillo")
_check("odio -> negative_preference", r.memory_type == "negative_preference")
_check("should_remember", r.should_remember)

r = analyze_text("Prefiero Python")
_check("positive_preference -> positive_preference", r.memory_type == "positive_preference")

r = analyze_text("Estoy creando un juego con Unity")
_check("project_fact -> project_fact", r.memory_type == "project_fact")

print()

print("=== Assistant memory ===")

r = analyze_assistant_text("Aqu\u00ed tienes una receta con ingredientes y pasos...")
_check("recipe -> should_remember False", not r.should_remember)
_check("reason", r.reason in ("assistant_answer", "assistant_skip"))

r = analyze_assistant_text("Mi color favorito es rojo.")
_check("self fact -> assistant_preference", r.memory_type == "assistant_preference")
_check("should_remember", r.should_remember)
_check("extracted fact not full text", len(r.content) < 50)

r = analyze_assistant_text("Mi color favorito es rojo. \u00bfY el tuyo?")
_check("self fact with question tail -> assistant_preference", r.memory_type == "assistant_preference")
_check("should_remember", r.should_remember)

r = analyze_assistant_text("\u00bfTe gustar\u00eda que te explique m\u00e1s?")
_check("question -> should_remember False", not r.should_remember)

r = analyze_assistant_text("Te gusta programar en Python, \u00bfverdad?")
_check("user fact with question -> should_remember False", not r.should_remember)

r = analyze_assistant_text("Tu nombre es Pablo")
_check("user fact -> assistant_claim_about_user", r.memory_type == "assistant_claim_about_user")
_check("content prefix", "Afirmaci\u00f3n del asistente" in r.content)

r = analyze_assistant_text("No me gusta el ruido")
_check("assistant dislike -> assistant_negative_preference", r.memory_type == "assistant_negative_preference")
_check("should_remember", r.should_remember)

r = analyze_assistant_text("S\u00ed, me gustan los n\u00fameros")
_check("me gustan plural -> assistant_preference", r.memory_type == "assistant_preference")
_check("should_remember", r.should_remember)
_check("extracted fact", "n\u00fameros" in r.content)

r = analyze_assistant_text("Me llamo Juan")
_check("name -> assistant_identity", r.memory_type == "assistant_identity")
_check("extracted fact", "Juan" in r.content)

r = analyze_assistant_text("Claro, puedo ayudarte con eso")
_check("generic -> should_remember False", not r.should_remember)

r = analyze_assistant_text("\u00a1Ah, s\u00ed! Conozco una receta...")
_check("short recipe -> should_remember False", not r.should_remember)

print()
print("=== Edge cases ===")

_check("is_noise('no')", is_noise("no"))
_check("is_noise('hola')", is_noise("hola"))
_check("not is_noise('no emojis')", not is_noise("no emojis"))
_check("not is_noise('sin markdown')", not is_noise("sin markdown"))

r = analyze_text("Quiero que respondas corto")
_check("quiero que -> assistant_instruction", r.memory_type == "assistant_instruction")

r = analyze_assistant_text("Si te gusta Python, puedes probar...")
_check("conditional -> should_remember False", not r.should_remember)

r = analyze_assistant_text("Mi sueno es viajar por el mundo")
_check("sueno fallback -> 0.70 confidence", abs(r.confidence - 0.70) < 0.01)

r = analyze_text("Me llamo pedro")
_check("me llamo -> positive_preference", r.memory_type == "positive_preference")
_check("should_remember", r.should_remember)

print()
print("=== ALL REGRESSION TESTS PASSED ===")
