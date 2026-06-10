"""Verification test for improved patterns.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.semantic_memory import (
    analyze_text, analyze_assistant_text, is_noise,
    extract_tags, detect_type, detect_negative_memory,
)

ok = 0
fail = 0

def check(label, got, expected):
    global ok, fail
    status = "OK" if got == expected else "FAIL"
    if status == "FAIL":
        fail += 1
        print(f"  FAIL  {label}: got={got} expected={expected}")
    else:
        ok += 1

print("=== 1. Noise filter ===")
check("hola", is_noise("hola"), True)
check("buenos dias", is_noise("buenos dias"), True)
check("xd", is_noise("xd"), True)
check("nos vemos", is_noise("nos vemos"), True)
check("jajajaja", is_noise("jajajaja"), True)
check("cool", is_noise("cool"), True)
check("no emojis", is_noise("no emojis"), False)
check("sin markdown", is_noise("sin markdown"), False)
check("me gusta python", is_noise("Me gusta programar en Python"), False)

print("\n=== 2. Tag extraction ===")
tags = extract_tags("Estoy usando React con TypeScript en Docker")
check("react in tags", "react" in tags, True)
check("typescript in tags", "typescript" in tags, True)
check("docker in tags", "docker" in tags, True)

tags2 = extract_tags("Mi perro se llama Max y vivo en Mexico")
check("mascotas in tags2", "mascotas" in tags2, True)
check("ubicacion in tags2", "ubicacion" in tags2, True)

tags3 = extract_tags("Uso PyTorch con CUDA en Ubuntu")
check("pytorch in tags3", "pytorch" in tags3, True)
check("cuda in tags3", "cuda" in tags3, True)
check("linux in tags3", "linux" in tags3, True)

print("\n=== 3. detect_type ===")
check("negative_inst", detect_type("no quiero que uses emojis"), "negative_instruction")
check("negative_pref", detect_type("odio el JavaScript"), "negative_preference")
check("positive_pref", detect_type("me gusta Python"), "positive_preference")
check("project_fact", detect_type("mi proyecto se llama SIAS"), "project_fact")
check("environment", detect_type("uso wsl con Ubuntu"), "environment")
check("personal_id", detect_type("vivo en Mexico"), "personal_identity")
check("asst_inst", detect_type("de ahora en adelante responde en ingles"), "assistant_instruction")
check("deja de", detect_type("deja de usar markdown"), "negative_instruction")

print("\n=== 4. Negative patterns ===")
neg1 = detect_negative_memory("deja de usar emojis")
check("deja de", neg1 is not None and neg1.should_remember, True)

neg2 = detect_negative_memory("no vuelvas a incluir tablas")
check("no vuelvas", neg2 is not None and neg2.should_remember, True)

neg3 = detect_negative_memory("me desagrada el codigo desordenado")
check("desagrada", neg3 is not None and neg3.should_remember, True)

neg4 = detect_negative_memory("me aburre la programacion funcional")
check("aburre", neg4 is not None and neg4.should_remember, True)

print("\n=== 5. Assistant analysis ===")
# Should NOT remember
r = analyze_assistant_text("Claro, te explico paso a paso")
check("explain skip", r.should_remember, False)

r = analyze_assistant_text("Quieres que te ayude con eso?")
check("question skip", r.should_remember, False)

r = analyze_assistant_text("Si quisieras, podriamos intentarlo")
check("conditional skip", r.should_remember, False)

r = analyze_assistant_text("Como modelo de lenguaje no puedo hacer eso")
check("disclaimer skip", r.should_remember, False)

# SHOULD remember (self-facts)
r = analyze_assistant_text("Mi color favorito es el azul")
check("self color", r.should_remember, True)
check("self color type", r.memory_type, "assistant_preference")

r = analyze_assistant_text("No me gusta el cafe amargo")
check("self dislike", r.should_remember, True)

r = analyze_assistant_text("Me llamo Luna")
check("self name", r.should_remember, True)

# SHOULD remember (user facts)
r = analyze_assistant_text("Tu nombre es Pablo y te gusta Python")
check("user fact", r.should_remember, True)
check("user fact type", r.memory_type, "assistant_claim_about_user")

r = analyze_assistant_text("Tu mascota se llama Luna")
check("user pet", r.should_remember, True)

print("\n=== 6. User analysis (analyze_text) ===")
r = analyze_text("recuerda que mi color favorito es el verde")
check("explicit", r.should_remember, True)
check("explicit type", r.memory_type, "explicit")

r = analyze_text("Como se hace una funcion en Python?")
check("question skip", r.should_remember, False)

r = analyze_text("no me gusta JavaScript")
check("negative pref", r.should_remember, True)

r = analyze_text("mi proyecto se llama SIAS y estoy usando FastAPI")
check("project", r.should_remember, True)

print("\n=== 7. Correction detection ===")
from src.semantic_memory import detect_correction_memory

c1 = detect_correction_memory("ahora prefiero TypeScript en vez de JavaScript")
check("correction pattern", c1 is not None and c1.should_remember, True)
if c1:
    check("correction tag", "correction" in c1.tags, True)
    check("correction conf", c1.confidence >= 0.90, True)

c2 = detect_correction_memory("en realidad mi color favorito es el azul")
check("correction signal", c2 is not None and c2.should_remember, True)
if c2:
    check("signal tag", "correction" in c2.tags, True)

c3 = detect_correction_memory("antes usaba Windows, ahora uso Linux")
check("before/now", c3 is not None and c3.should_remember, True)

c4 = detect_correction_memory("cambié de opinión sobre Python")
check("changed mind", c4 is not None and c4.should_remember, True)

c5 = detect_correction_memory("me gusta Python")
check("not correction", c5 is None, True)

print("\n=== 8. Accent normalization ===")
from src.patterns import normalize_accents

check("cafe", normalize_accents("café"), "cafe")
check("mexico", normalize_accents("México"), "mexico")
check("naci", normalize_accents("Nací en"), "naci en")
check("no change", normalize_accents("hola"), "hola")

print("\n=== 9. New noise words ===")
check("...", is_noise("..."), True)
check("hmm", is_noise("hmm"), True)
check("cuídate", is_noise("cuídate"), True)
check("tremendo", is_noise("tremendo"), True)
check("solo codigo", is_noise("solo codigo"), False)

print("\n=== 10. Expanded tags ===")
tags_new = extract_tags("Uso Angular con Firebase desplegado en Vercel")
check("angular", "angular" in tags_new, True)
check("firebase", "firebase" in tags_new, True)
check("vercel", "vercel" in tags_new, True)

tags_lang = extract_tags("Trabajo con LangChain y embeddings de Claude")
check("langchain", "langchain" in tags_lang, True)
check("claude", "claude" in tags_lang, True)
check("embeddings", "embeddings" in tags_lang, True)

print("\n=== 11. Assistant code block filter ===")
r = analyze_assistant_text("```python\ndef hello():\n    pass\n```")
check("code block skip", r.should_remember, False)

r = analyze_assistant_text("import os; print('hello')")
check("import skip", r.should_remember, False)

print("\n=== 12. spaCy integration tests ===")
from src.nlp_engine import is_spacy_available, get_analyzer, extract_fact_triple

if is_spacy_available():
    analyzer = get_analyzer()
    check("spaCy available", is_spacy_available(), True)
    
    # Test lemmatization and negation
    r = analyzer.analyze("No me gustaban las fresas")
    check("lemma root_lemma", r.root_lemma if r else "", "gustar")
    check("lemma has_negation", r.has_negation if r else False, True)
    
    # Test extract_fact_triple
    triple = extract_fact_triple(r)
    check("triple structure", triple is not None, True)
    if triple:
        check("triple verb", triple[1], "gustar")
        check("triple object", "fresas" in triple[2], True)
        
    # Test analyze_text with spaCy negative lemma fallback
    r_neg = analyze_text("No me gustaban las fresas")
    check("lemma negative should_remember", r_neg.should_remember, True)
    check("lemma negative memory_type", r_neg.memory_type, "negative_preference")
    check("lemma negative fact_key", r_neg.fact_key, "me")
    check("lemma negative fact_value", r_neg.fact_value, "fresas")
    
    # Test analyze_assistant_text with assistant self preference lemma fallback
    r_asst = analyze_assistant_text("Me encantaban los videojuegos")
    check("asst lemma should_remember", r_asst.should_remember, True)
    check("asst lemma memory_type", r_asst.memory_type, "assistant_preference")
    check("asst lemma fact_key", r_asst.fact_key, "me")
    check("asst lemma fact_value", r_asst.fact_value, "videojuegos")
else:
    print("spaCy is not available. Skipping spaCy integration tests.")

print(f"\n=== Results: {ok} passed, {fail} failed ===")
if fail > 0:
    sys.exit(1)
print("ALL CHECKS PASSED")
