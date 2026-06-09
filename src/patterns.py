"""
Patrones y constantes para la detección de memorias.

Organización:
  1. Ruido y filtros rápidos
  2. Prefijos explícitos (/remember, recuerda que...)
  3. Hints por tipo de memoria (usuario)
  4. Patrones negativos (regex con captura)
  5. Patrones de corrección/actualización temporal
  6. Keywords para tags
  7. Patrones del asistente (facts sobre usuario + self-facts)
  8. Marcadores de filtro del asistente
"""

from __future__ import annotations

# ══════════════════════════════════════════════════════════════
# 1. RUIDO Y FILTROS RÁPIDOS
# ══════════════════════════════════════════════════════════════

_NOISE = {
    # Saludos
    "hola", "buenas", "buenos días", "buenos dias",
    "buenas tardes", "buenas noches",
    "hey", "hi", "hello", "sup",
    "qué tal", "que tal", "cómo estás", "como estas",
    "qué onda", "que onda", "qué hay", "que hay",
    # Confirmaciones
    "ok", "okay", "okey", "dale", "perfecto", "listo", "claro", "vale",
    "entendido", "de acuerdo", "ya", "ajá", "aja", "va",
    "correcto", "exacto", "así es", "asi es",
    # Agradecimientos
    "gracias", "muchas gracias", "thanks", "thx", "ty",
    "te lo agradezco", "mil gracias",
    # Afirmaciones / negaciones
    "sí", "si", "no", "nop", "nope", "sep", "sip", "nel", "simón", "simon",
    # Risas / filler
    "jajaja", "jajajaja", "jeje", "jejeje", "lol", "xd", "xdd",
    "jaja", "ja", "haha", "hahaha", "lmao", "rofl",
    # Continuadores
    "continua", "continúa", "sigue", "adelante", "siguiente",
    "go", "vamos", "dale", "ándale", "andale",
    # Evaluaciones cortas
    "excelente", "genial", "sale", "bueno", "bien", "nice",
    "cool", "wow", "increíble", "increible", "brutal", "épico",
    "chido", "bacán", "bacan", "maravilloso", "buenísimo", "buenisimo",
    "tremendo", "estupendo", "fantástico", "fantastico",
    # Despedidas
    "adiós", "adios", "chao", "bye", "hasta luego", "nos vemos",
    "hasta pronto", "cuídate", "cuidate", "chau",
    # Vacíos / emojis solos / puro formato
    "...", "..", "???", "!!!", "hmm", "mmm", "ah", "oh", "uh",
}

# Frases cortas (<8 chars) que SÍ son instrucciones válidas
_SHORT_MEANINGFUL = {
    "no emojis", "sin emojis",
    "no markdown", "sin markdown",
    "no tablas", "sin tablas",
    "sin explicaciones largas", "sin tanto texto",
    "sé breve", "se breve", "sé conciso", "se conciso",
    "en inglés", "en ingles", "en español",
    "solo código", "solo codigo", "solo texto",
}


# ══════════════════════════════════════════════════════════════
# 2. PREFIJOS EXPLÍCITOS
# ══════════════════════════════════════════════════════════════

_EXPLICIT_PREFIXES = [
    "/remember", "/mem", "/recuerda",
    "recuerda que", "acuérdate de que", "acuerdate de que",
    "guarda esto", "guarda que", "ten en cuenta que",
    "no olvides que", "anota que", "apunta que",
    "quiero que recuerdes", "necesito que recuerdes",
    "importante:", "dato:",
]


# ══════════════════════════════════════════════════════════════
# 3. HINTS POR TIPO DE MEMORIA (USUARIO)
# ══════════════════════════════════════════════════════════════

_MEMORY_HINTS: dict[str, list[str]] = {
    "negative_instruction": [
        "no quiero que", "nunca hagas", "nunca uses",
        "evita", "no uses", "no hagas", "no me respondas",
        "deja de", "para de", "no vuelvas a",
        "sin emojis", "sin markdown", "sin tablas",
        "sin tanto texto", "sin explicaciones largas",
        "no me hables de", "no me digas",
        "no incluyas", "no agregues", "no añadas",
        "no menciones", "omite", "elimina",
    ],
    "negative_preference": [
        "no me gusta", "no me gustan",
        "odio", "detesto", "no soporto",
        "me molesta", "me incomoda",
        "me desagrada", "me disgusta",
        "no tolero", "me cae mal",
        "me aburre", "me aburren",
        "me da asco", "me da pereza",
        "no aguanto", "me harta",
    ],
    "assistant_instruction": [
        "quiero que", "me gustaría que",
        "de ahora en adelante", "a partir de ahora",
        "siempre que", "para futuras conversaciones",
        "responde siempre", "respóndeme siempre",
        "háblame en", "hablame en",
        "usa siempre", "utiliza siempre",
        "compórtate como", "actúa como",
        "cuando te pregunte", "cada vez que",
        "tu tono debe ser", "sé más",
        "responde como", "actúa de",
    ],
    "positive_preference": [
        "me gusta", "me gustan", "me encanta", "me encantan",
        "prefiero", "mi favorito", "mi favorita",
        "favorito", "favorita", "suelo usar",
        "me llamo", "mi nombre es",
        "me apasiona", "me fascina",
        "amo", "adoro", "me mola",
        "disfruto", "me divierte",
        "me interesa", "me atrae",
    ],
    "project_fact": [
        "estoy creando", "estoy desarrollando", "mi proyecto",
        "mi app", "mi juego", "mi librería", "se llama",
        "estoy haciendo", "estoy programando", "estoy armando",
        "estoy trabajando en", "mi repositorio", "mi repo",
        "mi api", "mi backend", "mi frontend",
        "mi base de datos", "mi servidor",
        "mi página web", "mi sitio web", "mi bot",
        "mi plugin", "mi extensión", "mi script",
    ],
    "environment": [
        "mi pc tiene", "tengo una", "uso windows", "uso linux",
        "uso wsl", "trabajo con", "uso mac", "uso macos",
        "mi computadora", "mi laptop", "mi ordenador",
        "mi gpu es", "mi tarjeta gráfica",
        "tengo instalado", "mi versión de",
        "mi editor es", "uso vscode", "uso vim", "uso neovim",
        "mi terminal es", "mi shell es",
        "mi ram es", "mi procesador es", "mi cpu es",
        "mi monitor", "mi teclado", "mi mouse",
        "mi resolución es", "mi pantalla",
    ],
    "personal_identity": [
        "tengo años", "mi edad es",
        "vivo en", "soy de", "nací en", "naci en",
        "trabajo como", "soy programador", "soy desarrollador",
        "soy ingeniero", "soy diseñador", "soy artista",
        "estudio", "soy estudiante",
        "hablo", "mi idioma", "mi lengua",
        "mi cumpleaños es", "mi cumpleanos es",
        "mi signo es", "mi signo zodiacal",
    ],
}

# Orden de prioridad para detect_type()
# Los negativos van primero para que "no me gusta X" no matchee como "me gusta X"
_DETECT_TYPE_PRIORITY = [
    "negative_instruction",
    "negative_preference",
    "assistant_instruction",
    "personal_identity",
    "positive_preference",
    "project_fact",
    "environment",
]


# ══════════════════════════════════════════════════════════════
# 4. PATRONES NEGATIVOS (regex con captura)
# ══════════════════════════════════════════════════════════════

_NEGATIVE_PATTERNS: list[tuple[str, str, str]] = [
    # Actualizaciones ("ya no...")
    (r"\bya no me gusta\s+(.+)", "negative_preference",
     "Preferencia actualizada del usuario: ya no le gusta {}."),
    (r"\bya no me gustan\s+(.+)", "negative_preference",
     "Preferencia actualizada del usuario: ya no le gustan {}."),
    (r"\bya no quiero\s+(.+)", "negative_instruction",
     "Restricción actualizada del usuario: ya no quiere {}."),
    (r"\bya no uso\s+(.+)", "environment",
     "Entorno actualizado del usuario: ya no usa {}."),
    (r"\bdeja de\s+(.+)", "negative_instruction",
     "Restricción persistente del usuario: debe dejar de {}."),
    (r"\bpara de\s+(.+)", "negative_instruction",
     "Restricción persistente del usuario: debe parar de {}."),
    (r"\bno vuelvas a\s+(.+)", "negative_instruction",
     "Restricción persistente del usuario: no debe volver a {}."),

    # Disgustos directos
    (r"\bno me gusta\s+(.+)", "negative_preference",
     "Preferencia negativa del usuario: no le gusta {}."),
    (r"\bno me gustan\s+(.+)", "negative_preference",
     "Preferencia negativa del usuario: no le gustan {}."),
    (r"\bodio\s+(.+)", "negative_preference",
     "Preferencia negativa del usuario: odia {}."),
    (r"\bdetesto\s+(.+)", "negative_preference",
     "Preferencia negativa del usuario: detesta {}."),
    (r"\bno soporto\s+(.+)", "negative_preference",
     "Preferencia negativa del usuario: no soporta {}."),
    (r"\bme molesta\s+(.+)", "negative_preference",
     "Preferencia negativa del usuario: le molesta {}."),
    (r"\bme incomoda\s+(.+)", "negative_preference",
     "Preferencia negativa del usuario: le incomoda {}."),
    (r"\bme desagrada\s+(.+)", "negative_preference",
     "Preferencia negativa del usuario: le desagrada {}."),
    (r"\bme aburre\s+(.+)", "negative_preference",
     "Preferencia negativa del usuario: le aburre {}."),
    (r"\bno aguanto\s+(.+)", "negative_preference",
     "Preferencia negativa del usuario: no aguanta {}."),
    (r"\bme harta\s+(.+)", "negative_preference",
     "Preferencia negativa del usuario: le harta {}."),

    # Instrucciones restrictivas
    (r"\bno quiero que\s+(.+)", "negative_instruction",
     "Restricción persistente del usuario: no quiere que {}."),
    (r"\bevita\s+(.+)", "negative_instruction",
     "Restricción persistente del usuario: debe evitar {}."),
    (r"\bno uses\s+(.+)", "negative_instruction",
     "Restricción persistente del usuario: no debe usar {}."),
    (r"\bno hagas\s+(.+)", "negative_instruction",
     "Restricción persistente del usuario: no debe hacer {}."),
    (r"\bno incluyas\s+(.+)", "negative_instruction",
     "Restricción persistente del usuario: no debe incluir {}."),
    (r"\bno me hables de\s+(.+)", "negative_instruction",
     "Restricción persistente del usuario: no le hables de {}."),
    (r"\bno menciones\s+(.+)", "negative_instruction",
     "Restricción persistente del usuario: no debe mencionar {}."),
    (r"\bomite\s+(.+)", "negative_instruction",
     "Restricción persistente del usuario: debe omitir {}."),

    # Formato / estilo
    (r"\bsin\s+(emojis|emoji|markdown|tablas|tanto texto|explicaciones largas|relleno|rodeos|formato|código|codigo)\b",
     "negative_instruction",
     "Restricción persistente del usuario: no quiere respuestas con {}."),
]


# ══════════════════════════════════════════════════════════════
# 5. PATRONES DE CORRECCIÓN / ACTUALIZACIÓN TEMPORAL
# ══════════════════════════════════════════════════════════════
# Señales que indican que el usuario está corrigiendo o
# actualizando una preferencia previamente almacenada.
# Estos triggers deben generar alta confianza y potencialmente
# archivar la versión anterior del hecho.

_CORRECTION_SIGNALS: list[str] = [
    # Correcciones explícitas
    "en realidad", "la verdad es que", "corrección:",
    "me equivoqué", "me equivoque",
    "no era así", "no era asi",
    "te dije mal", "eso no es correcto",
    # Cambios de opinión
    "cambié de opinión", "cambie de opinion",
    "ya no", "ahora prefiero", "ahora me gusta",
    "ahora uso", "antes usaba", "antes me gustaba",
    "ya cambié", "ya cambie",
    # Actualizaciones
    "actualización:", "actualiza eso",
    "ahora es", "en vez de", "en lugar de",
    "mejor dicho", "quise decir",
    "olvida lo anterior", "ignora lo que dije",
]

# Regex con captura para correcciones tipo "ahora X es Y" / "ya no X, ahora Y"
_CORRECTION_PATTERNS: list[tuple[str, str, str]] = [
    (r"\bahora (?:mi |el )?(.+?) es (.+)",
     "preference_update",
     "Actualización: ahora {} es {}."),
    (r"\bahora (?:prefiero|uso|me gusta)\s+(.+?)(?:\s+en vez de|\s+en lugar de)\s+(.+)",
     "preference_update",
     "Actualización: ahora prefiere {} en vez de {}."),
    (r"\bantes (?:usaba|me gustaba|prefería)\s+(.+?)(?:,?\s+(?:pero\s+)?ahora)\s+(.+)",
     "preference_update",
     "Actualización: antes le gustaba {}, ahora {}."),
    (r"\bcambié? de\s+(.+?)\s+a\s+(.+)",
     "preference_update",
     "Actualización: cambió de {} a {}."),
]


# ══════════════════════════════════════════════════════════════
# 6. KEYWORDS PARA TAGS
# ══════════════════════════════════════════════════════════════

_TECH_KEYWORDS: dict[str, list[str]] = {
    # Lenguajes
    "python": ["python"],
    "javascript": ["javascript", "js"],
    "typescript": ["typescript", "ts"],
    "rust": ["rust", "cargo"],
    "c++": ["c++", "cpp"],
    "c#": ["c#", "csharp"],
    "go": ["golang"],
    "java": ["java"],
    "lua": ["lua"],
    "ruby": ["ruby"],
    "php": ["php"],
    "swift": ["swift"],
    "kotlin": ["kotlin"],
    "html": ["html"],
    "css": ["css", "scss", "sass"],
    "sql": ["sql"],
    "bash": ["bash", "shell", "zsh", "powershell"],
    # Frameworks / librerías
    "react": ["react", "reactjs"],
    "vue": ["vue", "vuejs"],
    "angular": ["angular"],
    "svelte": ["svelte"],
    "next.js": ["next.js", "nextjs"],
    "nuxt": ["nuxt", "nuxtjs"],
    "django": ["django"],
    "flask": ["flask"],
    "fastapi": ["fastapi"],
    "express": ["express"],
    "nestjs": ["nestjs", "nest.js"],
    "spring": ["spring", "spring boot"],
    "laravel": ["laravel"],
    "rails": ["rails", "ruby on rails"],
    "tailwind": ["tailwind", "tailwindcss"],
    "bootstrap": ["bootstrap"],
    "electron": ["electron"],
    # Sistemas operativos
    "windows": ["windows", "windows 11", "windows 10"],
    "linux": ["linux", "ubuntu", "debian", "arch", "fedora", "manjaro", "mint"],
    "macos": ["macos", "mac os", "macbook"],
    "wsl": ["wsl", "wsl2"],
    "android": ["android"],
    "ios": ["ios"],
    # IA / ML
    "llama.cpp": ["llama.cpp", "llama cpp", "llamacpp"],
    "chromadb": ["chroma", "chromadb", "chroma db"],
    "ollama": ["ollama"],
    "gguf": ["gguf"],
    "pytorch": ["pytorch", "torch"],
    "tensorflow": ["tensorflow", "keras"],
    "transformers": ["transformers", "huggingface", "hugging face"],
    "openai": ["openai", "chatgpt", "gpt"],
    "claude": ["claude", "anthropic"],
    "gemini": ["gemini"],
    "stable_diffusion": ["stable diffusion", "comfyui", "comfy ui"],
    "langchain": ["langchain"],
    "embeddings": ["embeddings", "embedding"],
    # Bases de datos
    "sqlite": ["sqlite", "sqlite3"],
    "postgres": ["postgres", "postgresql"],
    "mysql": ["mysql", "mariadb"],
    "mongodb": ["mongodb", "mongo"],
    "redis": ["redis"],
    "supabase": ["supabase"],
    "firebase": ["firebase", "firestore"],
    # DevOps / infra
    "docker": ["docker", "dockerfile", "docker-compose"],
    "kubernetes": ["kubernetes", "k8s"],
    "git": ["git", "github", "gitlab", "bitbucket"],
    "nginx": ["nginx"],
    "aws": ["aws", "amazon web services", "s3", "ec2", "lambda"],
    "azure": ["azure"],
    "gcp": ["gcp", "google cloud"],
    "vercel": ["vercel"],
    "netlify": ["netlify"],
    "cloudflare": ["cloudflare"],
    # Herramientas
    "node": ["node", "nodejs", "npm", "yarn", "pnpm", "bun"],
    "vscode": ["vscode", "vs code", "visual studio code"],
    "vim": ["vim", "neovim", "nvim"],
    "cursor": ["cursor"],
    "jetbrains": ["jetbrains", "intellij", "pycharm", "webstorm"],
    "cuda": ["cuda", "nvidia"],
    "unity": ["unity"],
    "unreal": ["unreal", "unreal engine"],
    "godot": ["godot"],
    "blender": ["blender"],
    "figma": ["figma"],
    "postman": ["postman"],
}

_GENERAL_KEYWORDS: dict[str, list[str]] = {
    "comida": [
        "galleta", "galletas", "comida", "pizza", "chocolate",
        "café", "helado", "sushi", "tacos", "hamburguesa",
        "pasta", "arroz", "pollo", "ensalada", "fruta",
        "pan", "postre", "dulce", "té", "cerveza",
        "cocinar", "receta", "restaurante", "cocina",
    ],
    "nombre": ["me llamo", "mi nombre es", "llámame", "llamame"],
    "gustos": [
        "me gusta", "prefiero", "favorito", "favorita",
        "me encanta", "adoro", "amo", "disfruto",
        "me fascina", "me apasiona", "me interesa",
    ],
    "disgustos": [
        "no me gusta", "odio", "detesto",
        "no soporto", "me molesta", "me incomoda",
        "me desagrada", "me aburre", "no aguanto",
        "me harta", "me da asco",
    ],
    "restricciones": [
        "no quiero que", "evita", "no uses",
        "nunca", "no hagas", "sin", "deja de",
        "no incluyas", "no me hables de",
        "no menciones", "omite",
    ],
    "mascotas": [
        "mi perro", "mi gato", "mi mascota",
        "tengo un perro", "tengo un gato",
        "tengo una mascota", "mi cachorro",
        "mi gatito", "mi perrito",
    ],
    "musica": [
        "mi canción favorita", "mi banda favorita",
        "mi artista favorito", "escucho", "mi playlist",
        "mi género musical", "mi musica favorita",
        "mi cantante favorito", "mi album favorito",
    ],
    "hobbies": [
        "mi hobby", "mi pasatiempo", "en mi tiempo libre",
        "me dedico a", "me divierto con",
        "juego", "leo", "dibujo", "pinto",
        "hago ejercicio", "entreno",
    ],
    "ubicacion": [
        "vivo en", "soy de", "nací en", "naci en",
        "mi país", "mi pais", "mi ciudad",
        "mi zona horaria", "mi región", "mi region",
    ],
    "actualizacion": [
        "cambié de", "cambie de", "ahora prefiero",
        "ahora uso", "ya no", "antes usaba",
        "en realidad", "corrección",
    ],
    "edad": [
        "tengo años", "mi edad", "mi cumpleaños",
        "nací en", "naci en",
    ],
    "trabajo": [
        "trabajo como", "trabajo en", "soy programador",
        "soy desarrollador", "soy ingeniero", "soy diseñador",
        "mi empresa", "mi equipo", "mi jefe",
    ],
    "idioma": [
        "hablo", "mi idioma", "en español",
        "en inglés", "en ingles", "bilingüe",
    ],
}


# ══════════════════════════════════════════════════════════════
# 7. PATRONES DEL ASISTENTE
# ══════════════════════════════════════════════════════════════

# --- 7a. Facts que el asistente afirma SOBRE EL USUARIO ---

_ASSISTANT_USER_FACT_PATTERNS: list[str] = [
    # Identidad
    r"\btu nombre es\b",
    r"\bte llamas\b",
    r"\btienes \d+ años\b",
    r"\btu edad\b",
    r"\btu cumpleaños\b",
    # Preferencias
    r"\btu color favorito\b",
    r"\btu comida favorita\b",
    r"\bte gusta\b",
    r"\bte encanta\b",
    r"\bprefieres\b",
    r"\btu favorit[oa]\b",
    r"\bno te gusta\b",
    r"\bodias\b",
    r"\bte molesta\b",
    r"\bte aburre\b",
    # Entorno / trabajo
    r"\busas\b",
    r"\btu proyecto\b",
    r"\btu pc tiene\b",
    r"\btu sistema operativo\b",
    r"\bvives en\b",
    r"\beres de\b",
    r"\btrabajas con\b",
    r"\btrabajas como\b",
    r"\btrabajas en\b",
    r"\bestudiaste\b",
    r"\bestudias\b",
    r"\btu lenguaje favorito\b",
    # Mascotas / personal
    r"\btu mascota\b",
    r"\btu perro\b",
    r"\btu gato\b",
    r"\btu hobby\b",
    r"\btu pasatiempo\b",
    r"\btu familia\b",
]

# --- 7b. Facts que el asistente dice SOBRE SÍ MISMO ---

_ASSISTANT_SELF_PATTERNS: list[str] = [
    # Identidad
    r"\bmi nombre es\b",
    r"\bme llamo\b",
    r"\bnací en\b",
    r"\bnaci en\b",
    r"\bsoy de\b",
    r"\btengo \d+ años\b",
    # Preferencias positivas
    r"\bmi color favorito\b",
    r"\bmi comida favorita\b",
    r"\bmi canción favorita\b",
    r"\bmi cancion favorita\b",
    r"\bmi película favorita\b",
    r"\bmi pelicula favorita\b",
    r"\bmi libro favorito\b",
    r"\bmi serie favorita\b",
    r"\bmi juego favorito\b",
    r"\bmi hobby\b",
    r"\bmi pasatiempo\b",
    r"\bmi favorito\b",
    r"\bmi favorita\b",
    r"\bme gusta[n]?\b",
    r"\bme encanta\b",
    r"\bprefiero\b",
    r"\bme apasiona\b",
    r"\bme interesa\b",
    r"\bme fascina\b",
    r"\badoro\b",
    r"\bdisfruto\b",
    # Preferencias negativas
    r"\bno me gusta[n]?\b",
    r"\bodio\b",
    r"\bdetesto\b",
    r"\bme molesta\b",
    r"\bme incomoda\b",
    r"\bno soporto\b",
    r"\bme aburre\b",
    r"\bno aguanto\b",
    # Emociones / estados / backstory
    r"\btengo miedo de\b",
    r"\bmi mayor miedo\b",
    r"\bme da miedo\b",
    r"\bmi sueño es\b",
    r"\bmi sueno es\b",
    r"\bmi meta es\b",
    r"\bmi objetivo es\b",
    r"\bme siento\b",
    r"\bme pone triste\b",
    r"\bme alegra\b",
    r"\bme emociona\b",
    r"\bme preocupa\b",
    r"\bme inspira\b",
    # Habilidades / conocimiento (solo frases compuestas, no palabras sueltas)
    r"\bsé hacer\b",
    r"\bse hacer\b",
    r"\bmi especialidad\b",
    r"\bmi fuerte es\b",
]


# ══════════════════════════════════════════════════════════════
# 8. MARCADORES DE FILTRO DEL ASISTENTE
# ══════════════════════════════════════════════════════════════

# Marcadores de que el asistente está respondiendo/explicando
# (NO son hechos sobre sí mismo, se descartan)
_ASSISTANT_ANSWER_MARKERS: list[str] = [
    # Formato de instrucciones/recetas
    "**ingredientes:", "**instrucciones:", "**pasos:",
    "**requisitos:", "**instalación:",
    "*   ",
    # Ofrecimiento de ayuda
    "aquí te", "aqui te",
    "te explico", "te muestro", "te comparto",
    "te dejo", "te presento",
    "paso a paso",
    "veamos cómo", "veamos como",
    "vamos a ver", "vamos a hacer",
    # Preguntas al usuario
    "¿te gustaría", "¿quieres que", "¿quieres saber",
    "te gustaría", "quieres que", "quieres saber",
    "¿necesitas", "necesitas que te",
    "¿te ayudo", "¿puedo ayudarte",
    "¿algo más", "algo más?",
    # Disclaimers
    "como modelo de lenguaje", "como ia",
    "como inteligencia artificial",
    "como asistente virtual",
    "no tengo la capacidad de",
    "no puedo", "lamentablemente",
    "no tengo acceso a",
    "no tengo información sobre",
    # Meta-respuestas / estructura
    "en resumen:", "en conclusión:",
    "aquí tienes", "aqui tienes",
    "a continuación", "a continuacion",
    "por ejemplo:", "nota:",
    "importante:", "advertencia:",
    # Código / formato técnico
    "```", "def ", "class ", "import ",
    "function ", "const ", "let ", "var ",
]

# Marcadores condicionales (filtran facts condicionales)
_CONDITIONAL_MARKERS: list[str] = [
    "si ", "si\t",
    "quizás", "quizas", "tal vez",
    "podrías", "podrias",
    "puede que", "supongo",
    "parece que", "probablemente",
    "creo que", "me parece que",
    "posiblemente", "a lo mejor",
    "depende", "no estoy seguro",
    "hipotéticamente", "hipoteticamente",
    "en teoría", "en teoria",
    "imagina que", "supongamos que",
]


# ══════════════════════════════════════════════════════════════
# 9. NORMALIZACIÓN DE ACENTOS (lookup rápido)
# ══════════════════════════════════════════════════════════════
# Tabla para normalizar texto sin tildes a texto con tildes,
# útil para matching más robusto en español.

_ACCENT_NORMALIZE: dict[str, str] = {
    "a": "á", "e": "é", "i": "í", "o": "ó", "u": "ú", "n": "ñ",
}

_ACCENT_EQUIVALENCES: list[tuple[str, str]] = [
    ("á", "a"), ("é", "e"), ("í", "i"), ("ó", "o"), ("ú", "u"),
    ("ñ", "n"), ("ü", "u"),
]


def normalize_accents(text: str) -> str:
    """Normaliza un texto quitando acentos para matching más flexible.

    'Prefiero café' -> 'prefiero cafe'
    'Nací en México' -> 'naci en mexico'
    """
    t = text.lower()
    for accented, plain in _ACCENT_EQUIVALENCES:
        t = t.replace(accented, plain)
    return t


# ══════════════════════════════════════════════════════════════
# 10. PATRONES POR LEMMA (spaCy)
# ══════════════════════════════════════════════════════════════
# Estos patterns matchean sobre el LEMMA (forma base) del verbo,
# no sobre la conjugación. Esto permite que "me gustaban",
# "me gustaría", "me gusta" matcheen por "gustar".

_LEMMA_MEMORY_HINTS: dict[str, list[str]] = {
    "positive_preference": [
        "gustar", "encantar", "preferir", "fascinar",
        "apasionar", "adorar", "amar", "disfrutar",
        "interesar", "atraer", "molar",
    ],
    "negative_preference": [
        "odiar", "detestar", "molestar", "incomodar",
        "aburrir", "desagradar", "disgustar", "hartar",
        "soportar",  # + negación = "no soporto"
    ],
    "negative_instruction": [
        "evitar", "eliminar", "omitir", "prohibir",
    ],
    "assistant_instruction": [
        "comportar", "actuar", "responder", "hablar",
    ],
    "project_fact": [
        "crear", "desarrollar", "programar", "construir",
        "diseñar", "implementar", "armar",
    ],
    "environment": [
        "usar", "instalar", "tener", "configurar",
    ],
    "personal_identity": [
        "llamar", "vivir", "nacer", "estudiar", "trabajar",
    ],
}

# Lemmas que indican primera persona (incrementan memory score)
_LEMMA_PERSONAL_MARKERS: list[str] = [
    "gustar", "encantar", "preferir", "querer",
    "odiar", "detestar", "molestar", "llamar",
    "usar", "tener", "vivir", "trabajar",
    "estudiar", "necesitar", "crear", "programar",
]


# ══════════════════════════════════════════════════════════════
# 11. NER → TAGS (spaCy)
# ══════════════════════════════════════════════════════════════
# Mapa de labels NER de spaCy a prefijos de tags internos.

_NER_TAG_MAP: dict[str, str] = {
    "PER": "persona",
    "LOC": "ubicacion",
    "ORG": "organizacion",
    "MISC": "otro",
}


# ══════════════════════════════════════════════════════════════
# 12. LEMMAS DE PREGUNTA (spaCy)
# ══════════════════════════════════════════════════════════════
# Lemmas de verbos/adverbios interrogativos en español.

_QUESTION_LEMMAS: list[str] = [
    "cómo", "como", "qué", "que", "cuál", "cual",
    "dónde", "donde", "cuándo", "cuando", "cuánto", "cuanto",
    "por_qué",  # spaCy a veces lematiza "por qué" junto
]

