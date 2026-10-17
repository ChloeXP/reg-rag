"""Baseline RAG generation (step 4): versioned prompt -> context -> Ollama -> cited answer."""

from __future__ import annotations

import hashlib
import re
import string

import yaml

from rag.config import PROJECT_ROOT

PLACEHOLDERS = {"context", "question", "answer_language"}
# Context header written by chunking.unit_header: "[DORA · Article 19 Title] body..."
HEADER = re.compile(r"^\[([^\[\]\n]+)\]\s*")
MAX_LABEL = 120  # some headers carry a whole title page (ACPR: 317 chars)

# Language detection (decision 4.3b-B): the corpus and the questions are FR or EN only.
LANGUAGES = {"fr": "French", "en": "English"}
# Citations: [1] [1][3] [1, 3] [1; 3] [1-3]. Any other [...] is counted, never parsed (P35).
CITATION = re.compile(r"\[(\d+(?:\s*[,;\-–]\s*\d+)*)\]")
BRACKETS = re.compile(r"\[[^\[\]\n]*\]")
WORD = re.compile(r"[^\W\d_]+")  # letters only; apostrophes split "l'article" -> l, article
FR_ACCENTS = set("éèêëàâçùûüôîïœ")
FR_WORDS = frozenset(
    """le la les l de des du d un une et est sont que qu qui quel quelle quels quelles quoi
    comment pourquoi dans pour sur par au aux ce cette ces il ils elle doit doivent selon
    entre avec sans peut leur leurs""".split()
)
EN_WORDS = frozenset(
    """the of and to in is are what which who how does do must shall should be for with
    under by an this that these when where why can their its""".split()
)


def _fields(template: str) -> set[str]:
    return {f for _, f, _, _ in string.Formatter().parse(template) if f is not None}


def load_prompt(cfg: dict, version: str | None = None) -> dict:
    """Load prompts/<version>.yaml, check its placeholders, add a content hash.

    The hash (of system + user) is logged with every answer: if someone edits a version
    in place after it was evaluated, the logs show two different hashes for one name.
    """
    version = version or cfg["llm"]["prompt_version"]
    path = PROJECT_ROOT / cfg["paths"]["prompts_dir"] / f"{version}.yaml"
    prompt = yaml.safe_load(path.read_text(encoding="utf-8"))

    missing = {"version", "system", "user", "refusal"} - prompt.keys()
    if missing:
        raise ValueError(f"{path.name}: missing keys {sorted(missing)}")
    if prompt["version"] != version:
        raise ValueError(f"{path.name}: version field is {prompt['version']!r}")
    if _fields(prompt["system"]):
        raise ValueError(f"{path.name}: system must not contain placeholders")
    if _fields(prompt["user"]) != PLACEHOLDERS:
        raise ValueError(
            f"{path.name}: user placeholders {_fields(prompt['user'])} != {PLACEHOLDERS}"
        )

    for lang, sentence in prompt["refusal"].items():
        if f'"{sentence}"' not in prompt["system"]:
            raise ValueError(f"{path.name}: refusal[{lang}] not quoted verbatim in system")

    digest = hashlib.sha256((prompt["system"] + "\x00" + prompt["user"]).encode()).hexdigest()
    return {**prompt, "sha256": digest[:12]}


def work_label(work_id: str) -> str:
    """'eba_gl_2019_02' -> 'EBA GL 2019 02' (same rule as chunking.unit_header)."""
    return work_id.replace("_", " ").upper()


def _pages(hit: dict) -> str:
    a, b = hit.get("page_start"), hit.get("page_end")
    if a is None:  # XHTML (CRR) has no pages: say nothing rather than invent (#23)
        return ""
    return f"p. {a}" if b in (None, a) else f"p. {a}-{b}"


def split_passage(hit: dict) -> tuple[str, str]:
    """(label, body) shown to the model. Square brackets are reserved for [n] citations.

    structure chunks: their own header becomes the label ("DORA · Article 19 X" -> "DORA — Article
    19 X") and is removed from the body. fixed chunks have no header (one even starts with a
    real "[cs.LG]"), so they are never stripped: label = work + pages from the metadata.
    """
    text = hit["text"]
    m = HEADER.match(text) if "::structure::" in hit["chunk_id"] else None
    if m:
        label, body = m.group(1).replace(" · ", " — ", 1), text[m.end() :]
    else:
        label = " — ".join(x for x in (work_label(hit["work_id"]), _pages(hit)) if x)
        body = text
    if len(label) > MAX_LABEL:
        label = label[: MAX_LABEL - 1].rstrip() + "…"
    return label, body.strip()


def format_context(hits: list[dict]) -> str:
    """Numbered passages: "[1] DORA — Article 19 ...\n<body>", separated by blank lines.

    Number n = position in `hits` (1-based); parse_citations maps [n] back to hits[n-1].
    """
    blocks = []
    for n, hit in enumerate(hits, start=1):
        label, body = split_passage(hit)
        blocks.append(f"[{n}] {label}\n{body}")
    return "\n\n".join(blocks)


def detect_language(question: str, default: str) -> str:
    """'fr' | 'en' by counting function words (+1 for FR if any French accent).

    No signal or a tie -> `default` (config generation.default_language).
    """
    q = question.lower()
    words = WORD.findall(q)
    fr = sum(w in FR_WORDS for w in words) + (1 if FR_ACCENTS & set(q) else 0)
    en = sum(w in EN_WORDS for w in words)
    if fr == en:
        return default
    return "fr" if fr > en else "en"


def build_prompt(question: str, hits: list[dict], prompt: dict, cfg: dict, lang: str | None = None):
    """-> {system, user, language}. `lang` overrides detection (CLI --lang)."""
    lang = lang or detect_language(question, cfg["generation"]["default_language"])
    if lang not in LANGUAGES:
        raise ValueError(f"unsupported language {lang!r}, expected one of {sorted(LANGUAGES)}")
    user = prompt["user"].format(
        context=format_context(hits), question=question.strip(), answer_language=LANGUAGES[lang]
    )
    return {"system": prompt["system"], "user": user, "language": lang}


def _numbers(group: str) -> list[int]:
    """'1, 3' -> [1, 3]; '2-4' -> [2, 3, 4] (an absurd range like '1-900' keeps its two ends)."""
    out = []
    for part in re.split(r"\s*[,;]\s*", group):
        m = re.fullmatch(r"(\d+)\s*[\-–]\s*(\d+)", part)
        if not m:
            out.append(int(part))
            continue
        a, b = int(m[1]), int(m[2])
        out.extend(range(a, b + 1) if 0 <= b - a < 20 else [a, b])
    return out


def _norm(text: str) -> str:
    return " ".join(text.lower().replace("’", "'").strip(' "«»').split())


def parse_citations(answer: str, hits: list[dict], prompt: dict) -> dict:
    """Check the model's [n] against the passages it was given (decision 4.4-A: report, never fix).

    -> cited (valid n), invalid (n outside 1..len(hits)), is_refusal, uncited (no valid [n] and
    not a refusal), other_brackets (e.g. a copied "[DORA — Article 19]"), references.
    """
    numbers = [n for g in CITATION.findall(answer) for n in _numbers(g)]
    cited = sorted({n for n in numbers if 1 <= n <= len(hits)})
    invalid = sorted({n for n in numbers if not 1 <= n <= len(hits)})
    is_refusal = any(_norm(r) in _norm(answer) for r in prompt["refusal"].values())
    return {
        "cited": cited,
        "invalid": invalid,
        "is_refusal": is_refusal,
        "uncited": not cited and not is_refusal,
        "other_brackets": len(BRACKETS.findall(answer)) - len(CITATION.findall(answer)),
        "references": references([(n, hits[n - 1]) for n in cited]),
    }


def references(numbered: list[tuple[int, dict]]) -> list[dict]:
    """Merge cited passages of the same legal unit: Art. 19 cut in 3 chunks -> one reference."""
    refs: dict[tuple, dict] = {}
    for n, hit in numbered:
        label, _ = split_passage(hit)
        ref = refs.setdefault(
            (hit["doc_id"], label),
            {"n": [], "doc_id": hit["doc_id"], "work_id": hit["work_id"], "label": label,
             "article": hit.get("article"), "pages": set(), "chunk_ids": []},
        )  # fmt: skip
        ref["n"].append(n)
        ref["chunk_ids"].append(hit["chunk_id"])
        ref["pages"].update(
            p for p in (hit.get("page_start"), hit.get("page_end")) if p is not None
        )
    for ref in refs.values():
        pages = sorted(ref.pop("pages"))
        ref["pages"] = "" if not pages else _pages({"page_start": pages[0], "page_end": pages[-1]})
    return list(refs.values())
