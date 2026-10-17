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

    missing = {"version", "system", "user"} - prompt.keys()
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
