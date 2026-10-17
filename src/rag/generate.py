"""Baseline RAG generation (step 4): versioned prompt -> context -> Ollama -> cited answer."""

from __future__ import annotations

import hashlib
import string

import yaml

from rag.config import PROJECT_ROOT

PLACEHOLDERS = {"context", "question", "answer_language"}


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
