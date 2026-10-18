"""Load the YAML config (single source of truth for parameters)."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/rag/config.py -> project root
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"


def apply_overrides(cfg: dict, pairs: list[str] | None) -> dict:
    """Apply ["retrieval.mode=hybrid", "retrieval.top_k=10"] in place; values parsed as YAML.

    Only EXISTING keys can be overridden, so a typo fails loudly instead of being ignored.
    """
    for pair in pairs or []:
        key, sep, raw = pair.partition("=")
        if not sep:
            raise ValueError(f"override must look like a.b=value, got {pair!r}")
        *parents, leaf = key.split(".")
        node = cfg
        for part in parents:
            if not isinstance(node.get(part), dict):
                raise KeyError(f"unknown config section {key!r}")
            node = node[part]
        if leaf not in node:
            raise KeyError(f"unknown config key {key!r}")
        node[leaf] = yaml.safe_load(raw)
    return cfg


def load_config(path: str | Path | None = None) -> dict:
    """Load config from `path`, $REG_RAG_CONFIG, or the default file (in that order)."""
    path = Path(path or os.environ.get("REG_RAG_CONFIG", DEFAULT_CONFIG))
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
