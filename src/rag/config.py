"""Load the YAML config (single source of truth for parameters)."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/rag/config.py -> project root
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Load config from `path`, $REG_RAG_CONFIG, or the default file (in that order)."""
    path = Path(path or os.environ.get("REG_RAG_CONFIG", DEFAULT_CONFIG))
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
