"""Minimal client for the local Ollama server (no LangChain)."""

from __future__ import annotations

import sys
import time

import httpx

from rag.config import load_config


def generate(prompt: str, cfg: dict, system: str | None = None) -> dict:
    """Send one prompt to Ollama, return text + basic telemetry."""
    llm = cfg["llm"]
    payload = {
        "model": llm["model"],
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": llm["temperature"], "num_ctx": llm["num_ctx"]},
    }
    if system:
        payload["system"] = system

    t0 = time.perf_counter()
    resp = httpx.post(f"{llm['base_url']}/api/generate", json=payload, timeout=llm["timeout_s"])
    resp.raise_for_status()
    data = resp.json()

    return {
        "text": data["response"].strip(),
        "model": data["model"],
        "latency_s": round(time.perf_counter() - t0, 2),
        "prompt_tokens": data.get("prompt_eval_count"),
        "output_tokens": data.get("eval_count"),
    }


if __name__ == "__main__":
    # Usage: uv run python -m rag.llm "your question"
    cfg = load_config()
    question = " ".join(sys.argv[1:]) or "Bonjour, qui es-tu ?"
    out = generate(question, cfg)
    tokens = f"in={out['prompt_tokens']} out={out['output_tokens']}"
    print(f"[{out['model']}] {out['latency_s']}s | {tokens}")
    print(out["text"])
