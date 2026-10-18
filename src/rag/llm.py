"""Minimal client for the local Ollama server (no LangChain)."""

from __future__ import annotations

import sys
import time

import httpx

from rag.config import load_config


def generate(
    prompt: str, cfg: dict, system: str | None = None, fmt: str | dict | None = None
) -> dict:
    """Send one prompt to Ollama, return text + basic telemetry. fmt: "json" or a JSON schema."""
    llm = cfg["llm"]
    payload = {
        "model": llm["model"],
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": llm["temperature"], "num_ctx": llm["num_ctx"]},
    }
    if system:
        payload["system"] = system
    if fmt:
        payload["format"] = fmt  # step 5.3: golden-set drafts are JSON objects

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
        # Ollama durations are in ns: model loading, reading the prompt (prefill), writing (decode)
        "load_s": _seconds(data.get("load_duration")),
        "prefill_s": _seconds(data.get("prompt_eval_duration")),
        "decode_s": _seconds(data.get("eval_duration")),
    }


def unload(cfg: dict) -> None:
    """Unload the model from Ollama (keep_alive=0): its KV / prompt cache goes with it.

    Step 6.6c (P67): run B started right after run A re-used A's cached prompt prefixes
    (same system prompt + same first passages) -> prefill 15x faster, latency not comparable.
    Unloading before each evaluation run gives every run the same cold start.
    """
    llm = cfg["llm"]
    resp = httpx.post(
        f"{llm['base_url']}/api/generate",
        json={"model": llm["model"], "keep_alive": 0},
        timeout=llm["timeout_s"],
    )
    resp.raise_for_status()


def _seconds(ns: int | None) -> float | None:
    return None if ns is None else round(ns / 1e9, 2)


if __name__ == "__main__":
    # Usage: uv run python -m rag.llm "your question"
    cfg = load_config()
    question = " ".join(sys.argv[1:]) or "Bonjour, qui es-tu ?"
    out = generate(question, cfg)
    tokens = f"in={out['prompt_tokens']} out={out['output_tokens']}"
    split = f"load {out['load_s']}s prefill {out['prefill_s']}s decode {out['decode_s']}s"
    print(f"[{out['model']}] {out['latency_s']}s ({split}) | {tokens}")
    print(out["text"])
