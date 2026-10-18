"""Baseline RAG end to end (step 4.5): question -> retrieve -> prompt -> Ollama -> cited answer.

Usage: uv run python -m rag.ask "Quelles sont les obligations de notification ... ?"
       [--lang fr|en] [--top-k 5] [--model qwen2.5:7b] [--no-log]
Every call appends one JSON line to runs/ask_log.jsonl (decision 4.5-B: pointers + hashes,
not copies of the texts, which already live in chunks.parquet).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime

from rag.config import PROJECT_ROOT, load_config
from rag.generate import build_prompt, load_prompt, parse_citations
from rag.llm import generate
from rag.retrieve import search


def git_commit() -> str:
    """Short HEAD sha, '-dirty' if the working tree has changes, 'unknown' outside git."""
    try:
        run = dict(cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], **run).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], **run).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def ask(question: str, cfg: dict, lang: str | None = None) -> dict:
    """Run the whole pipeline once and return the full record (also what gets logged)."""
    prompt = load_prompt(cfg)
    t0 = time.perf_counter()
    hits = search(question, cfg)
    retrieve_s = round(time.perf_counter() - t0, 2)

    built = build_prompt(question, hits, prompt, cfg, lang)
    out = generate(built["user"], cfg, system=built["system"])
    checks = parse_citations(out["text"], hits, prompt)

    r, llm = cfg["retrieval"], cfg["llm"]
    return {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_commit(),
        "question": question,
        "language": built["language"],
        "config": {
            "model": out["model"],
            "temperature": llm["temperature"],
            "num_ctx": llm["num_ctx"],
            "embedding": cfg["embedding"]["model"],
            "strategy": cfg["retrieval"]["strategy"],
            "mode": r["mode"],
            "top_k": r["top_k"],
            "prompt_version": prompt["version"],
            "prompt_sha256": prompt["sha256"],
        },  # fmt: skip
        "hits": [
            {
                "rank": h["rank"],
                "score": round(h["score"], 4),
                "chunk_id": h["chunk_id"],
                "text_sha256": hashlib.sha256(h["text"].encode()).hexdigest()[:12],
            }
            for h in hits
        ],  # fmt: skip
        "answer": out["text"],
        **checks,
        "retrieve_s": retrieve_s,
        "llm_s": out["latency_s"],
        "load_s": out["load_s"],
        "prefill_s": out["prefill_s"],
        "decode_s": out["decode_s"],
        "prompt_tokens": out["prompt_tokens"],
        "output_tokens": out["output_tokens"],
    }


def append_log(record: dict, cfg: dict) -> str:
    path = PROJECT_ROOT / cfg["paths"]["runs_dir"] / "ask_log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return str(path.relative_to(PROJECT_ROOT))


def show(rec: dict) -> None:
    c = rec["config"]
    print(f"\n{rec['answer']}\n")
    for ref in rec["references"]:
        n = "".join(f"[{i}]" for i in ref["n"])
        print(f"  {n} {ref['label']}" + (f" ({ref['pages']})" if ref["pages"] else ""))
    flags = [f"invalid citations {rec['invalid']}"] if rec["invalid"] else []
    flags += ["no citation"] if rec["uncited"] else []
    flags += [f"{rec['other_brackets']} non-numeric [..]"] if rec["other_brackets"] > 0 else []
    for flag in flags:
        print(f"  WARNING: {flag}")
    print(
        f"\n{c['model']} | {c['strategy']}/{c['mode']} top-{c['top_k']}"
        f" | prompt {c['prompt_version']}"
        f" | lang {rec['language']} | refusal {rec['is_refusal']} | retrieve {rec['retrieve_s']}s"
        f" | llm {rec['llm_s']}s (load {rec['load_s']} + prefill {rec['prefill_s']}"
        f" + decode {rec['decode_s']})"
        f" | tokens in={rec['prompt_tokens']} out={rec['output_tokens']}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Ask the regulatory RAG assistant (baseline).")
    ap.add_argument("question", nargs="+")
    ap.add_argument("--lang", choices=["fr", "en"], help="override answer-language detection")
    ap.add_argument("--top-k", type=int, help="override retrieval.top_k")
    ap.add_argument("--model", help="override llm.model, e.g. qwen2.5:7b")
    ap.add_argument("--no-log", action="store_true", help="do not append to runs/ask_log.jsonl")
    args = ap.parse_args()

    cfg = copy.deepcopy(load_config())
    if args.top_k:
        cfg["retrieval"]["top_k"] = args.top_k
    if args.model:
        cfg["llm"]["model"] = args.model

    rec = ask(" ".join(args.question), cfg, args.lang)
    show(rec)
    if not args.no_log:
        print(f"logged -> {append_log(rec, cfg)}")


if __name__ == "__main__":
    main()
