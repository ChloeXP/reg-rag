"""Step 6.1 - retrieval-layer evaluation (decision 6.0-B): golden set -> search -> location metrics.

Run:  uv run python -m rag.eval_retrieval [--set retrieval.mode=hybrid] [--tag hybrid]
No LLM is called. One process: bge-m3 is loaded once (warm-up query, timed separately).
Writes runs/eval/<timestamp>__<tag>/{retrieval.jsonl, summary.json, config.yaml}, then logs
that directory to MLflow (step 6.2; --no-mlflow to skip, rag.tracking to log it later).
The test split is refused unless --final is given (decision #64: tune on dev only).
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

from rag.ask import git_commit
from rag.config import PROJECT_ROOT, apply_overrides, load_config
from rag.metrics import aggregate, grouped, location_ranks, retrieval_scores
from rag.retrieve import search
from rag.tracking import file_md5, log_run

GROUP_FIELDS = ["type", "verification", "cross_lingual", "question_language"]


def load_items(cfg: dict, split: str) -> list[dict]:
    """Golden items of one split that have expected locations (no_answer / G038 have none)."""
    path = PROJECT_ROOT / cfg["eval"]["golden_set"]
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [g for g in rows if g["split"] == split and g["expected"]]


def check_depth(cfg: dict) -> int:
    depth, ks = cfg["eval"]["retrieval_depth"], cfg["eval"]["ks"]
    if max(ks) > depth:
        raise ValueError(f"eval.ks {ks} go beyond eval.retrieval_depth {depth}")
    if cfg["retrieval"]["mode"] == "hybrid" and depth > cfg["retrieval"]["hybrid"]["candidates"]:
        # a deeper fused list would use more candidates than search(top_k=5) does in production
        raise ValueError("eval.retrieval_depth must be <= retrieval.hybrid.candidates")
    rr = cfg["retrieval"]["rerank"]
    if rr["enabled"] and depth > rr["candidates"]:
        # deeper than the reranked pool would mix reranked and first-stage order
        raise ValueError("eval.retrieval_depth must be <= retrieval.rerank.candidates")
    return depth


def evaluate(items: list[dict], cfg: dict) -> list[dict]:
    depth, ks = check_depth(cfg), cfg["eval"]["ks"]
    rows = []
    for g in items:
        t0 = time.perf_counter()
        hits = search(g["question"], cfg, top_k=depth)
        retrieve_s = round(time.perf_counter() - t0, 3)
        ranks = location_ranks(hits, g["expected"])
        rows.append(
            {
                "id": g["id"],
                **{f: g[f] for f in GROUP_FIELDS},
                "ranks": ranks,
                "scores": retrieval_scores(ranks, ks),
                "top_chunk_ids": [h["chunk_id"] for h in hits],
                "retrieve_s": retrieve_s,
            }
        )
    return rows


def metric_keys(cfg: dict) -> list[str]:
    ks = cfg["eval"]["ks"]
    return [f"hit@{k}" for k in ks] + [f"recall@{k}" for k in ks] + ["rr"]


def summarize(rows: list[dict], cfg: dict) -> dict:
    keys = metric_keys(cfg)
    lat = sorted(r["retrieve_s"] for r in rows)
    return {
        "overall": aggregate(rows, keys),
        **{f"by_{f}": grouped(rows, f, keys) for f in GROUP_FIELDS},
        "latency_s": {"mean": sum(lat) / len(lat), "p50": lat[len(lat) // 2], "max": lat[-1]},
    }


def default_tag(cfg: dict) -> str:
    r = cfg["retrieval"]
    tag = f"{r['strategy']}-{r['mode']}-{cfg['embedding']['model'].split('/')[-1]}"
    return tag + ("-rerank" if r["rerank"]["enabled"] else "")


def write_run(rows: list[dict], summary: dict, cfg: dict, meta: dict) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = PROJECT_ROOT / cfg["paths"]["runs_dir"] / cfg["eval"]["runs_subdir"]
    out = out / f"{stamp}__{meta['tag']}"
    out.mkdir(parents=True)
    with open(out / "retrieval.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(
        json.dumps({"meta": meta, **summary}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out


def print_summary(summary: dict, rows: list[dict], cfg: dict) -> None:
    cols = ["hit@1", "hit@3", "hit@5", "hit@10", "hit@20", "recall@5", "rr"]
    cols = [c for c in cols if c in summary["overall"]]
    print(f"\n{'group':<28}{'n':>4}" + "".join(f"{c:>10}" for c in cols))

    def line(name: str, agg: dict) -> None:
        print(f"{name:<28}{agg['n']:>4}" + "".join(f"{agg[c]:>10.2f}" for c in cols))

    line("ALL", summary["overall"])
    for f in GROUP_FIELDS:
        for value, agg in summary[f"by_{f}"].items():
            line(f"  {f}={value}", agg)
    k = cfg["retrieval"]["top_k"]
    misses = [r for r in rows if r["scores"][f"hit@{k}"] < 1]
    print(f"\nnot fully found in top-{k} ({len(misses)}): location ranks (None = not in top-20)")
    for r in misses:
        print(f"  {r['id']} {r['type']:<14} {r['question_language']} ranks={r['ranks']}")
    lat = summary["latency_s"]
    print(
        f"\nretrieve latency: mean {lat['mean']:.2f}s p50 {lat['p50']:.2f}s max {lat['max']:.2f}s"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Retrieval-layer evaluation on the golden set.")
    ap.add_argument("--set", action="append", metavar="KEY=VALUE", help="config override")
    ap.add_argument("--tag", help="run name suffix (default: strategy-mode-embedding)")
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--final", action="store_true", help="required to touch the test split")
    ap.add_argument("--no-mlflow", action="store_true", help="only write the run directory")
    args = ap.parse_args()
    if args.split == "test" and not args.final:
        ap.error("the test split is evaluated once, at the end: add --final if this is it")

    cfg = apply_overrides(copy.deepcopy(load_config()), args.set)
    items = load_items(cfg, args.split)

    t0 = time.perf_counter()
    search("warm-up", cfg, top_k=1)  # loads bge-m3 once; kept out of per-question latency
    warmup_s = round(time.perf_counter() - t0, 1)

    rows = evaluate(items, cfg)
    summary = summarize(rows, cfg)
    meta = {
        "tag": args.tag or default_tag(cfg),
        "layer": "retrieval",
        "split": args.split,
        "n_items": len(rows),
        "overrides": args.set or [],
        "git": git_commit(),
        "warmup_s": warmup_s,
        # data identity: which questions and which chunk texts this run was scored on
        "golden_md5": file_md5(PROJECT_ROOT / cfg["eval"]["golden_set"]),
        "chunks_md5": file_md5(PROJECT_ROOT / cfg["paths"]["processed_dir"] / "chunks.parquet"),
    }
    out = write_run(rows, summary, cfg, meta)
    print(f"{meta['tag']} | {args.split} | {len(rows)} items | warm-up {warmup_s}s")
    print_summary(summary, rows, cfg)
    print(f"\n-> {out}")
    if not args.no_mlflow:
        try:
            print(f"mlflow run {log_run(out, cfg)}")
        except Exception as e:  # server down etc.: the run directory is already safe on disk
            print(f"WARNING mlflow logging failed ({type(e).__name__}: {e})")
            print(f"  retry later: uv run python -m rag.tracking {out}")


if __name__ == "__main__":
    main()
