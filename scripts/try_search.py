"""Step 3.4 - run the 5 smoke queries on both chunking strategies and mark the hits.

Run: uv run python scripts/try_search.py   (needs Qdrant up + rag.index done)
"""

from __future__ import annotations

import time

import yaml

from rag.config import PROJECT_ROOT, load_config
from rag.retrieve import search

STRATEGIES = ["structure", "fixed"]


def is_hit(hit: dict, exp: dict) -> bool:
    if hit["work_id"] != exp["work_id"]:
        return False
    if hit["article"] in exp.get("articles", []):
        return True
    if any((hit["section"] or "").startswith(s) for s in exp.get("sections", [])):
        return True
    return any(m in hit["text"] for m in exp.get("markers", []))


def preview(text: str, n: int = 90) -> str:
    return " ".join(text.split())[:n]


def main() -> None:
    cfg = load_config()
    queries = yaml.safe_load(open(PROJECT_ROOT / "eval" / "smoke_queries.yaml", encoding="utf-8"))
    top_k = cfg["retrieval"]["top_k"]
    summary = []
    for q in queries:
        print(f"\n=== {q['id']} [{q['type']}] {q['query']}")
        row = [q["id"], q["type"]]
        for strategy in STRATEGIES:
            t0 = time.perf_counter()
            hits = search(q["query"], cfg, top_k=top_k, strategy=strategy)
            ranks = [h["rank"] for h in hits if is_hit(h, q["expected"])]
            first = ranks[0] if ranks else None
            row.append(f"@{first}" if first else "miss")
            print(f"  -- {strategy}: first hit {row[-1]} ({time.perf_counter() - t0:.1f}s)")
            for h in hits:
                mark = "✓" if is_hit(h, q["expected"]) else " "
                loc = f"{h['doc_id']} art={h['article'] or '-'}"
                print(f"   {h['rank']} {mark} {h['score']:.3f} {loc:<32} {preview(h['text'])}")
        summary.append(row)
    print(f"\n{'query':<6}{'type':<17}" + "".join(f"{s:<11}" for s in STRATEGIES))
    for row in summary:
        print(f"{row[0]:<6}{row[1]:<17}" + "".join(f"{c:<11}" for c in row[2:]))


if __name__ == "__main__":
    main()
