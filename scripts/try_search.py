"""Steps 3.4-3.6 - run the 5 smoke queries per retrieval mode x chunking strategy, mark hits.

Run: uv run python scripts/try_search.py [--modes dense bm25] [--show 3]
"""

from __future__ import annotations

import argparse

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


def preview(text: str, n: int = 80) -> str:
    return " ".join(text.split())[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["dense", "bm25", "hybrid"])
    ap.add_argument("--show", type=int, default=3, help="result lines printed per run")
    args = ap.parse_args()

    cfg = load_config()
    queries = yaml.safe_load(open(PROJECT_ROOT / "eval" / "smoke_queries.yaml", encoding="utf-8"))
    top_k = cfg["retrieval"]["top_k"]
    runs = [(m, s) for m in args.modes for s in STRATEGIES]
    summary = []
    for q in queries:
        print(f"\n=== {q['id']} [{q['type']}] {q['query']}")
        row = [q["id"], q["type"]]
        for mode, strategy in runs:
            hits = search(q["query"], cfg, top_k=top_k, strategy=strategy, mode=mode)
            ranks = [h["rank"] for h in hits if is_hit(h, q["expected"])]
            row.append(f"@{ranks[0]}" if ranks else "miss")
            print(f"  -- {mode}/{strategy}: first hit {row[-1]}")
            for h in hits[: args.show]:
                mark = "✓" if is_hit(h, q["expected"]) else " "
                loc = f"{h['doc_id']} art={h['article'] or '-'}"
                src = " ".join(f"{s[0]}@{r}" for s, r in h.get("ranks", {}).items())
                print(
                    f"   {h['rank']} {mark} {h['score']:7.3f} {loc:<30} {src:<9} "
                    f"{preview(h['text'], 70)}"
                )
        summary.append(row)
    print(f"\n{'query':<6}{'type':<17}" + "".join(f"{m + '/' + s:<17}" for m, s in runs))
    for row in summary:
        print(f"{row[0]:<6}{row[1]:<17}" + "".join(f"{c:<17}" for c in row[2:]))


if __name__ == "__main__":
    main()
