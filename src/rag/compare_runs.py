"""Step 6.3 - paired comparison of two retrieval runs on the SAME questions.

Run: uv run python -m rag.compare_runs runs/eval/<A> runs/eval/<B> [--k 5]
With n = 33 one question = 3 points, so averages alone mislead: this lists every question that
flipped at k (gained / lost) and gives an exact McNemar p-value on those flips.
Refuses to compare runs scored on different data (golden_md5 / chunks_md5 differ, decision #88).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.metrics import mcnemar_exact

DATA_KEYS = ["split", "golden_md5", "chunks_md5", "hit_rule"]  # same questions, texts, rule
HEADLINE = ["hit@1", "hit@3", "hit@5", "hit@10", "hit@20", "recall@5", "rr"]


def load_run(run_dir: str | Path) -> dict:
    d = Path(run_dir)
    summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    rows = [json.loads(x) for x in (d / "retrieval.jsonl").read_text(encoding="utf-8").splitlines()]
    return {"name": d.name, "meta": summary["meta"], "overall": summary["overall"], "rows": rows}


def data_mismatch(a: dict, b: dict) -> list[str]:
    """Data identity keys that differ or are missing (a missing md5 is NOT a match)."""
    return [
        k for k in DATA_KEYS if a["meta"].get(k) is None or a["meta"].get(k) != b["meta"].get(k)
    ]


def flips(a: dict, b: dict, k: int) -> dict:
    ra, rb = {r["id"]: r for r in a["rows"]}, {r["id"]: r for r in b["rows"]}
    if ra.keys() != rb.keys():
        raise ValueError(f"different question sets: {sorted(ra.keys() ^ rb.keys())}")
    key = f"hit@{k}"
    gained = [i for i in sorted(ra) if ra[i]["scores"][key] < 1 <= rb[i]["scores"][key]]
    lost = [i for i in sorted(ra) if rb[i]["scores"][key] < 1 <= ra[i]["scores"][key]]
    moved = [
        (i, ra[i]["ranks"], rb[i]["ranks"]) for i in sorted(ra) if ra[i]["ranks"] != rb[i]["ranks"]
    ]
    return {
        "gained": gained,
        "lost": lost,
        "moved": moved,
        "p": mcnemar_exact(len(gained), len(lost)),
        # direction of rank change: sees moves that never cross the k cut-off (sign test)
        "rr_up": [i for i in sorted(ra) if rb[i]["scores"]["rr"] > ra[i]["scores"]["rr"]],
        "rr_down": [i for i in sorted(ra) if rb[i]["scores"]["rr"] < ra[i]["scores"]["rr"]],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired comparison of two retrieval runs.")
    ap.add_argument("run_a", help="reference run (e.g. baseline)")
    ap.add_argument("run_b", help="candidate run")
    ap.add_argument("--k", type=int, default=5, help="cut-off for the flip count (default 5)")
    ap.add_argument("--force", action="store_true", help="compare even if the data differ")
    args = ap.parse_args()

    a, b = load_run(args.run_a), load_run(args.run_b)
    bad = data_mismatch(a, b)
    if bad and not args.force:
        ap.error(f"runs not scored on the same data ({', '.join(bad)} differ or missing)")

    print(f"A = {a['name']}\nB = {b['name']}\n")
    print(f"{'metric':<10}{'A':>8}{'B':>8}{'B-A':>8}")
    for m in HEADLINE:
        if m in a["overall"] and m in b["overall"]:
            va, vb = a["overall"][m], b["overall"][m]
            print(f"{m:<10}{va:>8.2f}{vb:>8.2f}{vb - va:>+8.2f}")

    f = flips(a, b, args.k)
    print(
        f"\nflips at hit@{args.k}: gained {len(f['gained'])} {f['gained']}"
        f" | lost {len(f['lost'])} {f['lost']} | exact McNemar p = {f['p']:.3f}"
    )
    up, down = len(f["rr_up"]), len(f["rr_down"])
    print(f"rr direction: better {up} | worse {down} | sign test p = {mcnemar_exact(up, down):.3f}")
    print(f"\nrank changes ({len(f['moved'])} questions; None = not in top-20):")
    for qid, ranks_a, ranks_b in f["moved"]:
        print(f"  {qid}: {ranks_a} -> {ranks_b}")


if __name__ == "__main__":
    main()
