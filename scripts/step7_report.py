"""Step 7 - print the README tables for structured output (no number typed by hand, #115).

Run: uv run python scripts/step7_report.py [--runs-dir runs/eval] > /tmp/step7_tables.md
All runs are scored with golden-v3 and share the chunks (both md5 checked, #91); within a model,
every compared pair saw the same retrieved passages (checked item by item). v2 / v3 3b runs are
the 7.8 policy replays of the original runs (rag.replay_policy); the 7b v3 run used the policy
directly. Paired tests on the dev split: sign test (key-fact recall), exact McNemar (binary).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rag.metrics import mcnemar_exact  # noqa: E402

RUNS = [  # label, tag
    ("3b · v1 free text", "e2e-3b-rerank-top5-g3"),
    ("3b · v2 JSON", "e2e-3b-rerank-top5-v2-replay-flag-drop-g3"),
    ("3b · v3 JSON + completeness", "e2e-3b-rerank-top5-v3-replay-flag-drop-g3"),
    ("7b · v1 free text", "e2e-7b-rerank-top5-g3"),
    ("7b · v3 JSON + completeness", "e2e-7b-rerank-top5-v3"),
]
PAIRS = [(0, 1), (0, 2), (1, 2), (3, 4)]
PAIRED = [("key_fact_recall", "key-fact recall"), ("all_facts", "all key facts"),
          ("invalid_cite", "non-existent citation (1st answer)"),
          ("false_refusal", "refused answerable")]  # fmt: skip


def run_dir(root: Path, tag: str) -> Path:
    found = sorted(root.glob(f"*__{tag}"))
    if not found:
        raise SystemExit(f"no run directory for {tag} in {root}")
    return found[-1]


def load(root: Path) -> list[tuple[str, dict, dict]]:
    runs = []
    for label, tag in RUNS:
        d = run_dir(root, tag)
        summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        rows = {r["id"]: r for r in map(json.loads, (d / "answers.jsonl").read_text().splitlines())}
        runs.append((label, summary, rows))
    for key in ("golden_md5", "chunks_md5"):  # #91: never compare runs scored on different data
        values = {s["meta"][key] for _, s, _ in runs}
        if len(values) != 1:
            raise SystemExit(f"runs disagree on {key}: {values}")
    return runs


def passages(row: dict) -> list[str]:
    return [h["chunk_id"] for h in row["variants"][0]["record"]["hits"]]


def mean(rows: dict, f) -> float:
    vals = [f(r) for r in rows.values()]
    vals = [v for v in vals if v is not None]
    return statistics.mean(vals) if vals else float("nan")


def count(rows: dict, key: str) -> str:
    vals = [r["scores"][key] for r in rows.values() if key in r["scores"]]
    return f"{sum(vals):.0f} / {len(vals)}"


def tables(runs) -> list[str]:
    out = [
        "| Content and safety (dev, 40 items) | key-fact recall | all key facts "
        "| refused answerable | refused no-answer | talks about the absent fact "
        "| resisted injection |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, s, rows in runs:
        o = s["overall"]
        out.append(
            f"| {label} | {o['key_fact_recall']:.2f} | {o['all_facts']:.2f} "
            f"| {count(rows, 'false_refusal')} | {count(rows, 'refused')} "
            f"| {count(rows, 'mentions_absent')} | {count(rows, 'resisted')} |"
        )
    out += [
        "",
        "| Traceability, robustness, cost | valid citations | non-existent citation "
        "(1st answer) | passages cited | cited passages holding evidence | cites the injected "
        "passage | fallback | retries (fixed) | answer length (chars) | s / question |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for label, s, rows in runs:
        o = s["overall"]
        retried = sum(r["scores"].get("retried", 0) > 0 for r in rows.values())
        fixed = sum(r["scores"].get("retry_success", 0) > 0 for r in rows.values())
        retry = (
            f"{retried} ({fixed})" if any("retried" in r["scores"] for r in rows.values()) else "—"
        )
        fallback = f"{count(rows, 'fallback')}" if "fallback" in o else "—"
        chars = mean(rows, lambda r: len(r["variants"][0]["record"]["answer"])
                     if r["expected_behavior"] == "answer" else None)  # fmt: skip
        out.append(
            f"| {label} | {o['citation_ok']:.2f} | {count(rows, 'invalid_cite')} "
            f"| {o['cited_n']:.1f} | {o['cite_precision']:.2f} | {count(rows, 'cites_poison')} "
            f"| {fallback} | {retry} | {chars:.0f} | {s['latency_s']['total_mean']:.1f} |"
        )
    return out


def paired_table(runs) -> list[str]:
    head = " | ".join(name for _, name in PAIRED)
    out = [f"| Paired, better / worse (p) | {head} |", "|---" * (len(PAIRED) + 1) + "|"]
    for a, b in PAIRS:
        (la, _, ra), (lb, _, rb) = runs[a], runs[b]
        if any(passages(ra[i]) != passages(rb[i]) for i in ra):
            raise SystemExit(f"{la} vs {lb}: different retrieved passages")
        cells = []
        for key, _ in PAIRED:
            ids = [i for i in ra if key in ra[i]["scores"] and key in rb[i]["scores"]]
            lower_is_better = key in ("invalid_cite", "false_refusal")
            up = sum(rb[i]["scores"][key] > ra[i]["scores"][key] for i in ids)
            down = sum(rb[i]["scores"][key] < ra[i]["scores"][key] for i in ids)
            better, worse = (down, up) if lower_is_better else (up, down)
            cells.append(f"{better} / {worse} (p = {mcnemar_exact(better, worse):.3f})")
        out.append(f"| {la} → {lb} | " + " | ".join(cells) + " |")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs/eval")
    runs = load(Path(ap.parse_args().runs_dir))
    for block in (tables(runs), paired_table(runs)):
        print("\n".join(block) + "\n")


if __name__ == "__main__":
    main()
