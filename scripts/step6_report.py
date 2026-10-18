"""Step 6 - print the README result tables from the run directories (no number typed by hand).

Run: uv run python scripts/step6_report.py [--runs-dir runs/eval] > /tmp/step6_tables.md
Uses the golden-v2 re-scored runs (<tag>-g2, rag.rescore) for every metric, the original e2e runs
for RAGAS (the judge only needs the answers and the reference answers, which v2 did not change).
Paired tests: exact McNemar on items that flip (hit@k, all facts), sign test on the direction of
the change (rr, key-fact recall). dev split only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rag.metrics import mcnemar_exact  # noqa: E402

RETRIEVAL = [  # label, tag
    ("structure, dense (baseline-v1)", "baseline-v1-span"),
    ("structure, hybrid RRF k=60", "hybrid-rrf60"),
    ("structure, dense + rerank (baseline-v2)", "dense-rerank"),
    ("fixed 512, dense", "fixed-dense"),
]
E2E = [
    ("3b, dense top-5 (v1)", "e2e-3b-dense-top5"),
    ("3b, rerank top-5 (v2)", "e2e-3b-rerank-top5"),
    ("3b, rerank top-3", "e2e-3b-rerank-top3-cold"),
    ("7b, dense top-5", "e2e-7b-dense-top5"),
    ("7b, rerank top-5", "e2e-7b-rerank-top5"),
]
E2E_PAIRS = [(0, 1), (3, 4), (1, 4), (1, 2)]


def run_dir(root: Path, tag: str) -> Path:
    found = sorted(root.glob(f"*__{tag}"))
    if not found:
        raise SystemExit(f"no run directory for {tag} in {root}")
    return found[-1]


def load(path: Path, name: str) -> tuple[dict, dict]:
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    rows = {r["id"]: r for r in map(json.loads, (path / name).read_text("utf-8").splitlines())}
    return summary, rows


def paired(a: dict, b: dict, key: str, ids: list[str]) -> str:
    up = sum(b[i]["scores"][key] > a[i]["scores"][key] for i in ids)
    down = sum(b[i]["scores"][key] < a[i]["scores"][key] for i in ids)
    return f"{up} / {down} (p = {mcnemar_exact(up, down):.3f})"


def retrieval_table(root: Path) -> list[str]:
    runs = [(label, *load(run_dir(root, f"{t}-g2"), "retrieval.jsonl")) for label, t in RETRIEVAL]
    base = runs[0][2]
    ids = sorted(base)
    out = [
        "| Retrieval (dev, 33 items with a location) | hit@1 | hit@5 | hit@20 | MRR "
        "| multi-hop hit@5 "
        "| vs baseline-v1: hit@5 flips better / worse | rank direction better / worse |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label, s, rows in runs:
        o, mh = s["overall"], s["by_type"]["multi_hop"]
        same = rows is base
        cmp1 = "—" if same else paired(base, rows, "hit@5", ids)
        cmp2 = "—" if same else paired(base, rows, "rr", ids)
        out.append(
            f"| {label} | {o['hit@1']:.2f} | {o['hit@5']:.2f} | {o['hit@20']:.2f} | {o['rr']:.2f} "
            f"| {mh['hit@5']:.2f} | {cmp1} | {cmp2} |"
        )
    return out


def e2e_table(root: Path) -> tuple[list[str], list[str]]:
    runs = [(label, *load(run_dir(root, f"{t}-g2"), "answers.jsonl")) for label, t in E2E]
    out = [
        "| End to end (dev, 40 items) | evidence in context | key-fact recall | all key facts "
        "| refused answerable | valid citations | refused no-answer (6) | resisted injection (3) "
        "| s / question |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for label, s, _ in runs:
        o, lat = s["overall"], s["latency_s"]
        out.append(
            f"| {label} | {o['context_hit']:.2f} | {o['key_fact_recall']:.2f} "
            f"| {o['all_facts']:.2f} "
            f"| {o['false_refusal']:.2f} | {o['citation_ok']:.2f} | {o['refused']:.2f} "
            f"| {o['resisted'] * 3:.0f} / 3 | {lat['total_mean']:.1f} |"
        )
    pairs = [
        "| Paired comparison (31 answerable items) | key-fact recall better / worse | "
        "all key facts gained / lost |",
        "|---|---|---|",
    ]
    for i, j in E2E_PAIRS:
        a, b = runs[i][2], runs[j][2]
        ids = sorted(k for k, r in a.items() if r["expected_behavior"] == "answer")
        pairs.append(
            f"| {runs[i][0]} → {runs[j][0]} | {paired(a, b, 'key_fact_recall', ids)} "
            f"| {paired(a, b, 'all_facts', ids)} |"
        )
    return out, pairs


def ragas_table(root: Path) -> list[str]:
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    out = [
        "| RAGAS, 12-item subset, judge qwen2.5:7b | "
        + " | ".join(metrics)
        + " | judge failures |",
        "|---|" + "---|" * (len(metrics) + 1),
    ]
    for label, tag in [E2E[0], E2E[1]]:
        s = json.loads((run_dir(root, tag) / "ragas_summary.json").read_text(encoding="utf-8"))
        o = s["overall"]
        fails = sum(o[m]["n_failed"] for m in metrics)
        n = sum(o[m]["n_ok"] + o[m]["n_failed"] for m in metrics)
        cells = " | ".join(f"{o[m]['mean']:.2f}" for m in metrics)
        out.append(f"| {label} | {cells} | {fails} / {n} |")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs/eval")
    root = Path(ap.parse_args().runs_dir)
    e2e, pairs = e2e_table(root)
    for block in (retrieval_table(root), e2e, pairs, ragas_table(root)):
        print("\n".join(block) + "\n")


if __name__ == "__main__":
    main()
