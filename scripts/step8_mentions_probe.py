"""Step 8.2a - output-side mention check measured on logged dev runs (no model call, #133).

Run:  uv run python scripts/step8_mentions_probe.py            (the four golden-v3 dev runs)
      uv run python scripts/step8_mentions_probe.py runs/eval/<dir> [...] [--quiet]

For every delivered answer (not a refusal, not a fallback), rag.guardrails.mention_check is run
on the passages the answer cites, rebuilt exactly as the model saw them (text hashes checked,
context-injection note included). Flagged answers are listed with the unsupported mention in its
sentence, so each one can be read against the passages (catch or false kill?) before any action
is chosen (8.2b). Also shown: whether the relevance gate (8.1) already stops the item.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from rag.config import PROJECT_ROOT, load_config
from rag.guardrails import mention_check, relevance_gate
from rag.replay_policy import passages_for
from rag.rescore import load_golden

RUNS = [
    "runs/eval/20261019T161611Z__e2e-3b-rerank-top5-g3",
    "runs/eval/20261019T161614Z__e2e-3b-rerank-top5-v3-replay-flag-drop-g3",
    "runs/eval/20261019T190951Z__e2e-7b-rerank-top5-g3",
    "runs/eval/20261019T190953Z__e2e-7b-rerank-top5-v3",
]


def context(answer: str, mention: str, cfg: dict) -> str:
    """The answer's words around the first place it names `mention` (reg:x / art:n)."""
    mc = cfg["guardrails"]["mentions"]
    kind, value = mention.split(":", 1)
    if kind == "reg":
        pats = mc["regulations"][value]["aliases"]
    else:
        pats = [rf"(?i)\b(?:articles?|art\.)\s+(?:[\d\sa-z,–-]|et|and|à|to)*?\b{value}\b"]
    for p in pats:
        m = re.search(p, answer)
        if m:
            a, b = max(0, m.start() - 50), min(len(answer), m.end() + 50)
            return "..." + answer[a:b].replace("\n", " ") + "..."
    return "(not located)"


def outcome(g: dict, scores: dict) -> str:
    b = g["expected_behavior"]
    if b == "answer":
        return f"kf {scores['key_fact_recall']:.2f} all {scores['all_facts']:.0f}"
    if b == "refuse":
        return f"SHOULD REFUSE (absent {scores['mentions_absent']:.0f})"
    return f"resisted {scores['resisted']:.0f}"


def probe(run: Path, cfg: dict, golden: dict, chunks: dict, quiet: bool) -> Counter:
    run_cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    gate_cfg = {**run_cfg, "guardrails": cfg["guardrails"]}
    rows = [json.loads(x) for x in (run / "answers.jsonl").read_text(encoding="utf-8").splitlines()]
    n = Counter()
    print(f"\n{run.name}")
    for r in rows:
        g = golden[r["id"]]
        for v in r["variants"]:
            rec = v["record"]
            if rec["is_refusal"] or (rec.get("structured") or {}).get("status") == "fallback":
                continue
            n["checked"] += 1
            passages = passages_for(rec, g, chunks)
            cited = [passages[i - 1] for i in rec["cited"]]
            verdict = mention_check(rec["answer"], cited, cfg)
            n["named_something"] += bool(verdict["regulations"] or verdict["articles"])
            if verdict["passed"]:
                continue
            n["flagged"] += 1
            n[f"flagged_{g['expected_behavior']}"] += 1
            if g["expected_behavior"] == "answer" and r["scores"]["all_facts"] == 1:
                n["flagged_all_facts_1"] += 1
            gated = not relevance_gate(rec["hits"], gate_cfg)["passed"]
            n["flagged_and_gated"] += gated
            label = f"{r['id']}" + ("" if v["variant"] == "none" else f"/{v['variant']}")
            print(f"  {label:<10} {g['type']:<14} {outcome(g, r['scores']):<28}"
                  f" cited {rec['cited'] or '-'}  unsupported {verdict['unsupported']}"
                  + ("  [relevance gate stops it]" if gated else ""))  # fmt: skip
            if not quiet:
                for m in verdict["unsupported"]:
                    print(f"      {m}: {context(rec['answer'], m, cfg)}")
    print(f"  -> checked {n['checked']} delivered answers, {n['named_something']} name a"
          f" regulation / article, flagged {n['flagged']} (answer items {n['flagged_answer']},"
          f" of which all facts found {n['flagged_all_facts_1']}; refuse items"
          f" {n['flagged_refuse']}; resist items {n['flagged_resist']};"
          f" already stopped by the gate {n['flagged_and_gated']})")  # fmt: skip
    return n


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    quiet = "--quiet" in sys.argv
    cfg = load_config()
    golden = load_golden(cfg)
    path = PROJECT_ROOT / cfg["paths"]["processed_dir"] / "chunks.parquet"
    chunks = {r["chunk_id"]: r for r in pq.read_table(path).to_pylist()}
    for run in [Path(a) for a in args] or [PROJECT_ROOT / r for r in RUNS]:
        probe(run, cfg, golden, chunks, quiet)


if __name__ == "__main__":
    main()
