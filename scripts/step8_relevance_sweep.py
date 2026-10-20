"""Step 8.1 - relevance threshold: sensitivity curve on logged dev runs (no model call, #133).

Run:  uv run python scripts/step8_relevance_sweep.py            (the four golden-v3 dev runs)
      uv run python scripts/step8_relevance_sweep.py runs/eval/<dir> [...]

For each threshold tau, every logged answer whose best passage scores < tau is replaced by the
refusal a gated pipeline would have returned (rag.guardrails: same function as in production)
and re-scored with rag.answer_metrics. Exact counterfactual: retrieval is deterministic and the
gate runs before the LLM, so a non-gated item keeps its logged answer. Dev split only.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import yaml

from rag.answer_metrics import score_answer
from rag.config import PROJECT_ROOT, load_config
from rag.generate import load_prompt
from rag.guardrails import refusal_fields, relevance_gate
from rag.tracking import file_md5

RUNS = [
    "runs/eval/20261019T161611Z__e2e-3b-rerank-top5-g3",
    "runs/eval/20261019T161614Z__e2e-3b-rerank-top5-v3-replay-flag-drop-g3",
    "runs/eval/20261019T190951Z__e2e-7b-rerank-top5-g3",
    "runs/eval/20261019T190953Z__e2e-7b-rerank-top5-v3",
]
TAUS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.45]


def load(run: Path, cfg: dict) -> tuple[list[dict], dict, dict]:
    meta = json.loads((run / "summary.json").read_text(encoding="utf-8"))["meta"]
    golden_path = PROJECT_ROOT / cfg["eval"]["golden_set"]
    if meta["split"] != "dev" or meta["golden_md5"] != file_md5(golden_path):
        raise ValueError(f"{run.name}: not a dev run scored on the current golden set")
    golden = {}
    for line in golden_path.read_text(encoding="utf-8").splitlines():
        g = json.loads(line)
        golden[g["id"]] = g
    run_cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    prompt = load_prompt(run_cfg)
    if prompt["sha256"] != meta["prompt_sha256"]:
        raise ValueError(f"{run.name}: prompt {prompt['version']} changed since the run")
    rows = [json.loads(x) for x in (run / "answers.jsonl").read_text(encoding="utf-8").splitlines()]
    return rows, golden, prompt


def sweep(rows: list[dict], golden: dict, prompt: dict, cfg: dict, tau: float) -> dict:
    c = copy.deepcopy(cfg)
    c["guardrails"]["relevance"].update(enabled=True, min_top_score=tau)
    out = {"refused": 0, "n_refuse": 0, "false_refusal": 0, "gated_answerable": 0, "n_answer": 0,
           "resisted": 0, "n_resist": 0, "llm_calls_saved": 0, "gated_ids": []}  # fmt: skip
    for r in rows:
        g = golden[r["id"]]
        worst: dict[str, float] = {}
        for v in r["variants"]:  # context-injection items: worst case over positions
            rec = v["record"]
            if not relevance_gate(rec["hits"], c)["passed"]:
                rec = {**rec, **refusal_fields(prompt, rec["language"])}
                out["llm_calls_saved"] += 1
                out["gated_ids"].append(r["id"])
            s = score_answer(g, rec)["scores"]
            for k, val in s.items():
                worst[k] = (
                    val if k not in worst else (max if k == "false_refusal" else min)(worst[k], val)
                )
        b = g["expected_behavior"]
        if b == "refuse":
            out["n_refuse"] += 1
            out["refused"] += worst["refused"]
        elif b == "answer":
            out["n_answer"] += 1
            out["false_refusal"] += worst["false_refusal"]
            out["gated_answerable"] += r["id"] in out["gated_ids"]
        else:
            out["n_resist"] += 1
            out["resisted"] += worst["resisted"]
    return out


def main() -> None:
    cfg = load_config()
    tau_cfg = cfg["guardrails"]["relevance"]["min_top_score"]
    runs = [Path(a) for a in sys.argv[1:]] or [PROJECT_ROOT / r for r in RUNS]
    for run in runs:
        rows, golden, prompt = load(run, cfg)
        base = sweep(rows, golden, prompt, cfg, 0.0)  # identity check: tau=0 == the run itself
        logged = json.loads((run / "summary.json").read_text(encoding="utf-8"))["overall"]
        for key, n in [("refused", "n_refuse"), ("false_refusal", "n_answer"),
                       ("resisted", "n_resist")]:  # fmt: skip
            if round(base[key] / base[n], 6) != round(logged[key], 6):
                raise ValueError(f"{run.name}: {key} at tau=0 != logged summary (scoring drift)")
        tops = {r["id"]: max(h["score"] for h in r["variants"][0]["record"]["hits"]) for r in rows}
        ans = sorted((tops[i], i) for i in tops if golden[i]["expected_behavior"] == "answer")
        print(f"\n{run.name}  (prompt {prompt['version']}, {len(rows)} dev items)")
        print(f"  lowest answerable top-1: {', '.join(f'{i} {s:.3f}' for s, i in ans[:3])}")
        print(f"  {'tau':>5} | refused | false_ref (gate) | resisted | llm saved | gated")
        for tau in sorted({*TAUS, tau_cfg}):
            o = sweep(rows, golden, prompt, cfg, tau)
            mark = "  <- config" if tau == tau_cfg else ""
            print(
                f"  {tau:>5.2f} | {o['refused']:.0f}/{o['n_refuse']:<5} |"
                f" {o['false_refusal']:.0f}/{o['n_answer']:<3} ({o['gated_answerable']})"
                f"      | {o['resisted']:.0f}/{o['n_resist']:<6} | {o['llm_calls_saved']:>9} |"
                f" {' '.join(sorted(set(o['gated_ids'])))}{mark}"
            )
        near = [(i, s) for i, s in tops.items() if abs(s - tau_cfg) <= 0.01]
        print(f"  items within +-0.01 of tau={tau_cfg}: {near or 'none'}")


if __name__ == "__main__":
    main()
