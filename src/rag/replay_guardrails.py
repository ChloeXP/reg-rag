"""Step 8.1c - replay the CURRENT input guardrails over finished e2e runs (no model call, #133).

Run:  uv run python -m rag.replay_guardrails runs/eval/<run dir> [more dirs] [--no-mlflow]

Input guardrails look only at what was retrieved, before the LLM, and every retrieved score is
logged at gate precision (rag.guardrails.SCORE_DECIMALS). So their effect on a finished run is
exact: a record the gates pass keeps its logged answer (+ the verdicts); a record they stop becomes
the refusal rag.ask would have returned (same functions: rag.guardrails + rag.ask.NO_LLM). The
gates are evaluated on the run's OWN retrieval config (the scale its scores were produced on). A
replay that would let through a question the run refused before the LLM is impossible (that
answer was never generated) and is refused. Re-scored like rag.rescore; the original directory is
never modified (#87): a new one <timestamp>__<tag>-gr-<suffix> carries meta.replayed_from and
meta.guardrails.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from rag.ask import NO_LLM
from rag.config import PROJECT_ROOT, load_config
from rag.eval_answers import summarize as summarize_e2e
from rag.eval_retrieval import ranker
from rag.generate import load_prompt
from rag.guardrails import refusal_fields, relevance_gate
from rag.rescore import load_golden, rescore_e2e
from rag.tracking import file_md5, log_run

KEYS = ["refused", "mentions_absent", "false_refusal", "resisted", "key_fact_recall",
        "all_facts", "answerable_acc", "citation_ok"]  # fmt: skip


def apply_input_guardrails(record: dict, gate_cfg: dict, prompt: dict) -> dict:
    """One logged record -> the record the current input guardrails would have produced."""
    gates = [relevance_gate(record["hits"], gate_cfg)]
    passed = all(v["passed"] for v in gates)
    old = record.get("guardrails")
    if old and not old["llm_called"] and passed:
        raise ValueError("these guardrails let through a question the run refused before the "
                         "LLM: its answer was never generated (not replayable)")  # fmt: skip
    rec = {**record, "guardrails": {"llm_called": passed, "input": gates}}
    if not passed:
        rec.update(NO_LLM)
        rec.update(refusal_fields(prompt, record["language"]))
    return rec


def suffix(guardrails: dict) -> str:
    rel = guardrails["relevance"]
    return "gr-" + (f"rel{rel['min_top_score']}" if rel["enabled"] else "none")


def replay(run: Path, cfg: dict, golden: dict, chunks: dict) -> Path:
    meta = dict(json.loads((run / "summary.json").read_text(encoding="utf-8"))["meta"])
    run_cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    if meta.get("chunks_md5") != file_md5(
        PROJECT_ROOT / cfg["paths"]["processed_dir"] / "chunks.parquet"
    ):
        raise ValueError(f"{run.name}: chunks.parquet changed since the run")
    prompt = load_prompt(run_cfg)
    if prompt["sha256"] != meta["prompt_sha256"]:
        raise ValueError(f"{run.name}: prompt {prompt['version']} changed since the run")
    gate_cfg = {**run_cfg, "guardrails": cfg["guardrails"]}  # run's retrieval = run's scale
    rows = [json.loads(x) for x in (run / "answers.jsonl").read_text(encoding="utf-8").splitlines()]
    for r in rows:
        for v in r["variants"]:
            v["record"] = apply_input_guardrails(v["record"], gate_cfg, prompt)
    rescore_cfg = {**run_cfg, "eval": {**run_cfg["eval"],
                   **{k: cfg["eval"][k] for k in ("hit_rule", "span")}}}  # fmt: skip
    rows = rescore_e2e(rows, golden, chunks, ranker(rescore_cfg, [golden[r["id"]] for r in rows]))
    for r in rows:  # item-level timings follow the replayed records
        r["llm_s"] = round(sum(v["record"]["llm_s"] or 0 for v in r["variants"]), 2)
        r["total_s"] = round(r["retrieve_s"] + r["llm_s"], 2)
    summary = summarize_e2e(rows)
    meta.update(
        tag=f"{meta['tag']}-{suffix(cfg['guardrails'])}",
        replayed_from=run.name,
        guardrails=cfg["guardrails"],
        golden_version=next(iter(golden.values())).get("golden_version", "v1"),
        golden_md5=file_md5(PROJECT_ROOT / cfg["eval"]["golden_set"]),
    )
    out = run.parent / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}__{meta['tag']}"
    out.mkdir()
    with open(out / "answers.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "config.yaml").write_text(yaml.safe_dump(gate_cfg, sort_keys=False), "utf-8")
    shutil.copy(run / "config.yaml", out / "config.original.yaml")
    (out / "summary.json").write_text(
        json.dumps({"meta": meta, **summary}, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )
    return out


def print_changes(run: Path, out: Path) -> None:
    before = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    after = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    fmt = lambda x: "-" if x is None else f"{x:.3f}"  # noqa: E731
    for k in KEYS:
        b, a = before["overall"].get(k), after["overall"].get(k)
        if b is not None or a is not None:
            print(f"     {k:<16} {fmt(b):>7} -> {fmt(a):>7}")
    lb, la = before["latency_s"], after["latency_s"]
    print(f"     {'llm_s mean':<16} {lb['llm_mean']:>7.1f} -> {la['llm_mean']:>7.1f}")
    old = {json.loads(x)["id"]: json.loads(x) for x in (run / "answers.jsonl").open()}
    for x in (out / "answers.jsonl").open():
        r = json.loads(x)
        gated = [v["variant"] for v in r["variants"] if not v["record"]["guardrails"]["llm_called"]]
        changed = {k: (round(old[r["id"]]["scores"].get(k, -1), 2), round(v, 2))
                   for k, v in r["scores"].items()
                   if k in KEYS and old[r["id"]]["scores"].get(k) != v}  # fmt: skip
        if gated or changed:
            print(f"     {r['id']}: gated {gated or '-'}  {changed or 'scores unchanged'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay the current input guardrails over runs.")
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    golden = load_golden(cfg)
    path = PROJECT_ROOT / cfg["paths"]["processed_dir"] / "chunks.parquet"
    chunks = {r["chunk_id"]: r for r in pq.read_table(path).to_pylist()}
    for d in args.run_dirs:
        run = Path(d)
        out = replay(run, cfg, golden, chunks)
        print(f"{run.name}\n  -> {out.name}")
        print_changes(run, out)
        if not args.no_mlflow:
            try:
                print(f"  mlflow run {log_run(out, cfg)}")
            except Exception as e:  # the directory is the source of truth (#87)
                print(f"  WARNING mlflow logging failed ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
