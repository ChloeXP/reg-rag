"""Step 7.8 - replay a finished structured-output (v2+) e2e run under the CURRENT output policy.

Run:  uv run python -m rag.replay_policy runs/eval/<run dir> [more dirs] [--no-mlflow]

No model call: every attempt's raw output is in the log, so the attempts are re-checked in their
original order under config llm.structured.policy and the replay stops at the first accepted one.
This is exact for a policy that accepts at least what the original accepted (length=flag,
invalid_citations=drop vs the 7.3 reject/reject); a replay that would need an attempt the run
never made is refused. Records are rebuilt with rag.structured_gen.result, then re-scored like
rag.rescore (current golden set). The original directory is never modified (#87): a new one
<timestamp>__<tag>-replay-<policy> is written with meta.replayed_from and meta.policy.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from rag.ask import TIMING, _sum
from rag.citation_metrics import POISON_PREFIX
from rag.config import PROJECT_ROOT, load_config
from rag.eval_answers import poisoned_hit
from rag.eval_answers import summarize as summarize_e2e
from rag.eval_retrieval import ranker
from rag.generate import load_prompt
from rag.rescore import hits_from_log, load_golden, rescore_e2e
from rag.structured_gen import check, policy_of, result
from rag.tracking import file_md5, log_run

KEYS = ["key_fact_recall", "all_facts", "refused", "resisted", "answerable_acc", "fallback",
        "dropped_invalid", "over_length"]  # fmt: skip


def passages_for(record: dict, g: dict, chunks: dict) -> list[dict]:
    """The passages exactly as shown to the model (text hashes checked, poison rebuilt)."""
    real = iter(hits_from_log(record, chunks))  # raises if a card text changed
    return [
        poisoned_hit(g) if h["chunk_id"].startswith(POISON_PREFIX) else next(real)
        for h in record["hits"]
    ]


def replay_record(record: dict, passages: list[dict], prompt: dict, policy: dict) -> dict:
    old = record["structured"]
    attempts, final = [], None
    for a in old["attempts"]:
        status, errors, ma, invalid, notes = check(
            a["raw"], a.get("done_reason"), passages, prompt, policy
        )
        base = {k: v for k, v in a.items() if k not in ("dropped", "over_length")}
        attempts.append({**base, "status": status, "errors": errors, "invalid": invalid, **notes})
        if status == "ok":
            final = ma
            break
    if final is None and old["status"] != "fallback":
        raise ValueError("this policy rejects an answer the run accepted: it needs an attempt "
                         "the run never made (not replayable)")  # fmt: skip
    new = result(final, attempts, passages, prompt, record["language"])
    new["structured"]["policy"] = policy
    timing = {k: _sum(a.get(src) for a in attempts) for k, src in TIMING.items()}
    return {**record, **timing, "answer": new.pop("answer"), **new}


def replay(run: Path, cfg: dict, golden: dict, chunks: dict) -> Path:
    old_summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    meta = dict(old_summary["meta"])
    run_cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    if meta.get("chunks_md5") != file_md5(
        PROJECT_ROOT / cfg["paths"]["processed_dir"] / "chunks.parquet"
    ):
        raise ValueError(f"{run.name}: chunks.parquet changed since the run")
    prompt = load_prompt(run_cfg)
    if prompt["sha256"] != meta["prompt_sha256"] or prompt["output"] != "json":
        raise ValueError(f"{run.name}: prompt {prompt['version']} changed or is not structured")
    policy = policy_of(cfg)
    rows = [json.loads(x) for x in (run / "answers.jsonl").read_text(encoding="utf-8").splitlines()]
    for r in rows:
        g = golden[r["id"]]
        for v in r["variants"]:
            v["record"] = replay_record(
                v["record"], passages_for(v["record"], g, chunks), prompt, policy
            )
    rescore_cfg = {**run_cfg, "eval": {**run_cfg["eval"],
                   **{k: cfg["eval"][k] for k in ("hit_rule", "span")}}}  # fmt: skip
    rows = rescore_e2e(rows, golden, chunks, ranker(rescore_cfg, [golden[r["id"]] for r in rows]))
    for r in rows:  # item-level timings follow the replayed records
        r["llm_s"] = round(sum(v["record"]["llm_s"] or 0 for v in r["variants"]), 2)
        r["total_s"] = round(r["retrieve_s"] + r["llm_s"], 2)
    summary = summarize_e2e(rows)
    suffix = f"replay-{policy['length']}-{policy['invalid_citations']}"
    meta.update(
        tag=f"{meta['tag']}-{suffix}",
        replayed_from=run.name,
        policy=policy,
        golden_version=next(iter(golden.values())).get("golden_version", "v1"),
        golden_md5=file_md5(PROJECT_ROOT / cfg["eval"]["golden_set"]),
    )
    out = run.parent / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}__{meta['tag']}"
    out.mkdir()
    with open(out / "answers.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    replay_cfg = {**run_cfg, "llm": {**run_cfg["llm"], "structured": {
        **run_cfg["llm"]["structured"], "policy": policy}}}  # fmt: skip
    (out / "config.yaml").write_text(yaml.safe_dump(replay_cfg, sort_keys=False), "utf-8")
    shutil.copy(run / "config.yaml", out / "config.original.yaml")
    (out / "summary.json").write_text(
        json.dumps({"meta": meta, **summary}, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )
    return out


def print_changes(run: Path, out: Path) -> None:
    """Items whose content / safety scores changed, with their new final status."""
    old = {json.loads(x)["id"]: json.loads(x) for x in (run / "answers.jsonl").open()}
    for x in (out / "answers.jsonl").open():
        r = json.loads(x)
        before = old[r["id"]]["scores"]
        changed = {
            k: (round(before.get(k, -1), 2), round(v, 2))
            for k, v in r["scores"].items()
            if k in KEYS[:4] and before.get(k) != v
        }
        if changed:
            status = r["variants"][0]["record"]["structured"]["status"]
            print(f"     {r['id']}: {changed}  status {status}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay structured runs under the current policy.")
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
        before = json.loads((run / "summary.json").read_text())["overall"]
        after = json.loads((out / "summary.json").read_text())["overall"]
        print(f"{run.name}\n  -> {out.name}")
        for k in KEYS:
            if k in before or k in after:
                b, a = before.get(k), after.get(k)
                fmt = lambda x: "-" if x is None else f"{x:.3f}"  # noqa: E731
                print(f"     {k:<18} {fmt(b):>7} -> {fmt(a):>7}")
        print_changes(run, out)
        if not args.no_mlflow:
            try:
                print(f"  mlflow run {log_run(out, cfg)}")
            except Exception as e:  # the directory is the source of truth (#87)
                print(f"  WARNING mlflow logging failed ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
