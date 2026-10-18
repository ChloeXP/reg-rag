"""Step 6.7 - RAGAS (0.4 "collections" metrics) on a FINISHED end-to-end run, local judge.

Run:  uv run --group eval python -m rag.ragas_eval runs/eval/<e2e run dir> [--ids G001 G003]
Reads the run's answers.jsonl; never re-runs retrieval or generation (the judge and the judged
model are never in memory at the same time). Appends one line per (item, metric) to
<run dir>/ragas.jsonl (resumable) and writes ragas_summary.json.

Boundaries (decision 6.7-A): this is the ONLY module that imports ragas (and through it
LangChain); it lives in the `eval` dependency group, not in the RAG code or the future image.
Telemetry: ragas sends usage analytics by default -> switched off before the import (#15).
Scope: answerable items only (context recall / precision need a reference answer).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from importlib import metadata
from pathlib import Path

import pyarrow.parquet as pq

from rag.config import PROJECT_ROOT, load_config
from rag.generate import split_passage
from rag.tracking import file_md5

os.environ["RAGAS_DO_NOT_TRACK"] = "true"  # before any ragas import: nothing leaves the machine
OUT, SUMMARY = "ragas.jsonl", "ragas_summary.json"


def load_texts(cfg: dict) -> dict[str, dict]:
    path = PROJECT_ROOT / cfg["paths"]["processed_dir"] / "chunks.parquet"
    return {r["chunk_id"]: r for r in pq.read_table(path).to_pylist()}


def passages(record: dict, chunks: dict[str, dict]) -> list[str]:
    """The passages the model saw (label + body, as in the prompt), checked against the log.

    The run log stores pointers + hashes (#59), so a chunk whose text changed since the run
    is detected here instead of silently judging the answer against a different text.
    """
    out = []
    for h in record["hits"]:
        c = chunks.get(h["chunk_id"])
        if c is None:
            raise ValueError(f"{h['chunk_id']} not in chunks.parquet")
        if hashlib.sha256(c["text"].encode()).hexdigest()[:12] != h["text_sha256"]:
            raise ValueError(f"{h['chunk_id']}: text changed since the run (sha mismatch)")
        label, body = split_passage(c)
        out.append(f"{label}\n{body}")
    return out


def ragas_subset(items: list[dict], rule: dict) -> list[str]:
    """Deterministic stratified subset of answerable dev items (decision 6.7-C).

    Per type quota; anchors first; then greedy by priority: human line-by-line verified (4),
    cross-lingual (2), a work not covered yet (1); ties broken by a seeded shuffle.
    Cross-lingual items are deliberately over-represented: the judge's reliability on them
    is one of the questions of step 6.7.
    """
    pool = [g for g in items if g["split"] == "dev" and g["expected_behavior"] == "answer"]
    by_id = {g["id"]: g for g in pool}
    chosen = [by_id[a] for a in rule["anchors"]]
    order = {g["id"]: k for k, g in enumerate(random.Random(rule["seed"]).sample(pool, len(pool)))}
    for kind, quota in rule["quotas"].items():
        while sum(g["type"] == kind for g in chosen) < quota:
            works = {g["expected"][0]["work_id"] for g in chosen}
            cands = [g for g in pool if g["type"] == kind and g not in chosen]
            if not cands:
                raise ValueError(f"not enough {kind} items for quota {quota}")
            chosen.append(min(cands, key=lambda g, w=works: _priority(g, w, order)))
    return sorted(g["id"] for g in chosen)


def _priority(g: dict, works: set[str], order: dict[str, int]) -> tuple:
    prio = 4 * (g["verification"] == "human_line_by_line") + 2 * g["cross_lingual"]
    prio += g["expected"][0]["work_id"] not in works
    return (-prio, order[g["id"]])


def metric_inputs(name: str, g: dict, answer: str, ctx: list[str]) -> dict:
    q, ref = g["question"], g["ground_truth"]
    return {
        "faithfulness": {"user_input": q, "response": answer, "retrieved_contexts": ctx},
        "answer_relevancy": {"user_input": q, "response": answer},
        "context_precision": {"user_input": q, "reference": ref, "retrieved_contexts": ctx},
        "context_recall": {"user_input": q, "retrieved_contexts": ctx, "reference": ref},
    }[name]


def build_metrics(cfg: dict) -> dict:
    """RAGAS metric objects with the local judge (Ollama, OpenAI-compatible) + local bge-m3."""
    from openai import AsyncOpenAI
    from ragas.embeddings import HuggingFaceEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithReference,
        ContextRecall,
        Faithfulness,
    )

    r = cfg["eval"]["ragas"]
    client = AsyncOpenAI(base_url=r["base_url"], api_key="ollama", timeout=r["timeout_s"])
    llm = llm_factory(r["judge_model"], client=client, temperature=0.0, max_tokens=r["max_tokens"])
    emb = HuggingFaceEmbeddings(model=cfg["embedding"]["model"], device=cfg["embedding"]["device"])
    every = {
        "faithfulness": lambda: Faithfulness(llm=llm),
        "answer_relevancy": lambda: AnswerRelevancy(llm=llm, embeddings=emb),
        "context_precision": lambda: ContextPrecisionWithReference(llm=llm),
        "context_recall": lambda: ContextRecall(llm=llm),
    }
    return {name: every[name]() for name in r["metrics"]}


def done_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    return {(r["id"], r["metric"]) for r in map(json.loads, path.read_text().splitlines())}


async def judge(rows: list[dict], items: dict, chunks: dict, metrics: dict, out: Path) -> None:
    done = done_keys(out)
    todo = [(r, m) for r in rows for m in metrics if (r["id"], m) not in done]
    print(f"{len(rows)} items x {len(metrics)} metrics: {len(done)} done, {len(todo)} to go")
    with open(out, "a", encoding="utf-8") as f:
        for i, (row, name) in enumerate(todo, start=1):
            rec = row["variants"][0]["record"]
            t0 = time.perf_counter()
            line = {"id": row["id"], "metric": name}
            try:
                ctx = passages(rec, chunks)
                res = await metrics[name].ascore(
                    **metric_inputs(name, items[row["id"]], rec["answer"], ctx)
                )
                line["value"] = None if res.value is None else float(res.value)
            except Exception as e:  # local judges do fail (bad JSON, timeouts): count, never hide
                line["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            line["seconds"] = round(time.perf_counter() - t0, 1)
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            f.flush()
            shown = line.get("value", line.get("error", "")[:60])
            print(f"  [{i}/{len(todo)}] {row['id']} {name:<18} {line['seconds']:>6.1f}s {shown}")


def summarize(lines: list[dict], rows: list[dict], cfg: dict) -> dict:
    by_id = {r["id"]: r for r in rows}
    per: dict[str, list[dict]] = defaultdict(list)
    for x in lines:
        per[x["metric"]].append(x)
    overall, groups = {}, defaultdict(dict)
    for name, xs in per.items():
        ok = [x for x in xs if x.get("value") is not None]
        overall[name] = {
            "mean": sum(x["value"] for x in ok) / len(ok) if ok else None,
            "n_ok": len(ok), "n_failed": len(xs) - len(ok),
            "seconds_mean": sum(x["seconds"] for x in xs) / len(xs),
        }  # fmt: skip
        for field in ["cross_lingual", "verification", "type"]:
            vals = defaultdict(list)
            for x in ok:
                vals[str(by_id[x["id"]][field])].append(x["value"])
            groups[f"by_{field}"][name] = {k: sum(v) / len(v) for k, v in sorted(vals.items())}
    r = cfg["eval"]["ragas"]
    meta = {
        "judge_model": r["judge_model"],
        "judge_modelfile_md5": file_md5(PROJECT_ROOT / r["judge_modelfile"]),
        "ragas": metadata.version("ragas"),
        "langchain_community": metadata.version("langchain-community"),
    }
    return {"meta": meta, "overall": overall, **groups}


def main() -> None:
    ap = argparse.ArgumentParser(description="RAGAS on a finished end-to-end run (local judge).")
    ap.add_argument("run_dir")
    ap.add_argument("--ids", nargs="+", help="only these items (trial runs)")
    ap.add_argument("--subset", action="store_true", help="the config's 12-item subset (6.7-C)")
    ap.add_argument("--mlflow", action="store_true", help="also log to the run's MLflow entry")
    args = ap.parse_args()

    run = Path(args.run_dir)
    cfg = load_config()  # judge settings come from the CURRENT config, recorded in the summary
    golden = {
        g["id"]: g
        for g in map(json.loads, (PROJECT_ROOT / cfg["eval"]["golden_set"]).open(encoding="utf-8"))
    }
    rows = [json.loads(x) for x in (run / "answers.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r["expected_behavior"] == "answer"]
    ids = (
        ragas_subset(list(golden.values()), cfg["eval"]["ragas"]["subset"])
        if args.subset
        else args.ids
    )
    if ids:
        rows = [r for r in rows if r["id"] in set(ids)]
        print("items:", " ".join(ids))
    chunks = load_texts(cfg)
    asyncio.run(judge(rows, golden, chunks, build_metrics(cfg), run / OUT))

    lines = [json.loads(x) for x in (run / OUT).read_text(encoding="utf-8").splitlines()]
    lines = [x for x in lines if x["id"] in {r["id"] for r in rows}]
    summary = summarize(lines, rows, cfg)
    summary["meta"]["items"] = sorted(r["id"] for r in rows)
    (run / SUMMARY).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["overall"], indent=2))
    if args.mlflow:
        log_ragas(run, summary, cfg)


def log_ragas(run: Path, summary: dict, cfg: dict) -> None:
    """Add the RAGAS scores to the run's existing MLflow entry (needs the MLflow container)."""
    import mlflow

    run_id = (run / "mlflow_run_id.txt").read_text(encoding="utf-8").strip()
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    with mlflow.start_run(run_id=run_id):
        for name, s in summary["overall"].items():
            mlflow.log_metric(f"ragas/{name}", s["mean"] if s["mean"] is not None else float("nan"))
            mlflow.log_metric(f"ragas/{name}_n_failed", s["n_failed"])
        mlflow.log_metric("ragas/n_items", len(summary["meta"]["items"]))
        mlflow.set_tag("ragas_judge", summary["meta"]["judge_model"])
        mlflow.log_artifact(str(run / OUT))
        mlflow.log_artifact(str(run / SUMMARY))
    print(f"logged to mlflow run {run_id}")


if __name__ == "__main__":
    main()
