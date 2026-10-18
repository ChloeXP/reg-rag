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
    args = ap.parse_args()

    run = Path(args.run_dir)
    cfg = load_config()  # judge settings come from the CURRENT config, recorded in the summary
    golden = {
        g["id"]: g
        for g in map(json.loads, (PROJECT_ROOT / cfg["eval"]["golden_set"]).open(encoding="utf-8"))
    }
    rows = [json.loads(x) for x in (run / "answers.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r["expected_behavior"] == "answer"]
    if args.ids:
        rows = [r for r in rows if r["id"] in set(args.ids)]
    chunks = load_texts(cfg)
    asyncio.run(judge(rows, golden, chunks, build_metrics(cfg), run / OUT))

    lines = [json.loads(x) for x in (run / OUT).read_text(encoding="utf-8").splitlines()]
    lines = [x for x in lines if x["id"] in {r["id"] for r in rows}]
    summary = summarize(lines, rows, cfg)
    (run / SUMMARY).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
