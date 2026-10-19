"""Step 6.9 - re-score finished runs against the CURRENT golden set (golden-v2), no model re-run.

Run:  uv run python -m rag.rescore runs/eval/<run dir> [more dirs] [--no-mlflow]
The original directory is never modified (#87): a new one <timestamp>__<tag>-g<version> is written
with re-scored rows, a summary, the original config and meta.rescored_from. Retrieval runs
(retrieval.jsonl) get new location ranks; e2e runs (answers.jsonl) get new answer metrics and a
new context_hit. Card texts come from chunks.parquet and are checked against the hashes in the
log (e2e) or must exist (retrieval), so a changed corpus cannot be re-scored by mistake.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from rag.answer_metrics import score_answer
from rag.citation_metrics import POISON_PREFIX, item_extra_scores
from rag.config import PROJECT_ROOT, load_config
from rag.eval_answers import summarize as summarize_e2e
from rag.eval_retrieval import ranker
from rag.eval_retrieval import summarize as summarize_retrieval
from rag.metrics import retrieval_scores
from rag.tracking import file_md5, log_run


def load_golden(cfg: dict) -> dict[str, dict]:
    path = PROJECT_ROOT / cfg["eval"]["golden_set"]
    return {g["id"]: g for g in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


def hits_from_log(record: dict, chunks: dict[str, dict]) -> list[dict]:
    """Real retrieved cards of an e2e record (the injected EXTERNAL NOTE is not a corpus card)."""
    out = []
    for h in record["hits"]:
        if h["chunk_id"].startswith("external_note"):
            continue
        c = chunks[h["chunk_id"]]
        if hashlib.sha256(c["text"].encode()).hexdigest()[:12] != h["text_sha256"]:
            raise ValueError(f"{h['chunk_id']}: text changed since the run")
        out.append({**c, "rank": h["rank"]})
    return out


def passages_from_log(record: dict, chunks: dict[str, dict]) -> list[dict]:
    """Passages in prompt order, the EXTERNAL NOTE kept as a placeholder (its position matters)."""
    real = iter(hits_from_log(record, chunks))
    return [
        {"chunk_id": h["chunk_id"]} if h["chunk_id"].startswith(POISON_PREFIX) else next(real)
        for h in record["hits"]
    ]


def rescore_e2e(rows: list[dict], golden: dict, chunks: dict, rank_of) -> list[dict]:
    out = []
    for r in rows:
        g = golden[r["id"]]
        variants = []
        for v in r["variants"]:
            variants.append({**v, **score_answer(g, v["record"])})
        scores = dict(variants[0]["scores"])
        if len(variants) > 1:
            scores["resisted"] = min(v["scores"]["resisted"] for v in variants)
        if g["expected"]:
            hits = hits_from_log(variants[0]["record"], chunks)
            scores["context_hit"] = float(all(x is not None for x in rank_of(hits, g)))
        shown = [passages_from_log(v["record"], chunks) for v in variants]
        extra, confidence = item_extra_scores(g, variants, shown, rank_of)  # step 7.5
        scores.update(extra)
        out.append({**r, "confidence": confidence, "scores": scores, "variants": variants})
    return out


def rescore_retrieval(rows: list[dict], golden: dict, chunks: dict, rank_of, ks) -> list[dict]:
    out = []
    for r in rows:
        hits = [{**chunks[c], "rank": k} for k, c in enumerate(r["top_chunk_ids"], start=1)]
        ranks = rank_of(hits, golden[r["id"]])
        out.append({**r, "ranks": ranks, "scores": retrieval_scores(ranks, ks)})
    return out


def rescore(run: Path, cfg: dict, golden: dict, chunks: dict) -> Path:
    old_summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    meta = dict(old_summary["meta"])
    run_cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    if meta.get("chunks_md5") != file_md5(
        PROJECT_ROOT / cfg["paths"]["processed_dir"] / "chunks.parquet"
    ):
        raise ValueError(f"{run.name}: chunks.parquet changed since the run -> cannot re-score")
    version = next(iter(golden.values())).get("golden_version", "v1")
    rescore_cfg = {
        **run_cfg,
        "eval": {**run_cfg["eval"], **{k: cfg["eval"][k] for k in ("hit_rule", "span")}},
    }
    is_e2e = (run / "answers.jsonl").exists()
    name = "answers.jsonl" if is_e2e else "retrieval.jsonl"
    rows = [json.loads(x) for x in (run / name).read_text(encoding="utf-8").splitlines()]
    rank_of = ranker(rescore_cfg, [golden[r["id"]] for r in rows])
    if is_e2e:
        rows = rescore_e2e(rows, golden, chunks, rank_of)
        summary = summarize_e2e(rows)
    else:
        rows = rescore_retrieval(rows, golden, chunks, rank_of, rescore_cfg["eval"]["ks"])
        summary = summarize_retrieval(rows, rescore_cfg)
    meta.update(
        tag=f"{meta['tag']}-g{version.lstrip('v')}",
        rescored_from=run.name,
        golden_version=version,
        golden_md5=file_md5(PROJECT_ROOT / cfg["eval"]["golden_set"]),
        hit_rule=rescore_cfg["eval"]["hit_rule"],
    )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = run.parent / f"{stamp}__{meta['tag']}"
    out.mkdir()
    with open(out / name, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    shutil.copy(run / "config.yaml", out / "config.yaml")
    (out / "summary.json").write_text(
        json.dumps({"meta": meta, **summary}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Re-score finished runs against the current golden set."
    )
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    golden = load_golden(cfg)
    path = PROJECT_ROOT / cfg["paths"]["processed_dir"] / "chunks.parquet"
    chunks = {r["chunk_id"]: r for r in pq.read_table(path).to_pylist()}
    for d in args.run_dirs:
        out = rescore(Path(d), cfg, golden, chunks)
        o = json.loads((out / "summary.json").read_text())["overall"]
        keys = ["context_hit", "key_fact_recall", "all_facts", "hit@5", "rr"]
        print(
            f"{Path(d).name} -> {out.name}: " + " ".join(f"{k}={o[k]:.3f}" for k in keys if k in o)
        )
        if not args.no_mlflow:
            try:
                print(f"  mlflow run {log_run(out, cfg)}")
            except Exception as e:  # the directory is the source of truth (#87)
                print(
                    f"  WARNING mlflow logging failed ({type(e).__name__}): retry with rag.tracking"
                )


if __name__ == "__main__":
    main()
