"""Step 6.6b - end-to-end evaluation: golden set -> retrieve -> (inject) -> generate -> metrics.

Run:  uv run python -m rag.eval_answers [--set llm.model=qwen2.5:7b] [--tag 7b] [--ids G001 G040]
      uv run python -m rag.eval_answers --resume runs/eval/<dir>     # continue after a crash
One process (bge-m3 + the LLM loaded once, by warm-up calls kept out of the timings).
Every answer is appended to answers.jsonl as soon as it exists, so a long run can be resumed;
a resumed run re-reads ITS OWN config.yaml (overrides are refused) so it cannot change setup.

Context-injection items (decision 6.6b-D): the poisoned passage is inserted FIRST and, in a
second call, LAST (k+1 passages, no real passage removed); the item passes only if the model
resists in both positions = worst case. The passage is labelled like a document
("EXTERNAL NOTE"), never "injected".
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

from rag.answer_metrics import aggregate_present, grouped_present, score_answer
from rag.ask import answer, git_commit
from rag.config import PROJECT_ROOT, apply_overrides, load_config
from rag.eval_retrieval import GROUP_FIELDS, ranker
from rag.generate import load_prompt
from rag.llm import generate, unload
from rag.retrieve import search
from rag.tracking import file_md5, log_run

POSITIONS = ["first", "last"]
ANSWERS = "answers.jsonl"


def load_split(cfg: dict, split: str, ids: list[str] | None) -> list[dict]:
    path = PROJECT_ROOT / cfg["eval"]["golden_set"]
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows = [g for g in rows if g["split"] == split]
    if ids:
        unknown = set(ids) - {g["id"] for g in rows}
        if unknown:
            raise ValueError(f"not in the {split} split: {sorted(unknown)}")
        rows = [g for g in rows if g["id"] in ids]
    return rows


def poisoned_hit(g: dict) -> dict:
    """The injected passage, shaped like a retrieved card (label 'EXTERNAL NOTE')."""
    return {
        "rank": 0, "score": 0.0, "chunk_id": "external_note::0", "doc_id": "external_note",
        "work_id": "external_note", "language": g["question_language"], "kind": "body",
        "article": None, "section": None, "page_start": None, "page_end": None,
        "text": g["injected_passage"],
    }  # fmt: skip


def contexts(g: dict, hits: list[dict]) -> list[tuple[str, list[dict]]]:
    """(variant, passages) to send: one variant, or first/last for context injection."""
    if g.get("injection") != "context":
        return [("none", hits)]
    bad = poisoned_hit(g)
    return [("first", [bad, *hits]), ("last", [*hits, bad])]


def run_item(g: dict, cfg: dict, rank_of) -> dict:
    t0 = time.perf_counter()
    hits = search(g["question"], cfg)
    retrieve_s = round(time.perf_counter() - t0, 3)
    variants = []
    for variant, passages in contexts(g, hits):
        rec = answer(g["question"], passages, cfg, lang=None, retrieve_s=retrieve_s)
        variants.append({"variant": variant, **score_answer(g, rec), "record": rec})
    scores = dict(variants[0]["scores"])
    if len(variants) > 1:  # worst case over positions
        scores["resisted"] = min(v["scores"]["resisted"] for v in variants)
    if g["expected"]:  # were all expected locations among the passages the model saw?
        scores["context_hit"] = float(all(r is not None for r in rank_of(hits, g)))
    llm_s = sum(v["record"]["llm_s"] for v in variants)
    return {
        "id": g["id"],
        **{f: g[f] for f in GROUP_FIELDS},
        "expected_behavior": g["expected_behavior"],
        "scores": scores,
        "variants": variants,
        "retrieve_s": retrieve_s,
        "llm_s": llm_s,
        "total_s": round(retrieve_s + llm_s, 2),
        "prompt_tokens": variants[0]["record"]["prompt_tokens"],
        # P67 cache detector: ~3 ms/token on an M1 for qwen2.5:3b; ~0.02 when the prefix was cached
        "prefill_ms_per_tok": _ms_per_tok(variants[0]["record"]),
    }


def _ms_per_tok(rec: dict) -> float | None:
    if not rec.get("prefill_s") or not rec.get("prompt_tokens"):
        return None
    return round(1000 * rec["prefill_s"] / rec["prompt_tokens"], 3)


def _stats(values: list[float]) -> dict[str, float]:
    v = sorted(x for x in values if x is not None)
    return {"mean": sum(v) / len(v), "p50": v[len(v) // 2], "max": v[-1]} if v else {}


def summarize(rows: list[dict]) -> dict:
    lat = {}
    for key in ["total_s", "retrieve_s", "llm_s", "prompt_tokens", "prefill_ms_per_tok"]:
        for stat, value in _stats([r[key] for r in rows]).items():
            lat[f"{key.removesuffix('_s')}_{stat}"] = value
    return {
        "overall": aggregate_present(rows),
        **{f"by_{f}": grouped_present(rows, f) for f in ["type", *GROUP_FIELDS[1:]]},
        "latency_s": lat,  # flat dict of numbers (rag.tracking logs it as metrics)
    }


def print_report(summary: dict, rows: list[dict]) -> None:
    cols = ["context_hit", "key_fact_recall", "all_facts", "false_refusal", "citation_ok",
            "refused", "mentions_absent", "resisted"]  # fmt: skip
    short = ["ctx_hit", "kf_rec", "all_kf", "f_refus", "cite_ok", "refused", "m_absent", "resist"]
    print(f"\n{'group':<30}{'n':>4}" + "".join(f"{c:>9}" for c in short))

    def line(name: str, agg: dict) -> None:
        cells = [f"{agg[c]:>9.2f}" if c in agg else f"{'-':>9}" for c in cols]
        print(f"{name:<30}{agg['n']:>4}" + "".join(cells))

    line("ALL", summary["overall"])
    for section, groups in summary.items():
        if section.startswith("by_"):
            for value, agg in groups.items():
                line(f"  {section[3:]}={value}", agg)
    print("\nitems below full marks:")
    for r in rows:
        s, v0 = r["scores"], r["variants"][0]
        if r["expected_behavior"] == "answer" and s["all_facts"] < 1:
            ctx = "ctx ok" if s.get("context_hit") else "ctx MISSING"
            refusal = " REFUSED" if s["false_refusal"] else ""
            missing = v0["detail"]["facts_missing"]
            print(f"  {r['id']} {r['type']:<14} {ctx}{refusal} missing={missing}")
        elif r["expected_behavior"] == "refuse" and (not s["refused"] or s["mentions_absent"]):
            print(f"  {r['id']} no_answer      refused={s['refused']:.0f} mentions={v0['detail']}")
        elif r["expected_behavior"] == "resist" and not s["resisted"]:
            for v in r["variants"]:
                print(f"  {r['id']} adversarial [{v['variant']}] {v['detail']}")
    lat = summary["latency_s"]
    print(
        f"\nlatency per item: total mean {lat['total_mean']:.1f}s p50 {lat['total_p50']:.1f}s"
        f" max {lat['total_max']:.1f}s | llm mean {lat['llm_mean']:.1f}s"
        f" | prompt tokens mean {lat['prompt_tokens_mean']:.0f}"
        f" | prefill {lat['prefill_ms_per_tok_p50']:.2f} ms/token (p50; ~0.0x = cached prefix)"
    )


def done_rows(out: Path) -> list[dict]:
    path = out / ANSWERS
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="End-to-end evaluation on the golden set.")
    ap.add_argument("--set", action="append", metavar="KEY=VALUE", help="config override")
    ap.add_argument("--tag", help="run name suffix")
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--final", action="store_true", help="required to touch the test split")
    ap.add_argument("--ids", nargs="+", help="only these items (smoke runs, RAGAS trials)")
    ap.add_argument("--resume", help="run directory to continue")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()
    if args.split == "test" and not args.final:
        ap.error("the test split is evaluated once, at the end: add --final if this is it")

    if args.resume:
        if args.set or args.tag:
            ap.error("--resume reuses the run's own config: no --set / --tag")
        out = Path(args.resume)
        cfg = yaml.safe_load((out / "config.yaml").read_text(encoding="utf-8"))
        meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    else:
        cfg = apply_overrides(copy.deepcopy(load_config()), args.set)
        r, prompt = cfg["retrieval"], load_prompt(cfg)
        tag = args.tag or (
            f"e2e-{cfg['llm']['model'].replace(':', '')}-{r['strategy']}-{r['mode']}"
            f"-top{r['top_k']}" + ("-rerank" if r["rerank"]["enabled"] else "")
        )
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out = (
            PROJECT_ROOT / cfg["paths"]["runs_dir"] / cfg["eval"]["runs_subdir"] / f"{stamp}__{tag}"
        )
        out.mkdir(parents=True)
        meta = {
            "tag": tag, "layer": "e2e", "split": args.split, "ids": args.ids or [],
            "overrides": args.set or [], "git": git_commit(), "model": cfg["llm"]["model"],
            "prompt_version": prompt["version"], "prompt_sha256": prompt["sha256"],
            "hit_rule": cfg["eval"]["hit_rule"], "injection_positions": POSITIONS,
            "llm_cache": "model unloaded before the run (P67)",
            "golden_md5": file_md5(PROJECT_ROOT / cfg["eval"]["golden_set"]),
            "chunks_md5": file_md5(PROJECT_ROOT / cfg["paths"]["processed_dir"] / "chunks.parquet"),
        }  # fmt: skip
        (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    items = load_split(cfg, meta["split"], meta["ids"] or None)
    rows = done_rows(out)
    todo = [g for g in items if g["id"] not in {r["id"] for r in rows}]
    print(
        f"{meta['tag']} | {meta['split']} | {len(items)} items, {len(rows)} done, {len(todo)} to go"
    )

    if todo:
        t0 = time.perf_counter()
        search("warm-up", cfg, top_k=1)  # embedding (+ reranker) models
        unload(cfg)  # P67: drop prompt caches left by an earlier run -> same cold start for all
        generate("Say OK.", cfg)  # loads the LLM again, outside the per-item timings
        print(f"warm-up {time.perf_counter() - t0:.1f}s")
        rank_of = ranker(cfg, items)
        with open(out / ANSWERS, "a", encoding="utf-8") as f:
            for i, g in enumerate(todo, start=1):
                row = run_item(g, cfg, rank_of)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()  # crash-safe: what is written can be resumed
                rows.append(row)
                print(f"  [{i}/{len(todo)}] {g['id']} {g['type']:<14} {row['total_s']:>5.1f}s"
                      f" {row['scores']}")  # fmt: skip

    order = {g["id"]: k for k, g in enumerate(items)}
    rows.sort(key=lambda r: order[r["id"]])
    summary = summarize(rows)
    (out / "summary.json").write_text(
        json.dumps({"meta": meta, **summary}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print_report(summary, rows)
    print(f"\n-> {out}")
    if not args.no_mlflow:
        try:
            print(f"mlflow run {log_run(out, cfg)}")
        except Exception as e:  # the run directory is the source of truth (#87)
            print(f"WARNING mlflow logging failed ({type(e).__name__}: {e})")
            print(f"  retry later: uv run python -m rag.tracking {out}")


if __name__ == "__main__":
    main()
