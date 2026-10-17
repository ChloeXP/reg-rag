"""Step 4.6 - end-to-end smoke test: 5 smoke queries x N models through rag.ask, one process.

Run: uv run python scripts/smoke_ask.py [--models qwen2.5:3b qwen2.5:7b] [--ids Q1 Q5] [--no-log]

Automatic (objective) checks per answer: requested vs actual answer language, refusal,
invalid / missing citations, and whether a CITED passage is at the expected location
(same rule as try_search.is_hit).
Whether the answer itself is right is judged by hand against the cited text.
"""

from __future__ import annotations

import argparse
import copy

import pyarrow.parquet as pq
import yaml
from try_search import is_hit  # same hit rule as the retrieval smoke test (step 3.4)

from rag.ask import append_log, ask, show
from rag.config import PROJECT_ROOT, load_config
from rag.generate import detect_language


def expected_language(qtype: str) -> str:
    return "fr" if qtype.startswith(("fr_", "cross_fr")) else "en"


def chunk_lookup(cfg: dict) -> dict[str, dict]:
    """chunk_id -> metadata: the log stores pointers only (decision 4.5-B), resolve them here."""
    path = PROJECT_ROOT / cfg["paths"]["processed_dir"] / "chunks.parquet"
    cols = ["chunk_id", "work_id", "article", "section", "text"]
    return {r["chunk_id"]: r for r in pq.read_table(path, columns=cols).to_pylist()}


def judge(rec: dict, q: dict, chunks: dict[str, dict]) -> dict:
    exp = q["expected"]
    hit_ok = [is_hit(chunks[h["chunk_id"]], exp) for h in rec["hits"]]
    cited_ok = any(hit_ok[n - 1] for n in rec["cited"])
    return {
        "retrieved@": next((i for i, ok in enumerate(hit_ok, 1) if ok), None),
        "cited_ok": cited_ok,
        "lang_ok": rec["language"] == expected_language(q["type"]),
        # language the model ACTUALLY answered in ("?" = no signal), vs the one it was asked for
        "answer_lang": detect_language(rec["answer"], default="?"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", help="default: llm.model from config")
    ap.add_argument("--ids", nargs="+", help="subset of query ids, e.g. Q1 Q5")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    base = load_config()
    queries = yaml.safe_load(open(PROJECT_ROOT / "eval" / "smoke_queries.yaml", encoding="utf-8"))
    queries = [q for q in queries if not args.ids or q["id"] in args.ids]
    chunks = chunk_lookup(base)

    rows = []
    for model in args.models or [base["llm"]["model"]]:
        cfg = copy.deepcopy(base)
        cfg["llm"]["model"] = model
        for q in queries:
            print(f"\n{'=' * 100}\n{q['id']} [{q['type']}] {model}: {q['query']}")
            rec = ask(q["query"], cfg)
            show(rec)
            if not args.no_log:
                append_log(rec, cfg)
            rows.append((q["id"], model, rec, judge(rec, q, chunks)))

    print(
        f"\n{'id':<4}{'model':<12}{'lang':<6}{'ans':<5}{'retr@':<6}{'cited_ok':<9}{'refusal':<8}"
        f"{'invalid':<8}{'uncited':<8}{'[..]':<5}{'prefill':>8}{'decode':>7}{'out':>5}"
    )
    for qid, model, rec, j in rows:
        print(
            f"{qid:<4}{model:<12}{rec['language'] + ('' if j['lang_ok'] else '!'):<6}"
            f"{j['answer_lang'] + ('' if j['answer_lang'] == rec['language'] else '!'):<5}"
            f"{j['retrieved@'] or '-':<6}{'yes' if j['cited_ok'] else 'NO':<9}"
            f"{str(rec['is_refusal']):<8}{str(rec['invalid'] or '-'):<8}{str(rec['uncited']):<8}"
            f"{rec['other_brackets']:<5}{rec['prefill_s']:>8}{rec['decode_s']:>7}"
            f"{rec['output_tokens']:>5}"
        )


if __name__ == "__main__":
    main()
