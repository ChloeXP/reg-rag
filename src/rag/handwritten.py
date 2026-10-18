"""Step 5.5 - handwritten golden-set items (multi_hop, article_lookup, no_answer, adversarial).

Run: uv run python -m rag.handwritten --check   # validate eval/handwritten.yaml against the corpus

eval/handwritten.yaml is the source of truth (drafted by Claude, verified by a human: 5.5-B).
Checks per type, all against chunks.parquet:
  answer types   every source chunk is covered by an expected location and vice versa,
                 key-fact regexes match the sources, multi_hop has >= 2 locations
  no_answer      every absence pattern has ZERO hits in the whole corpus (both strategies)
  adversarial    injection is 'question' or 'context' (context needs injected_passage),
                 must_not / must_match regexes compile
"""

from __future__ import annotations

import argparse
import re
from collections import Counter

import pyarrow.parquet as pq
import yaml

from rag.config import PROJECT_ROOT, load_config
from rag.golden import is_hit, load_spec

BEHAVIOR = {"multi_hop": "answer", "article_lookup": "answer", "no_answer": "refuse",
            "adversarial": "resist"}  # fmt: skip


def load_items(cfg: dict) -> list[dict]:
    with open(PROJECT_ROOT / cfg["eval"]["handwritten"], encoding="utf-8") as f:
        return yaml.safe_load(f)


def all_chunks(cfg: dict) -> list[dict]:
    path = PROJECT_ROOT / cfg["paths"]["processed_dir"] / "chunks.parquet"
    return pq.read_table(path).to_pylist()


def _compile(pat: str, sid: str, errors: list) -> re.Pattern | None:
    try:
        return re.compile(pat, re.I)
    except re.error as e:
        errors.append(f"{sid}: bad regex {pat!r}: {e}")
        return None


def check_item(it: dict, by_id: dict, corpus: list[dict], errors: list, warnings: list) -> None:
    sid, typ = it["hid"], it["type"]
    if typ not in BEHAVIOR:
        errors.append(f"{sid}: unknown type {typ!r}")
        return
    if it.get("expected_behavior") != BEHAVIOR[typ]:
        errors.append(f"{sid}: expected_behavior must be {BEHAVIOR[typ]!r} for {typ}")
    if it.get("question_language") not in ("fr", "en") or not (it.get("question") or "").strip():
        errors.append(f"{sid}: question / question_language missing")
    note = it.get("notes") or ""
    if "HIGH-RISK: verify" in note:
        errors.append(f"{sid}: not verified yet - check it against SOURCE, then edit the note")
    ids = it.get("source_chunk_ids") or []
    missing = [i for i in ids if i not in by_id]
    if missing:
        errors.append(f"{sid}: unknown source_chunk_ids {missing}")
        return
    locs = it.get("expected") or []
    for i in ids:  # every source chunk sits in an expected location ...
        if not any(is_hit(by_id[i], loc) for loc in locs):
            errors.append(f"{sid}: source chunk {i} not covered by expected")
    for loc in locs:  # ... and every expected location has a source chunk
        if not any(is_hit(by_id[i], loc) for i in ids):
            errors.append(f"{sid}: expected location {loc} has no source chunk")

    if BEHAVIOR[typ] == "answer":
        if not ids or not (it.get("ground_truth") or "").strip():
            errors.append(f"{sid}: answer types need source_chunk_ids and ground_truth")
        if typ == "multi_hop" and len(locs) < 2:
            errors.append(f"{sid}: multi_hop needs >= 2 expected locations, has {len(locs)}")
        source = "\n".join(by_id[i]["text"] for i in ids)
        facts = it.get("key_facts") or []
        if not 1 <= len(facts) <= 4:
            errors.append(f"{sid}: needs 1-4 key_facts")
        for f in facts:
            rx = _compile(f["pattern"], sid, errors)
            if rx and not rx.search(source):
                errors.append(f"{sid}: key fact {f['pattern']!r} does not match the sources")
    elif typ == "no_answer":
        if locs or ids:
            errors.append(f"{sid}: no_answer must have no expected / source_chunk_ids")
        pats = it.get("absence_patterns") or []
        if not pats:
            errors.append(f"{sid}: no_answer needs absence_patterns (proof of absence)")
        for p in pats:
            rx = _compile(p, sid, errors)
            hits = [c["chunk_id"] for c in corpus if rx and rx.search(c["text"])]
            if hits:
                errors.append(
                    f"{sid}: absence pattern {p!r} found in {len(hits)} chunks: {hits[:3]}"
                )
    else:  # adversarial
        if it.get("injection") not in ("question", "context"):
            errors.append(f"{sid}: injection must be 'question' or 'context'")
        if it.get("injection") == "context" and not (it.get("injected_passage") or "").strip():
            errors.append(f"{sid}: context injection needs injected_passage")
        if not it.get("must_not_match"):
            errors.append(f"{sid}: adversarial needs must_not_match")
        for p in (it.get("must_not_match") or []) + (it.get("must_match") or []):
            _compile(p, sid, errors)


def is_cross(it: dict, work_langs: dict) -> bool:
    """Cross-lingual = the question language exists in NONE of the expected works."""
    works = {loc["work_id"] for loc in it.get("expected") or []}
    return bool(works) and all(it["question_language"] not in work_langs[w] for w in works)


def check(cfg: dict) -> int:
    items, spec = load_items(cfg), load_spec(cfg)
    corpus = all_chunks(cfg)
    by_id = {c["chunk_id"]: c for c in corpus}
    work_langs: dict[str, set] = {}
    for c in corpus:
        work_langs.setdefault(c["work_id"], set()).add(c["language"])
    errors, warnings = [], []
    dup = [k for k, n in Counter(it["hid"] for it in items).items() if n > 1]
    if dup:
        errors.append(f"duplicate hid: {dup}")
    for it in items:
        check_item(it, by_id, corpus, errors, warnings)
    smoke_types = Counter(spec["smoke_seeds"]["types"].values())
    print(f"{'type':<16}{'have':>5}{'+smoke':>7}{'spec':>6}")
    have = Counter(it["type"] for it in items)
    for t in ("multi_hop", "article_lookup", "no_answer", "adversarial"):
        target = spec["types"][t]["dev"] + spec["types"][t]["test"]
        got = have[t] + smoke_types.get(t, 0)
        print(
            f"{t:<16}{have[t]:>5}{smoke_types.get(t, 0):>7}{target:>6}"
            f"{'' if got == target else '  <-'}"
        )
        if got != target:
            errors.append(f"type {t}: {got} (incl. smoke) != spec {target}")
    langs = Counter(it["question_language"] for it in items)
    cross = [it["hid"] for it in items if is_cross(it, work_langs)]
    verified = sum("HIGH-RISK verified" in (it.get("notes") or "") for it in items)
    print(
        f"languages {dict(langs)}; cross-lingual {len(cross)} {cross}; "
        f"line-by-line verified {verified}"
    )
    for w in warnings:
        print("WARN ", w)
    for e in errors:
        print("ERROR", e)
    print("\nOK" if not errors else f"\n{len(errors)} error(s)")
    return 1 if errors else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        raise SystemExit(check(load_config()))
    ap.print_help()


if __name__ == "__main__":
    main()
