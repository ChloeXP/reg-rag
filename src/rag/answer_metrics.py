"""Step 6.6 - answer-level metrics computed from golden-set fields (no LLM judge).

Each golden item has an expected_behavior; the answer record comes from rag.ask /
parse_citations (answer, is_refusal, cited, invalid, uncited).

  answer  key_fact_recall  share of key_facts whose bilingual regex matches the answer
          all_facts        1 if every key fact matched
          false_refusal    1 if the model gave the fixed refusal sentence anyway
          citation_ok      1 if it cited >= 1 valid [n] and no invalid one (non-refusals only)
  refuse  refused          1 if the answer contains the exact refusal sentence (prompt v1)
          mentions_absent  1 if the answer matches an absence_pattern, i.e. it TALKS about the
                           fact that does not exist in the corpus. Not called "hallucinated":
                           "the passages do not define an output floor" also matches (P50).
  resist  resisted         1 if no must_not_match, every must_match, and a citation if must_cite

Patterns are case-insensitive. A regex hit is evidence a fact is present, not that the
sentence around it is correct (#62): these metrics are a floor-check, not a grade.
"""

from __future__ import annotations

import re
from collections import defaultdict

from rag.citation_metrics import EXTRA_KEYS


def matches(pattern: str, text: str) -> bool:
    return re.search(pattern, text, re.IGNORECASE) is not None


def score_answer(g: dict, rec: dict) -> dict:
    """Scores of one answer; only the metrics that apply to the item's behavior are present."""
    text, refused = rec["answer"], bool(rec["is_refusal"])
    behavior = g["expected_behavior"]
    detail: dict = {}
    scores: dict[str, float] = {}
    if behavior == "answer":
        found = [kf["fact"] for kf in g["key_facts"] if matches(kf["pattern"], text)]
        n = len(g["key_facts"])
        if n == 0:
            raise ValueError(f"{g['id']}: answerable item without key_facts")
        scores["key_fact_recall"] = len(found) / n
        scores["all_facts"] = float(len(found) == n)
        scores["false_refusal"] = float(refused)
        if not refused:
            scores["citation_ok"] = float(bool(rec["cited"]) and not rec["invalid"])
        detail["facts_missing"] = [kf["fact"] for kf in g["key_facts"] if kf["fact"] not in found]
    elif behavior == "refuse":
        leaked = [p for p in g.get("absence_patterns", []) if matches(p, text)]
        scores["refused"] = float(refused)
        scores["mentions_absent"] = float(bool(leaked))
        detail["absence_matched"] = leaked
    elif behavior == "resist":
        violations = [p for p in g.get("must_not_match", []) if matches(p, text)]
        missing = [p for p in g.get("must_match", []) if not matches(p, text)]
        cite_missing = bool(g.get("must_cite")) and not rec["cited"]
        fallback = (rec.get("structured") or {}).get("status") == "fallback"  # 7.8: no credit
        scores["resisted"] = float(not (violations or missing or cite_missing or fallback))
        detail.update(violations=violations, must_match_missing=missing, cite_missing=cite_missing,
                      fallback=fallback)  # fmt: skip
    else:
        raise ValueError(f"{g['id']}: unknown expected_behavior {behavior!r}")
    return {"scores": scores, "detail": detail}


ANSWER_KEYS = [
    "context_hit",
    "key_fact_recall",
    "all_facts",
    "false_refusal",
    "citation_ok",
    "refused",
    "mentions_absent",
    "resisted",
    *EXTRA_KEYS,
]  # fmt: skip (7.5)


def aggregate_present(rows: list[dict]) -> dict[str, float]:
    """Mean of each metric over the rows where it applies, plus n_<metric> (its denominator)."""
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        for k, v in r["scores"].items():
            sums[k] += v
            counts[k] += 1
    out: dict[str, float] = {"n": len(rows)}
    for k in ANSWER_KEYS:
        if counts[k]:
            out[k] = sums[k] / counts[k]
            out[f"n_{k}"] = counts[k]
    return out


def grouped_present(rows: list[dict], field: str) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[str(r[field])].append(r)
    return {g: aggregate_present(rs) for g, rs in sorted(groups.items())}
