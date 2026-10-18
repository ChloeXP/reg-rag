"""Step 6 - evaluation metrics computed from golden-set fields (no LLM judge).

6.1 retrieval layer: where does each expected location first appear in ONE ranking?
  hit@k     1 if ALL expected locations are in the top k (multi_hop needs every hop), else 0
  recall@k  share of expected locations in the top k (partial credit for multi_hop)
  rr        1 / rank of the first relevant hit (0 if none in the ranking) -> mean = MRR
"""

from __future__ import annotations

from collections import defaultdict
from math import comb

from rag.golden import is_hit  # the ONE hit rule (decision #68)


def location_ranks(hits: list[dict], expected: list[dict]) -> list[int | None]:
    """For each expected location: rank of the first hit that matches it, None if absent."""
    return [next((h["rank"] for h in hits if is_hit(h, exp)), None) for exp in expected]


def retrieval_scores(ranks: list[int | None], ks: list[int]) -> dict[str, float]:
    if not ranks:
        raise ValueError("retrieval metrics need at least one expected location")
    out: dict[str, float] = {}
    for k in ks:
        found = [r is not None and r <= k for r in ranks]
        out[f"hit@{k}"] = float(all(found))
        out[f"recall@{k}"] = sum(found) / len(found)
    first = min((r for r in ranks if r is not None), default=None)
    out["rr"] = 1.0 / first if first else 0.0
    return out


def aggregate(rows: list[dict], metric_keys: list[str]) -> dict[str, float]:
    """Mean of each metric over rows (each row has a 'scores' dict) + n."""
    if not rows:
        return {"n": 0}
    out = {k: sum(r["scores"][k] for r in rows) / len(rows) for k in metric_keys}
    return {"n": len(rows), **out}


def grouped(rows: list[dict], field: str, metric_keys: list[str]) -> dict[str, dict]:
    """aggregate() per value of a row field, e.g. 'type' or 'verification'."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[str(r[field])].append(r)
    return {g: aggregate(rs, metric_keys) for g, rs in sorted(groups.items())}


def mcnemar_exact(gained: int, lost: int) -> float:
    """Two-sided exact McNemar p-value on the questions that FLIPPED between two runs.

    Questions both runs got right (or both wrong) say nothing about which run is better;
    under "no difference" each flip is a coin toss, so p = P(at least this lopsided | n, 0.5).
    """
    n = gained + lost
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(gained, lost) + 1)) / 2**n
    return min(1.0, 2 * tail)
