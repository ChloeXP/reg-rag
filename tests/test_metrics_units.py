"""Step 6.1 - retrieval metrics on hand-made rankings (no Qdrant, no model)."""

import pytest

from rag.metrics import aggregate, grouped, location_ranks, mcnemar_exact, retrieval_scores

KS = [1, 3, 5]


def hit(rank, work_id, article=None, section=None, text=""):
    return {"rank": rank, "work_id": work_id, "article": article, "section": section, "text": text}


def test_location_ranks_first_match_per_location():
    hits = [hit(1, "dora", "5"), hit(2, "crr", "392"), hit(3, "crr", "395"), hit(4, "crr", "392")]
    exp = [{"work_id": "crr", "articles": ["392"]}, {"work_id": "crr", "articles": ["999"]}]
    assert location_ranks(hits, exp) == [2, None]


def test_multi_hop_hit_needs_every_location_recall_gives_partial_credit():
    s = retrieval_scores([2, 4], KS)
    assert (s["hit@1"], s["hit@3"], s["hit@5"]) == (0.0, 0.0, 1.0)
    assert s["recall@3"] == 0.5
    assert s["rr"] == 0.5


def test_missing_location_never_counts():
    s = retrieval_scores([None], KS)
    assert s["hit@5"] == 0.0 and s["recall@5"] == 0.0 and s["rr"] == 0.0


def test_no_expected_location_is_an_error():
    with pytest.raises(ValueError):
        retrieval_scores([], KS)


def test_aggregate_and_grouped():
    rows = [
        {"type": "factual", "scores": {"hit@1": 1.0}},
        {"type": "factual", "scores": {"hit@1": 0.0}},
        {"type": "multi_hop", "scores": {"hit@1": 0.0}},
    ]
    assert aggregate(rows, ["hit@1"]) == {"n": 3, "hit@1": pytest.approx(1 / 3)}
    g = grouped(rows, "type", ["hit@1"])
    assert g["factual"] == {"n": 2, "hit@1": 0.5}
    assert g["multi_hop"]["n"] == 1


def test_mcnemar_exact_known_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(3, 0) == pytest.approx(0.25)  # 2 * (1/8): 3 gains alone prove nothing
    assert mcnemar_exact(6, 0) == pytest.approx(2 / 64)  # first count below 0.05
    assert mcnemar_exact(2, 2) == 1.0
    assert mcnemar_exact(1, 4) == mcnemar_exact(4, 1)
