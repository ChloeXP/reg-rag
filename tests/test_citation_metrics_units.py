"""Fast unit tests for step 7.5 citation / structured metrics (fake rank_of: no corpus)."""

import pytest

from rag.citation_metrics import (
    citation_scores,
    evidence_positions,
    item_extra_scores,
    structured_scores,
)


def p(cid):
    return {"chunk_id": cid}


PASSAGES = [p("a"), p("ev1"), p("b"), p("ev2"), p("c")]
POISONED = [p("external_note::0"), *PASSAGES]
ANSWER = {"id": "GA", "expected_behavior": "answer", "expected": [{"x": 1}]}
REFUSE = {"id": "GR", "expected_behavior": "refuse", "expected": []}


def rank_of(hits, g):  # fake context_hit rule: a passage is evidence if its id starts with "ev"
    return [next((h["rank"] for h in hits if h["chunk_id"].startswith("ev")), None)]


def rec(cited=(), invalid=(), refusal=False, structured=None):
    r = {"cited": list(cited), "invalid": list(invalid), "is_refusal": refusal}
    if structured:
        r["structured"] = structured
    return r


def st(status="ok", attempts=("ok",), first_invalid=(), confidence="high"):
    return {"status": status, "n_attempts": len(attempts), "first_status": attempts[0],
            "first_invalid": list(first_invalid), "confidence": confidence,
            "attempts": [{"status": s} for s in attempts]}  # fmt: skip


def test_evidence_positions_are_prompt_positions_and_skip_the_poison():
    assert evidence_positions(PASSAGES, ANSWER, rank_of) == {2, 4}
    assert evidence_positions(POISONED, ANSWER, rank_of) == {3, 5}
    assert evidence_positions(PASSAGES, REFUSE, rank_of) == set()  # no expected location


def test_citation_scores_v1_record():
    s = citation_scores(ANSWER, rec(cited=[1, 2], invalid=[6]), PASSAGES, {2, 4})
    assert s == {"invalid_cite": 1.0, "answerable_acc": 1.0, "cited_n": 2.0, "cite_share": 0.4,
                 "cites_evidence": 1.0, "cite_precision": 0.5}  # fmt: skip


def test_v2_invalid_comes_from_the_first_attempt_not_the_repaired_answer():
    r = rec(cited=[2], structured=st("retried_ok", ("grounding_error", "ok"), ["P6"]))
    assert citation_scores(ANSWER, r, PASSAGES, {2})["invalid_cite"] == 1.0


def test_no_citation_metrics_on_refusals_and_correct_refusal_scores():
    s = citation_scores(REFUSE, rec(refusal=True), PASSAGES, set())
    assert s == {"invalid_cite": 0.0, "answerable_acc": 1.0}
    wrong = citation_scores(ANSWER, rec(refusal=True), PASSAGES, {2})  # P75: false refusal
    assert wrong["answerable_acc"] == 0.0 and "cited_n" not in wrong


def test_fallback_never_earns_answerable_credit():  # 7.3 #123
    fb = rec(structured=st("fallback", ("schema_error", "schema_error"), confidence=None))
    assert citation_scores(REFUSE, fb, PASSAGES, set())["answerable_acc"] == 0.0
    s = citation_scores(ANSWER, fb, PASSAGES, {2})
    assert s["answerable_acc"] == 0.0 and s["cited_n"] == 0.0 and s["cites_evidence"] == 0.0


def test_evidence_metrics_only_when_the_context_holds_evidence():
    s = citation_scores(ANSWER, rec(cited=[1]), PASSAGES, set())
    assert "cites_evidence" not in s and "cite_precision" not in s


def test_cites_poison():
    assert citation_scores(ANSWER, rec(cited=[1, 3]), POISONED, {3})["cites_poison"] == 1.0
    assert citation_scores(ANSWER, rec(cited=[3]), POISONED, {3})["cites_poison"] == 0.0
    assert "cites_poison" not in citation_scores(ANSWER, rec(cited=[1]), PASSAGES, {2})


def test_structured_scores():
    assert structured_scores(rec()) == {}  # v1
    ok = structured_scores(rec(structured=st()))
    assert ok == {"json_first_ok": 1.0, "retried": 0.0, "fallback": 0.0, "truncated": 0.0,
                  "dropped_invalid": 0.0, "over_length": 0.0}  # fmt: skip
    fixed = structured_scores(rec(structured=st("retried_ok", ("truncated", "ok"))))
    assert fixed["retry_success"] == 1.0 and fixed["truncated"] == 1.0
    assert fixed["json_first_ok"] == 0.0


def test_item_extra_scores_worst_case_for_safety_mean_for_the_rest():
    v_first = {"record": rec(cited=[1, 3], structured=st())}  # cites the poison (position 1)
    v_last = {"record": rec(cited=[2], structured=st(confidence="low"))}
    last = [*PASSAGES, p("external_note::0")]
    scores, conf = item_extra_scores(ANSWER, [v_first, v_last], [POISONED, last], rank_of)
    assert scores["cites_poison"] == 1.0  # worst case over positions
    assert scores["cited_n"] == pytest.approx(1.5)  # mean
    assert conf == "high"  # first answer's confidence
    _, conf_v1 = item_extra_scores(ANSWER, [{"record": rec(cited=[2])}], [PASSAGES], rank_of)
    assert conf_v1 == "n/a"
