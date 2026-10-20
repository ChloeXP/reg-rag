"""Fast unit tests for step 8 guardrails (no model, no corpus)."""

import copy
import json

import pytest

from rag.answer_metrics import score_answer
from rag.config import load_config
from rag.generate import load_prompt, parse_citations
from rag.guardrails import refusal_fields, relevance_gate, top_score

CFG = load_config()


def cfg_with(**relevance):
    c = copy.deepcopy(CFG)
    c["retrieval"]["rerank"]["enabled"] = True
    c["guardrails"]["relevance"].update({"enabled": True, "min_top_score": 0.05, **relevance})
    return c


def hits(*scores):
    return [{"rank": i, "score": s, "chunk_id": f"c{i}"} for i, s in enumerate(scores, 1)]


def test_config_threshold_is_on_the_reranker_scale():
    rc = CFG["guardrails"]["relevance"]
    assert rc["score_model"] == CFG["retrieval"]["rerank"]["model"]
    assert 0 < rc["min_top_score"] < 0.213  # dev: lowest answerable top-1 (8.1)


@pytest.mark.parametrize(
    ("scores", "passed"),
    [((0.9, 0.2), True), ((0.05,), True), ((0.0499, 0.01), False), ((), False)],
)
def test_gate_threshold_inclusive(scores, passed):
    v = relevance_gate(hits(*scores), cfg_with())
    assert v["passed"] is passed and v["threshold"] == 0.05


def test_gate_uses_log_precision():
    # 0.04996 is logged as 0.05: live and replayed decisions must agree
    assert top_score(hits(0.04996)) == 0.05
    assert relevance_gate(hits(0.04996), cfg_with())["passed"]


def test_gate_uses_max_not_first_passage():
    poison_first = [{"rank": 0, "score": 0.0, "chunk_id": "external_note::0"}, *hits(0.9)]
    assert relevance_gate(poison_first, cfg_with())["passed"]


def test_gate_disabled_always_passes():
    v = relevance_gate(hits(0.0), cfg_with(enabled=False))
    assert v == {"name": "relevance", "enabled": False, "passed": True}


def test_gate_refuses_other_score_scales():
    c = cfg_with()
    c["retrieval"]["rerank"]["enabled"] = False  # dense cosine scores: other scale
    with pytest.raises(ValueError, match="recalibrate"):
        relevance_gate(hits(0.9), c)
    c = cfg_with(score_model="some/other-reranker")
    with pytest.raises(ValueError, match="recalibrate"):
        relevance_gate(hits(0.9), c)


@pytest.mark.parametrize("version", ["v1", "v3"])
@pytest.mark.parametrize("lang", ["fr", "en"])
def test_gated_refusal_is_detected_like_a_model_refusal(version, lang):
    prompt = load_prompt(CFG, version)
    rec = refusal_fields(prompt, lang)
    assert parse_citations(rec["answer"], hits(0.9), prompt)["is_refusal"]
    refuse = {"id": "X", "expected_behavior": "refuse", "absence_patterns": [r"\bSFDR\b"]}
    assert score_answer(refuse, rec)["scores"] == {"refused": 1.0, "mentions_absent": 0.0}
    answer = {
        "id": "Y",
        "expected_behavior": "answer",
        "key_facts": [{"fact": "f", "pattern": "x"}],
    }
    assert score_answer(answer, rec)["scores"]["false_refusal"] == 1.0


# --- 8.1b: the gate inside rag.ask.answer (what rag.ask and eval_answers call) ------------------
def card(n, score):
    return {"chunk_id": f"dora_fr::structure::{n:04d}", "doc_id": "dora_fr", "work_id": "dora",
            "article": str(n), "page_start": 1, "page_end": 1, "rank": n, "score": score,
            "text": f"[DORA · Article {n} Titre] Corps {n}."}  # fmt: skip


@pytest.fixture
def no_llm(monkeypatch):
    import rag.ask
    import rag.structured_gen

    def boom(*args, **kwargs):
        raise AssertionError("the LLM must not be called")

    monkeypatch.setattr(rag.ask, "generate", boom)
    monkeypatch.setattr(rag.structured_gen, "generate", boom)
    monkeypatch.setattr(rag.ask, "git_commit", lambda: "test")
    return rag.ask


@pytest.mark.parametrize("version", ["v1", "v3"])
def test_off_topic_question_is_refused_without_llm_call(no_llm, version):
    c = cfg_with()
    c["llm"]["prompt_version"] = version
    hits_ = [card(19, 0.0071), card(20, 0.002)]
    rec = no_llm.answer("Quel est le taux de la facilité de dépôt de la BCE ?", hits_, c)
    prompt = load_prompt(c)
    assert rec["answer"] == prompt["refusal"]["fr"] and rec["is_refusal"]
    assert rec["structured"] is None and rec["llm_s"] == 0.0 and rec["output_tokens"] == 0
    assert rec["guardrails"]["llm_called"] is False
    (v,) = rec["guardrails"]["input"]
    assert (v["name"], v["passed"], v["top_score"]) == ("relevance", False, 0.0071)
    assert rec["config"]["prompt_version"] == version and len(rec["hits"]) == 2
    g = {"id": "X", "expected_behavior": "refuse", "absence_patterns": [r"facilit"]}
    assert score_answer(g, rec)["scores"] == {"refused": 1.0, "mentions_absent": 0.0}
    no_llm.show(rec)  # the CLI printout handles a record without LLM output


def test_relevant_question_reaches_the_llm(no_llm, monkeypatch):
    raw = json.dumps({"citations": ["P1"], "answerable": True, "answer": "Article 19.",
                      "confidence": "high"})  # fmt: skip
    fake = {"text": raw, "model": "fake:3b", "done_reason": "stop", "latency_s": 1.0,
            "prompt_tokens": 100, "output_tokens": 10, "load_s": 0.0, "prefill_s": 0.3,
            "decode_s": 0.7}  # fmt: skip
    import rag.structured_gen

    monkeypatch.setattr(rag.structured_gen, "generate", lambda *a, **k: fake)
    c = cfg_with()
    c["llm"]["prompt_version"] = "v3"
    rec = no_llm.answer("Quelles entités déclarent les incidents ?", [card(19, 0.9)], c)
    assert rec["guardrails"]["llm_called"] is True and rec["guardrails"]["input"][0]["passed"]
    assert rec["structured"]["status"] == "ok" and not rec["is_refusal"]


# --- 8.1c: offline replay of the input guardrails ------------------------------------------------
def logged(score, guardrails=None):
    rec = {"language": "en", "answer": "Model answer.", "is_refusal": False, "cited": [2],
           "invalid": [], "uncited": False, "other_brackets": 0, "references": [{"n": [2]}],
           "structured": {"status": "ok"}, "llm_s": 41.0, "prompt_tokens": 2000,
           "output_tokens": 300, "load_s": 0.1, "prefill_s": 14.0, "decode_s": 27.0,
           "hits": [{"rank": 1, "score": score, "chunk_id": "c1", "text_sha256": "x"}]}  # fmt: skip
    if guardrails is not None:
        rec["guardrails"] = guardrails
    return rec


def test_replay_stops_off_topic_and_keeps_the_rest():
    from rag.replay_guardrails import apply_input_guardrails

    prompt = load_prompt(CFG, "v3")
    gated = apply_input_guardrails(logged(0.0073), cfg_with(), prompt)
    assert gated["answer"] == prompt["refusal"]["en"] and gated["is_refusal"]
    assert gated["llm_s"] == 0.0 and gated["structured"] is None and gated["cited"] == []
    assert gated["guardrails"]["llm_called"] is False
    kept = apply_input_guardrails(logged(0.9), cfg_with(), prompt)
    assert kept["answer"] == "Model answer." and kept["llm_s"] == 41.0
    assert kept["guardrails"] == {
        "llm_called": True,
        "input": [relevance_gate(kept["hits"], cfg_with())],
    }


def test_replay_refuses_to_invent_an_answer_never_generated():
    from rag.replay_guardrails import apply_input_guardrails

    prompt = load_prompt(CFG, "v3")
    old = logged(0.0073, guardrails={"llm_called": False, "input": []})
    with pytest.raises(ValueError, match="not replayable"):
        apply_input_guardrails(old, cfg_with(min_top_score=0.001), prompt)
