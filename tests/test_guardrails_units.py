"""Fast unit tests for step 8 guardrails (no model, no corpus)."""

import copy
import json

import pytest

from rag.answer_metrics import score_answer
from rag.config import load_config
from rag.generate import load_prompt, parse_citations
from rag.guardrails import mention_check, refusal_fields, relevance_gate, top_score

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
        "output": [],
    }


def test_replay_refuses_to_invent_an_answer_never_generated():
    from rag.replay_guardrails import apply_input_guardrails

    prompt = load_prompt(CFG, "v3")
    old = logged(0.0073, guardrails={"llm_called": False, "input": []})
    with pytest.raises(ValueError, match="not replayable"):
        apply_input_guardrails(old, cfg_with(min_top_score=0.001), prompt)


# --- 8.2: output-side mention check ---------------------------------------------------------
def passage(text, work_id="crr", article=None):
    return {"text": text, "work_id": work_id, "article": article}


@pytest.mark.parametrize(
    ("text", "numbers"),
    [("articles 92 et 465 du CRR", {"92", "465"}), ("Articles 399 to 403, to", {"399", "403"}),
     ("Article 26, paragraphes 9 et 10", {"26"}), ("Art. 92a", {"92a"}),
     ("Articles 26 and 27", {"26", "27"}), ("l’article 47;", {"47"}), ("article premier", set())],
)  # fmt: skip
def test_article_numbers(text, numbers):
    from rag.guardrails import article_numbers

    assert article_numbers(text, CFG["guardrails"]["mentions"]["article"]) == numbers


def test_mentions_supported_by_text_header_metadata_or_work():
    cited = [
        passage("[CRR · Article 92 Own funds requirements]\n1. ... Article 465 ...", article="92"),
        passage("[AI ACT · Article 6 Classification]\n... Regulation (EU) 2016/679 ...", "ai_act"),
    ]
    answer = "Under Article 92 of the CRR (see also article 465), the AI Act and the GDPR apply."
    v = mention_check(answer, cited, CFG)
    assert v["passed"] and v["regulations"] == ["ai_act", "crr", "gdpr"]
    assert v["articles"] == ["92", "465"]


def test_regulation_absent_from_the_cited_passages_is_unsupported():  # G037 (3b v3)
    cited = [passage("[CRR · Article 449a Disclosure of ESG risks]\n...", article="449a")]
    v = mention_check("Le règlement SFDR impose de publier ... (article 449a).", cited, CFG)
    assert not v["passed"] and v["unsupported"] == ["reg:sfdr"]


def test_section_is_not_an_article():  # G015 (3b): source says "Section 12.1"
    cited = [passage("... as referred to in Section 12.1. ...", "eba_gl_2019_02")]
    v = mention_check("comme mentionné dans l'Article 12.1", cited, CFG)
    assert v["unsupported"] == ["art:12"]


def test_uncited_answer_naming_things_is_unsupported_and_plain_text_passes():
    assert mention_check("Article 19 of DORA requires ...", [], CFG)["unsupported"] == [
        "reg:dora", "art:19"]  # fmt: skip
    v = mention_check("Les entités notifient les incidents majeurs.", [], CFG)
    assert v["passed"] and v["regulations"] == [] and v["articles"] == []


def test_mention_check_disabled():
    c = copy.deepcopy(CFG)
    c["guardrails"]["mentions"]["enabled"] = False
    assert mention_check("SFDR", [], c) == {"name": "mentions", "enabled": False, "passed": True}


# --- 8.2b: action on the mention check (decision 8.2b-C) -----------------------------------------
V3 = load_prompt(CFG, "v3")
CRR = card(92, 0.9) | {
    "work_id": "crr",
    "article": "92",
    "text": "[CRR · Article 92 Own funds requirements]\n1. ... 4,5 % ...",
}


def generated(answer, cited=(1,), structured=True):
    st = (
        {
            "status": "ok",
            "confidence": "high",
            "attempts": [],
            "api_answer": {
                "answer": answer,
                "citations": [{"passage": "P1"}],
                "confidence": "high",
                "answerable": True,
            },
        }
        if structured
        else None
    )
    return {"answer": answer, "cited": list(cited), "invalid": [], "is_refusal": False,
            "uncited": not cited, "other_brackets": 0, "references": [{"n": list(cited)}],
            "structured": st}  # fmt: skip


def with_refuse_on(value):
    c = copy.deepcopy(CFG)
    c["guardrails"]["mentions"]["refuse_on"] = value
    return c


def test_out_of_corpus_regulation_is_replaced_by_the_refusal():  # G034 (3b)
    from rag.guardrails import output_guardrails, undo_output_guardrails

    before = generated("The maximum fine under the GDPR is EUR 15 000 000 (Article 92).")
    after, (v,) = output_guardrails(before, [CRR], V3, "en", CFG)
    assert v["action"] == "refuse" and v["unsupported"] == ["reg:gdpr"]
    assert after["answer"] == V3["refusal"]["en"] and after["is_refusal"] and after["cited"] == []
    assert after["structured"]["api_answer"]["answerable"] is False
    assert after["structured"]["status"] == "ok"  # the JSON attempts are not rewritten
    rec = {**after, "guardrails": {"llm_called": True, "input": [], "output": [v]}}
    restored = undo_output_guardrails(rec)
    assert {k: restored[k] for k in before} == before  # replayable: nothing lost


def test_unsupported_article_or_corpus_name_is_only_flagged():  # G015, G023, G006
    from rag.guardrails import output_guardrails

    before = generated("Selon les orientations de l'ABE (Article 12.1), ...")
    after, (v,) = output_guardrails(before, [CRR], V3, "fr", CFG)
    assert after == before and v["action"] == "flag"
    assert v["unsupported"] == ["reg:eba", "art:12"]


@pytest.mark.parametrize(
    ("refuse_on", "sfdr", "article"),
    [("never", "flag", "flag"), ("out_of_corpus", "refuse", "flag"), ("any", "refuse", "refuse")],
)
def test_refuse_on_levels(refuse_on, sfdr, article):
    c = with_refuse_on(refuse_on)
    assert mention_check("Le règlement SFDR ...", [CRR], c)["action"] == sfdr
    assert mention_check("Article 12 ...", [CRR], c)["action"] == article
    assert mention_check("Article 92 ...", [CRR], c)["action"] == "none"
    with pytest.raises(ValueError, match="refuse_on"):
        mention_check("x", [CRR], with_refuse_on("sometimes"))


def test_refusals_and_fallbacks_are_not_checked():
    from rag.guardrails import output_guardrails

    refusal = {**generated("x"), **refusal_fields(V3, "en")}
    assert output_guardrails(refusal, [CRR], V3, "en", CFG) == (refusal, [])
    fb = generated("SFDR")
    fb["structured"]["status"] = "fallback"
    assert output_guardrails(fb, [CRR], V3, "en", CFG) == (fb, [])


def test_ask_answer_applies_the_output_check(no_llm, monkeypatch):
    import rag.structured_gen

    raw = json.dumps({"citations": ["P1"], "answerable": True, "confidence": "high",
                      "answer": "Under the GDPR the maximum fine is EUR 20 000 000."})  # fmt: skip
    fake = {"text": raw, "model": "fake:3b", "done_reason": "stop", "latency_s": 1.0,
            "prompt_tokens": 100, "output_tokens": 10, "load_s": 0.0, "prefill_s": 0.3,
            "decode_s": 0.7}  # fmt: skip
    monkeypatch.setattr(rag.structured_gen, "generate", lambda *a, **k: fake)
    c = cfg_with()
    c["llm"]["prompt_version"] = "v3"
    rec = no_llm.answer("What is the maximum fine for a breach of the GDPR?", [CRR], c)
    (v,) = rec["guardrails"]["output"]
    assert rec["guardrails"]["llm_called"] and v["action"] == "refuse" and rec["is_refusal"]
    assert v["replaced"]["answer"].startswith("Under the GDPR")
    assert rec["llm_s"] == 1.0  # the LLM did run: its cost is kept
    g = {"id": "G034", "expected_behavior": "refuse", "absence_patterns": [r"20\s?000\s?000"]}
    assert score_answer(g, rec)["scores"] == {"refused": 1.0, "mentions_absent": 0.0}
    no_llm.show(rec)


def test_replay_is_idempotent_and_guardrail_scores():
    from rag.citation_metrics import guardrail_scores
    from rag.replay_guardrails import apply_guardrails

    rec = {**logged(0.9), **generated("The SFDR requires ...", cited=(1,)),
           "language": "en"}  # fmt: skip
    once = apply_guardrails(rec, cfg_with(), V3, [CRR])
    twice = apply_guardrails(once, cfg_with(), V3, [CRR])
    assert once == twice and once["is_refusal"]
    assert guardrail_scores(once) == {"gated": 0.0, "mention_flag": 1.0, "mention_refused": 1.0}
    gated = apply_guardrails(logged(0.001), cfg_with(), V3, [CRR])
    assert guardrail_scores(gated) == {"gated": 1.0}
    assert guardrail_scores({"answer": "old run"}) == {}
