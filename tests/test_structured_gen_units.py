"""Fast unit tests for step 7.3 structured generation (scripted fake LLM: no Ollama, no Qdrant)."""

import copy
import json

import pytest

import rag.structured_gen as sg
from rag.config import load_config
from rag.generate import build_prompt, load_prompt

CFG = load_config()
V2 = load_prompt(CFG, "v2")


def hit(n):
    cid = f"dora_fr::structure::{n:04d}"
    text = f"[DORA · Article {n} Titre] Corps {n}."
    return {"chunk_id": cid, "doc_id": "dora_fr", "work_id": "dora", "article": str(n),
            "page_start": 1, "page_end": 1, "text": text, "rank": 1, "score": 0.5}  # fmt: skip


HITS = [hit(n) for n in (19, 20, 21, 22, 23)]
BUILT = build_prompt("Quelles entités déclarent les incidents ?", HITS, V2, CFG)  # lang fr
OK = {"citations": ["P1", "P3"], "answerable": True,
      "answer": "Les entités financières déclarent les incidents majeurs (article 19).",
      "confidence": "high"}  # fmt: skip
NO = {"citations": [], "answerable": False, "answer": "Je ne trouve pas.", "confidence": "low"}


def js(**changes):
    return json.dumps({**OK, **changes}, ensure_ascii=False)


@pytest.fixture
def llm(monkeypatch):
    """Scripted Ollama: each call pops the next (text, done_reason); calls are recorded."""
    script, calls = [], []

    def fake(prompt, cfg, system=None, fmt=None, num_predict=None):
        calls.append({"prompt": prompt, "system": system, "fmt": fmt, "num_predict": num_predict})
        text, done = script.pop(0)
        return {"text": text, "model": "fake:3b", "done_reason": done, "latency_s": 1.5,
                "prompt_tokens": 1000, "output_tokens": 60, "load_s": 0.1,
                "prefill_s": 3.0, "decode_s": 1.2}  # fmt: skip

    monkeypatch.setattr(sg, "generate", fake)
    return script, calls


def run(cfg=CFG):
    return sg.run(BUILT, HITS, V2, cfg)


def with_policy(**policy):
    cfg = copy.deepcopy(CFG)
    cfg["llm"]["structured"]["policy"] = policy
    return cfg


STRICT = with_policy(length="reject", invalid_citations="reject")  # the 7.3 behaviour


# --- first attempt valid -----------------------------------------------------------------------
def test_first_attempt_ok_maps_to_v1_fields(llm):
    script, calls = llm
    script.append((js(), "stop"))
    r = run()
    st = r["structured"]
    assert (st["status"], st["n_attempts"], st["first_status"]) == ("ok", 1, "ok")
    assert r["answer"] == OK["answer"] and r["cited"] == [1, 3] and r["invalid"] == []
    assert not r["is_refusal"] and not r["uncited"] and r["other_brackets"] == 0
    assert [ref["n"] for ref in r["references"]] == [[1], [3]]
    api = st["api_answer"]
    assert [c["chunk_id"] for c in api["citations"]] == ["dora_fr::structure::0019",
                                                         "dora_fr::structure::0021"]  # fmt: skip
    assert len(calls) == 1 and calls[0]["fmt"] == sg.model_schema()
    assert calls[0]["num_predict"] == CFG["llm"]["structured"]["num_predict"]
    assert calls[0]["system"] == V2["system"]


def test_answerable_false_shows_the_fixed_refusal_sentence(llm):
    script, _ = llm
    script.append((json.dumps(NO), "stop"))
    r = run()
    assert r["is_refusal"] and r["answer"] == V2["refusal"]["fr"] and r["cited"] == []
    assert not r["uncited"] and r["structured"]["answerable"] is False
    assert r["structured"]["api_answer"]["answer"] == V2["refusal"]["fr"]
    assert r["structured"]["attempts"][0]["raw"] == json.dumps(NO)  # model's own text is kept


# --- retry ---------------------------------------------------------------------------------------
def test_schema_error_is_retried_with_the_errors(llm):
    script, calls = llm
    script += [(js(citations=[]), "stop"), (js(), "stop")]  # P68 then corrected
    r = run()
    st = r["structured"]
    assert (st["status"], st["n_attempts"], st["first_status"]) == ("retried_ok", 2, "schema_error")
    retry = calls[1]["prompt"]
    assert retry.startswith(BUILT["user"]) and "Your previous output was rejected." in retry
    assert "citations is empty" in retry and '"citations": []' in retry
    assert r["cited"] == [1, 3]


def test_hallucinated_citation_is_retried_and_counted(llm):  # decision 7.3-B, P65
    script, calls = llm
    script += [(js(citations=["P2", "P6"]), "stop"), (js(citations=["P2"]), "stop")]
    r = run(STRICT)
    st = r["structured"]
    assert (st["status"], st["first_status"], st["first_invalid"]) == (
        "retried_ok", "grounding_error", ["P6"])  # fmt: skip
    assert "['P6'] do not exist: the passages are P1 to P5" in calls[1]["prompt"]
    assert r["cited"] == [2] and r["invalid"] == []


def test_refusal_sentence_with_answerable_true_is_a_contradiction(llm):  # P76, probe case 3
    script, _ = llm
    script += [(js(answer=V2["refusal"]["fr"]), "stop"), (json.dumps(NO), "stop")]
    r = run()
    assert r["structured"]["first_status"] == "schema_error" and r["is_refusal"]


# --- fallback ------------------------------------------------------------------------------------
def test_truncated_twice_falls_back_without_claiming_a_refusal(llm):  # P78
    script, _ = llm
    script += [(js()[:50], "length"), (js()[:50], "length")]
    r = run()
    st = r["structured"]
    assert st["status"] == "fallback" and [a["status"] for a in st["attempts"]] == ["truncated"] * 2
    assert r["answer"] == sg.FALLBACK["fr"] and not r["is_refusal"]  # no credit on no-answer items
    assert r["cited"] == [] and r["uncited"] and st["api_answer"] is None
    assert st["answerable"] is None and st["confidence"] is None


def test_invalid_json_without_length_is_a_schema_error(llm):
    script, _ = llm
    script += [("not json", "stop"), ("still not json", "stop")]
    st = run()["structured"]
    assert [a["status"] for a in st["attempts"]] == ["schema_error"] * 2


def test_max_retries_zero_means_one_call(llm):
    script, calls = llm
    script.append(("not json", "stop"))
    cfg = copy.deepcopy(CFG)
    cfg["llm"]["structured"]["max_retries"] = 0
    assert run(cfg)["structured"]["status"] == "fallback" and len(calls) == 1


def test_previous_output_is_capped_in_the_retry():
    long = "x" * 5000
    msg = sg.retry_user("U", V2, long, ["e1"], max_chars=100)
    assert ("x" * 100 + " [...]") in msg and ("x" * 101) not in msg and "- e1" in msg


# --- integration with rag.ask.answer (what eval_answers calls) ----------------------------------
def test_ask_answer_v2_record_keeps_v1_fields_and_is_scoreable(llm, monkeypatch):
    import rag.ask
    from rag.answer_metrics import score_answer

    script, _ = llm
    script += [(js(citations=["P9"]), "stop"), (js(), "stop")]
    monkeypatch.setattr(rag.ask, "git_commit", lambda: "test")  # no git call from unit tests
    cfg = copy.deepcopy(CFG)
    cfg["llm"]["prompt_version"] = "v2"
    rec = rag.ask.answer("Quelles entités déclarent les incidents ?", HITS, cfg)
    for key in ["answer", "cited", "invalid", "is_refusal", "uncited", "references", "llm_s",
                "prompt_tokens", "output_tokens", "prefill_s", "config", "hits"]:  # fmt: skip
        assert key in rec, key
    assert rec["config"]["prompt_version"] == "v2" and rec["config"]["model"] == "fake:3b"
    assert (rec["llm_s"], rec["prompt_tokens"], rec["prefill_s"]) == (3.0, 2000, 6.0)  # 2 calls
    assert rec["structured"]["status"] == "retried_ok"
    g = {"id": "GX", "expected_behavior": "answer",
         "key_facts": [{"fact": "entités", "pattern": r"entit\w+ financi"}]}  # fmt: skip
    assert score_answer(g, rec)["scores"] == {"key_fact_recall": 1.0, "all_facts": 1.0,
                                              "false_refusal": 0.0, "citation_ok": 1.0}  # fmt: skip


# --- step 7.8: policy (offline-replayable) -------------------------------------------------------
def test_default_config_policy_is_the_measured_one():
    assert sg.policy_of(CFG) == {"length": "flag", "invalid_citations": "drop"}
    assert sg.policy_of({"llm": {"structured": {}}}) == sg.DEFAULT_POLICY  # old run configs
    with pytest.raises(ValueError):
        sg.policy_of(with_policy(length="maybe"))


def test_drop_policy_keeps_the_answer_and_records_the_dropped_id(llm):  # P82 (G018)
    script, calls = llm
    script.append((js(citations=["P1", "P13"]), "stop"))
    r = run()
    st = r["structured"]
    assert (st["status"], st["n_attempts"], st["dropped"]) == ("ok", 1, ["P13"])
    assert st["first_invalid"] == ["P13"] and r["cited"] == [1] and len(calls) == 1
    assert st["policy"] == {"length": "flag", "invalid_citations": "drop"}


def test_drop_policy_still_rejects_when_no_valid_id_remains(llm):
    script, _ = llm
    script += [(js(citations=["P9"]), "stop"), (js(citations=["P7"]), "stop")]
    st = run()["structured"]
    assert st["status"] == "fallback" and st["first_status"] == "grounding_error"


def test_length_flag_keeps_a_complete_long_answer(llm):  # P84 (G002)
    script, _ = llm
    script.append((js(answer="x" * 2500), "stop"))  # v2 limit 2000
    st = run()["structured"]
    assert (st["status"], st["over_length"]) == ("ok", True)
    script.append((js(answer="x" * 2500), "stop"))
    script.append((js(answer="x" * 2500), "stop"))
    assert run(STRICT)["structured"]["status"] == "fallback"


def test_truncation_is_still_a_failure_under_the_flag_policy(llm):  # P85
    script, _ = llm
    script += [(js(answer="x" * 2500)[:-30], "length"), (js()[:40], "length")]
    st = run()["structured"]
    assert st["status"] == "fallback" and st["first_status"] == "truncated"
