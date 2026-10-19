"""Fast unit tests for step 7.8 offline policy replay (no model, no corpus)."""

import json

import pytest

from rag.config import load_config
from rag.generate import load_prompt
from rag.replay_policy import replay_record

CFG = load_config()
V2 = load_prompt(CFG, "v2")
DROP = {"length": "flag", "invalid_citations": "drop"}
STRICT = {"length": "reject", "invalid_citations": "reject"}


def hit(n):
    return {"chunk_id": f"dora_fr::structure::{n:04d}", "doc_id": "dora_fr", "work_id": "dora",
            "article": str(n), "page_start": 1, "page_end": 1,
            "text": f"[DORA · Article {n} Titre] Corps {n}."}  # fmt: skip


HITS = [hit(n) for n in (19, 20, 21, 22, 23)]
RAW = json.dumps({"citations": ["P1", "P13"], "answerable": True,
                  "answer": "Réponse correcte (article 19).", "confidence": "high"})  # fmt: skip


def attempt(n, status, raw=RAW):
    return {"attempt": n, "status": status, "errors": ["x"], "invalid": ["P13"], "raw": raw,
            "model": "fake", "done_reason": "stop", "latency_s": 2.0, "prompt_tokens": 1000,
            "output_tokens": 50, "load_s": 0.0, "prefill_s": 3.0, "decode_s": 1.0}  # fmt: skip


def g018_like():  # P82: correct answer + one non-existent id, retried identically, fallback
    return {"language": "fr", "answer": "fallback text", "llm_s": 4.0,
            "structured": {"status": "fallback",
                           "attempts": [attempt(1, "grounding_error"),
                                        attempt(2, "grounding_error")]}}  # fmt: skip


def test_drop_policy_rescues_the_first_attempt_and_recounts_time():
    rec = replay_record(g018_like(), HITS, V2, DROP)
    st = rec["structured"]
    assert (st["status"], st["n_attempts"], st["dropped"], st["first_invalid"]) == (
        "ok", 1, ["P13"], ["P13"])  # fmt: skip
    assert rec["answer"] == "Réponse correcte (article 19)." and rec["cited"] == [1]
    assert (rec["llm_s"], rec["prompt_tokens"]) == (2.0, 1000)  # only the attempt used
    assert st["policy"] == DROP


def test_same_policy_reproduces_the_original_outcome():
    rec = replay_record(g018_like(), HITS, V2, STRICT)
    assert rec["structured"]["status"] == "fallback" and rec["structured"]["n_attempts"] == 2


def test_a_stricter_policy_is_refused():
    ok_run = {"language": "fr", "structured": {"status": "ok", "attempts": [attempt(1, "ok")]}}
    with pytest.raises(ValueError, match="not replayable"):
        replay_record(ok_run, HITS, V2, STRICT)  # would need a retry the run never made
    assert replay_record(ok_run, HITS, V2, DROP)["structured"]["status"] == "ok"
