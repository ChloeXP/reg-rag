"""Step 6.4 - rerank() ordering logic with a fake cross-encoder (no model download)."""

import copy

import pytest

from rag import rerank as rr
from rag.config import load_config

CFG = load_config()


class FakeCrossEncoder:
    """Scores a pair by how many times the word 'answer' appears in the card."""

    def predict(self, pairs, batch_size, show_progress_bar):
        return [float(text.count("answer")) for _, text in pairs]


@pytest.fixture
def fake_model(monkeypatch):
    monkeypatch.setattr(rr, "_load_model", lambda *a: FakeCrossEncoder())


def hits(*texts):
    return [{"rank": i, "score": 1.0 - i / 10, "text": t} for i, t in enumerate(texts, start=1)]


def test_rerank_reorders_and_keeps_first_stage_rank(fake_model):
    out = rr.rerank("q", hits("noise", "answer", "answer answer"), CFG, top_k=2)
    assert [h["text"] for h in out] == ["answer answer", "answer"]
    assert [h["rank"] for h in out] == [1, 2]
    assert [h["first_stage_rank"] for h in out] == [3, 2]


def test_ties_keep_first_stage_order(fake_model):
    out = rr.rerank("q", hits("a", "b", "c"), CFG, top_k=3)
    assert [h["first_stage_rank"] for h in out] == [1, 2, 3]


def test_empty_hits(fake_model):
    assert rr.rerank("q", [], CFG, top_k=5) == []


def test_rerank_is_off_in_baseline_and_covers_eval_depth():
    r = CFG["retrieval"]["rerank"]
    assert r["enabled"] is False  # baseline-v1 has no reranker
    assert CFG["eval"]["retrieval_depth"] <= r["candidates"]
    assert r["max_length"] >= CFG["chunking"]["chunk_size"] + 64  # chunk + question fit


def test_override_turns_rerank_on():
    from rag.config import apply_overrides

    cfg = apply_overrides(copy.deepcopy(CFG), ["retrieval.rerank.enabled=true"])
    assert cfg["retrieval"]["rerank"]["enabled"] is True
