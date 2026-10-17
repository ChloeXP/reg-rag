"""Fast unit tests for step 3 (no Qdrant, no model, no corpus)."""

import pytest

from rag import bm25
from rag.index import point_id


def test_tokenize_keeps_numbers_strips_accents_and_stopwords():
    assert bm25.tokenize("What does CRR Article 395 say?") == ["crr", "article", "395"]
    assert bm25.tokenize("Déclaration des incidents liés aux TIC") == [
        "declaration", "incidents", "lies", "tic",
    ]  # fmt: skip


def test_bm25_doc_vector_matches_formula():
    k1, b = 1.2, 0.75
    vecs, avgdl = bm25.doc_vectors(["risk risk capital", "capital"], k1, b)
    assert avgdl == pytest.approx(2.0)  # (3 + 1) / 2 tokens
    idx, val = vecs[0]
    weights = dict(zip(idx, val, strict=True))
    norm = k1 * (1 - b + b * 3 / avgdl)
    assert weights[bm25.token_id("risk")] == pytest.approx(2 * (k1 + 1) / (2 + norm))
    assert weights[bm25.token_id("capital")] == pytest.approx(1 * (k1 + 1) / (1 + norm))


def test_query_vector_unique_terms_weight_one():
    idx, val = bm25.query_vector("Article 395 article")
    assert len(idx) == 2 and val == [1.0, 1.0]
    assert bm25.query_vector("the of ?") == ([], [])


def test_ids_are_deterministic():
    assert bm25.token_id("395") == bm25.token_id("395")
    assert point_id("crr_en::structure::0278") == point_id("crr_en::structure::0278")
    assert point_id("crr_en::structure::0278") != point_id("crr_en::fixed::0278")


def test_rrf_rewards_agreement_and_is_deterministic():
    from rag.retrieve import rrf_fuse

    def hits(*ids):
        return [{"chunk_id": c, "rank": r} for r, c in enumerate(ids, start=1)]

    fused = rrf_fuse({"dense": hits("a", "b", "c"), "bm25": hits("c", "d")}, k=60, top_k=3)
    assert [h["chunk_id"] for h in fused] == ["c", "a", "b"]  # c is in both lists
    assert fused[0]["ranks"] == {"dense": 3, "bm25": 1}
    assert fused[0]["score"] == pytest.approx(1 / 63 + 1 / 61)
    # b (dense@2) and d (bm25@2) tie on score and best rank -> chunk_id decides; same here:
    tie = rrf_fuse({"dense": hits("y"), "bm25": hits("x")}, k=60, top_k=2)
    assert [h["chunk_id"] for h in tie] == ["x", "y"]
