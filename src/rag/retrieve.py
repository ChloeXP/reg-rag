"""Retrieval over the Qdrant collections: dense (3.4), BM25 (3.5), hybrid = RRF of both (3.6)."""

from __future__ import annotations

from collections.abc import Sequence

from qdrant_client import models

from rag import bm25
from rag.embed import embed_query
from rag.index import DENSE, SPARSE, collection_name, get_client
from rag.rerank import rerank

# "strategy" is read by spans.body() to strip the structure header before span scoring: without
# it, live runs scored header + body while rag.rescore (chunks.parquet rows) scored the body only
# -> context_hit differed between a run and its re-score (P89, G025: long ACPR cover headers).
HIT_FIELDS = ["chunk_id", "doc_id", "work_id", "language", "kind", "article", "section",
              "page_start", "page_end", "text", "strategy"]  # fmt: skip


def _filter(filters: dict | None) -> models.Filter | None:
    """{'language': 'fr', 'work_id': 'crr'} -> AND of exact keyword matches."""
    if not filters:
        return None
    return models.Filter(
        must=[
            models.FieldCondition(key=k, match=models.MatchValue(value=v))
            for k, v in filters.items()
        ]
    )


def _query(cfg: dict, strategy: str, query, using: str, top_k: int, filters) -> list[dict]:
    res = get_client(cfg).query_points(
        collection_name(cfg, strategy),
        query=query,
        using=using,
        limit=top_k,
        query_filter=_filter(filters),
        with_payload=True,
    )
    return [
        {"rank": r, "score": p.score, **{f: p.payload.get(f) for f in HIT_FIELDS}}
        for r, p in enumerate(res.points, start=1)
    ]


def dense_search(
    query_vector: Sequence[float], cfg: dict, strategy: str, top_k: int, filters: dict | None = None
) -> list[dict]:
    """Nearest chunks to a query vector (cosine), best first."""
    return _query(cfg, strategy, list(query_vector), DENSE, top_k, filters)


def bm25_search(
    query: str, cfg: dict, strategy: str, top_k: int, filters: dict | None = None
) -> list[dict]:
    """Keyword search: BM25 score = sparse(query) . sparse(doc), IDF applied by Qdrant."""
    indices, values = bm25.query_vector(query)
    if not indices:  # only stopwords / punctuation
        return []
    sparse = models.SparseVector(indices=indices, values=values)
    return _query(cfg, strategy, sparse, SPARSE, top_k, filters)


def rrf_fuse(rankings: dict[str, list[dict]], k: int, top_k: int) -> list[dict]:
    """Reciprocal Rank Fusion (Cormack et al., 2009): score(d) = sum_r 1 / (k + rank_r(d)).

    Uses ranks only, never raw scores, so cosine (0-1) and BM25 (unbounded) need no scaling.
    A chunk missing from one list simply gets nothing from it. Ties -> best single rank,
    then chunk_id, so the order is deterministic. Each hit keeps its per-source ranks.
    """
    fused: dict[str, dict] = {}
    for source, hits in rankings.items():
        for h in hits:
            entry = fused.setdefault(h["chunk_id"], {**h, "score": 0.0, "ranks": {}})
            entry["score"] += 1.0 / (k + h["rank"])
            entry["ranks"][source] = h["rank"]
    order = sorted(
        fused.values(), key=lambda e: (-e["score"], min(e["ranks"].values()), e["chunk_id"])
    )
    return [{**e, "rank": r} for r, e in enumerate(order[:top_k], start=1)]


def hybrid_search(
    query: str, cfg: dict, strategy: str, top_k: int, filters: dict | None = None
) -> list[dict]:
    """Dense + BM25, each fetching `candidates` hits, fused with RRF."""
    h = cfg["retrieval"]["hybrid"]
    n = max(h["candidates"], top_k)
    rankings = {
        "dense": dense_search(embed_query(query, cfg), cfg, strategy, n, filters),
        "bm25": bm25_search(query, cfg, strategy, n, filters),
    }
    return rrf_fuse(rankings, h["rrf_k"], top_k)


def search(
    query: str,
    cfg: dict,
    top_k: int | None = None,
    strategy: str | None = None,
    filters: dict | None = None,
    mode: str | None = None,
) -> list[dict]:
    """Top_k chunks for a query. mode: 'dense' (bge-m3) | 'bm25' (keywords) | 'hybrid'.

    With retrieval.rerank.enabled, the first stage fetches rerank.candidates hits and the
    cross-encoder keeps the best top_k (step 6.4).
    """
    top_k = top_k or cfg["retrieval"]["top_k"]
    strategy = strategy or cfg["retrieval"]["strategy"]
    mode = mode or cfg["retrieval"]["mode"]
    rr = cfg["retrieval"]["rerank"]
    if rr["enabled"]:
        first = first_stage(query, cfg, max(rr["candidates"], top_k), strategy, filters, mode)
        return rerank(query, first, cfg, top_k)
    return first_stage(query, cfg, top_k, strategy, filters, mode)


def first_stage(
    query: str, cfg: dict, top_k: int, strategy: str, filters: dict | None, mode: str
) -> list[dict]:
    if mode == "dense":
        return dense_search(embed_query(query, cfg), cfg, strategy, top_k, filters)
    if mode == "bm25":
        return bm25_search(query, cfg, strategy, top_k, filters)
    if mode == "hybrid":
        return hybrid_search(query, cfg, strategy, top_k, filters)
    raise ValueError(f"unknown retrieval mode {mode!r}")
