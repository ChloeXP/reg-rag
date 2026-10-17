"""Retrieval over the Qdrant collections (step 3.4: dense; BM25 + hybrid follow in 3.5-3.6)."""

from __future__ import annotations

from collections.abc import Sequence

from qdrant_client import models

from rag.embed import embed_query
from rag.index import collection_name, get_client

HIT_FIELDS = ["chunk_id", "doc_id", "work_id", "language", "kind", "article", "section",
              "page_start", "page_end", "text"]  # fmt: skip


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


def dense_search(
    query_vector: Sequence[float], cfg: dict, strategy: str, top_k: int, filters: dict | None = None
) -> list[dict]:
    """Nearest chunks to a query vector, best first, as plain dicts (rank, score, payload)."""
    res = get_client(cfg).query_points(
        collection_name(cfg, strategy),
        query=list(query_vector),
        limit=top_k,
        query_filter=_filter(filters),
        with_payload=True,
    )
    return [
        {"rank": r, "score": p.score, **{f: p.payload.get(f) for f in HIT_FIELDS}}
        for r, p in enumerate(res.points, start=1)
    ]


def search(
    query: str,
    cfg: dict,
    top_k: int | None = None,
    strategy: str | None = None,
    filters: dict | None = None,
) -> list[dict]:
    """Semantic search: embed the query with bge-m3, return the top_k closest chunks."""
    top_k = top_k or cfg["retrieval"]["top_k"]
    strategy = strategy or cfg["chunking"]["strategy"]
    return dense_search(embed_query(query, cfg), cfg, strategy, top_k, filters)
