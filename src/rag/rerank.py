"""Step 6.4 - second-stage reranking with a cross-encoder (bge-reranker-v2-m3).

Dense retrieval compares two vectors computed separately ("does this card look like the
question?"); a cross-encoder reads question + card TOGETHER and scores how well the card
answers it. Too slow for the whole corpus, so it only re-orders the first-stage candidates.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _load_model(model_name: str, device: str, max_length: int):
    from sentence_transformers import CrossEncoder  # heavy import: only when reranking

    return CrossEncoder(model_name, device=device, max_length=max_length)


def rerank(query: str, hits: list[dict], cfg: dict, top_k: int) -> list[dict]:
    """Re-score hits with the cross-encoder; return the best top_k, renumbered 1..top_k.

    Each hit keeps its first-stage rank and score, so a run log can show what moved.
    Ties keep the first-stage order (stable sort), so the output is deterministic.
    """
    if not hits:
        return []
    r = cfg["retrieval"]["rerank"]
    model = _load_model(r["model"], cfg["embedding"]["device"], r["max_length"])
    scores = model.predict(
        [(query, h["text"]) for h in hits], batch_size=r["batch_size"], show_progress_bar=False
    )
    order = sorted(range(len(hits)), key=lambda i: -float(scores[i]))  # stable on ties
    return [
        {
            **hits[i],
            "rank": new_rank,
            "score": float(scores[i]),
            "first_stage_rank": hits[i]["rank"],
            "first_stage_score": hits[i]["score"],
        }
        for new_rank, i in enumerate(order[:top_k], start=1)
    ]
