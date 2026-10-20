"""Step 8 - guardrails: pure functions shared by rag.ask, the evaluation and the API (step 9).

Every guardrail returns a small JSON-able verdict dict that is logged with the answer
(record["guardrails"]), so each decision is traceable and can be replayed offline (#133).

Input side (before the LLM is called)
  relevance  refuse WITHOUT an LLM call when the best retrieved passage is clearly off-topic:
             max reranker score < guardrails.relevance.min_top_score (decision 8.0-B: 0.05).
             The threshold only means something on the scale of the model that produced the
             score, so the gate refuses to run on any other scale (#138). It is compared at the
             precision written in the logs (4 decimals): live decisions and offline replays of
             logged runs are identical by construction.
"""

from __future__ import annotations

SCORE_DECIMALS = 4  # = rag.ask logs hits[].score with round(score, 4)


def top_score(hits: list[dict]) -> float | None:
    """Best score among the passages, at log precision. max(), not hits[0]: a passage inserted
    in front (the evaluation's context-injection note, score 0.0) does not change the decision."""
    if not hits:
        return None
    return round(max(h["score"] for h in hits), SCORE_DECIMALS)


def relevance_gate(hits: list[dict], cfg: dict) -> dict:
    """-> {"name", "enabled", "passed", "top_score", "threshold", "score_model"}."""
    rc = cfg["guardrails"]["relevance"]
    verdict = {"name": "relevance", "enabled": bool(rc["enabled"]), "passed": True}
    if not rc["enabled"]:
        return verdict
    rr = cfg["retrieval"]["rerank"]
    if not rr["enabled"] or rr["model"] != rc["score_model"]:
        scale = rr["model"] if rr["enabled"] else f"first-stage {cfg['retrieval']['mode']}"
        raise ValueError(
            f"guardrails.relevance.min_top_score is calibrated on {rc['score_model']} scores, "
            f"but hits are scored by {scale}: recalibrate or disable the gate"
        )
    top = top_score(hits)
    verdict.update(
        passed=top is not None and top >= rc["min_top_score"],
        top_score=top,
        threshold=rc["min_top_score"],
        score_model=rc["score_model"],
    )
    return verdict


def refusal_fields(prompt: dict, lang: str) -> dict:
    """The answer-check fields of a refusal decided by a guardrail (no LLM output exists).

    Same keys as rag.generate.parse_citations / rag.structured_gen.result, with the prompt's
    own refusal sentence so every metric and every reader detects it exactly like a model
    refusal. structured is None: no JSON attempt was made, so the JSON metrics do not apply.
    """
    return {
        "answer": prompt["refusal"][lang],
        "cited": [],
        "invalid": [],
        "is_refusal": True,
        "uncited": False,
        "other_brackets": 0,
        "references": [],
        "structured": None,
    }
