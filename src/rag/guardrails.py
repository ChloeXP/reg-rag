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

Output side (after the LLM, on the final answer)
  mentions   every regulation and article number the answer NAMES must appear in the passages
             it CITES (text incl. header, article metadata, or the work itself). Necessary, not
             sufficient (P64): it proves the name is in the evidence, not that it is used right.
             Action (refuse_on, decision 8.2b-C): an answer naming a regulation that is not in
             the corpus and not in its cited passages is replaced by the refusal; any other
             unsupported name is only flagged (dev: it marked partly-right answers, 8.2a).
             The replaced fields are kept in the verdict, so the decision can be replayed.
"""

from __future__ import annotations

import re

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


NUMBER = re.compile(r"\d+[a-z]?", re.IGNORECASE)
REFUSE_ON = ("never", "out_of_corpus", "any")
# answer fields a guardrail may replace; the originals are kept in the verdict (replay, audit)
ANSWER_FIELDS = ("answer", "cited", "invalid", "is_refusal", "uncited", "other_brackets",
                 "references")  # fmt: skip


def article_numbers(text: str, pattern: str) -> set[str]:
    """{"92", "465"} from "articles 92 et 465"; a range "399 to 403" gives its two ends."""
    out: set[str] = set()
    for m in re.finditer(pattern, text):
        out.add(m.group(1).lower())
        out |= {n.lower() for n in NUMBER.findall(m.group(2))}
    return out


def regulations_named(text: str, groups: dict) -> set[str]:
    return {name for name, g in groups.items() if any(re.search(a, text) for a in g["aliases"])}


def mention_check(answer: str, cited: list[dict], cfg: dict) -> dict:
    """cited = the passages the answer cites (dicts with text, work_id, article).

    -> {"name", "enabled", "passed", "regulations", "articles", "unsupported"}; unsupported lists
    "reg:<group>" / "art:<n>" the answer names but no cited passage contains.
    """
    mc = cfg["guardrails"]["mentions"]
    verdict = {"name": "mentions", "enabled": bool(mc["enabled"]), "passed": True}
    if not mc["enabled"]:
        return verdict
    groups = mc["regulations"]
    haystack = "\n".join(p.get("text") or "" for p in cited)
    have_regs = regulations_named(haystack, groups) | {
        name for name, g in groups.items() if g.get("works")
        for p in cited if p.get("work_id") and re.search(g["works"], p["work_id"])
    }  # fmt: skip
    have_arts = article_numbers(haystack, mc["article"]) | {
        str(p["article"]).lower() for p in cited if p.get("article")
    }
    regs = regulations_named(answer, groups)
    arts = article_numbers(answer, mc["article"])
    unsupported = [f"reg:{r}" for r in sorted(regs - have_regs)]
    unsupported += [f"art:{a}" for a in sorted(arts - have_arts, key=lambda x: (len(x), x))]
    outside = [r for r in sorted(regs - have_regs) if not groups[r].get("works")]
    refuse_on = mc["refuse_on"]
    if refuse_on not in REFUSE_ON:
        raise ValueError(f"guardrails.mentions.refuse_on={refuse_on!r}: expected {REFUSE_ON}")
    refuse = bool(unsupported) and (
        refuse_on == "any" or (refuse_on == "out_of_corpus" and outside)
    )
    verdict.update(
        passed=not unsupported,
        action="refuse" if refuse else ("flag" if unsupported else "none"),
        regulations=sorted(regs),
        articles=sorted(arts, key=lambda x: (len(x), x)),
        unsupported=unsupported,
    )
    return verdict


def output_guardrails(checks: dict, passages: list[dict], prompt: dict, lang: str, cfg: dict):
    """checks = the answer fields of one generated answer (ANSWER_FIELDS + structured);
    passages = what the model was shown, in order (cited n = passages[n - 1]).

    -> (checks, possibly replaced by the refusal; list of output verdicts). Refusals and
    fallbacks make no claim to check: no verdict.
    """
    st = checks.get("structured")
    if checks["is_refusal"] or (st and st["status"] == "fallback"):
        return checks, []
    cited = [passages[n - 1] for n in checks["cited"]]
    verdict = mention_check(checks["answer"], cited, cfg)
    if verdict.get("action") != "refuse":
        return checks, [verdict]
    verdict["replaced"] = {k: checks[k] for k in ANSWER_FIELDS}
    new = {**checks, **refusal_fields(prompt, lang), "structured": st}
    if st:  # what the user gets (step 9 body) follows the guardrail; the JSON attempts stay
        verdict["replaced"]["api_answer"] = st["api_answer"]
        api = {"answer": new["answer"], "citations": [], "confidence": "low", "answerable": False}
        new["structured"] = {**st, "api_answer": api}
    return new, [verdict]


def undo_output_guardrails(record: dict) -> dict:
    """The record as generated, before any output guardrail replaced it (for an offline replay)."""
    verdicts = (record.get("guardrails") or {}).get("output", [])
    replaced = next((v["replaced"] for v in verdicts if "replaced" in v), None)
    if replaced is None:
        return record
    rec = {**record, **{k: replaced[k] for k in ANSWER_FIELDS}}
    if "api_answer" in replaced:
        rec["structured"] = {**record["structured"], "api_answer": replaced["api_answer"]}
    return rec
