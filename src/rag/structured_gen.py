"""Structured generation (step 7.3): Ollama JSON mode -> validate -> retry once -> fallback.

One call of `run()` = up to 1 + max_retries LLM calls. Every attempt is logged with its raw
output and status, so the first attempt can be scored on its own (hallucinated-citation rate)
and the final answer separately (what the user would get).

Attempt status (decision 7.3-B: grounding errors are retried too):
  ok              valid JSON, consistent fields, every cited id exists among the passages
  truncated       hit num_predict (P78) and the JSON is cut off
  schema_error    invalid JSON / type / extra key / answerable<->citations contradiction, or
                  answerable=true with the refusal sentence as the answer (P76)
  grounding_error well-formed id that is not a passage ("P6" with 5 passages, P65)
Final status: ok (first attempt) | retried_ok | fallback (no valid attempt; nothing invented).

Policy (config llm.structured.policy, step 7.8 - measured in 7.6 / 7.7, P82 / P84):
  length             reject: answer longer than the prompt's answer_max_chars = schema_error
                     flag:   a COMPLETE JSON answer is kept whatever its length (runaway is caught
                             by num_predict -> truncated); attempts[].over_length records it
  invalid_citations  reject: any non-existent id = grounding_error (7.3-B)
                     drop:   if at least one cited id exists, the others are removed and the
                             answer is kept (attempts[].dropped); no valid id left = error
The defaults (reject / reject) reproduce the 7.3 behaviour; the first attempt's invalid ids are
always recorded (invalid_cite metric), whatever the policy.
"""

from __future__ import annotations

from pydantic import ValidationError

from rag.generate import _norm, references
from rag.llm import generate
from rag.structured import ModelAnswer, error_lines, model_schema, parse, to_answer

# Shown to the user when no attempt is valid. Deliberately NOT the refusal sentence: a system
# failure must not earn the credit of a correct refusal on no-answer items (is_refusal False).
FALLBACK = {
    "fr": "Le système n'a pas pu produire une réponse fiable et vérifiable. Veuillez reformuler "
    "la question ou consulter directement les textes.",
    "en": "The system could not produce a reliable, verifiable answer. Please rephrase the "
    "question or consult the texts directly.",
}


DEFAULT_POLICY = {"length": "reject", "invalid_citations": "reject"}  # = 7.3 behaviour
POLICIES = {"length": {"reject", "flag"}, "invalid_citations": {"reject", "drop"}}


def policy_of(cfg: dict) -> dict:
    policy = {**DEFAULT_POLICY, **cfg["llm"]["structured"].get("policy", {})}
    for key, value in policy.items():
        if value not in POLICIES.get(key, set()):
            raise ValueError(f"llm.structured.policy.{key}={value!r}: expected {POLICIES[key]}")
    return policy


def check(raw: str, done_reason: str | None, hits: list[dict], prompt: dict, policy=None):
    """One attempt -> (status, errors for the retry, ModelAnswer | None, invalid ids, notes).

    notes: {"dropped": [...], "over_length": True} when the policy kept an answer anyway.
    """
    policy = policy or DEFAULT_POLICY
    limit = prompt["answer_max_chars"]
    try:
        ma = parse(raw, limit if policy["length"] == "reject" else None)
    except ValidationError as err:
        if done_reason == "length":
            return "truncated", ["the output was cut off: write a shorter answer"], None, [], {}
        return "schema_error", error_lines(err), None, [], {}
    notes: dict = {"over_length": True} if len(ma.answer) > limit else {}
    refusals = {_norm(r) for r in prompt["refusal"].values()}
    if ma.answerable and _norm(ma.answer) in refusals:
        msg = ("answer is the refusal sentence but answerable is true: if no passage contains "
               "the answer, set answerable to false and citations to []")  # fmt: skip
        return "schema_error", [msg], ma, [], notes
    _, invalid = to_answer(ma, hits)
    if invalid:
        if policy["invalid_citations"] == "drop" and len(invalid) < len(ma.citations):
            kept = [c for c in ma.citations if c not in invalid]
            return "ok", [], ma.model_copy(update={"citations": kept}), invalid, {
                **notes, "dropped": invalid}  # fmt: skip
        msg = (f"citations {invalid} do not exist: the passages are P1 to P{len(hits)}; "
               "cite only those ids")  # fmt: skip
        return "grounding_error", [msg], ma, invalid, notes
    return "ok", [], ma, [], notes


def retry_user(user: str, prompt: dict, raw: str, errors: list[str], max_chars: int) -> str:
    """Original user turn + the retry template (rejected output, capped, + its problems)."""
    previous = raw if len(raw) <= max_chars else raw[:max_chars] + " [...]"
    return (
        user
        + "\n\n"
        + prompt["retry"].format(previous=previous, errors="\n".join(f"- {e}" for e in errors))
    )


def run(built: dict, hits: list[dict], prompt: dict, cfg: dict) -> dict:
    """built = rag.generate.build_prompt(...) -> structured result (see module docstring)."""
    sc = cfg["llm"]["structured"]
    policy = policy_of(cfg)
    schema = model_schema(prompt["answer_max_chars"])
    attempts: list[dict] = []
    user = built["user"]
    final: ModelAnswer | None = None
    for i in range(1 + sc["max_retries"]):
        out = generate(user, cfg, system=built["system"], fmt=schema, num_predict=sc["num_predict"])
        status, errors, ma, invalid, notes = check(
            out["text"], out.get("done_reason"), hits, prompt, policy
        )
        telemetry = {k: v for k, v in out.items() if k != "text"}
        attempts.append({"attempt": i + 1, "status": status, "errors": errors, "invalid": invalid,
                         **notes, "raw": out["text"], **telemetry})  # fmt: skip
        if status == "ok":
            final = ma
            break
        user = retry_user(built["user"], prompt, out["text"], errors, sc["max_previous_chars"])
    out = result(final, attempts, hits, prompt, built["language"])
    out["structured"]["policy"] = policy
    return out


def result(final, attempts, hits, prompt, lang) -> dict:
    """Map to the v1 record fields (answer, cited, invalid, is_refusal, ...) + structured detail.

    answerable=false -> the fixed refusal sentence is shown (code decides the wording, as in v1
    the refusal stays exactly detectable); the model's own text stays in attempts[].raw.
    """
    if final is None:
        status, answer, cited, refs, api = "fallback", FALLBACK[lang], [], [], None
        is_refusal = False
    else:
        status = "ok" if len(attempts) == 1 else "retried_ok"
        ans, _ = to_answer(final, hits)
        is_refusal = not final.answerable
        answer = prompt["refusal"][lang] if is_refusal else final.answer
        cited = [int(c.passage[1:]) for c in ans.citations]
        refs = references([(n, hits[n - 1]) for n in cited])
        api = {**ans.model_dump(), "answer": answer}  # same text the user sees
    return {
        "answer": answer,
        "cited": sorted(set(cited)),
        "invalid": [],  # final answers never carry invalid ids (7.3-B); see attempts[0].invalid
        "is_refusal": is_refusal,
        "uncited": not cited and not is_refusal,
        "other_brackets": answer.count("["),  # rule 3: no brackets / ids inside the answer
        "references": refs,
        "structured": {
            "status": status,
            "n_attempts": len(attempts),
            "first_status": attempts[0]["status"],
            "first_invalid": attempts[0]["invalid"],
            "policy": None,  # filled by run() / replay
            "dropped": attempts[-1].get("dropped", []) if final is not None else [],
            "over_length": bool(attempts[-1].get("over_length")) if final is not None else False,
            "answerable": None if final is None else final.answerable,
            "confidence": None if final is None else final.confidence,
            "api_answer": api,  # rag.structured.Answer (step 9 response body)
            "attempts": attempts,
        },
    }
