"""Structured output (step 7): what the LLM fills (ModelAnswer) vs what the system returns (Answer).

Decision 7.0-B: the model cites passage ids ("P1".."Pk") only. Code resolves them to
doc_id / article / chunk_id from the retrieved hits, so the model never copies metadata.

Two kinds of errors are kept apart on purpose:
  - schema errors: bad/truncated JSON, wrong type, extra key, broken consistency rule
    -> pydantic.ValidationError -> the caller may retry with `error_lines(err)` as feedback (7.3)
  - grounding errors: a well-formed id that is not among the passages (e.g. "P6", P65)
    -> reported by `resolve_citations` as `invalid`; measured, never silently accepted
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from rag.generate import split_passage

PASSAGE_ID = r"^P[1-9][0-9]?$"  # "P1".."P99"; the "P" avoids clashes with "[52]" in the text (P38)
MAX_ANSWER_CHARS = 2000  # default (v2): ~6 sentences need < 1000; longer = runaway generation
# A prompt may set its own `answer_max_chars` (v3: complete lists, P83). The limit is passed to
# the schema and to validation, so v2's schema - hence its logged hash - stays byte-identical.
MAX_CITATIONS = 10
MAX_ERROR_LINES = 3  # retry feedback stays short: a 3b model drowns in long error dumps

Confidence = Literal["high", "medium", "low"]
PassageId = Annotated[str, Field(pattern=PASSAGE_ID)]


def passage_id(n: int) -> str:
    """1-based position in the hits -> "P<n>" (the label shown in the prompt context)."""
    return f"P{n}"


class ModelAnswer(BaseModel):
    """Exactly what the LLM must produce. Its JSON schema is sent to Ollama as `format`.

    strict=True: "true" (a string) is not a bool, "P1" is not 1 -> a sloppy output is an
    error we count, not something pydantic quietly repairs.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    # Field order = generation order (Ollama follows the schema order, probe 7.1).
    # Decision 7.2-C: evidence -> verdict -> answer. The model first picks the passages, then
    # judges answerability, then writes an answer anchored on what it just picked (P76: with
    # `answerable` last it was filled after the fact and contradicted the answer 2 times in 3).
    citations: list[PassageId] = Field(max_length=MAX_CITATIONS)
    answerable: bool
    answer: str = Field(min_length=1)  # max length: per prompt, see _length / model_schema
    confidence: Confidence

    @field_validator("answer")
    @classmethod
    def _length(cls, v: str, info: ValidationInfo) -> str:
        if not v.strip():
            raise ValueError("answer is blank")
        limit = (info.context or {}).get("max_answer_chars", MAX_ANSWER_CHARS)
        if limit is not None and len(v) > limit:  # None: no limit (policy length=flag, 7.8)
            raise ValueError(f"answer is too long ({len(v)} characters, maximum {limit})")
        return v.strip()

    @field_validator("citations")
    @classmethod
    def _dedupe(cls, v: list[str]) -> list[str]:
        return list(dict.fromkeys(v))  # normalisation, not an error: keep first-seen order

    # Semantic rules: a JSON schema cannot express them, so Ollama cannot enforce them.
    # This is why validation + retry exist even with constrained decoding.
    @model_validator(mode="after")
    def _consistent(self) -> ModelAnswer:
        if self.answerable and not self.citations:
            raise ValueError(
                'answerable is true but citations is empty: list the passage ids (e.g. "P1") '
                "that support the answer"
            )
        if not self.answerable and self.citations:
            raise ValueError("answerable is false but citations is not empty: use []")
        return self


def model_schema(max_answer_chars: int = MAX_ANSWER_CHARS) -> dict:
    """JSON schema for Ollama `format` (constrained decoding of the shape, not of the truth).

    The class docstring is dropped: it is for developers, and editing it must not change the
    schema (the schema becomes part of the prompt hash in 7.2).
    """
    schema = ModelAnswer.model_json_schema()
    schema.pop("description", None)
    schema["properties"]["answer"]["maxLength"] = max_answer_chars
    return schema


def parse(raw: str, max_answer_chars: int | None = MAX_ANSWER_CHARS) -> ModelAnswer:
    """Raw LLM text -> ModelAnswer. Invalid JSON is a ValidationError too (type json_invalid)."""
    return ModelAnswer.model_validate_json(raw, context={"max_answer_chars": max_answer_chars})


def error_lines(err: ValidationError, limit: int = MAX_ERROR_LINES) -> list[str]:
    """Short, model-readable error list: '<field path>: <message>' (no echo of the input)."""
    lines = []
    for e in err.errors(include_url=False, include_input=False)[:limit]:
        loc = ".".join(str(p) for p in e["loc"]) or "(root)"
        msg = e["msg"].removeprefix("Value error, ")
        lines.append(f"{loc}: {msg}")
    return lines


class Citation(BaseModel):
    """One resolved citation (framework: doc_id + article + chunk_id), built by code from a hit."""

    passage: str  # "P2" as written by the model
    doc_id: str
    article: str | None
    chunk_id: str
    label: str  # e.g. "DORA — Article 19 ..." (same label the model saw)


class Answer(BaseModel):
    """What the system returns (step 9 API reuses it). Citations are resolved, never invented."""

    answer: str
    citations: list[Citation]
    confidence: Confidence
    answerable: bool


def resolve_citations(ma: ModelAnswer, hits: list[dict]) -> tuple[list[Citation], list[str]]:
    """-> (valid citations in model order, invalid ids). Valid = "P<n>" with 1 <= n <= len(hits)."""
    valid, invalid = [], []
    for pid in ma.citations:
        n = int(pid[1:])
        if not 1 <= n <= len(hits):
            invalid.append(pid)
            continue
        hit = hits[n - 1]
        label, _ = split_passage(hit)
        valid.append(
            Citation(passage=pid, doc_id=hit["doc_id"], article=hit.get("article"),
                     chunk_id=hit["chunk_id"], label=label)
        )  # fmt: skip
    return valid, invalid


def to_answer(ma: ModelAnswer, hits: list[dict]) -> tuple[Answer, list[str]]:
    """ModelAnswer + hits -> (Answer with only grounded citations, invalid ids for the log)."""
    valid, invalid = resolve_citations(ma, hits)
    return (
        Answer(
            answer=ma.answer, citations=valid, confidence=ma.confidence, answerable=ma.answerable
        ),
        invalid,
    )
