"""Fast unit tests for step 7.1 structured output (no Qdrant, no Ollama, no corpus)."""

import json

import pytest
from pydantic import ValidationError

from rag.structured import (
    error_lines,
    model_schema,
    parse,
    passage_id,
    to_answer,
)


def hit(n):
    cid = f"dora_fr::structure::{n:04d}"
    text = f"[DORA · Article {n} Titre] Corps {n}."
    return {"chunk_id": cid, "doc_id": "dora_fr", "work_id": "dora", "article": str(n),
            "page_start": 1, "page_end": 1, "text": text}  # fmt: skip


HITS = [hit(n) for n in (19, 20, 21, 22, 23)]
OK = {"answer": "Les entités déclarent les incidents majeurs.", "citations": ["P1", "P3"],
      "confidence": "high", "answerable": True}  # fmt: skip


def raw(**changes):
    return json.dumps({**OK, **changes}, ensure_ascii=False)


def errors_of(text):
    with pytest.raises(ValidationError) as exc:
        parse(text)
    return exc.value


# --- happy path ----------------------------------------------------------------------------------
def test_valid_output_parses():
    ma = parse(raw())
    assert ma.citations == ["P1", "P3"] and ma.answerable is True


def test_refusal_shape_parses():
    ma = parse(raw(citations=[], answerable=False, confidence="low"))
    assert ma.citations == [] and ma.answerable is False


def test_duplicate_citations_are_normalised_not_rejected():
    assert parse(raw(citations=["P3", "P1", "P3"])).citations == ["P3", "P1"]


def test_passage_id_matches_pattern():
    assert passage_id(1) == "P1" and parse(raw(citations=[passage_id(5)])).citations == ["P5"]


# --- schema errors (-> retry in 7.3) -----------------------------------------------------------
@pytest.mark.parametrize(
    "bad", ["1", "[1]", "p1", "P0", "P", "Passage 1", "dora_fr::structure::0019"]
)
def test_citation_must_be_a_passage_id(bad):
    errors_of(raw(citations=[bad]))


def test_strict_types_no_silent_coercion():
    errors_of(raw(answerable="true"))  # a string is not a bool
    errors_of(raw(citations=[1]))  # an int is not "P1"


def test_unknown_confidence_and_extra_keys_rejected():
    errors_of(raw(confidence="very high"))
    errors_of(raw(sources=["P1"]))  # extra="forbid"


def test_missing_field_rejected():
    d = dict(OK)
    del d["answerable"]
    errors_of(json.dumps(d))


def test_truncated_json_is_a_validation_error():
    err = errors_of(raw()[:40])  # what a cut-off generation looks like
    assert err.errors()[0]["type"] == "json_invalid"


def test_blank_or_runaway_answer_rejected():
    errors_of(raw(answer="   "))
    errors_of(raw(answer="x" * 2001))


def test_consistency_rules():
    e1 = errors_of(raw(citations=[]))  # P68: answerable but no citation
    assert "citations is empty" in error_lines(e1)[0]
    e2 = errors_of(raw(answerable=False))  # P75: refusal followed by citations
    assert "use []" in error_lines(e2)[0]


def test_error_lines_short_and_without_input_echo():
    err = errors_of(json.dumps({"answer": "secret text", "citations": ["x"] * 5}))
    lines = error_lines(err)
    assert 1 <= len(lines) <= 3 and all("secret text" not in line for line in lines)
    assert lines[0].startswith("citations.0:")


# --- grounding (resolve against the retrieved hits) -----------------------------------------------
def test_resolve_maps_ids_to_hit_metadata_in_model_order():
    ans, invalid = to_answer(parse(raw(citations=["P3", "P1"])), HITS)
    assert [c.chunk_id for c in ans.citations] == ["dora_fr::structure::0021",
                                                    "dora_fr::structure::0019"]  # fmt: skip
    c = ans.citations[0]
    assert (c.passage, c.doc_id, c.article, c.label) == ("P3", "dora_fr", "21",
                                                          "DORA — Article 21 Titre")  # fmt: skip
    assert invalid == []


def test_out_of_range_id_is_reported_not_kept():
    ans, invalid = to_answer(parse(raw(citations=["P2", "P6"])), HITS)  # P65: cites a 6th passage
    assert [c.passage for c in ans.citations] == ["P2"] and invalid == ["P6"]


def test_fewer_hits_shrink_the_valid_range():
    _, invalid = to_answer(parse(raw(citations=["P3"])), HITS[:2])
    assert invalid == ["P3"]


# --- schema sent to Ollama -----------------------------------------------------------------------
def test_schema_shape():
    s = model_schema()
    assert set(s["required"]) == {"answer", "citations", "confidence", "answerable"}
    assert s["additionalProperties"] is False
    assert "description" not in s  # docstring edits must not change the schema
    assert s["properties"]["citations"]["items"]["pattern"] == r"^P[1-9][0-9]?$"
    assert s["properties"]["confidence"]["enum"] == ["high", "medium", "low"]
    assert s["properties"]["answerable"]["type"] == "boolean"
    assert list(s["properties"]) == ["answer", "citations", "confidence", "answerable"]
