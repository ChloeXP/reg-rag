"""Step 6.5 - span hit rule: unit tests + data checks that make P55 a regression test."""

import json

import pytest

from rag.config import PROJECT_ROOT, load_config
from rag.golden import is_hit, load_chunks
from rag.spans import body, location_span, shingles, span_hit, span_ranks

CFG = load_config()
RULE = CFG["eval"]["span"]
N = RULE["shingle_words"]
PARQUET = PROJECT_ROOT / CFG["paths"]["processed_dir"] / "chunks.parquet"
TEXT = " ".join(f"w{i}" for i in range(100))  # 100 distinct words -> 93 8-word shingles


def test_body_strips_only_the_structure_header():
    assert body({"strategy": "structure", "text": "[DORA · Article 19]\n1. Les entités"}) == (
        "1. Les entités"
    )
    assert body({"strategy": "fixed", "text": "[not a header] text"}) == "[not a header] text"


def test_shingles_ignore_case_and_punctuation():
    assert shingles("A, b. C d e f g H!", 8) == {("a", "b", "c", "d", "e", "f", "g", "h")}


def test_card_mostly_inside_location_is_a_hit():
    span = shingles(TEXT, N)
    straddling = " ".join(f"x{i}" for i in range(30)) + " " + TEXT  # 30 foreign + 100 own words
    assert span_hit(shingles(straddling, N), span, RULE)


def test_card_with_a_small_slice_of_a_long_location_is_not():
    span = shingles(TEXT, N)
    card = " ".join(f"x{i}" for i in range(200)) + " " + " ".join(f"w{i}" for i in range(12))
    assert not span_hit(shingles(card, N), span, RULE)


def test_short_location_inside_a_big_card_is_a_hit():
    short = " ".join(f"r{i}" for i in range(20))  # a 20-word recital
    card = " ".join(f"x{i}" for i in range(300)) + " " + short
    assert span_hit(shingles(card, N), shingles(short, N), RULE)


def test_span_ranks_first_matching_rank_per_location():
    hits = [{"rank": 1, "text": "unrelated words " * 20}, {"rank": 2, "text": TEXT}]
    assert span_ranks(hits, [shingles(TEXT, N), {("nope",) * N}], RULE) == [2, None]


@pytest.fixture(scope="module")
def corpus():
    if not PARQUET.exists():
        pytest.skip("run `python -m rag.ingest` first")
    items = [json.loads(x) for x in (PROJECT_ROOT / CFG["eval"]["golden_set"]).open()]
    return items, load_chunks(CFG, "structure"), load_chunks(CFG, "fixed")


@pytest.mark.data
def test_every_location_is_reachable_in_both_strategies(corpus):
    """P55 as a regression test: no expected location may be unreachable under any chunking."""
    items, structure, fixed = corpus
    missing = []
    for g in items:
        for i, e in enumerate(g["expected"]):
            span = location_span(e, structure, N)
            for name, cards in [("structure", structure), ("fixed", fixed)]:
                pool = [c for c in cards if c["work_id"] == e["work_id"]]
                if not any(span_hit(shingles(body(c), N), span, RULE) for c in pool):
                    missing.append((g["id"], i, name))
    assert not missing


@pytest.mark.data
def test_span_rule_keeps_every_metadata_hit_on_structure(corpus):
    """The new rule must not lose what the old rule accepted on the cards it was designed for."""
    items, structure, _ = corpus
    lost = []
    for g in items:
        for e in g["expected"]:
            span = location_span(e, structure, N)
            for c in structure:
                if is_hit(c, e) and not span_hit(shingles(body(c), N), span, RULE):
                    lost.append((g["id"], c["chunk_id"]))
    assert not lost
