"""Step 5.6 - tests on eval/golden_set.jsonl.

Schema tests need only the repo files (run everywhere). The `data` tests re-check every item
against data/processed/chunks.parquet (skipped when the corpus has not been ingested).
"""

import json
import re
from collections import Counter

import pytest
import yaml

from rag.config import PROJECT_ROOT, load_config
from rag.golden import load_spec

CFG = load_config()
SPEC = load_spec(CFG)
ROWS = [
    json.loads(line)
    for line in (PROJECT_ROOT / CFG["eval"]["golden_set"]).read_text(encoding="utf-8").splitlines()
]
PROMPT = yaml.safe_load((PROJECT_ROOT / "prompts" / "v1.yaml").read_text(encoding="utf-8"))
PARQUET = PROJECT_ROOT / CFG["paths"]["processed_dir"] / "chunks.parquet"
BEHAVIOR = {t: v["behavior"] for t, v in SPEC["types"].items()}


def test_size_ids_and_required_fields():
    assert len(ROWS) == SPEC["total"]
    assert [r["id"] for r in ROWS] == [f"G{i:03d}" for i in range(1, len(ROWS) + 1)]
    assert len({r["src_id"] for r in ROWS}) == len(ROWS)
    required = {"split", "type", "origin", "verification", "question", "question_language",
                "expected_behavior", "expected", "source_chunk_ids", "ground_truth", "key_facts",
                "lang_by_default", "cross_lingual"}  # fmt: skip
    for r in ROWS:
        assert required <= set(r), r["id"]
        assert r["question"].strip() and r["question_language"] in ("fr", "en")


def test_counts_match_the_blueprint():
    tab = Counter((r["type"], r["split"]) for r in ROWS)
    for t, v in SPEC["types"].items():
        assert tab[t, "dev"] == v["dev"] and tab[t, "test"] == v["test"], t
    assert dict(Counter(r["question_language"] for r in ROWS)) == SPEC["languages"]
    assert sum(r["cross_lingual"] for r in ROWS) >= SPEC["min_cross_lingual"]


def test_smoke_seeds_are_dev_only():
    smoke = [r for r in ROWS if r["origin"] == "smoke"]
    assert len(smoke) == 5 and all(r["split"] == "dev" for r in smoke)


@pytest.mark.parametrize("r", ROWS, ids=lambda r: r["id"])
def test_fields_per_behavior(r):
    assert r["expected_behavior"] == BEHAVIOR[r["type"]]
    if r["expected_behavior"] == "answer":
        assert r["ground_truth"] and r["source_chunk_ids"] and r["expected"]
        assert 1 <= len(r["key_facts"]) <= 4
        for f in r["key_facts"]:
            re.compile(f["pattern"])
    elif r["expected_behavior"] == "refuse":  # the exact refusal sentence of prompt v1
        assert r["ground_truth"] == PROMPT["refusal"][r["question_language"]]
        assert r["absence_patterns"] and not r["expected"] and not r["source_chunk_ids"]
    else:
        assert r["injection"] in ("question", "context") and r["must_not_match"]
        assert r["injection"] == "question" or r["injected_passage"]


def test_verification_labels_are_known():
    assert {r["verification"] for r in ROWS} <= {"human_location", "human_line_by_line",
                                                  "claude_only"}  # fmt: skip


@pytest.mark.data
@pytest.mark.skipif(not PARQUET.exists(), reason="run `python -m rag.ingest` first")
def test_every_item_checks_out_against_the_corpus():
    """Sources exist and match the locations, key facts match the sources, absences are real."""
    from rag.handwritten import all_chunks, check_item

    corpus = all_chunks(CFG)
    by_id = {c["chunk_id"]: c for c in corpus}
    errors: list[str] = []
    for r in ROWS:
        check_item({**r, "hid": r["id"]}, by_id, corpus, errors, [])
    assert not errors, errors
