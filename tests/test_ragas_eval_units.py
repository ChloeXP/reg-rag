"""Step 6.7 - RAGAS wiring without a judge: inputs per metric, hash check, failure counting."""

import hashlib
import os

import pytest

from rag import ragas_eval as R
from rag.config import load_config

CFG = load_config()
CHUNK = {
    "chunk_id": "dora_fr::structure::0157",
    "work_id": "dora",
    "text": "[DORA · Article 19 Déclaration]\n1. Les entités financières déclarent…",
    "page_start": None,
    "page_end": None,
}
SHA = hashlib.sha256(CHUNK["text"].encode()).hexdigest()[:12]


def test_telemetry_is_off_before_ragas_is_imported():
    assert os.environ["RAGAS_DO_NOT_TRACK"] == "true"


def test_passages_are_what_the_model_saw_and_are_hash_checked():
    rec = {"hits": [{"chunk_id": CHUNK["chunk_id"], "text_sha256": SHA}]}
    assert R.passages(rec, {CHUNK["chunk_id"]: CHUNK}) == [
        "DORA — Article 19 Déclaration\n1. Les entités financières déclarent…"
    ]
    changed = {CHUNK["chunk_id"]: {**CHUNK, "text": CHUNK["text"] + " (edited)"}}
    with pytest.raises(ValueError, match="sha mismatch"):
        R.passages(rec, changed)


def test_each_metric_gets_exactly_its_inputs():
    g = {"question": "q", "ground_truth": "ref"}
    keys = {m: set(R.metric_inputs(m, g, "a", ["c"])) for m in CFG["eval"]["ragas"]["metrics"]}
    assert keys == {
        "faithfulness": {"user_input", "response", "retrieved_contexts"},
        "answer_relevancy": {"user_input", "response"},
        "context_precision": {"user_input", "reference", "retrieved_contexts"},
        "context_recall": {"user_input", "retrieved_contexts", "reference"},
    }


def test_failed_judgements_are_counted_not_averaged():
    rows = [{"id": i, "cross_lingual": False, "verification": "claude_only", "type": "factual"}
            for i in ("G1", "G2")]  # fmt: skip
    lines = [
        {"id": "G1", "metric": "faithfulness", "value": 1.0, "seconds": 10},
        {"id": "G2", "metric": "faithfulness", "error": "ValueError: bad JSON", "seconds": 5},
    ]
    s = R.summarize(lines, rows, CFG)["overall"]["faithfulness"]
    assert (s["mean"], s["n_ok"], s["n_failed"]) == (1.0, 1, 1)


def test_subset_is_deterministic_and_meets_quotas():
    import json

    from rag.config import PROJECT_ROOT

    items = [json.loads(x) for x in (PROJECT_ROOT / CFG["eval"]["golden_set"]).open()]
    rule = CFG["eval"]["ragas"]["subset"]
    ids = R.ragas_subset(items, rule)
    assert ids == R.ragas_subset(items, rule) and len(ids) == sum(rule["quotas"].values())
    by_id = {g["id"]: g for g in items}
    assert set(rule["anchors"]) <= set(ids)
    assert all(by_id[i]["split"] == "dev" for i in ids)  # never the test split
    for kind, quota in rule["quotas"].items():
        assert sum(by_id[i]["type"] == kind for i in ids) == quota
