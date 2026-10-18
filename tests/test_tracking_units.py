"""Step 6.2 - MLflow logging helpers + a full log into a throwaway local store (no server)."""

import json
import re

import pytest
import yaml

from rag.config import load_config
from rag.tracking import RUN_ID_FILE, flatten, log_run, metric_name, run_metrics, run_params

CFG = load_config()
SUMMARY = {
    "meta": {"tag": "t", "layer": "retrieval", "split": "dev", "overrides": []},
    "overall": {"n": 2, "hit@5": 0.5, "rr": 0.75},
    "by_type": {"factual": {"n": 1, "hit@5": 1.0, "rr": 1.0}},
    "by_cross_lingual": {"True": {"n": 1, "hit@5": 0.0, "rr": 0.5}},
    "latency_s": {"mean": 0.1, "p50": 0.1, "max": 0.2},
}


def test_flatten_nested_config():
    assert flatten({"a": {"b": {"c": 1}}, "d": [1, 3]}) == {"a.b.c": "1", "d": "[1, 3]"}


def test_params_skip_locations_keep_experiment_variables():
    p = run_params(CFG)
    assert p["retrieval.strategy"] == "structure" and "llm.model" in p
    assert not any(k.startswith(("paths.", "mlflow.", "vector_store.")) for k in p)


def test_metric_names_are_valid_for_mlflow():
    names = run_metrics(SUMMARY)
    assert names["hit_at_5"] == 0.5 and names["by_type/factual/hit_at_5"] == 1.0
    assert all(re.fullmatch(r"[A-Za-z0-9_\-. /]+", n) for n in names)
    assert metric_name("by_x", "a=b", "recall@10") == "by_x/a_b/recall_at_10"


def test_log_run_into_local_store_once(tmp_path, monkeypatch):
    pytest.importorskip("mlflow")
    # throwaway file store: MLflow 3 refuses it unless explicitly allowed (fine for a unit test)
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    import mlflow

    run_dir = tmp_path / "20260101T000000Z__t"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(json.dumps(SUMMARY))
    (run_dir / "config.yaml").write_text(yaml.safe_dump(CFG))
    (run_dir / "retrieval.jsonl").write_text("{}\n")
    cfg = {**CFG, "mlflow": {"tracking_uri": (tmp_path / "mlruns").as_uri(), "experiment": "t"}}

    run_id = log_run(run_dir, cfg)
    run = mlflow.get_run(run_id)
    assert run.data.params["retrieval.mode"] == CFG["retrieval"]["mode"]
    assert run.data.metrics["by_cross_lingual/True/rr"] == 0.5
    assert run.data.tags["layer"] == "retrieval"
    files = {a.path for a in mlflow.MlflowClient().list_artifacts(run_id)}
    assert files == {"summary.json", "config.yaml", "retrieval.jsonl"}
    assert (run_dir / RUN_ID_FILE).read_text().strip() == run_id
    assert log_run(run_dir, cfg) == run_id  # second call: no duplicate run
