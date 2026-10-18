"""Step 6.2 - log one evaluation run directory to MLflow: params + tags + metrics + artifacts.

The run directory on disk is the source of truth (rag.eval_retrieval writes it first); MLflow is
the index on top. If the server is down the evaluation is not lost - log it later:
    uv run python -m rag.tracking runs/eval/<timestamp>__<tag> [more dirs...]
A directory is logged at most once: the MLflow run id is written next to it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from rag.config import load_config

RUN_ID_FILE = "mlflow_run_id.txt"
# where things live, not what was tested -> not params (data identity goes in tags as md5)
SKIP_SECTIONS = {"paths", "vector_store", "mlflow", "project"}


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def flatten(d: dict, prefix: str = "") -> dict[str, str]:
    """{'retrieval': {'hybrid': {'rrf_k': 60}}} -> {'retrieval.hybrid.rrf_k': '60'}."""
    out: dict[str, str] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = str(v)
    return out


def run_params(run_cfg: dict) -> dict[str, str]:
    return flatten({k: v for k, v in run_cfg.items() if k not in SKIP_SECTIONS})


def metric_name(*parts: str) -> str:
    """MLflow names allow [A-Za-z0-9_-. /] only: 'hit@5' -> 'hit_at_5'; groups joined by '/'."""
    return "/".join(p.replace("@", "_at_").replace("=", "_") for p in parts)


def run_metrics(summary: dict) -> dict[str, float]:
    out = {metric_name(k): v for k, v in summary["overall"].items()}
    for section, groups in summary.items():
        if section.startswith("by_"):
            for value, agg in groups.items():
                out.update({metric_name(section, value, k): v for k, v in agg.items()})
    out.update({metric_name("latency_s", k): v for k, v in summary["latency_s"].items()})
    return {k: float(v) for k, v in out.items()}


def log_run(run_dir: str | Path, cfg: dict) -> str:
    """Log run_dir to the experiment in cfg['mlflow']; return the MLflow run id."""
    import mlflow  # imported here so evaluation code does not need it to run with --no-mlflow

    run_dir = Path(run_dir)
    id_file = run_dir / RUN_ID_FILE
    if id_file.exists():
        run_id = id_file.read_text(encoding="utf-8").strip()
        print(f"already logged: {run_dir.name} -> {run_id}")
        return run_id

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    run_cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    meta = summary["meta"]

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    with mlflow.start_run(run_name=run_dir.name) as run:
        mlflow.log_params(run_params(run_cfg))
        mlflow.set_tags({k: str(v) for k, v in meta.items()})
        mlflow.log_metrics(run_metrics(summary))
        mlflow.log_artifacts(str(run_dir))
    id_file.write_text(run.info.run_id + "\n", encoding="utf-8")
    return run.info.run_id


def main() -> None:
    ap = argparse.ArgumentParser(description="Log evaluation run directories to MLflow.")
    ap.add_argument("run_dirs", nargs="+")
    args = ap.parse_args()
    cfg = load_config()
    for d in args.run_dirs:
        print(f"{Path(d).name} -> {log_run(d, cfg)}")


if __name__ == "__main__":
    main()
