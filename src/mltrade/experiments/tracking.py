"""Experiment-run tracking behind a small interface.

The runner persists canonical evidence first and only then calls a tracker.
A tracker failure degrades the run's ``tracking_status`` but never deletes the
canonical record.  ``NullRunTracker`` is the default no-op; ``MlflowRunTracker``
(added in the tracking task) logs to a local MLflow file store.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

from mltrade.experiments.records import ExperimentRunRecord

_MAX_PARAM_LEN = 250


class RunTracker(Protocol):
    def log(self, record: ExperimentRunRecord, artifact_dir: Path) -> str:
        """Log a completed run; return the external tracking id (or "")."""
        ...


class NullRunTracker:
    """A tracker that records nothing and returns an empty external id."""

    def log(self, record: ExperimentRunRecord, artifact_dir: Path) -> str:
        return ""


def _flatten_params(record: ExperimentRunRecord) -> dict[str, str]:
    flattened: dict[str, str] = {}
    for key, value in record.parameters.items():
        text = str(value)
        if len(text) > _MAX_PARAM_LEN:
            text = f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"
        flattened[key] = text
    return flattened


class MlflowRunTracker:
    """Log canonical runs to a local MLflow file store.

    MLflow is imported lazily so the core package stays free of the dependency
    unless tracking is actually used.
    """

    def __init__(self, tracking_root: Path) -> None:
        self._root = Path(tracking_root).resolve()

    def log(self, record: ExperimentRunRecord, artifact_dir: Path) -> str:
        import os

        # MLflow 3.x gates the local filesystem backend behind an explicit
        # opt-in.  A local file store is exactly this tool's intent.
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

        import mlflow

        self._root.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(self._root.as_uri())
        mlflow.set_experiment(record.experiment_name)
        with mlflow.start_run(run_name=record.run_id) as active:
            mlflow.log_params(_flatten_params(record))
            if record.metrics is not None:
                mlflow.log_metrics(record.metrics.model_dump())
            mlflow.set_tags(
                {
                    "mltrade.run_id": record.run_id,
                    "mltrade.snapshot_id": record.dataset_snapshot_id,
                    "mltrade.compatibility_key": record.compatibility_key,
                    "mltrade.git_dirty": str(record.provenance.git_dirty).lower(),
                    "mltrade.status": record.status,
                }
            )
            artifacts = Path(artifact_dir)
            if artifacts.is_dir():
                mlflow.log_artifacts(str(artifacts), artifact_path="research")
            run_id: str = active.info.run_id
        return run_id
