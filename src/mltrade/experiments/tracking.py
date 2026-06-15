"""Experiment-run tracking behind a small interface.

The runner persists canonical evidence first and only then calls a tracker.
A tracker failure degrades the run's ``tracking_status`` but never deletes the
canonical record.  ``NullRunTracker`` is the default no-op; ``MlflowRunTracker``
(added in the tracking task) logs to a local MLflow file store.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from mltrade.experiments.records import ExperimentRunRecord


class RunTracker(Protocol):
    def log(self, record: ExperimentRunRecord, artifact_dir: Path) -> str:
        """Log a completed run; return the external tracking id (or "")."""
        ...


class NullRunTracker:
    """A tracker that records nothing and returns an empty external id."""

    def log(self, record: ExperimentRunRecord, artifact_dir: Path) -> str:
        return ""
