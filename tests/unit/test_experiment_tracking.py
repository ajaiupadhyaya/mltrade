from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from mltrade.config import Environment, Settings
from mltrade.experiments.records import (
    ExperimentRunRecord,
    RunMetrics,
    RunProvenance,
)
from mltrade.experiments.runner import ExperimentRunner, ExperimentTrackingError
from mltrade.experiments.tracking import MlflowRunTracker, NullRunTracker

pytest.importorskip("mlflow")


def _record(**overrides) -> ExperimentRunRecord:
    base = dict(
        run_id="run-" + "0" * 20,
        experiment_name="ridge-baseline",
        status="complete",
        spec_sha256="a" * 64,
        dataset_sha256="b" * 64,
        dataset_snapshot_id="fixture-2026-06-12",
        compatibility_key="d" * 64,
        seed=42,
        started_at=datetime(2026, 6, 14, tzinfo=UTC),
        finished_at=datetime(2026, 6, 14, tzinfo=UTC),
        provenance=RunProvenance(
            git_commit="c" * 40,
            git_dirty=False,
            git_diff_sha256=None,
            python_version="3.13.1",
            platform="test",
            mltrade_version="0.1.0",
            dependencies={"pydantic": "2.11.0"},
            command=("mltrade", "experiment", "run"),
        ),
        parameters={"model.alpha": 1.0, "seed": 42},
        metrics=RunMetrics(
            annualized_return=0.1,
            annualized_volatility=0.1,
            sharpe=1.0,
            max_drawdown=-0.2,
            turnover=0.3,
            total_costs=10.0,
            hit_rate=0.5,
            equal_weight_return=0.05,
            cash_return=0.0,
            robust_sharpe=0.9,
            window_sharpe_std=0.1,
        ),
        artifacts=(),
    )
    base.update(overrides)
    return ExperimentRunRecord(**base)


def test_null_tracker_is_a_noop(tmp_path: Path) -> None:
    assert NullRunTracker().log(_record(), tmp_path) == ""


def test_local_mlflow_tracker_logs_record_and_artifacts(tmp_path: Path) -> None:
    from mlflow.tracking import MlflowClient

    record = _record()
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "report.md").write_text("# report", encoding="utf-8")

    tracking_root = tmp_path / "mlflow"
    external_id = MlflowRunTracker(tracking_root).log(record, artifact_dir)
    assert external_id

    client = MlflowClient(tracking_uri=tracking_root.resolve().as_uri())
    experiment = client.get_experiment_by_name(record.experiment_name)
    assert experiment is not None
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    assert any(
        run.data.tags.get("mltrade.run_id") == record.run_id for run in runs
    )


class _FailingTracker:
    def log(self, record: ExperimentRunRecord, artifact_dir: Path) -> str:
        raise RuntimeError("tracking backend unavailable")


def test_tracker_failure_degrades_without_losing_evidence(tmp_path: Path) -> None:
    settings = Settings(
        environment=Environment.TEST,
        data_root=tmp_path,
        database_url=f"sqlite+pysqlite:///{tmp_path / 'ops.db'}",
    )
    runner = ExperimentRunner(settings=settings, tracker=_FailingTracker())
    record = _record()
    runner._store.save(record)

    with pytest.raises(ExperimentTrackingError):
        runner._track(record)

    reloaded = runner._store.load(record.run_id)
    assert reloaded.tracking_status == "degraded"
    assert reloaded.failure is not None
    assert reloaded.failure.category == "tracking"
