from datetime import UTC, datetime

import pytest

from mltrade.experiments.records import (
    ExperimentRunRecord,
    RunMetrics,
    RunProvenance,
)
from mltrade.experiments.storage import RunStorageError, RunStore


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
        parameters={"model.alpha": 1.0},
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


def test_run_store_publishes_atomically(tmp_path) -> None:
    store = RunStore(tmp_path)
    record = _record()

    path = store.save(record)

    assert path == tmp_path / "runs" / record.run_id / "run.json"
    assert store.load(record.run_id) == record
    assert not list((tmp_path / "runs").glob(".*.tmp-*"))


def test_idempotent_save_returns_same_path(tmp_path) -> None:
    store = RunStore(tmp_path)
    record = _record()

    first = store.save(record)
    second = store.save(record)

    assert first == second


def test_same_run_id_with_different_content_is_rejected(tmp_path) -> None:
    store = RunStore(tmp_path)
    record = _record()
    store.save(record)

    other_metrics = record.metrics.model_copy(update={"sharpe": 2.0})
    with pytest.raises(RunStorageError, match="different content"):
        store.save(record.model_copy(update={"metrics": other_metrics}))


def test_replace_tracking_state_updates_run_json(tmp_path) -> None:
    store = RunStore(tmp_path)
    record = _record()
    store.save(record)

    store.replace_tracking_state(
        record.model_copy(update={"tracking_status": "degraded"})
    )

    assert store.load(record.run_id).tracking_status == "degraded"


def test_save_with_artifacts_writes_files(tmp_path) -> None:
    store = RunStore(tmp_path)
    record = _record()

    store.save(record, artifacts={"report.md": b"# report"})

    assert (
        tmp_path / "runs" / record.run_id / "report.md"
    ).read_bytes() == b"# report"


def test_list_records_round_trips(tmp_path) -> None:
    store = RunStore(tmp_path)
    store.save(_record())

    records = store.list_records()

    assert len(records) == 1
    assert records[0].run_id == "run-" + "0" * 20
