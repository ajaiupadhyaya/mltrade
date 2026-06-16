from __future__ import annotations

from datetime import UTC, datetime

from mltrade.experiments.comparison import compare_runs
from mltrade.experiments.records import (
    ExperimentRunRecord,
    RunMetrics,
    RunProvenance,
)

_T = datetime(2026, 6, 14, tzinfo=UTC)


def _record(
    run_id: str,
    robust: float,
    *,
    key: str = "d" * 64,
    dataset_sha: str = "b" * 64,
    status: str = "complete",
    dirty: bool = False,
    tracking: str = "pending",
    with_metrics: bool = True,
) -> ExperimentRunRecord:
    metrics = (
        RunMetrics(
            annualized_return=0.1,
            annualized_volatility=0.1,
            sharpe=1.0,
            max_drawdown=-0.2,
            turnover=0.3,
            total_costs=10.0,
            hit_rate=0.5,
            equal_weight_return=0.05,
            cash_return=0.0,
            robust_sharpe=robust,
            window_sharpe_std=0.1,
        )
        if with_metrics
        else None
    )
    return ExperimentRunRecord(
        run_id=run_id,
        experiment_name="ridge-baseline",
        status=status,
        spec_sha256="a" * 64,
        dataset_sha256=dataset_sha,
        dataset_snapshot_id="fixture-2026-06-12",
        compatibility_key=key,
        seed=42,
        started_at=_T,
        finished_at=_T,
        provenance=RunProvenance(
            git_commit="c" * 40,
            git_dirty=dirty,
            git_diff_sha256=("e" * 64) if dirty else None,
            python_version="3.13.1",
            platform="test",
            mltrade_version="0.1.0",
            dependencies={},
            command=("mltrade",),
        ),
        parameters={"model.alpha": 1.0},
        metrics=metrics,
        artifacts=(),
        tracking_status=tracking,
    )


def test_compatible_runs_rank_by_robust_sharpe() -> None:
    low = _record("run-a", 0.5)
    high = _record("run-b", 1.5)

    result = compare_runs((low, high))

    assert result.compatible is True
    assert result.ranking[0].run_id == "run-b"
    assert [r.rank for r in result.ranking] == [1, 2]


def test_incompatible_runs_have_no_winner() -> None:
    baseline = _record("run-a", 1.0, key="d" * 64, dataset_sha="b" * 64)
    different = _record("run-b", 1.0, key="f" * 64, dataset_sha="e" * 64)

    result = compare_runs((baseline, different))

    assert result.compatible is False
    assert result.ranking == ()
    assert "dataset_sha256" in result.differences
    assert result.compatibility_key is None


def test_dirty_runs_are_excluded_by_default() -> None:
    clean = _record("run-a", 1.0)
    dirty = _record("run-b", 2.0, dirty=True)

    result = compare_runs((clean, dirty))

    assert [item.run_id for item in result.ranking] == ["run-a"]
    assert "run-b" in result.excluded_run_ids


def test_include_dirty_flag_admits_dirty_runs() -> None:
    clean = _record("run-a", 1.0)
    dirty = _record("run-b", 2.0, dirty=True)

    result = compare_runs((clean, dirty), include_dirty=True)

    assert result.ranking[0].run_id == "run-b"


def test_blocked_and_degraded_runs_are_excluded() -> None:
    complete = _record("run-a", 1.0)
    blocked = _record("run-b", 5.0, status="blocked")
    degraded = _record("run-c", 9.0, tracking="degraded")

    result = compare_runs((complete, blocked, degraded))

    assert [item.run_id for item in result.ranking] == ["run-a"]
    assert set(result.excluded_run_ids) == {"run-b", "run-c"}
