from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from mltrade.experiments.records import (
    ExperimentRunRecord,
    RunIdentityContext,
    RunMetrics,
    RunProvenance,
    build_compatibility_key,
    build_run_id,
)


def _provenance() -> RunProvenance:
    return RunProvenance(
        git_commit="c" * 40,
        git_dirty=False,
        git_diff_sha256=None,
        python_version="3.13.1",
        platform="test-platform",
        mltrade_version="0.1.0",
        dependencies={"pydantic": "2.11.0"},
        command=("mltrade", "experiment", "run"),
    )


def _metrics() -> RunMetrics:
    return RunMetrics(
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
    )


BASE = dict(
    run_id="run-" + "0" * 20,
    experiment_name="ridge-baseline",
    spec_sha256="a" * 64,
    dataset_sha256="b" * 64,
    dataset_snapshot_id="fixture-2026-06-12",
    compatibility_key="d" * 64,
    seed=42,
    started_at=datetime(2026, 6, 14, tzinfo=UTC),
    finished_at=datetime(2026, 6, 14, tzinfo=UTC),
    provenance=_provenance(),
    parameters={"model.alpha": 1.0},
    metrics=_metrics(),
    artifacts=(),
)


def test_run_identity_is_content_addressed() -> None:
    context = RunIdentityContext(
        spec_sha256="a" * 64,
        dataset_sha256="b" * 64,
        git_commit="c" * 40,
        git_diff_sha256=None,
    )

    assert build_run_id(context) == build_run_id(context)
    assert build_run_id(context).startswith("run-")
    assert len(build_run_id(context)) == len("run-") + 20


def test_dirty_diff_changes_run_identity() -> None:
    clean = RunIdentityContext(
        spec_sha256="a" * 64,
        dataset_sha256="b" * 64,
        git_commit="c" * 40,
        git_diff_sha256=None,
    )
    dirty = clean.model_copy(update={"git_diff_sha256": "e" * 64})

    assert build_run_id(dirty) != build_run_id(clean)


def test_terminal_status_is_closed_set() -> None:
    ExperimentRunRecord(**BASE, status="complete")

    with pytest.raises(ValidationError):
        ExperimentRunRecord(**BASE, status="running")


def test_record_round_trips_through_json() -> None:
    record = ExperimentRunRecord(**BASE, status="complete")

    assert (
        ExperimentRunRecord.model_validate_json(record.model_dump_json()) == record
    )


def test_extra_fields_are_forbidden() -> None:
    with pytest.raises(ValidationError):
        ExperimentRunRecord(**BASE, status="complete", unexpected=True)


def test_compatibility_key_is_order_independent() -> None:
    payload = {"dataset_sha256": "b" * 64, "snapshot_id": "fixture-2026-06-12"}
    reordered = dict(reversed(list(payload.items())))

    assert build_compatibility_key(payload) == build_compatibility_key(reordered)
