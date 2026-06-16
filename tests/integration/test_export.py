"""Integration tests for the dashboard JSON export (schema v2, real snapshot).

``build_dashboard_payload`` runs the full real-data research pipeline off the
committed frozen snapshot, which takes ~35s, so the heavy payload is built once
in a module-scoped fixture and shared across assertions.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from mltrade.config import Settings
from mltrade.experiments import (
    ExperimentRunRecord,
    RunMetrics,
    RunProvenance,
    RunStore,
)
from mltrade.export import (
    _experiments_payload,
    build_dashboard_payload,
    write_dashboard_json,
)


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    # Built once: the real-data research pipeline is expensive (~35s).
    return build_dashboard_payload(Settings())


def test_schema_and_meta(payload: dict[str, Any]) -> None:
    assert payload["schema_version"] == 2

    meta = payload["meta"]
    assert meta["synthetic"] is False
    assert meta["data_mode"] == "real-frozen-snapshot"
    assert meta["live_trading_enabled"] is False
    assert meta["platform"] == "MLTrade"
    assert len(meta["universe"]) == 10


def test_headline_has_institutional_kpis(payload: dict[str, Any]) -> None:
    headline = payload["headline"]
    for key in ("sharpe", "alpha_tstat", "deflated_sharpe_ratio", "pbo"):
        assert key in headline


def test_overfitting_block_bounded(payload: dict[str, Any]) -> None:
    overfitting = payload["overfitting"]
    assert overfitting is not None
    assert 0.0 <= overfitting["pbo"] <= 1.0
    assert 0.0 <= overfitting["deflated_sharpe_ratio"] <= 1.0


def test_performance_equity_curve_shape(payload: dict[str, Any]) -> None:
    curve = payload["performance"]["equity_curve"]
    assert len(curve) > 0
    for point in curve:
        assert {"date", "strategy", "benchmark"} <= set(point)


def test_risk_and_execution_blocks(payload: dict[str, Any]) -> None:
    risk = payload["risk"]
    summary = risk["summary"]
    assert sum(summary.values()) == len(risk["checks"])

    assert payload["execution"]["count"] == 10


def test_attribution_has_five_exposures(payload: dict[str, Any]) -> None:
    exposures = payload["attribution"]["exposures"]
    assert len(exposures) == 5


def test_payload_is_deterministic(payload: dict[str, Any]) -> None:
    # The fixture build plus a fresh build must be byte-identical.
    second = build_dashboard_payload(Settings())
    assert json.dumps(payload, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_write_dashboard_json_trailing_newline(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "dashboard.json"
    written = write_dashboard_json(Settings(), out)

    assert written == out
    text = out.read_text(encoding="utf-8")
    assert text.endswith("\n")
    parsed = json.loads(text)
    assert parsed["schema_version"] == 2
    assert parsed["meta"]["platform"] == "MLTrade"


# ---------------------------------------------------------------------------
# Experiment-leaderboard payload (fast; uses a throwaway registry).
# ---------------------------------------------------------------------------


def _settings(root: Path) -> Settings:
    root.mkdir(parents=True, exist_ok=True)
    return Settings(
        data_root=root,
        database_url=f"sqlite:///{root / 'operations.db'}",
    )


def _run_record(
    run_id: str,
    status: str,
    *,
    alpha: float,
    robust: float,
    sharpe: float,
    drawdown: float = -0.3,
    turnover: float = 0.4,
    key: str = "compat-key-1",
    dirty: bool = False,
) -> ExperimentRunRecord:
    return ExperimentRunRecord(
        run_id=run_id,
        experiment_name="ridge-baseline",
        status=status,  # type: ignore[arg-type]
        spec_sha256="a" * 64,
        dataset_sha256="b" * 64,
        dataset_snapshot_id="fixture-2026-06-12",
        compatibility_key=key,
        seed=42,
        started_at=datetime(2026, 6, 12, tzinfo=UTC),
        finished_at=datetime(2026, 6, 12, tzinfo=UTC),
        provenance=RunProvenance(
            git_commit="c" * 40,
            git_dirty=dirty,
            git_diff_sha256=None,
            python_version="3.13.0",
            platform="test",
            mltrade_version="0.1.0",
            dependencies={},
            command=("mltrade", "experiment", "run"),
        ),
        parameters={"model.alpha": alpha},
        metrics=RunMetrics(
            annualized_return=0.1,
            annualized_volatility=0.1,
            sharpe=sharpe,
            max_drawdown=drawdown,
            turnover=turnover,
            total_costs=1.0,
            hit_rate=0.5,
            equal_weight_return=0.1,
            cash_return=0.0,
            robust_sharpe=robust,
            window_sharpe_std=0.1,
        ),
        artifacts=(),
    )


def test_experiments_payload_ranks_real_registry_runs(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    assert settings.experiment_root is not None
    store = RunStore(settings.experiment_root)
    store.save(_run_record("run-a1", "complete", alpha=3.2, robust=1.9, sharpe=2.2))
    store.save(_run_record("run-b2", "complete", alpha=1.0, robust=1.7, sharpe=2.1))
    store.save(_run_record("run-c3", "blocked", alpha=56.0, robust=-9.0, sharpe=0.5))

    leaderboard = _experiments_payload(settings)

    assert leaderboard["available"] is True
    ranked = [r for r in leaderboard["runs"] if r["rank"] != "—"]
    excluded = [r for r in leaderboard["runs"] if r["rank"] == "—"]

    # Compatible complete runs ranked by descending robust Sharpe.
    assert [r["run_id"] for r in ranked] == ["run-a1", "run-b2"]
    assert ranked[0]["robust_sharpe"] == "1.90"
    assert ranked[0]["alpha"] == "3.2"
    assert ranked[0]["sharpe"] == "2.20"
    # Blocked run stays visible but unranked.
    assert len(excluded) == 1
    assert excluded[0]["run_id"] == "run-c3"
    assert excluded[0]["status"] == "blocked"


def test_experiments_payload_empty_when_no_runs(tmp_path: Path) -> None:
    leaderboard = _experiments_payload(_settings(tmp_path))
    assert leaderboard["available"] is False
    assert leaderboard["runs"] == []
