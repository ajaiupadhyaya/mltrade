"""Integration tests for the dashboard JSON export."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

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


def _settings(root: Path) -> Settings:
    root.mkdir(parents=True, exist_ok=True)
    return Settings(
        data_root=root,
        database_url=f"sqlite:///{root / 'operations.db'}",
    )


def test_dashboard_payload_has_expected_shape(tmp_path: Path) -> None:
    payload = build_dashboard_payload(_settings(tmp_path))

    assert payload["schema_version"] == 1
    assert set(payload) >= {
        "meta",
        "backtest",
        "portfolio",
        "risk",
        "execution",
        "quality",
        "forecast",
        "experiments",
    }

    meta = payload["meta"]
    assert meta["snapshot_id"] == "fixture-2026-06-12"
    assert meta["live_trading_enabled"] is False
    assert meta["synthetic"] is True
    assert len(meta["universe"]) == 10


def test_backtest_block_carries_real_curve_and_metrics(tmp_path: Path) -> None:
    bt = build_dashboard_payload(_settings(tmp_path))["backtest"]

    # The engineered fixture produces a strongly positive Sharpe; assert a
    # tolerant lower bound rather than a brittle exact value.
    assert bt["sharpe"] > 1.5
    assert bt["sessions"] >= 1300
    assert len(bt["cost_sensitivity"]) == 3

    curve = bt["equity_curve"]
    assert len(curve) >= 2
    assert all(set(point) == {"date", "equity"} for point in curve)
    assert curve[0]["equity"] == 1_000_000.0
    assert curve[-1]["equity"] > curve[0]["equity"]
    # Dates are strictly increasing.
    dates = [point["date"] for point in curve]
    assert dates == sorted(dates)
    assert len(set(dates)) == len(dates)


def test_risk_execution_and_portfolio_blocks(tmp_path: Path) -> None:
    payload = build_dashboard_payload(_settings(tmp_path))

    execution = payload["execution"]
    assert execution["count"] == 10
    assert execution["preview_only"] is True

    risk = payload["risk"]
    # Steady-state caps gate two notional checks in the cold-start allocation.
    assert risk["summary"]["block"] == 2
    assert risk["blocked"] is True
    assert len(risk["checks"]) == sum(risk["summary"].values())

    portfolio = payload["portfolio"]
    assert abs(portfolio["cash_weight"] + portfolio["invested_weight"] - 1.0) < 1e-6

    # The platform ships before the experiment registry is merged on this branch.
    assert payload["experiments"]["available"] is False


def test_payload_is_deterministic(tmp_path: Path) -> None:
    first = build_dashboard_payload(_settings(tmp_path / "first"))
    second = build_dashboard_payload(_settings(tmp_path / "second"))
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_write_dashboard_json_creates_parents_and_trailing_newline(
    tmp_path: Path,
) -> None:
    out = tmp_path / "nested" / "dashboard.json"
    written = write_dashboard_json(_settings(tmp_path / "data"), out)

    assert written == out
    text = out.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert json.loads(text)["meta"]["platform"] == "MLTrade"


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

    payload = _experiments_payload(settings)

    assert payload["available"] is True
    ranked = [r for r in payload["runs"] if r["rank"] != "—"]
    excluded = [r for r in payload["runs"] if r["rank"] == "—"]

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
    payload = _experiments_payload(_settings(tmp_path))
    assert payload["available"] is False
    assert payload["runs"] == []
