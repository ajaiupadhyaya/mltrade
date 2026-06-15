"""Integration test for the research workflow (run_research).

run_research operates on an already-published snapshot: no fixture generation,
no broker, no DB writes. We publish a snapshot via run_demo, load its manifest,
then exercise the research pipeline.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from mltrade.backtest import BacktestConfig, run_backtest
from mltrade.config import Environment, Settings
from mltrade.data.publication import DailyBarPublisher
from mltrade.features.pipeline import build_feature_rows
from mltrade.models import RidgeForecastConfig
from mltrade.models.walk_forward import generate_forecast_batch
from mltrade.portfolio.targets import PortfolioLimits
from mltrade.storage.snapshots import SnapshotStore
from mltrade.workflows.demo import run_demo
from mltrade.workflows.research import run_research

_CLOCK = datetime(2026, 6, 13, tzinfo=UTC)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        environment=Environment.TEST,
        data_root=tmp_path,
        database_url=f"sqlite+pysqlite:///{tmp_path / 'ops.db'}",
    )


def test_run_research_on_published_snapshot(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    # Publish a snapshot via the demo, then load its manifest.
    demo = run_demo(settings, clock=_CLOCK)
    store = SnapshotStore(settings.data_root)
    manifest = store.load_manifest("daily_bars", demo.snapshot_id)

    result = run_research(settings, manifest)

    assert result.snapshot_id == demo.snapshot_id
    assert result.quality.blocked is False
    assert result.backtest.sessions > 250
    assert result.forecast_batch.forecasts
    # Research shares the production target code; result is a valid target.
    assert result.target.blocked is False


def test_run_research_is_deterministic(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    demo = run_demo(settings, clock=_CLOCK)
    manifest = SnapshotStore(settings.data_root).load_manifest(
        "daily_bars", demo.snapshot_id
    )

    first = run_research(settings, manifest)
    second = run_research(settings, manifest)

    assert first.backtest == second.backtest
    assert first.target == second.target


def test_run_research_uses_one_forecast_config_throughout(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    demo = run_demo(settings, clock=_CLOCK)
    manifest = SnapshotStore(settings.data_root).load_manifest(
        "daily_bars", demo.snapshot_id
    )
    config = BacktestConfig(
        forecast=RidgeForecastConfig(alpha=100.0, fit_intercept=False),
        retrain_every_sessions=7,
        cost_sensitivity_bps=(Decimal("1"), Decimal("3")),
        evaluation_window_sessions=126,
    )

    configured = run_research(settings, manifest, backtest_config=config)
    bars = DailyBarPublisher(SnapshotStore(settings.data_root)).load_verified(
        manifest
    )
    limits = PortfolioLimits(
        maximum_position_weight=settings.maximum_position_weight,
        minimum_cash_weight=settings.minimum_cash_weight,
        target_annual_volatility=settings.target_annual_volatility,
    )
    direct_backtest = run_backtest(bars, limits=limits, config=config)
    feature_rows = build_feature_rows(bars, manifest.snapshot_id)
    direct_forecast = generate_forecast_batch(
        feature_rows,
        configured.forecast_batch.decision_session,
        config=config.forecast,
    )

    assert configured.backtest == direct_backtest
    assert configured.forecast_batch == direct_forecast
