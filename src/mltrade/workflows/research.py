"""Research workflow — operates on an existing verified snapshot.

``run_research`` accepts a pre-existing
:class:`~mltrade.storage.manifests.DatasetManifest`, loads and verifies the
bars, then runs the full feature → backtest → forecast → target pipeline.

No fixture generation, no broker interaction, and no DB writes.  This workflow
is suitable for interactive research, parameter sweeps, and CI reproducibility
checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from mltrade.backtest.engine import BacktestConfig, run_backtest
from mltrade.backtest.reporting import BacktestResult
from mltrade.config import Settings
from mltrade.data.publication import DailyBarPublisher
from mltrade.data.quality import DataQualityReport, validate_daily_bars
from mltrade.features.pipeline import build_feature_rows
from mltrade.models.forecasts import ForecastBatch
from mltrade.models.walk_forward import generate_forecast_batch
from mltrade.portfolio.optimizer import build_target
from mltrade.portfolio.targets import OptimizationResult, PortfolioLimits
from mltrade.storage.manifests import DatasetManifest
from mltrade.storage.snapshots import SnapshotStore
from mltrade.universe import MVP_UNIVERSE


@dataclass(frozen=True)
class ResearchResult:
    """Immutable result bundle from :func:`run_research`."""

    quality: DataQualityReport
    backtest: BacktestResult
    target: OptimizationResult
    forecast_batch: ForecastBatch
    snapshot_id: str


def _build_limits(settings: Settings) -> PortfolioLimits:
    return PortfolioLimits(
        maximum_position_weight=settings.maximum_position_weight,
        minimum_cash_weight=settings.minimum_cash_weight,
        target_annual_volatility=settings.target_annual_volatility,
    )


def run_research(
    settings: Settings,
    manifest: DatasetManifest,
    *,
    backtest_config: BacktestConfig | None = None,
) -> ResearchResult:
    """Run the research pipeline on an existing verified snapshot.

    Parameters
    ----------
    settings:
        Runtime configuration (limits, data root).
    manifest:
        The :class:`~mltrade.storage.manifests.DatasetManifest` describing the
        snapshot to load.  Must have been previously written by
        :class:`~mltrade.data.publication.DailyBarPublisher`.
    backtest_config:
        Optional experiment boundaries shared by the backtest and final
        forecast batch.

    Returns
    -------
    ResearchResult
        Immutable bundle with quality, backtest, target, and forecast_batch.
    """
    store = SnapshotStore(settings.data_root)
    publisher = DailyBarPublisher(store)
    bars = publisher.load_verified(manifest)

    # Derive the decision session from the snapshot metadata
    last_session_str = manifest.metadata.get("last_session")
    decision_session: date
    if last_session_str is None:
        # Fall back to max session in bars
        decision_session = max(bar.session for bar in bars)
    else:
        decision_session = date.fromisoformat(last_session_str)

    # Quality check (non-blocking: we always return the report)
    quality = validate_daily_bars(
        bars,
        universe=MVP_UNIVERSE,
        expected_last_session=decision_session,
    )

    limits = _build_limits(settings)

    # Feature rows
    feature_rows = build_feature_rows(bars, manifest.snapshot_id)

    # Backtest
    backtest = run_backtest(
        bars,
        limits=limits,
        config=backtest_config,
    )

    # Forecast batch
    forecast_config = (
        backtest_config.forecast
        if backtest_config is not None
        else None
    )
    forecast_batch = generate_forecast_batch(
        feature_rows,
        decision_session,
        config=forecast_config,
    )

    # Portfolio target
    forecasts_map: dict[str, float] = {
        fc.symbol: fc.predicted_forward_return
        for fc in forecast_batch.forecasts
    }
    vol_map: dict[str, float] = {}
    for row in feature_rows:
        if row.decision_session == decision_session and not row.missing:
            vol_map[row.symbol] = row.realized_volatility_21

    target = build_target(
        forecasts=forecasts_map,
        trailing_volatility=vol_map,
        limits=limits,
    )

    return ResearchResult(
        quality=quality,
        backtest=backtest,
        target=target,
        forecast_batch=forecast_batch,
        snapshot_id=manifest.snapshot_id,
    )
