"""Real-market-data research pipeline (offline, deterministic).

Runs the full research slice on the frozen point-in-time real-data snapshot:
walk-forward backtest, the analytics layer (performance, benchmark, attribution),
and the *current* decision (forecast → portfolio target → pre-trade risk preview)
at the last observed session.  Unlike :func:`mltrade.workflows.demo.run_demo`
this path reads real, split/dividend-adjusted bars from a committed Parquet panel
and does not persist to the operations database — it exists to feed the dashboard
export with genuinely out-of-sample numbers.

Everything is deterministic: the snapshot is frozen, the engine is seedless and
sorted, so identical inputs always produce identical output.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from mltrade.analytics.returns import align_by_session, equity_to_returns
from mltrade.backtest.engine import compute_equity_curve, run_backtest
from mltrade.backtest.reporting import BacktestResult
from mltrade.config import Settings
from mltrade.data.bars import DailyBar
from mltrade.data.snapshot import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_PANEL_PATH,
    SnapshotBarSource,
    load_manifest,
)
from mltrade.execution.broker import BrokerAccount
from mltrade.execution.reconciliation import InternalState
from mltrade.execution.service import ExecutionService, Preview
from mltrade.execution.simulated import SimulatedBroker
from mltrade.features.definitions import FEATURE_VERSION
from mltrade.features.pipeline import build_feature_rows
from mltrade.models.forecasts import MODEL_VERSION, ForecastBatch
from mltrade.models.walk_forward import generate_forecast_batch
from mltrade.portfolio.optimizer import build_target
from mltrade.portfolio.targets import OptimizationResult, PortfolioLimits
from mltrade.universe import MVP_UNIVERSE

_BENCHMARK = "SPY"
_INITIAL_EQUITY = 1_000_000.0


@dataclass(frozen=True)
class RealResearchResult:
    """Immutable bundle from :func:`run_real_research`."""

    manifest: dict[str, Any]
    last_session: date
    snapshot_id: str
    bars: tuple[DailyBar, ...]
    backtest: BacktestResult
    equity_curve: list[tuple[date, float]]
    aligned_sessions: list[date]
    strategy_returns: list[float]
    benchmark_returns: list[float]
    factor_returns: dict[str, list[float]]
    forecast_batch: ForecastBatch
    target: OptimizationResult
    preview: Preview
    prices: dict[str, Decimal]


def _limits(settings: Settings) -> PortfolioLimits:
    return PortfolioLimits(
        maximum_position_weight=settings.maximum_position_weight,
        minimum_cash_weight=settings.minimum_cash_weight,
        target_annual_volatility=settings.target_annual_volatility,
    )


def _symbol_close_returns(
    bars: tuple[DailyBar, ...],
) -> dict[str, tuple[list[date], list[float]]]:
    """Per-symbol close-to-close simple returns keyed by session (ascending)."""
    closes: dict[str, dict[date, float]] = defaultdict(dict)
    for bar in bars:
        closes[bar.instrument.symbol][bar.session] = float(bar.close)
    out: dict[str, tuple[list[date], list[float]]] = {}
    for symbol, by_session in closes.items():
        items = sorted(by_session.items())
        sessions = [s for s, _ in items][1:]
        values = [items[i][1] / items[i - 1][1] - 1.0 for i in range(1, len(items))]
        out[symbol] = (sessions, values)
    return out


def _target_positions(
    weights: dict[str, Decimal],
    equity: Decimal,
    prices: dict[str, Decimal],
) -> dict[str, int]:
    result: dict[str, int] = {}
    for symbol, weight in weights.items():
        price = prices.get(symbol)
        if price is None or price <= Decimal("0"):
            continue
        shares = int((equity * weight) / price)
        if shares > 0:
            result[symbol] = shares
    return result


def run_real_research(
    settings: Settings,
    *,
    panel_path: Path = DEFAULT_PANEL_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> RealResearchResult:
    """Run the real-data backtest + current decision off the frozen snapshot."""
    manifest = load_manifest(manifest_path)
    start = date.fromisoformat(manifest["start_session"])
    last_session = date.fromisoformat(manifest["end_session"])
    snapshot_id = f"real-{manifest['as_of']}"
    ingested_at = datetime(
        last_session.year, last_session.month, last_session.day, tzinfo=UTC
    )

    source = SnapshotBarSource(panel_path)
    bars = source.fetch(MVP_UNIVERSE, start, last_session, ingested_at)

    limits = _limits(settings)
    backtest = run_backtest(bars, limits=limits)
    equity_curve = compute_equity_curve(bars, limits=limits)

    # Strategy per-session returns aligned to the benchmark and factor returns.
    strat_sessions = [d for d, _ in equity_curve]
    strat_returns_full = equity_to_returns(
        [eq for _, eq in equity_curve], initial=_INITIAL_EQUITY
    )
    sym_returns = _symbol_close_returns(bars)
    bench_sessions, bench_values = sym_returns[_BENCHMARK]
    aligned_sessions, strat_returns, benchmark_returns = align_by_session(
        strat_sessions, strat_returns_full, bench_sessions, bench_values
    )
    typed_sessions: list[date] = [s for s in aligned_sessions]  # narrow for callers

    factor_returns: dict[str, list[float]] = {}
    for symbol, (f_sessions, f_values) in sym_returns.items():
        _, _, aligned = align_by_session(
            typed_sessions, strat_returns, f_sessions, f_values
        )
        factor_returns[symbol] = aligned

    # ------------------------------------------------------------------
    # Current decision at the last observed session (forecast → target → risk).
    # ------------------------------------------------------------------
    feature_rows = build_feature_rows(bars, snapshot_id)
    forecast_batch = generate_forecast_batch(feature_rows, last_session)
    forecasts_map = {
        fc.symbol: fc.predicted_forward_return for fc in forecast_batch.forecasts
    }
    vol_map: dict[str, float] = {
        row.symbol: row.realized_volatility_21
        for row in feature_rows
        if row.decision_session == last_session and not row.missing
    }
    target = build_target(
        forecasts=forecasts_map, trailing_volatility=vol_map, limits=limits
    )

    prices: dict[str, Decimal] = {
        bar.instrument.symbol: bar.close
        for bar in bars
        if bar.session == last_session
    }
    broker = SimulatedBroker(
        BrokerAccount(
            id="research-paper-account",
            status="ACTIVE",
            cash=settings.reference_equity,
            equity=settings.reference_equity,
            account_blocked=False,
            trading_blocked=False,
            pattern_day_trader=False,
        )
    )
    preview = ExecutionService(broker).preview(
        target_positions=_target_positions(
            target.weights, settings.reference_equity, prices
        ),
        internal_state=InternalState(
            cash=settings.reference_equity, positions={}, open_client_order_ids=()
        ),
        settings=settings,
        strategy_version=MODEL_VERSION,
        decision_session=last_session,
        environment="local",
        prices=prices,
        snapshot_blocked=False,
        snapshot_last_session=last_session,
        expected_last_session=last_session,
        expected_decision_session=last_session,
        model_version=MODEL_VERSION,
        feature_version=FEATURE_VERSION,
        expected_model_version=MODEL_VERSION,
        expected_feature_version=FEATURE_VERSION,
    )

    return RealResearchResult(
        manifest=manifest,
        last_session=last_session,
        snapshot_id=snapshot_id,
        bars=bars,
        backtest=backtest,
        equity_curve=equity_curve,
        aligned_sessions=typed_sessions,
        strategy_returns=strat_returns,
        benchmark_returns=benchmark_returns,
        factor_returns=factor_returns,
        forecast_batch=forecast_batch,
        target=target,
        preview=preview,
        prices=prices,
    )
