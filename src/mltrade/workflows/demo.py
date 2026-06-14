"""Offline demo workflow — no network access required.

``run_demo`` orchestrates the full pipeline using deterministic fixture bars:

1. Generate bars via :class:`~mltrade.data.fixtures.DeterministicBarSource`.
2. Validate quality via :func:`~mltrade.data.quality.validate_daily_bars`.
3. Publish to Parquet via :class:`~mltrade.data.publication.DailyBarPublisher`.
4. Reload and verify via ``publisher.load_verified(manifest)``.
5. Build feature rows via :func:`~mltrade.features.pipeline.build_feature_rows`.
6. Run backtest via :func:`~mltrade.backtest.engine.run_backtest`.
7. Generate forecast batch via
   :func:`~mltrade.models.walk_forward.generate_forecast_batch`.
8. Build portfolio target via
   :func:`~mltrade.portfolio.optimizer.build_target`.
9. Build a :class:`~mltrade.execution.simulated.SimulatedBroker` and call
   :meth:`~mltrade.execution.service.ExecutionService.preview` with real
   provenance so the risk gates actually fire.
10. Persist all evidence to the operations DB in per-stage session scopes.

All steps are deterministic given identical ``clock`` and ``settings``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal

from sqlalchemy import Engine

from mltrade.backtest.engine import run_backtest
from mltrade.backtest.reporting import BacktestResult
from mltrade.calendar import XNYSCalendar
from mltrade.config import Settings
from mltrade.data.fixtures import DeterministicBarSource
from mltrade.data.publication import DailyBarPublisher
from mltrade.data.quality import DataQualityReport, validate_daily_bars
from mltrade.execution.broker import BrokerAccount
from mltrade.execution.reconciliation import InternalState
from mltrade.execution.service import ExecutionService, Preview
from mltrade.execution.simulated import SimulatedBroker
from mltrade.features.definitions import FEATURE_VERSION
from mltrade.features.pipeline import build_feature_rows
from mltrade.models.forecasts import MODEL_VERSION, ForecastBatch
from mltrade.models.walk_forward import generate_forecast_batch
from mltrade.operations.database import build_engine, session_scope
from mltrade.operations.repositories import OperationsRepository
from mltrade.portfolio.optimizer import build_target
from mltrade.portfolio.targets import OptimizationResult, PortfolioLimits
from mltrade.storage.manifests import DatasetManifest
from mltrade.storage.snapshots import SnapshotStore
from mltrade.universe import MVP_UNIVERSE

_CALENDAR = XNYSCalendar()

_FIXTURE_START = date(2019, 1, 2)
_DEFAULT_CLOCK = datetime(2026, 6, 13, tzinfo=UTC)


@dataclass(frozen=True)
class DemoResult:
    """Immutable result bundle from :func:`run_demo`."""

    quality: DataQualityReport
    backtest: BacktestResult
    target: OptimizationResult
    preview: Preview
    snapshot_id: str
    forecast_batch: ForecastBatch


def _build_paper_account(reference_equity: Decimal) -> BrokerAccount:
    return BrokerAccount(
        id="demo-paper-account",
        status="ACTIVE",
        cash=reference_equity,
        equity=reference_equity,
        account_blocked=False,
        trading_blocked=False,
        pattern_day_trader=False,
    )


def _build_limits(settings: Settings) -> PortfolioLimits:
    return PortfolioLimits(
        maximum_position_weight=settings.maximum_position_weight,
        minimum_cash_weight=settings.minimum_cash_weight,
        target_annual_volatility=settings.target_annual_volatility,
    )


def _target_positions_from_weights(
    weights: dict[str, Decimal],
    equity: Decimal,
    prices: dict[str, Decimal],
) -> dict[str, int]:
    """Convert optimizer weights to integer share quantities."""
    result: dict[str, int] = {}
    for symbol, weight in weights.items():
        price = prices.get(symbol)
        if price is None or price <= Decimal("0"):
            continue
        shares = int((equity * weight) / price)
        if shares > 0:
            result[symbol] = shares
    return result


def run_demo(
    settings: Settings,
    *,
    clock: datetime | None = None,
) -> DemoResult:
    """Run the full offline demo workflow.

    Parameters
    ----------
    settings:
        Runtime configuration (limits, data root, database URL).
    clock:
        UTC datetime used to determine the last completed XNYS session.
        Defaults to ``datetime(2026, 6, 13, tzinfo=UTC)``.

    Returns
    -------
    DemoResult
        Immutable bundle with quality, backtest, target, preview, snapshot_id,
        and forecast_batch.

    Raises
    ------
    ValueError
        If the data quality gate blocks (``quality.blocked is True``).
    """
    effective_clock: datetime = clock if clock is not None else _DEFAULT_CLOCK
    last_session: date = _CALENDAR.last_completed_session(effective_clock)
    snapshot_id = f"fixture-{last_session.isoformat()}"
    correlation_id = snapshot_id

    engine: Engine = build_engine(settings.database_url)

    # -----------------------------------------------------------------------
    # Step 1: Generate fixture bars
    # -----------------------------------------------------------------------
    ingested_at = effective_clock
    source = DeterministicBarSource(seed=42)
    bars = source.fetch(MVP_UNIVERSE, _FIXTURE_START, last_session, ingested_at)

    # -----------------------------------------------------------------------
    # Step 2: Validate quality
    # -----------------------------------------------------------------------
    quality = validate_daily_bars(
        bars,
        universe=MVP_UNIVERSE,
        expected_last_session=last_session,
    )
    if quality.blocked:
        raise ValueError(
            f"Data quality gate blocked for snapshot '{snapshot_id}': "
            f"{quality.issues}"
        )

    # Persist quality report
    with session_scope(engine) as session:
        repo = OperationsRepository(session)
        repo.save_quality_report(
            payload=json.loads(quality.model_dump_json()),
            correlation_id=correlation_id,
            decision_session=last_session.isoformat(),
        )

    # -----------------------------------------------------------------------
    # Step 3: Publish to Parquet snapshot
    # -----------------------------------------------------------------------
    store = SnapshotStore(settings.data_root)
    publisher = DailyBarPublisher(store)

    # If the snapshot already exists on disk from a previous run, load it
    # instead of re-publishing (idempotent demo replay).
    manifest: DatasetManifest
    try:
        published = publisher.publish(
            bars=bars,
            quality=quality,
            snapshot_id=snapshot_id,
            created_at=ingested_at,
        )
        manifest = published.manifest
    except FileExistsError:
        manifest = store.load_manifest("daily_bars", snapshot_id)

    # Persist snapshot evidence
    with session_scope(engine) as session:
        repo = OperationsRepository(session)
        repo.save_snapshot(
            payload={
                "snapshot_id": snapshot_id,
                "dataset": manifest.dataset,
                "row_count": manifest.row_count,
                "content_sha256": manifest.content_sha256,
                "last_session": last_session.isoformat(),
            },
            correlation_id=correlation_id,
            decision_session=last_session.isoformat(),
        )

    # -----------------------------------------------------------------------
    # Step 4: Reload and verify bars from disk
    # -----------------------------------------------------------------------
    verified_bars = publisher.load_verified(manifest)

    # -----------------------------------------------------------------------
    # Step 5: Build feature rows
    # -----------------------------------------------------------------------
    feature_rows = build_feature_rows(verified_bars, snapshot_id)

    # -----------------------------------------------------------------------
    # Step 6: Run backtest
    # -----------------------------------------------------------------------
    limits = _build_limits(settings)
    backtest = run_backtest(verified_bars, limits=limits)

    with session_scope(engine) as session:
        repo = OperationsRepository(session)
        repo.save_backtest_run(
            payload=json.loads(backtest.model_dump_json()),
            correlation_id=correlation_id,
            decision_session=last_session.isoformat(),
        )

    # -----------------------------------------------------------------------
    # Step 7: Generate forecast batch at the decision session
    # -----------------------------------------------------------------------
    forecast_batch = generate_forecast_batch(feature_rows, last_session)

    with session_scope(engine) as session:
        repo = OperationsRepository(session)
        repo.save_forecast_batch(
            payload=json.loads(forecast_batch.model_dump_json()),
            correlation_id=correlation_id,
            decision_session=last_session.isoformat(),
        )

    # -----------------------------------------------------------------------
    # Step 8: Build portfolio target
    # -----------------------------------------------------------------------
    forecasts_map: dict[str, float] = {
        fc.symbol: fc.predicted_forward_return
        for fc in forecast_batch.forecasts
    }

    # Extract trailing vol from feature rows at the decision session
    vol_map: dict[str, float] = {}
    for row in feature_rows:
        if row.decision_session == last_session and not row.missing:
            vol_map[row.symbol] = row.realized_volatility_21

    target = build_target(
        forecasts=forecasts_map,
        trailing_volatility=vol_map,
        limits=limits,
    )

    with session_scope(engine) as session:
        repo = OperationsRepository(session)
        repo.save_portfolio_target(
            payload=json.loads(target.model_dump_json()),
            correlation_id=correlation_id,
            decision_session=last_session.isoformat(),
        )

    # -----------------------------------------------------------------------
    # Step 9: Build execution preview with real provenance
    # -----------------------------------------------------------------------
    broker_account = _build_paper_account(settings.reference_equity)
    broker = SimulatedBroker(broker_account)

    # Build approximate prices from the last session's bars
    prices: dict[str, Decimal] = {}
    for bar in verified_bars:
        if bar.session == last_session:
            prices[bar.instrument.symbol] = bar.close

    # Convert target weights to share quantities
    target_pos = _target_positions_from_weights(
        target.weights,
        settings.reference_equity,
        prices,
    )

    # Fresh account → no internal positions or open orders
    internal_state = InternalState(
        cash=settings.reference_equity,
        positions={},
        open_client_order_ids=(),
    )

    svc = ExecutionService(broker)
    preview = svc.preview(
        target_positions=target_pos,
        internal_state=internal_state,
        settings=settings,
        strategy_version=MODEL_VERSION,
        decision_session=last_session,
        environment="local",
        prices=prices,
        # Real provenance — these wire the actual risk gates
        snapshot_blocked=quality.blocked,
        snapshot_last_session=last_session,
        expected_last_session=last_session,
        expected_decision_session=last_session,
        model_version=MODEL_VERSION,
        feature_version=FEATURE_VERSION,
        expected_model_version=MODEL_VERSION,
        expected_feature_version=FEATURE_VERSION,
    )

    # Persist risk report and preview
    with session_scope(engine) as session:
        repo = OperationsRepository(session)
        repo.save_risk_report(
            payload=json.loads(preview.risk_report.model_dump_json()),
            correlation_id=correlation_id,
            decision_session=last_session.isoformat(),
        )
        repo.save_preview(
            preview,
            correlation_id=correlation_id,
            decision_session=last_session.isoformat(),
        )
        for intent in preview.intents:
            repo.save_intent(
                intent,
                correlation_id=correlation_id,
                decision_session=last_session.isoformat(),
            )

    return DemoResult(
        quality=quality,
        backtest=backtest,
        target=target,
        preview=preview,
        snapshot_id=snapshot_id,
        forecast_batch=forecast_batch,
    )
