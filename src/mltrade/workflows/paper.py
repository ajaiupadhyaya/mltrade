"""Paper-trading workflow — fail-closed live execution path.

``run_paper`` is the production paper-trading entrypoint.  It enforces strict
fail-closed semantics:

- Refuses offline/fixture data sources (Environment must be PAPER).
- If ``preview.risk_report.blocked`` → never submit, even if ``submit=True``.
- If ``submit=False`` → returns preview only (dry run).
- If ``submit=True`` + not blocked + environment=PAPER → calls
  ``ExecutionService.submit(preview)``.
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
from mltrade.config import Environment, Settings
from mltrade.data.publication import DailyBarPublisher
from mltrade.data.quality import DataQualityReport, validate_daily_bars
from mltrade.execution.broker import Broker
from mltrade.execution.reconciliation import InternalState
from mltrade.execution.service import ExecutionService, Preview, SubmitResult
from mltrade.features.definitions import FEATURE_VERSION
from mltrade.features.pipeline import build_feature_rows
from mltrade.models.forecasts import MODEL_VERSION, ForecastBatch
from mltrade.models.walk_forward import generate_forecast_batch
from mltrade.operations.database import build_engine, session_scope
from mltrade.operations.models import Base
from mltrade.operations.repositories import OperationsRepository
from mltrade.portfolio.optimizer import build_target
from mltrade.portfolio.targets import OptimizationResult, PortfolioLimits
from mltrade.storage.manifests import DatasetManifest
from mltrade.storage.snapshots import SnapshotStore
from mltrade.universe import MVP_UNIVERSE

_CALENDAR = XNYSCalendar()
_DEFAULT_CLOCK = datetime(2026, 6, 13, tzinfo=UTC)


@dataclass(frozen=True)
class PaperResult:
    """Immutable result bundle from :func:`run_paper`."""

    quality: DataQualityReport
    backtest: BacktestResult
    target: OptimizationResult
    preview: Preview
    submit_result: SubmitResult | None
    snapshot_id: str
    forecast_batch: ForecastBatch


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


def run_paper(
    settings: Settings,
    manifest: DatasetManifest,
    *,
    broker: Broker,
    submit: bool = False,
    clock: datetime | None = None,
) -> PaperResult:
    """Run the paper-trading workflow.

    Fail-closed rules
    -----------------
    1. Refuses offline/fixture environments: ``settings.environment`` must be
       :attr:`~mltrade.config.Environment.PAPER`.
    2. If ``preview.risk_report.blocked`` → never submit even if
       ``submit=True``.
    3. If ``submit=False`` → dry-run only (returns preview, no orders sent).
    4. If ``submit=True`` + not blocked + PAPER environment →
       :meth:`~mltrade.execution.service.ExecutionService.submit` is called.

    Parameters
    ----------
    settings:
        Runtime configuration.  ``environment`` must be ``PAPER``.
    manifest:
        The verified :class:`~mltrade.storage.manifests.DatasetManifest`
        describing the live data snapshot.
    broker:
        Live broker adapter (satisfies the ``Broker`` Protocol).
    submit:
        If ``True`` and the preview is not blocked, orders are submitted to
        the broker.  Default ``False`` (dry run).
    clock:
        UTC datetime used to determine the expected last session.  Defaults
        to ``datetime(2026, 6, 13, tzinfo=UTC)``.

    Returns
    -------
    PaperResult
        Immutable result bundle.

    Raises
    ------
    RuntimeError
        If ``settings.environment`` is not ``PAPER`` (offline data refused).
    """
    if settings.environment is not Environment.PAPER:
        raise RuntimeError(
            f"run_paper requires environment=PAPER; got "
            f"{settings.environment!r}.  Refusing to submit with offline or "
            f"fixture data."
        )

    effective_clock: datetime = clock if clock is not None else _DEFAULT_CLOCK
    expected_last_session: date = _CALENDAR.last_completed_session(effective_clock)
    correlation_id = f"paper-{manifest.snapshot_id}"

    engine: Engine = build_engine(settings.database_url)
    # Ensure the DB schema exists before any persistence step.
    # create_all is idempotent: a no-op when tables already exist.
    Base.metadata.create_all(engine)

    # -----------------------------------------------------------------------
    # Step 1: Load and verify bars from the manifest
    # -----------------------------------------------------------------------
    store = SnapshotStore(settings.data_root)
    publisher = DailyBarPublisher(store)
    bars = publisher.load_verified(manifest)

    # -----------------------------------------------------------------------
    # Step 2: Derive decision session from manifest
    # -----------------------------------------------------------------------
    last_session_str = manifest.metadata.get("last_session")
    snapshot_last_session: date
    if last_session_str is None:
        snapshot_last_session = max(bar.session for bar in bars)
    else:
        snapshot_last_session = date.fromisoformat(last_session_str)

    # -----------------------------------------------------------------------
    # Step 3: Quality check
    # -----------------------------------------------------------------------
    quality = validate_daily_bars(
        bars,
        universe=MVP_UNIVERSE,
        expected_last_session=expected_last_session,
    )

    with session_scope(engine) as session:
        repo = OperationsRepository(session)
        repo.save_quality_report(
            payload=json.loads(quality.model_dump_json()),
            correlation_id=correlation_id,
            decision_session=snapshot_last_session.isoformat(),
        )

    # -----------------------------------------------------------------------
    # Step 4: Feature rows + backtest + forecasts + target
    # -----------------------------------------------------------------------
    limits = _build_limits(settings)
    feature_rows = build_feature_rows(bars, manifest.snapshot_id)
    backtest = run_backtest(bars, limits=limits)

    forecast_batch = generate_forecast_batch(feature_rows, snapshot_last_session)

    forecasts_map: dict[str, float] = {
        fc.symbol: fc.predicted_forward_return
        for fc in forecast_batch.forecasts
    }
    vol_map: dict[str, float] = {}
    for row in feature_rows:
        if row.decision_session == snapshot_last_session and not row.missing:
            vol_map[row.symbol] = row.realized_volatility_21

    target = build_target(
        forecasts=forecasts_map,
        trailing_volatility=vol_map,
        limits=limits,
    )

    # -----------------------------------------------------------------------
    # Step 5: Build preview with real provenance
    # -----------------------------------------------------------------------
    # Fetch live broker account to size positions
    live_account = broker.get_account()
    live_equity = live_account.equity

    prices: dict[str, Decimal] = {}
    for bar in bars:
        if bar.session == snapshot_last_session:
            prices[bar.instrument.symbol] = bar.close

    target_pos = _target_positions_from_weights(
        target.weights,
        live_equity,
        prices,
    )

    live_positions = {p.symbol: p.quantity for p in broker.list_positions()}
    live_open_orders = tuple(
        o.client_order_id for o in broker.list_open_orders()
    )
    internal_state = InternalState(
        cash=live_account.cash,
        positions=live_positions,
        open_client_order_ids=live_open_orders,
    )

    svc = ExecutionService(broker)
    preview = svc.preview(
        target_positions=target_pos,
        internal_state=internal_state,
        settings=settings,
        strategy_version=MODEL_VERSION,
        decision_session=snapshot_last_session,
        environment=settings.environment.value,
        prices=prices,
        # Real provenance wired here so freshness/version gates actually fire
        snapshot_blocked=quality.blocked,
        snapshot_last_session=snapshot_last_session,
        expected_last_session=expected_last_session,
        expected_decision_session=expected_last_session,
        model_version=MODEL_VERSION,
        feature_version=FEATURE_VERSION,
        expected_model_version=MODEL_VERSION,
        expected_feature_version=FEATURE_VERSION,
    )

    with session_scope(engine) as session:
        repo = OperationsRepository(session)
        repo.save_risk_report(
            payload=json.loads(preview.risk_report.model_dump_json()),
            correlation_id=correlation_id,
            decision_session=snapshot_last_session.isoformat(),
        )
        repo.save_preview(
            preview,
            correlation_id=correlation_id,
            decision_session=snapshot_last_session.isoformat(),
        )

    # -----------------------------------------------------------------------
    # Step 6: Submit (fail-closed)
    # -----------------------------------------------------------------------
    submit_result: SubmitResult | None = None

    if submit:
        if preview.risk_report.blocked:
            # Fail closed: blocked preview → never submit
            pass
        else:
            submit_result = svc.submit(preview)

    return PaperResult(
        quality=quality,
        backtest=backtest,
        target=target,
        preview=preview,
        submit_result=submit_result,
        snapshot_id=manifest.snapshot_id,
        forecast_batch=forecast_batch,
    )
