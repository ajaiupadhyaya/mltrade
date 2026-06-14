"""Unit tests for OperationsRepository (Task 13).

Coverage
--------
1.  test_repository_persists_preview_and_checks  — save + load round-trip equality
2.  test_execution_intent_client_id_is_unique    — idempotent save: count stays 1
3.  test_save_snapshot                           — DatasetSnapshot persists
4.  test_save_quality_report                     — QualityReport persists
5.  test_save_forecast_batch                     — ForecastBatch persists
6.  test_save_backtest_run                       — BacktestRun persists
7.  test_save_portfolio_target                   — PortfolioTarget persists
8.  test_save_risk_report                        — RiskReportRow persists
9.  test_save_broker_order                       — BrokerOrder persists
10. test_save_reconciliation_run                 — ReconciliationRun persists
11. test_two_saves_in_one_session_both_persist   — two writes visible before commit
12. test_error_rolls_back                        — session_scope rollback on error
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from mltrade.execution.broker import OrderSide
from mltrade.execution.intents import ExecutionIntent, build_intent
from mltrade.execution.reconciliation import ReconciliationResult
from mltrade.execution.service import Preview
from mltrade.operations.database import build_engine, session_scope
from mltrade.operations.models import (
    BacktestRun,
    Base,
    BrokerOrder,
    DatasetSnapshot,
    ForecastBatch,
    PortfolioTarget,
    QualityReport,
    ReconciliationRun,
    RiskReportRow,
)
from mltrade.operations.repositories import OperationsRepository
from mltrade.risk.checks import CheckStatus, RiskCheck, RiskReport

# ---------------------------------------------------------------------------
# Session fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def sqlite_session() -> Session:  # type: ignore[return]
    """In-memory SQLite session for fast isolated unit tests.

    Yields a Session with an open transaction so that flush() works and all
    rows are visible within the same session, mirroring how session_scope is
    used in production (but without commit, to keep tests independent).
    """
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        with session.begin():
            yield session
    engine.dispose()


# ---------------------------------------------------------------------------
# Helpers: build realistic domain objects
# ---------------------------------------------------------------------------

_SESSION_DATE = date(2026, 6, 12)
_CORRELATION_ID = "test-run-001"
_DECISION_SESSION = _SESSION_DATE.isoformat()


def _make_intent(
    symbol: str = "SPY",
    side: OrderSide = OrderSide.BUY,
) -> ExecutionIntent:
    return build_intent(
        environment="paper",
        strategy_version="ridge-trend-v1",
        decision_session=_SESSION_DATE,
        symbol=symbol,
        side=side,
        target_quantity=Decimal("10"),
    )


def _make_preview(blocked: bool = False) -> Preview:
    """Build a minimal Preview without hitting a live broker."""
    intent = _make_intent()
    risk_status = CheckStatus.BLOCK if blocked else CheckStatus.PASS
    risk_report = RiskReport(
        checks=(
            RiskCheck(
                code="live_trading_disabled",
                status=risk_status,
                message=(
                    "live trading is disabled"
                    if blocked
                    else "live trading disabled — paper only"
                ),
            ),
        )
    )
    reconciliation = ReconciliationResult(differences=())
    return Preview(
        intents=(intent,),
        reconciliation=reconciliation,
        risk_report=risk_report,
    )


# ---------------------------------------------------------------------------
# 1. Preview round-trip
# ---------------------------------------------------------------------------


def test_repository_persists_preview_and_checks(sqlite_session: Session) -> None:
    repo = OperationsRepository(sqlite_session)
    original = _make_preview()

    preview_id = repo.save_preview(
        original,
        correlation_id=_CORRELATION_ID,
        decision_session=_DECISION_SESSION,
    )

    assert isinstance(preview_id, UUID)

    loaded = repo.load_preview(preview_id)
    assert loaded == original


def test_load_preview_raises_for_missing_id(sqlite_session: Session) -> None:
    repo = OperationsRepository(sqlite_session)

    with pytest.raises(KeyError):
        repo.load_preview(uuid4())


def test_blocked_preview_round_trips(sqlite_session: Session) -> None:
    repo = OperationsRepository(sqlite_session)
    blocked = _make_preview(blocked=True)
    assert blocked.blocked is True

    pid = repo.save_preview(
        blocked,
        correlation_id="run-b",
        decision_session=_DECISION_SESSION,
    )
    loaded = repo.load_preview(pid)
    assert loaded == blocked
    assert loaded.blocked is True


# ---------------------------------------------------------------------------
# 2. ExecutionIntent idempotency
# ---------------------------------------------------------------------------


def test_execution_intent_client_id_is_unique(sqlite_session: Session) -> None:
    repo = OperationsRepository(sqlite_session)
    intent = _make_intent()

    id1 = repo.save_intent(
        intent,
        correlation_id=_CORRELATION_ID,
        decision_session=_DECISION_SESSION,
    )
    id2 = repo.save_intent(
        intent,
        correlation_id=_CORRELATION_ID,
        decision_session=_DECISION_SESSION,
    )

    assert id1 == id2, "second save must return the same UUID (idempotent)"
    assert repo.count_intents(intent.client_order_id) == 1


def test_different_intents_get_different_rows(sqlite_session: Session) -> None:
    repo = OperationsRepository(sqlite_session)
    intent_buy = _make_intent("SPY", OrderSide.BUY)
    intent_sell = _make_intent("QQQ", OrderSide.SELL)

    id_buy = repo.save_intent(intent_buy)
    id_sell = repo.save_intent(intent_sell)

    assert id_buy != id_sell
    assert repo.count_intents(intent_buy.client_order_id) == 1
    assert repo.count_intents(intent_sell.client_order_id) == 1


# ---------------------------------------------------------------------------
# 3-10. save_* methods
# ---------------------------------------------------------------------------


def test_save_snapshot(sqlite_session: Session) -> None:
    repo = OperationsRepository(sqlite_session)
    row_id = repo.save_snapshot(
        {"symbols": ["SPY", "QQQ"], "bar_count": 1000},
        correlation_id=_CORRELATION_ID,
        decision_session=_DECISION_SESSION,
    )

    assert isinstance(row_id, UUID)
    stored = sqlite_session.scalar(
        select(DatasetSnapshot).where(DatasetSnapshot.id == row_id)
    )
    assert stored is not None
    assert stored.payload["bar_count"] == 1000


def test_save_quality_report(sqlite_session: Session) -> None:
    repo = OperationsRepository(sqlite_session)
    row_id = repo.save_quality_report(
        {"issues": [], "blocked": False},
        correlation_id=_CORRELATION_ID,
        decision_session=_DECISION_SESSION,
    )

    assert isinstance(row_id, UUID)
    stored = sqlite_session.scalar(
        select(QualityReport).where(QualityReport.id == row_id)
    )
    assert stored is not None
    assert stored.payload["blocked"] is False


def test_save_forecast_batch(sqlite_session: Session) -> None:
    repo = OperationsRepository(sqlite_session)
    row_id = repo.save_forecast_batch(
        {"model": "ridge-trend-v1", "predictions": {"SPY": 0.02, "QQQ": -0.01}},
        correlation_id=_CORRELATION_ID,
        decision_session=_DECISION_SESSION,
    )

    assert isinstance(row_id, UUID)
    stored = sqlite_session.scalar(
        select(ForecastBatch).where(ForecastBatch.id == row_id)
    )
    assert stored is not None
    assert stored.payload["model"] == "ridge-trend-v1"


def test_save_backtest_run(sqlite_session: Session) -> None:
    repo = OperationsRepository(sqlite_session)
    row_id = repo.save_backtest_run(
        {"sharpe": 1.42, "max_drawdown": 0.08},
        correlation_id=_CORRELATION_ID,
        decision_session=_DECISION_SESSION,
    )

    assert isinstance(row_id, UUID)
    stored = sqlite_session.scalar(
        select(BacktestRun).where(BacktestRun.id == row_id)
    )
    assert stored is not None
    assert stored.payload["sharpe"] == 1.42


def test_save_portfolio_target(sqlite_session: Session) -> None:
    repo = OperationsRepository(sqlite_session)
    row_id = repo.save_portfolio_target(
        {"SPY": 10, "QQQ": 5},
        correlation_id=_CORRELATION_ID,
        decision_session=_DECISION_SESSION,
    )

    assert isinstance(row_id, UUID)
    stored = sqlite_session.scalar(
        select(PortfolioTarget).where(PortfolioTarget.id == row_id)
    )
    assert stored is not None
    assert stored.payload["SPY"] == 10


def test_save_risk_report(sqlite_session: Session) -> None:
    repo = OperationsRepository(sqlite_session)
    row_id = repo.save_risk_report(
        {
            "blocked": False,
            "checks": [{"code": "live_trading_disabled", "status": "pass"}],
        },
        correlation_id=_CORRELATION_ID,
        decision_session=_DECISION_SESSION,
    )

    assert isinstance(row_id, UUID)
    stored = sqlite_session.scalar(
        select(RiskReportRow).where(RiskReportRow.id == row_id)
    )
    assert stored is not None
    assert stored.payload["blocked"] is False


def test_save_broker_order(sqlite_session: Session) -> None:
    repo = OperationsRepository(sqlite_session)
    row_id = repo.save_broker_order(
        {"order_id": "broker-abc-123", "symbol": "SPY", "status": "filled"},
        correlation_id=_CORRELATION_ID,
        decision_session=_DECISION_SESSION,
    )

    assert isinstance(row_id, UUID)
    stored = sqlite_session.scalar(
        select(BrokerOrder).where(BrokerOrder.id == row_id)
    )
    assert stored is not None
    assert stored.payload["status"] == "filled"


def test_save_reconciliation_run(sqlite_session: Session) -> None:
    repo = OperationsRepository(sqlite_session)
    row_id = repo.save_reconciliation_run(
        {"differences": [], "blocked": False},
        correlation_id=_CORRELATION_ID,
        decision_session=_DECISION_SESSION,
    )

    assert isinstance(row_id, UUID)
    stored = sqlite_session.scalar(
        select(ReconciliationRun).where(ReconciliationRun.id == row_id)
    )
    assert stored is not None
    assert stored.payload["differences"] == []


# ---------------------------------------------------------------------------
# 11. Two writes in one session
# ---------------------------------------------------------------------------


def test_two_saves_in_one_session_both_persist(sqlite_session: Session) -> None:
    repo = OperationsRepository(sqlite_session)

    id1 = repo.save_snapshot(
        {"run": 1},
        correlation_id="run-1",
        decision_session=_DECISION_SESSION,
    )
    id2 = repo.save_snapshot(
        {"run": 2},
        correlation_id="run-2",
        decision_session=_DECISION_SESSION,
    )

    assert id1 != id2

    count = sqlite_session.scalar(
        select(func.count()).select_from(DatasetSnapshot)
    )
    assert count == 2

    row1 = sqlite_session.scalar(
        select(DatasetSnapshot).where(DatasetSnapshot.id == id1)
    )
    row2 = sqlite_session.scalar(
        select(DatasetSnapshot).where(DatasetSnapshot.id == id2)
    )
    assert row1 is not None and row1.payload["run"] == 1
    assert row2 is not None and row2.payload["run"] == 2


# ---------------------------------------------------------------------------
# 12. Error rolls back via session_scope
# ---------------------------------------------------------------------------


def test_error_rolls_back(tmp_path: object) -> None:
    """session_scope must roll back all writes when an exception is raised."""
    engine = build_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)

    with pytest.raises(RuntimeError, match="forced rollback"):
        with session_scope(engine) as session:
            repo = OperationsRepository(session)
            repo.save_snapshot(
                {"step": "before-error"},
                correlation_id="rollback-test",
            )
            raise RuntimeError("forced rollback")

    # After the rollback, the table must be empty.
    with session_scope(engine) as session:
        count = session.scalar(
            select(func.count()).select_from(DatasetSnapshot)
        )
    assert count == 0

    engine.dispose()
