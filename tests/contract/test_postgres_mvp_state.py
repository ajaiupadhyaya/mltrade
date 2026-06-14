"""PostgreSQL contract tests for operational evidence persistence (Task 13).

Requires a running Postgres instance and the environment variable
``MLTRADE_TEST_DATABASE_URL`` to be set, e.g.::

    MLTRADE_TEST_DATABASE_URL=postgresql+psycopg://mltrade:mltrade@localhost:5432/mltrade
        pytest tests/contract/test_postgres_mvp_state.py -v

Tests are marked ``contract`` and are skipped by default when the env var is
absent (via conftest / pytest mark filtering).
"""

from __future__ import annotations

import os
from datetime import date
from decimal import Decimal

import pytest

from mltrade.execution.broker import OrderSide
from mltrade.execution.intents import build_intent
from mltrade.execution.reconciliation import ReconciliationResult
from mltrade.execution.service import Preview
from mltrade.operations.database import build_engine, session_scope
from mltrade.operations.models import Base
from mltrade.operations.repositories import OperationsRepository
from mltrade.risk.checks import CheckStatus, RiskCheck, RiskReport

pytestmark = pytest.mark.contract

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_SESSION_DATE = date(2026, 6, 12)
_DECISION_SESSION = _SESSION_DATE.isoformat()


def _make_preview() -> Preview:
    intent = build_intent(
        environment="paper",
        strategy_version="ridge-trend-v1",
        decision_session=_SESSION_DATE,
        symbol="SPY",
        side=OrderSide.BUY,
        target_quantity=Decimal("10"),
    )
    risk_report = RiskReport(
        checks=(
            RiskCheck(
                code="live_trading_disabled",
                status=CheckStatus.PASS,
                message="live trading disabled — paper only",
            ),
        )
    )
    return Preview(
        intents=(intent,),
        reconciliation=ReconciliationResult(differences=()),
        risk_report=risk_report,
    )


# ---------------------------------------------------------------------------
# Postgres session fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def pg_session():  # type: ignore[return]
    database_url = os.environ["MLTRADE_TEST_DATABASE_URL"]
    engine = build_engine(database_url)
    Base.metadata.create_all(engine)
    with session_scope(engine) as session:
        yield session
    engine.dispose()


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


def test_postgres_preview_round_trip(pg_session: object) -> None:
    """save_preview → load_preview must reconstruct the exact Preview on Postgres."""
    repo = OperationsRepository(pg_session)  # type: ignore[arg-type]
    original = _make_preview()

    preview_id = repo.save_preview(
        original,
        correlation_id="pg-contract-001",
        decision_session=_DECISION_SESSION,
    )

    loaded = repo.load_preview(preview_id)
    assert loaded == original


def test_postgres_intent_idempotency(pg_session: object) -> None:
    """save_intent twice with same client_order_id must return same UUID."""
    repo = OperationsRepository(pg_session)  # type: ignore[arg-type]
    intent = build_intent(
        environment="paper",
        strategy_version="ridge-trend-v1",
        decision_session=_SESSION_DATE,
        symbol="QQQ",
        side=OrderSide.SELL,
        target_quantity=Decimal("5"),
    )

    id1 = repo.save_intent(
        intent,
        correlation_id="pg-contract-002",
        decision_session=_DECISION_SESSION,
    )
    id2 = repo.save_intent(
        intent,
        correlation_id="pg-contract-002",
        decision_session=_DECISION_SESSION,
    )

    assert id1 == id2
    assert repo.count_intents(intent.client_order_id) == 1
