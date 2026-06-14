"""Integration tests for ExecutionService with SimulatedBroker (Task 12).

These tests drive the full preview → submit pipeline using the real
SimulatedBroker rather than a stub.

Coverage
--------
- test_complete_fill_submit              — happy path, submitted==1
- test_timeout_after_acceptance          — TIMEOUT_AFTER → submitted==1, no dup
- test_timeout_before_retries_once       — TIMEOUT_BEFORE → retry, if both fail
                                           → submitted==0
- test_blocked_preview_refused          — blocked preview → submitted==0
- test_reconciliation_mismatch_blocks   — cash mismatch → reconciliation.blocked
- test_idempotent_resubmit              — submit same preview twice, no duplicates
- test_partial_fill_counted_as_submitted— PARTIAL_FILL → submitted incremented
- test_rejected_order_not_counted       — REJECTED → submitted==0
- test_minimum_notional_filter          — tiny delta below $500 → no intents
- test_timeout_before_retry_succeeds    — TIMEOUT_BEFORE first, then FILL → submitted==1
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from mltrade.config import Settings
from mltrade.execution.broker import BrokerAccount, BrokerTimeout
from mltrade.execution.reconciliation import InternalState
from mltrade.execution.service import ExecutionService
from mltrade.execution.simulated import SimulatedBroker, SubmitOutcome

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SESSION = date(2026, 6, 12)
_ENV = "paper"
_STRAT = "ridge-trend-v1"


def _account(cash: Decimal = Decimal("100000")) -> BrokerAccount:
    return BrokerAccount(
        id="paper-001",
        status="ACTIVE",
        cash=cash,
        equity=cash,
        account_blocked=False,
        trading_blocked=False,
        pattern_day_trader=False,
    )


def _settings() -> Settings:
    return Settings(
        minimum_order_notional=Decimal("500"),
        maximum_position_weight=Decimal("0.25"),
        minimum_cash_weight=Decimal("0.05"),
        maximum_order_weight=Decimal("0.50"),
        maximum_rebalance_weight=Decimal("0.90"),
        live_trading_enabled=False,
    )


def _internal(cash: Decimal = Decimal("100000")) -> InternalState:
    return InternalState(
        cash=cash,
        positions={},
        open_client_order_ids=(),
    )


def _prices(**kwargs: int) -> dict[str, Decimal]:
    return {sym: Decimal(str(price)) for sym, price in kwargs.items()}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_complete_fill_submit() -> None:
    """Happy path: one BUY intent, COMPLETE_FILL → submitted==1."""
    broker = SimulatedBroker(_account(), default_outcome=SubmitOutcome.COMPLETE_FILL)
    svc = ExecutionService(broker)

    preview = svc.preview(
        target_positions={"SPY": 10},
        internal_state=_internal(),
        settings=_settings(),
        strategy_version=_STRAT,
        decision_session=_SESSION,
        environment=_ENV,
        prices=_prices(SPY=200),
    )

    assert not preview.blocked, (
        f"Expected unblocked preview; report: {preview.risk_report}"
    )
    result = svc.submit(preview)

    assert result.submitted == 1
    assert result.blocked is False
    assert len(result.outcomes) == 1
    assert result.outcomes[0].status == "submitted"


def test_timeout_after_acceptance_does_not_duplicate() -> None:
    """TIMEOUT_AFTER: order recorded once; submit returns submitted==1, no dup."""
    broker = SimulatedBroker(_account(), default_outcome=SubmitOutcome.TIMEOUT_AFTER)
    svc = ExecutionService(broker)

    preview = svc.preview(
        target_positions={"SPY": 10},
        internal_state=_internal(),
        settings=_settings(),
        strategy_version=_STRAT,
        decision_session=_SESSION,
        environment=_ENV,
        prices=_prices(SPY=200),
    )

    assert not preview.blocked
    result = svc.submit(preview)

    # Exactly one order recorded at broker
    assert len(broker.list_orders()) == 1
    # Counted as submitted (order is live as NEW)
    assert result.submitted == 1
    assert result.outcomes[0].status == "submitted"


def test_timeout_before_retries_once_then_fails() -> None:
    """TIMEOUT_BEFORE twice → submitted==0, status='timeout_failed'."""
    broker = SimulatedBroker(_account(), default_outcome=SubmitOutcome.TIMEOUT_BEFORE)
    svc = ExecutionService(broker)

    preview = svc.preview(
        target_positions={"SPY": 10},
        internal_state=_internal(),
        settings=_settings(),
        strategy_version=_STRAT,
        decision_session=_SESSION,
        environment=_ENV,
        prices=_prices(SPY=200),
    )

    assert not preview.blocked
    result = svc.submit(preview)

    assert result.submitted == 0
    assert result.blocked is False
    assert result.outcomes[0].status == "timeout_failed"
    # Nothing recorded at broker
    assert len(broker.list_orders()) == 0


def test_timeout_before_retry_succeeds() -> None:
    """TIMEOUT_BEFORE on first attempt, COMPLETE_FILL on retry → submitted==1."""
    # We need the first call to the same intent to timeout, then succeed.
    # Use a custom outcome_map that can be mutated between calls by subclassing.
    call_count: dict[str, int] = {}

    class _RetryBroker(SimulatedBroker):
        def submit(self, intent):  # type: ignore[override]
            cid = intent.client_order_id
            n = call_count.get(cid, 0)
            call_count[cid] = n + 1
            if n == 0:
                # First call: TIMEOUT_BEFORE (raise before recording)
                raise BrokerTimeout("simulated timeout before")
            # Second call: let parent handle (COMPLETE_FILL default)
            return super().submit(intent)

    broker = _RetryBroker(_account(), default_outcome=SubmitOutcome.COMPLETE_FILL)
    svc = ExecutionService(broker)

    preview = svc.preview(
        target_positions={"SPY": 10},
        internal_state=_internal(),
        settings=_settings(),
        strategy_version=_STRAT,
        decision_session=_SESSION,
        environment=_ENV,
        prices=_prices(SPY=200),
    )

    assert not preview.blocked
    result = svc.submit(preview)

    assert result.submitted == 1
    assert result.outcomes[0].status == "submitted"


def test_blocked_preview_refused() -> None:
    """Blocked preview → submit returns submitted==0, all outcomes 'blocked'."""
    broker = SimulatedBroker(_account(), default_outcome=SubmitOutcome.COMPLETE_FILL)
    svc = ExecutionService(broker)

    # cash mismatch creates a reconciliation block
    mismatched_internal = InternalState(
        cash=Decimal("999999"),  # different from broker's 100000
        positions={},
        open_client_order_ids=(),
    )
    preview = svc.preview(
        target_positions={"SPY": 10},
        internal_state=mismatched_internal,
        settings=_settings(),
        strategy_version=_STRAT,
        decision_session=_SESSION,
        environment=_ENV,
        prices=_prices(SPY=200),
    )

    assert preview.blocked
    result = svc.submit(preview)

    assert result.submitted == 0
    assert result.blocked is True
    for outcome in result.outcomes:
        assert outcome.status == "blocked"


def test_reconciliation_mismatch_blocks_preview() -> None:
    """Cash mismatch → reconciliation.blocked=True → preview.blocked=True."""
    broker = SimulatedBroker(_account(cash=Decimal("100000")))
    svc = ExecutionService(broker)

    wrong_internal = InternalState(
        cash=Decimal("50000"),  # wrong!
        positions={},
        open_client_order_ids=(),
    )
    preview = svc.preview(
        target_positions={"SPY": 10},
        internal_state=wrong_internal,
        settings=_settings(),
        strategy_version=_STRAT,
        decision_session=_SESSION,
        environment=_ENV,
        prices=_prices(SPY=200),
    )

    assert preview.reconciliation.blocked is True
    assert preview.blocked is True


def test_open_order_mismatch_blocks_and_refuses_submit() -> None:
    """An internal open order the broker doesn't know about blocks submission."""
    broker = SimulatedBroker(_account(), default_outcome=SubmitOutcome.COMPLETE_FILL)
    svc = ExecutionService(broker)

    stale_internal = InternalState(
        cash=Decimal("100000"),
        positions={},
        open_client_order_ids=("mlt-20260612-deadbeefdeadbeefdeadbeef",),
    )
    preview = svc.preview(
        target_positions={"SPY": 10},
        internal_state=stale_internal,
        settings=_settings(),
        strategy_version=_STRAT,
        decision_session=_SESSION,
        environment=_ENV,
        prices=_prices(SPY=200),
    )

    assert preview.reconciliation.blocked is True
    assert any(d.kind == "open_order" for d in preview.reconciliation.differences)
    assert preview.blocked is True

    result = svc.submit(preview)
    assert result.submitted == 0
    assert len(broker.list_orders()) == 0


def test_idempotent_resubmit() -> None:
    """Submitting the same preview twice → same result, no duplicates at broker."""
    broker = SimulatedBroker(_account(), default_outcome=SubmitOutcome.COMPLETE_FILL)
    svc = ExecutionService(broker)

    preview = svc.preview(
        target_positions={"SPY": 10},
        internal_state=_internal(),
        settings=_settings(),
        strategy_version=_STRAT,
        decision_session=_SESSION,
        environment=_ENV,
        prices=_prices(SPY=200),
    )

    assert not preview.blocked

    r1 = svc.submit(preview)
    r2 = svc.submit(preview)

    assert r1.submitted == 1
    assert r2.submitted == 1  # already_existed still counts
    # Only 1 order recorded at broker
    assert len(broker.list_orders()) == 1
    assert r2.outcomes[0].status == "already_existed"


def test_partial_fill_counted_as_submitted() -> None:
    """PARTIAL_FILL → submitted count incremented (order is live)."""
    broker = SimulatedBroker(_account(), default_outcome=SubmitOutcome.PARTIAL_FILL)
    svc = ExecutionService(broker)

    preview = svc.preview(
        target_positions={"SPY": 10},
        internal_state=_internal(),
        settings=_settings(),
        strategy_version=_STRAT,
        decision_session=_SESSION,
        environment=_ENV,
        prices=_prices(SPY=200),
    )

    assert not preview.blocked
    result = svc.submit(preview)

    assert result.submitted == 1
    assert result.outcomes[0].status == "submitted"


def test_rejected_order_not_counted() -> None:
    """REJECTED → submitted count stays 0."""
    broker = SimulatedBroker(_account(), default_outcome=SubmitOutcome.REJECTED)
    svc = ExecutionService(broker)

    preview = svc.preview(
        target_positions={"SPY": 10},
        internal_state=_internal(),
        settings=_settings(),
        strategy_version=_STRAT,
        decision_session=_SESSION,
        environment=_ENV,
        prices=_prices(SPY=200),
    )

    assert not preview.blocked
    result = svc.submit(preview)

    assert result.submitted == 0
    assert result.outcomes[0].status == "rejected"


def test_minimum_notional_filter() -> None:
    """Delta below minimum notional → filtered out, zero intents generated."""
    broker = SimulatedBroker(_account(), default_outcome=SubmitOutcome.COMPLETE_FILL)
    svc = ExecutionService(broker)

    # SPY price=10, target=1 share → notional=$10, well below $500 minimum
    preview = svc.preview(
        target_positions={"SPY": 1},
        internal_state=_internal(),
        settings=_settings(),
        strategy_version=_STRAT,
        decision_session=_SESSION,
        environment=_ENV,
        prices=_prices(SPY=10),
    )

    assert len(preview.intents) == 0
    # No intents → not blocked on any order-level check
    # (risk checks on 0-intent batch should all pass)
    result = svc.submit(preview)
    assert result.submitted == 0
    assert len(result.outcomes) == 0


def test_multiple_intents_partial_success() -> None:
    """SPY fills, QQQ rejects → submitted==1."""
    broker_account = _account()
    broker = SimulatedBroker(
        broker_account, default_outcome=SubmitOutcome.COMPLETE_FILL
    )
    svc = ExecutionService(broker)

    preview = svc.preview(
        target_positions={"SPY": 10, "QQQ": 5},
        internal_state=_internal(),
        settings=_settings(),
        strategy_version=_STRAT,
        decision_session=_SESSION,
        environment=_ENV,
        prices=_prices(SPY=200, QQQ=400),
    )

    assert not preview.blocked
    assert len(preview.intents) == 2

    # Override QQQ to reject after preview
    for intent in preview.intents:
        if intent.symbol == "QQQ":
            broker._outcome_map[intent.client_order_id] = SubmitOutcome.REJECTED

    result = svc.submit(preview)

    spy_outcome = next(o for o in result.outcomes if o.symbol == "SPY")
    qqq_outcome = next(o for o in result.outcomes if o.symbol == "QQQ")
    assert spy_outcome.status == "submitted"
    assert qqq_outcome.status == "rejected"
    assert result.submitted == 1


def test_preview_blocked_property() -> None:
    """Preview.blocked reflects both reconciliation and risk gates."""
    broker = SimulatedBroker(_account())
    svc = ExecutionService(broker)

    # Perfect match
    preview = svc.preview(
        target_positions={},
        internal_state=_internal(),
        settings=_settings(),
        strategy_version=_STRAT,
        decision_session=_SESSION,
        environment=_ENV,
        prices={},
    )

    assert preview.blocked == (
        preview.reconciliation.blocked or preview.risk_report.blocked
    )
