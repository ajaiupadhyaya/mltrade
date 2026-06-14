"""Tests for SimulatedBroker.

Covers
------
- Dedup (same client_order_id → idempotent return, no side effects)
- All five SubmitOutcome values
- list_open_orders (excludes FILLED/REJECTED)
- list_orders (includes all)
- list_recent_fills (all fills)
- get_order_by_client_id (found + not found)
- get_account
- list_positions (initially empty)
- Fill UTC timestamp enforcement
- Deterministic broker order IDs (sim-0, sim-1, ...)
- outcome_map overrides default_outcome
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from mltrade.execution import (
    BrokerAccount,
    BrokerTimeout,
    ExecutionIntent,
    OrderSide,
    OrderStatus,
    SimulatedBroker,
    SubmitOutcome,
    build_intent,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def paper_account() -> BrokerAccount:
    return BrokerAccount(
        id="paper-001",
        status="ACTIVE",
        cash=Decimal("100000"),
        equity=Decimal("100000"),
        account_blocked=False,
        trading_blocked=False,
        pattern_day_trader=False,
    )


def make_intent(**overrides: object) -> ExecutionIntent:
    defaults: dict[str, object] = dict(
        environment="paper",
        strategy_version="ridge-trend-v1",
        decision_session=date(2026, 6, 12),
        symbol="SPY",
        side=OrderSide.BUY,
        target_quantity=Decimal("10"),
    )
    defaults.update(overrides)
    return build_intent(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def test_dedup_same_client_order_id() -> None:
    """Submitting the same intent twice: same order returned, only 1 entry recorded."""
    broker = SimulatedBroker(paper_account())
    intent = make_intent()

    first = broker.submit(intent)
    second = broker.submit(intent)

    assert first.id == second.id
    assert len(broker.list_orders()) == 1


def test_dedup_returns_same_order_object() -> None:
    """The second submit call must return the identical BrokerOrder."""
    broker = SimulatedBroker(paper_account())
    intent = make_intent()

    first = broker.submit(intent)
    second = broker.submit(intent)

    assert first.id == second.id
    assert first.client_order_id == second.client_order_id


def test_dedup_does_not_add_extra_fill() -> None:
    """Dedup must not create an extra fill record."""
    broker = SimulatedBroker(paper_account())
    intent = make_intent()

    broker.submit(intent)
    broker.submit(intent)  # dedup

    assert len(broker.list_recent_fills()) == 1


# ---------------------------------------------------------------------------
# Multiple distinct intents
# ---------------------------------------------------------------------------


def test_different_intents_create_separate_orders() -> None:
    """Distinct client_order_ids must create independent order entries."""
    broker = SimulatedBroker(paper_account())
    spy = make_intent(symbol="SPY")
    qqq = make_intent(symbol="QQQ")

    broker.submit(spy)
    broker.submit(qqq)

    assert len(broker.list_orders()) == 2


# ---------------------------------------------------------------------------
# Outcome: COMPLETE_FILL
# ---------------------------------------------------------------------------


def test_complete_fill_outcome() -> None:
    """COMPLETE_FILL → status=FILLED, filled_quantity==quantity, fill created."""
    broker = SimulatedBroker(
        paper_account(), default_outcome=SubmitOutcome.COMPLETE_FILL
    )
    order = broker.submit(make_intent(target_quantity=Decimal("7")))

    assert order.status is OrderStatus.FILLED
    assert order.filled_quantity == 7
    fills = broker.list_recent_fills()
    assert len(fills) == 1
    assert fills[0].quantity == 7


# ---------------------------------------------------------------------------
# Outcome: PARTIAL_FILL
# ---------------------------------------------------------------------------


def test_partial_fill_outcome() -> None:
    """PARTIAL_FILL → status=PARTIALLY_FILLED, 0 < filled_quantity < quantity."""
    broker = SimulatedBroker(
        paper_account(),
        default_outcome=SubmitOutcome.PARTIAL_FILL,
        partial_fill_fraction=0.5,
    )
    order = broker.submit(make_intent(target_quantity=Decimal("10")))

    assert order.status is OrderStatus.PARTIALLY_FILLED
    assert 0 < order.filled_quantity < 10
    fills = broker.list_recent_fills()
    assert len(fills) == 1
    assert fills[0].quantity == order.filled_quantity


def test_partial_fill_fraction_applied() -> None:
    """Filled quantity should be floor(quantity * partial_fill_fraction)."""
    broker = SimulatedBroker(
        paper_account(),
        default_outcome=SubmitOutcome.PARTIAL_FILL,
        partial_fill_fraction=0.3,
    )
    order = broker.submit(make_intent(target_quantity=Decimal("10")))
    # floor(10 * 0.3) == 3
    assert order.filled_quantity == 3


# ---------------------------------------------------------------------------
# Outcome: REJECTED
# ---------------------------------------------------------------------------


def test_rejected_outcome() -> None:
    """REJECTED → status=REJECTED, filled_quantity==0, no fill record."""
    broker = SimulatedBroker(paper_account(), default_outcome=SubmitOutcome.REJECTED)
    intent = make_intent()
    order = broker.submit(intent)

    assert order.status is OrderStatus.REJECTED
    assert order.filled_quantity == 0

    # No fill for this order
    fills = [
        f
        for f in broker.list_recent_fills()
        if f.client_order_id == intent.client_order_id
    ]
    assert fills == []


# ---------------------------------------------------------------------------
# Outcome: TIMEOUT_BEFORE
# ---------------------------------------------------------------------------


def test_timeout_before_outcome() -> None:
    """TIMEOUT_BEFORE → BrokerTimeout raised; order NOT recorded."""
    broker = SimulatedBroker(
        paper_account(), default_outcome=SubmitOutcome.TIMEOUT_BEFORE
    )
    intent = make_intent()

    with pytest.raises(BrokerTimeout):
        broker.submit(intent)

    assert broker.get_order_by_client_id(intent.client_order_id) is None
    assert len(broker.list_orders()) == 0


def test_timeout_before_no_fills() -> None:
    """TIMEOUT_BEFORE must not create any fill records."""
    broker = SimulatedBroker(
        paper_account(), default_outcome=SubmitOutcome.TIMEOUT_BEFORE
    )

    with pytest.raises(BrokerTimeout):
        broker.submit(make_intent())

    assert broker.list_recent_fills() == ()


# ---------------------------------------------------------------------------
# Outcome: TIMEOUT_AFTER
# ---------------------------------------------------------------------------


def test_timeout_after_outcome() -> None:
    """TIMEOUT_AFTER → BrokerTimeout raised; order IS recorded as NEW."""
    broker = SimulatedBroker(
        paper_account(), default_outcome=SubmitOutcome.TIMEOUT_AFTER
    )
    intent = make_intent()

    with pytest.raises(BrokerTimeout):
        broker.submit(intent)

    order = broker.get_order_by_client_id(intent.client_order_id)
    assert order is not None
    assert order.status is OrderStatus.NEW


def test_timeout_after_no_fills() -> None:
    """TIMEOUT_AFTER must not create a fill (order is NEW, not filled)."""
    broker = SimulatedBroker(
        paper_account(), default_outcome=SubmitOutcome.TIMEOUT_AFTER
    )

    with pytest.raises(BrokerTimeout):
        broker.submit(make_intent())

    assert broker.list_recent_fills() == ()


# ---------------------------------------------------------------------------
# list_open_orders
# ---------------------------------------------------------------------------


def test_list_open_orders_excludes_filled() -> None:
    """FILLED orders must NOT appear in list_open_orders."""
    broker = SimulatedBroker(
        paper_account(), default_outcome=SubmitOutcome.COMPLETE_FILL
    )
    broker.submit(make_intent())

    assert broker.list_open_orders() == ()


def test_list_open_orders_excludes_rejected() -> None:
    """REJECTED orders must NOT appear in list_open_orders."""
    broker = SimulatedBroker(paper_account(), default_outcome=SubmitOutcome.REJECTED)
    broker.submit(make_intent())

    assert broker.list_open_orders() == ()


def test_list_open_orders_includes_new() -> None:
    """NEW orders (from TIMEOUT_AFTER) must appear in list_open_orders."""
    broker = SimulatedBroker(
        paper_account(), default_outcome=SubmitOutcome.TIMEOUT_AFTER
    )
    intent = make_intent()

    with pytest.raises(BrokerTimeout):
        broker.submit(intent)

    open_orders = broker.list_open_orders()
    assert len(open_orders) == 1
    assert open_orders[0].client_order_id == intent.client_order_id


def test_list_open_orders_includes_partially_filled() -> None:
    """PARTIALLY_FILLED orders must appear in list_open_orders."""
    broker = SimulatedBroker(
        paper_account(), default_outcome=SubmitOutcome.PARTIAL_FILL
    )
    intent = make_intent()
    broker.submit(intent)

    open_orders = broker.list_open_orders()
    assert len(open_orders) == 1
    assert open_orders[0].status is OrderStatus.PARTIALLY_FILLED


# ---------------------------------------------------------------------------
# list_orders
# ---------------------------------------------------------------------------


def test_list_orders_includes_all() -> None:
    """list_orders() must return FILLED, REJECTED, and NEW orders."""
    spy_intent = make_intent(symbol="SPY")
    qqq_intent = make_intent(symbol="QQQ")
    iwm_intent = make_intent(symbol="IWM")

    outcome_map = {
        spy_intent.client_order_id: SubmitOutcome.COMPLETE_FILL,
        qqq_intent.client_order_id: SubmitOutcome.REJECTED,
        iwm_intent.client_order_id: SubmitOutcome.TIMEOUT_AFTER,
    }
    broker = SimulatedBroker(paper_account(), outcome_map=outcome_map)

    broker.submit(spy_intent)
    broker.submit(qqq_intent)
    with pytest.raises(BrokerTimeout):
        broker.submit(iwm_intent)

    all_orders = broker.list_orders()
    assert len(all_orders) == 3
    statuses = {o.status for o in all_orders}
    assert OrderStatus.FILLED in statuses
    assert OrderStatus.REJECTED in statuses
    assert OrderStatus.NEW in statuses


# ---------------------------------------------------------------------------
# get_account / list_positions
# ---------------------------------------------------------------------------


def test_get_account() -> None:
    """get_account must return the account passed at construction."""
    account = paper_account()
    broker = SimulatedBroker(account)
    assert broker.get_account() is account


def test_list_positions_initially_empty() -> None:
    """A freshly constructed SimulatedBroker has no positions."""
    broker = SimulatedBroker(paper_account())
    assert broker.list_positions() == ()


# ---------------------------------------------------------------------------
# Fill timestamps are UTC
# ---------------------------------------------------------------------------


def test_fills_have_utc_timestamps() -> None:
    """All fill timestamps must be UTC-aware."""
    broker = SimulatedBroker(
        paper_account(), default_outcome=SubmitOutcome.COMPLETE_FILL
    )
    broker.submit(make_intent())

    for fill in broker.list_recent_fills():
        assert fill.timestamp.tzinfo is not None
        assert fill.timestamp.tzinfo is UTC


def test_fill_timestamp_is_deterministic_by_default() -> None:
    """Two fresh brokers must produce identical fill timestamps (no wall clock)."""
    first = SimulatedBroker(paper_account())
    second = SimulatedBroker(paper_account())
    first.submit(make_intent())
    second.submit(make_intent())

    assert (
        first.list_recent_fills()[0].timestamp
        == second.list_recent_fills()[0].timestamp
    )


def test_fill_timestamp_injection() -> None:
    """An injected fill_timestamp is used (normalized to UTC)."""
    stamp = datetime(2026, 6, 12, 20, 0, tzinfo=UTC)
    broker = SimulatedBroker(paper_account(), fill_timestamp=stamp)
    broker.submit(make_intent())

    assert broker.list_recent_fills()[0].timestamp == stamp


# ---------------------------------------------------------------------------
# Deterministic order IDs
# ---------------------------------------------------------------------------


def test_deterministic_order_ids() -> None:
    """Broker-assigned IDs must follow sim-0, sim-1, sim-2 in insertion order."""
    broker = SimulatedBroker(paper_account())

    i0 = make_intent(symbol="SPY")
    i1 = make_intent(symbol="QQQ")
    i2 = make_intent(symbol="IWM")

    o0 = broker.submit(i0)
    o1 = broker.submit(i1)
    o2 = broker.submit(i2)

    assert o0.id == "sim-0"
    assert o1.id == "sim-1"
    assert o2.id == "sim-2"


# ---------------------------------------------------------------------------
# outcome_map overrides default
# ---------------------------------------------------------------------------


def test_outcome_map_overrides_default() -> None:
    """An outcome_map entry takes precedence over default_outcome."""
    spy_intent = make_intent(symbol="SPY")
    qqq_intent = make_intent(symbol="QQQ")

    # Default: COMPLETE_FILL, but SPY → REJECTED via map
    outcome_map = {spy_intent.client_order_id: SubmitOutcome.REJECTED}
    broker = SimulatedBroker(
        paper_account(),
        default_outcome=SubmitOutcome.COMPLETE_FILL,
        outcome_map=outcome_map,
    )

    spy_order = broker.submit(spy_intent)
    qqq_order = broker.submit(qqq_intent)

    assert spy_order.status is OrderStatus.REJECTED
    assert qqq_order.status is OrderStatus.FILLED


# ---------------------------------------------------------------------------
# get_order_by_client_id
# ---------------------------------------------------------------------------


def test_get_order_by_client_id_found() -> None:
    """get_order_by_client_id must return the order for a known client_order_id."""
    broker = SimulatedBroker(paper_account())
    intent = make_intent()
    submitted = broker.submit(intent)

    found = broker.get_order_by_client_id(intent.client_order_id)
    assert found is not None
    assert found.id == submitted.id


def test_get_order_by_client_id_not_found() -> None:
    """get_order_by_client_id must return None for an unknown client_order_id."""
    broker = SimulatedBroker(paper_account())
    assert broker.get_order_by_client_id("no-such-id") is None


def test_get_order_by_client_id() -> None:
    """Combined found + not-found in a single test."""
    broker = SimulatedBroker(paper_account())
    intent = make_intent()
    order = broker.submit(intent)

    # Found
    result = broker.get_order_by_client_id(intent.client_order_id)
    assert result is not None
    assert result.id == order.id

    # Not found
    assert broker.get_order_by_client_id("mlt-99999999-notexistent000000000000") is None
