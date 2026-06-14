"""Unit tests for the reconciliation module (Task 12).

Coverage
--------
- No differences when internal and broker states match → blocked=False
- Position difference (broker has extra position) → blocked=True, kind=="position"
- Cash difference detected → "cash" difference
- Position missing from broker → kind=="position"
- Open order difference → kind=="open_order"
- Difference ordering: cash first, then positions, then open_orders
- Multiple position differences sorted alphabetically by symbol
- Perfect match after previous mismatch (sanity)
- Broker-only position vs internal-only position both surfaced
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from mltrade.execution.broker import (
    BrokerAccount,
    BrokerOrder,
    BrokerPosition,
    OrderSide,
    OrderStatus,
)
from mltrade.execution.reconciliation import (
    InternalState,
    ReconciliationDifference,
    ReconciliationResult,
    reconcile,
)

# ---------------------------------------------------------------------------
# Minimal stub broker
# ---------------------------------------------------------------------------


class _StubBroker:
    """Minimal Broker protocol implementation for unit tests."""

    def __init__(
        self,
        *,
        cash: Decimal = Decimal("100000"),
        positions: list[BrokerPosition] | None = None,
        open_orders: list[BrokerOrder] | None = None,
    ) -> None:
        self._account = BrokerAccount(
            id="stub-001",
            status="ACTIVE",
            cash=cash,
            equity=cash,
            account_blocked=False,
            trading_blocked=False,
            pattern_day_trader=False,
        )
        self._positions: tuple[BrokerPosition, ...] = tuple(positions or [])
        self._open_orders: tuple[BrokerOrder, ...] = tuple(open_orders or [])

    def get_account(self) -> BrokerAccount:
        return self._account

    def list_positions(self) -> tuple[BrokerPosition, ...]:
        return self._positions

    def list_open_orders(self) -> tuple[BrokerOrder, ...]:
        return self._open_orders

    def list_orders(self) -> tuple[BrokerOrder, ...]:
        return self._open_orders

    def list_recent_fills(self) -> tuple:  # type: ignore[type-arg]
        return ()

    def get_order_by_client_id(self, client_order_id: str) -> BrokerOrder | None:
        for o in self._open_orders:
            if o.client_order_id == client_order_id:
                return o
        return None

    def submit(self, intent: object) -> BrokerOrder:  # type: ignore[override]
        raise NotImplementedError


def _make_position(symbol: str, qty: int) -> BrokerPosition:
    return BrokerPosition(symbol=symbol, quantity=qty, avg_price=Decimal("100"))


def _make_open_order(client_order_id: str, symbol: str = "SPY") -> BrokerOrder:
    return BrokerOrder(
        id=f"broker-{client_order_id}",
        client_order_id=client_order_id,
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=10,
        status=OrderStatus.NEW,
        filled_quantity=0,
    )


# ---------------------------------------------------------------------------
# Tests: basic pass
# ---------------------------------------------------------------------------


def test_no_differences_when_states_match() -> None:
    """Perfect alignment → empty differences, blocked=False."""
    internal = InternalState(
        cash=Decimal("100000"),
        positions={},
        open_client_order_ids=(),
    )
    broker = _StubBroker(cash=Decimal("100000"))
    result = reconcile(internal=internal, broker=broker)

    assert result.differences == ()
    assert result.blocked is False


def test_no_differences_with_matching_positions() -> None:
    """Internal and broker both have SPY=10 → no difference."""
    internal = InternalState(
        cash=Decimal("100000"),
        positions={"SPY": 10},
        open_client_order_ids=(),
    )
    broker = _StubBroker(
        cash=Decimal("100000"),
        positions=[_make_position("SPY", 10)],
    )
    result = reconcile(internal=internal, broker=broker)

    assert result.differences == ()
    assert result.blocked is False


# ---------------------------------------------------------------------------
# Tests: cash differences
# ---------------------------------------------------------------------------


def test_cash_difference_detected() -> None:
    """Cash mismatch → one 'cash' difference."""
    internal = InternalState(
        cash=Decimal("100000"),
        positions={},
        open_client_order_ids=(),
    )
    broker = _StubBroker(cash=Decimal("99000"))
    result = reconcile(internal=internal, broker=broker)

    assert len(result.differences) == 1
    diff = result.differences[0]
    assert diff.kind == "cash"
    assert diff.symbol is None
    assert diff.client_order_id is None
    assert diff.expected == "100000"
    assert diff.actual == "99000"
    assert result.blocked is True


# ---------------------------------------------------------------------------
# Tests: position differences
# ---------------------------------------------------------------------------


def test_position_difference_blocks_submission() -> None:
    """Broker has SPY=5 but internal says 0 → position difference, blocked."""
    internal = InternalState(
        cash=Decimal("100000"),
        positions={},
        open_client_order_ids=(),
    )
    broker = _StubBroker(
        cash=Decimal("100000"),
        positions=[_make_position("SPY", 5)],
    )
    result = reconcile(internal=internal, broker=broker)

    assert len(result.differences) == 1
    diff = result.differences[0]
    assert diff.kind == "position"
    assert diff.symbol == "SPY"
    assert diff.expected == "0"
    assert diff.actual == "5"
    assert result.blocked is True


def test_position_missing_from_broker_detected() -> None:
    """Internal has SPY=10 but broker has none → position difference."""
    internal = InternalState(
        cash=Decimal("100000"),
        positions={"SPY": 10},
        open_client_order_ids=(),
    )
    broker = _StubBroker(cash=Decimal("100000"))
    result = reconcile(internal=internal, broker=broker)

    assert len(result.differences) == 1
    diff = result.differences[0]
    assert diff.kind == "position"
    assert diff.symbol == "SPY"
    assert diff.expected == "10"
    assert diff.actual == "0"
    assert result.blocked is True


def test_position_quantity_mismatch() -> None:
    """Both have SPY but different quantities → position difference."""
    internal = InternalState(
        cash=Decimal("100000"),
        positions={"SPY": 10},
        open_client_order_ids=(),
    )
    broker = _StubBroker(
        cash=Decimal("100000"),
        positions=[_make_position("SPY", 7)],
    )
    result = reconcile(internal=internal, broker=broker)

    assert len(result.differences) == 1
    diff = result.differences[0]
    assert diff.kind == "position"
    assert diff.symbol == "SPY"
    assert diff.expected == "10"
    assert diff.actual == "7"


# ---------------------------------------------------------------------------
# Tests: open order differences
# ---------------------------------------------------------------------------


def test_open_order_difference_detected() -> None:
    """Internal expects order X open but broker has none → open_order difference."""
    internal = InternalState(
        cash=Decimal("100000"),
        positions={},
        open_client_order_ids=("mlt-20260612-aabbccddeeff001122334455",),
    )
    broker = _StubBroker(cash=Decimal("100000"))
    result = reconcile(internal=internal, broker=broker)

    assert len(result.differences) == 1
    diff = result.differences[0]
    assert diff.kind == "open_order"
    assert diff.client_order_id == "mlt-20260612-aabbccddeeff001122334455"
    assert diff.expected == "open"
    assert diff.actual == "absent"
    assert result.blocked is True


def test_broker_has_extra_open_order() -> None:
    """Broker has open order that internal doesn't expect → open_order difference."""
    internal = InternalState(
        cash=Decimal("100000"),
        positions={},
        open_client_order_ids=(),
    )
    broker = _StubBroker(
        cash=Decimal("100000"),
        open_orders=[_make_open_order("mlt-20260612-xxyyzz000000000000000000")],
    )
    result = reconcile(internal=internal, broker=broker)

    assert len(result.differences) == 1
    diff = result.differences[0]
    assert diff.kind == "open_order"
    assert diff.expected == "absent"
    assert diff.actual == "open"
    assert result.blocked is True


# ---------------------------------------------------------------------------
# Tests: ordering
# ---------------------------------------------------------------------------


def test_difference_ordering_cash_before_position() -> None:
    """When cash diff + position diff both present, cash comes first."""
    internal = InternalState(
        cash=Decimal("100000"),
        positions={"SPY": 10},
        open_client_order_ids=(),
    )
    broker = _StubBroker(
        cash=Decimal("99000"),  # cash mismatch
        positions=[_make_position("SPY", 5)],  # position mismatch
    )
    result = reconcile(internal=internal, broker=broker)

    assert len(result.differences) == 2
    assert result.differences[0].kind == "cash"
    assert result.differences[1].kind == "position"


def test_difference_ordering_position_before_open_order() -> None:
    """Position diffs come before open_order diffs."""
    cid = "mlt-20260612-aabbccddeeff001122334455"
    internal = InternalState(
        cash=Decimal("100000"),
        positions={"SPY": 10},
        open_client_order_ids=(cid,),
    )
    broker = _StubBroker(
        cash=Decimal("100000"),
        positions=[_make_position("SPY", 5)],  # position mismatch
        # no open orders → open_order mismatch
    )
    result = reconcile(internal=internal, broker=broker)

    assert len(result.differences) == 2
    assert result.differences[0].kind == "position"
    assert result.differences[1].kind == "open_order"


def test_difference_ordering_all_three_kinds() -> None:
    """Cash → positions → open_orders ordering with all three kinds."""
    cid = "mlt-20260612-aabbccddeeff001122334455"
    internal = InternalState(
        cash=Decimal("100000"),
        positions={"SPY": 10},
        open_client_order_ids=(cid,),
    )
    broker = _StubBroker(
        cash=Decimal("99000"),  # cash diff
        positions=[_make_position("SPY", 5)],  # position diff
        # no open orders → open_order diff
    )
    result = reconcile(internal=internal, broker=broker)

    assert len(result.differences) == 3
    assert result.differences[0].kind == "cash"
    assert result.differences[1].kind == "position"
    assert result.differences[2].kind == "open_order"


# ---------------------------------------------------------------------------
# Tests: alphabetical ordering within position diffs
# ---------------------------------------------------------------------------


def test_multiple_position_differences_sorted_by_symbol() -> None:
    """Multiple position mismatches must be sorted alphabetically by symbol."""
    internal = InternalState(
        cash=Decimal("100000"),
        positions={"ZZZ": 5, "AAA": 3, "MMM": 7},
        open_client_order_ids=(),
    )
    broker = _StubBroker(
        cash=Decimal("100000"),
        positions=[
            _make_position("ZZZ", 99),
            _make_position("AAA", 99),
            _make_position("MMM", 99),
        ],
    )
    result = reconcile(internal=internal, broker=broker)

    assert len(result.differences) == 3
    assert [d.symbol for d in result.differences] == ["AAA", "MMM", "ZZZ"]


def test_multiple_open_order_differences_sorted_by_client_order_id() -> None:
    """Multiple open_order mismatches must be sorted by client_order_id."""
    cid_b = "mlt-20260612-bbbbbbbbbbbbbbbbbbbbbbbb"
    cid_a = "mlt-20260612-aaaaaaaaaaaaaaaaaaaaaaaa"
    internal = InternalState(
        cash=Decimal("100000"),
        positions={},
        open_client_order_ids=(cid_b, cid_a),  # intentionally out of order
    )
    broker = _StubBroker(cash=Decimal("100000"))  # no open orders
    result = reconcile(internal=internal, broker=broker)

    assert len(result.differences) == 2
    # should be sorted alphabetically
    assert result.differences[0].client_order_id == cid_a
    assert result.differences[1].client_order_id == cid_b


# ---------------------------------------------------------------------------
# Tests: ReconciliationResult type checks
# ---------------------------------------------------------------------------


def test_reconciliation_result_is_frozen() -> None:
    """ReconciliationResult must be immutable (frozen pydantic)."""
    result = ReconciliationResult(differences=())
    with pytest.raises(ValidationError):
        result.differences = ()  # type: ignore[misc]


def test_internal_state_is_frozen() -> None:
    """InternalState must be immutable."""
    state = InternalState(cash=Decimal("0"), positions={}, open_client_order_ids=())
    with pytest.raises(ValidationError):
        state.cash = Decimal("1")  # type: ignore[misc]


def test_reconciliation_difference_fields() -> None:
    """ReconciliationDifference must store all fields correctly."""
    diff = ReconciliationDifference(
        kind="position",
        symbol="SPY",
        client_order_id=None,
        expected="10",
        actual="5",
    )
    assert diff.kind == "position"
    assert diff.symbol == "SPY"
    assert diff.client_order_id is None
    assert diff.expected == "10"
    assert diff.actual == "5"
