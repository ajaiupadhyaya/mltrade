"""Simulated (in-memory) broker for testing and offline workflows.

``SimulatedBroker`` implements the :class:`~mltrade.execution.broker.Broker`
Protocol and provides fine-grained outcome control for tests.

Outcome surface
---------------
``SubmitOutcome`` controls what happens when :meth:`~SimulatedBroker.submit`
is called for a *new* ``client_order_id``:

- ``COMPLETE_FILL``   — order recorded, status = FILLED, fill created.
- ``PARTIAL_FILL``    — order recorded, status = PARTIALLY_FILLED, partial
                        fill created (fraction configurable via
                        ``partial_fill_fraction``, default 0.5).
- ``REJECTED``        — order recorded, status = REJECTED, no fill.
- ``TIMEOUT_BEFORE``  — order NOT recorded; :exc:`~.BrokerTimeout` raised.
- ``TIMEOUT_AFTER``   — order IS recorded (status = NEW); THEN
                        :exc:`~.BrokerTimeout` raised.  This models a
                        network partition after the exchange accepted the
                        order — Task 12 reconciliation must handle it.

Dedup logic
-----------
If ``submit`` is called with a ``client_order_id`` that already exists in the
internal registry, the existing :class:`~.BrokerOrder` is returned immediately
with no side effects (idempotent).

Order IDs
---------
Broker-assigned order IDs follow ``"sim-{n}"`` where ``n`` is the 0-indexed
insertion counter (first accepted order = ``"sim-0"``).
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from math import floor

from mltrade.domain.time import require_utc
from mltrade.execution.broker import (
    BrokerAccount,
    BrokerError,  # noqa: F401 — re-exported for callers
    BrokerFill,
    BrokerOrder,
    BrokerPosition,
    BrokerTimeout,
    OrderSide,
    OrderStatus,
)
from mltrade.execution.intents import ExecutionIntent

# ---------------------------------------------------------------------------
# Outcome enum
# ---------------------------------------------------------------------------


class SubmitOutcome(StrEnum):
    """Controls the result of a new ``submit`` call in :class:`SimulatedBroker`.

    Values
    ------
    COMPLETE_FILL:
        Order recorded as FILLED with a single fill for the full quantity.
    PARTIAL_FILL:
        Order recorded as PARTIALLY_FILLED with a fill for
        ``floor(quantity * partial_fill_fraction)`` shares.
    REJECTED:
        Order recorded as REJECTED; no fill is created.
    TIMEOUT_BEFORE:
        :exc:`~mltrade.execution.broker.BrokerTimeout` raised *before* the
        order is recorded.  ``get_order_by_client_id`` returns ``None``.
    TIMEOUT_AFTER:
        Order recorded with status NEW; then
        :exc:`~mltrade.execution.broker.BrokerTimeout` raised.
        ``get_order_by_client_id`` returns the NEW order.
    """

    COMPLETE_FILL = "complete_fill"
    PARTIAL_FILL = "partial_fill"
    REJECTED = "rejected"
    TIMEOUT_BEFORE = "timeout_before"
    TIMEOUT_AFTER = "timeout_after"


# ---------------------------------------------------------------------------
# SimulatedBroker
# ---------------------------------------------------------------------------

_OPEN_STATUSES: frozenset[OrderStatus] = frozenset(
    {OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED}
)

# Deterministic default fill timestamp. The simulated broker must be fully
# reproducible (persisted demo evidence is compared across runs), so it never
# reads the wall clock; callers needing real times inject ``fill_timestamp``.
_DEFAULT_FILL_TIMESTAMP = datetime(2026, 1, 1, tzinfo=UTC)


class SimulatedBroker:
    """In-memory broker simulation for testing and offline workflows.

    Satisfies the :class:`~mltrade.execution.broker.Broker` Protocol.

    Parameters
    ----------
    account:
        Static account snapshot returned by :meth:`get_account`.
    default_outcome:
        Applied to all ``submit`` calls not in ``outcome_map``.
    outcome_map:
        ``client_order_id`` → :class:`SubmitOutcome` overrides.
    partial_fill_fraction:
        Fraction of quantity filled when outcome is ``PARTIAL_FILL``
        (0 < fraction < 1).
    """

    def __init__(
        self,
        account: BrokerAccount,
        default_outcome: SubmitOutcome = SubmitOutcome.COMPLETE_FILL,
        outcome_map: dict[str, SubmitOutcome] | None = None,
        partial_fill_fraction: float = 0.5,
        fill_timestamp: datetime | None = None,
    ) -> None:
        if not 0.0 < partial_fill_fraction < 1.0:
            raise ValueError("partial_fill_fraction must be in the open (0, 1)")
        self._account = account
        self._default_outcome = default_outcome
        self._outcome_map: dict[str, SubmitOutcome] = outcome_map or {}
        self._partial_fill_fraction = partial_fill_fraction
        self._fill_timestamp = (
            require_utc(fill_timestamp)
            if fill_timestamp is not None
            else _DEFAULT_FILL_TIMESTAMP
        )

        # Ordered insertion registry: client_order_id → BrokerOrder
        self._orders: dict[str, BrokerOrder] = {}
        # All fills, in insertion order
        self._fills: list[BrokerFill] = []
        # Counter for deterministic broker order IDs
        self._counter: int = 0

    # ------------------------------------------------------------------
    # Broker Protocol implementation
    # ------------------------------------------------------------------

    def get_account(self) -> BrokerAccount:
        """Return the static account snapshot provided at construction."""
        return self._account

    def list_positions(self) -> tuple[BrokerPosition, ...]:
        """Return current positions (always empty for simulated broker)."""
        return ()

    def list_open_orders(self) -> tuple[BrokerOrder, ...]:
        """Return only NEW and PARTIALLY_FILLED orders."""
        return tuple(
            order
            for order in self._orders.values()
            if order.status in _OPEN_STATUSES
        )

    def list_orders(self) -> tuple[BrokerOrder, ...]:
        """Return all orders (open + closed), in insertion order."""
        return tuple(self._orders.values())

    def list_recent_fills(self) -> tuple[BrokerFill, ...]:
        """Return all fills in insertion order."""
        return tuple(self._fills)

    def get_order_by_client_id(self, client_order_id: str) -> BrokerOrder | None:
        """Look up an order by client_order_id.  Returns ``None`` if absent."""
        return self._orders.get(client_order_id)

    def submit(self, intent: ExecutionIntent) -> BrokerOrder:
        """Submit an execution intent.

        Deduplication
        -------------
        If ``intent.client_order_id`` is already in the registry, the existing
        :class:`~mltrade.execution.broker.BrokerOrder` is returned immediately
        with no side effects.

        Outcome dispatch
        ----------------
        The outcome is resolved from ``outcome_map`` (if the client_order_id
        appears there) or ``default_outcome``.  See :class:`SubmitOutcome` for
        per-outcome semantics.

        Raises
        ------
        BrokerTimeout
            On ``TIMEOUT_BEFORE`` (before recording) or ``TIMEOUT_AFTER``
            (after recording as NEW).
        """
        client_order_id = intent.client_order_id

        # --- Dedup: return existing order immediately ---
        if client_order_id in self._orders:
            return self._orders[client_order_id]

        outcome = self._outcome_map.get(client_order_id, self._default_outcome)
        quantity = int(intent.target_quantity)

        # --- TIMEOUT_BEFORE: raise without recording ---
        if outcome is SubmitOutcome.TIMEOUT_BEFORE:
            raise BrokerTimeout(
                f"Simulated timeout before recording order {client_order_id!r}"
            )

        # --- All other outcomes: assign broker ID and record ---
        broker_id = f"sim-{self._counter}"
        self._counter += 1

        # Compute fill quantity and final status
        if outcome is SubmitOutcome.COMPLETE_FILL:
            filled_qty = quantity
            status = OrderStatus.FILLED
        elif outcome is SubmitOutcome.PARTIAL_FILL:
            filled_qty = floor(quantity * self._partial_fill_fraction)
            status = OrderStatus.PARTIALLY_FILLED
        elif outcome is SubmitOutcome.REJECTED:
            filled_qty = 0
            status = OrderStatus.REJECTED
        else:
            # TIMEOUT_AFTER: record as NEW, then raise
            filled_qty = 0
            status = OrderStatus.NEW

        order = BrokerOrder(
            id=broker_id,
            client_order_id=client_order_id,
            symbol=intent.symbol,
            side=intent.side,
            quantity=quantity,
            status=status,
            filled_quantity=filled_qty,
            limit_price=None,
        )
        self._orders[client_order_id] = order

        # Create fill record(s) for COMPLETE_FILL and PARTIAL_FILL
        if outcome in (SubmitOutcome.COMPLETE_FILL, SubmitOutcome.PARTIAL_FILL):
            fill = BrokerFill(
                order_id=broker_id,
                client_order_id=client_order_id,
                symbol=intent.symbol,
                quantity=filled_qty,
                price=Decimal("100.00"),  # synthetic price for simulation
                timestamp=self._fill_timestamp,
            )
            self._fills.append(fill)

        # --- TIMEOUT_AFTER: raise after recording ---
        if outcome is SubmitOutcome.TIMEOUT_AFTER:
            raise BrokerTimeout(
                f"Simulated timeout after recording order {client_order_id!r}"
            )

        return order

    # ------------------------------------------------------------------
    # Side/size helpers (used by tests for OrderSide)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_buy(side: OrderSide) -> bool:
        return side is OrderSide.BUY
