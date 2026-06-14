"""Reconciliation between internal state and broker state.

Public API
----------
- ``ReconciliationDifference`` — frozen value object describing one discrepancy.
- ``ReconciliationResult``     — frozen collection of differences; ``blocked``
                                  property is True iff any differences exist.
- ``InternalState``            — frozen snapshot of what the system believes to
                                  be true: cash, positions, and open order IDs.
- ``reconcile``                — compare InternalState against a live Broker and
                                  return a ReconciliationResult.

Difference ordering
-------------------
Differences are emitted in a deterministic, stable order:
1. Cash difference (at most one), if any.
2. Position differences, sorted alphabetically by symbol.
3. Open-order differences, sorted alphabetically by client_order_id.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal

from pydantic import BaseModel, ConfigDict

from mltrade.execution.broker import Broker

# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


class ReconciliationDifference(BaseModel):
    """One reconciliation discrepancy between internal and broker state.

    Attributes
    ----------
    kind:
        ``"cash"`` | ``"position"`` | ``"open_order"``
    symbol:
        Instrument ticker.  ``None`` for ``kind == "cash"`` and
        ``kind == "open_order"`` differences.
    client_order_id:
        Client-order identifier.  ``None`` unless ``kind == "open_order"``.
    expected:
        String representation of the value the internal ledger believes.
    actual:
        String representation of the value observed from the broker.
    """

    model_config = ConfigDict(frozen=True)

    kind: str
    symbol: str | None
    client_order_id: str | None
    expected: str
    actual: str


class ReconciliationResult(BaseModel):
    """Immutable collection of reconciliation differences.

    Attributes
    ----------
    differences:
        All discrepancies, in deterministic order (cash → positions A-Z →
        open_orders A-Z).

    Properties
    ----------
    blocked:
        ``True`` when any difference is present.
    """

    model_config = ConfigDict(frozen=True)

    differences: tuple[ReconciliationDifference, ...] = ()

    @property
    def blocked(self) -> bool:
        """True when at least one difference is present."""
        return len(self.differences) > 0


class InternalState(BaseModel):
    """Snapshot of the system's internal beliefs about account state.

    Attributes
    ----------
    cash:
        Cash the system believes the account holds.
    positions:
        Symbol → share quantity the system believes is currently held.
    open_client_order_ids:
        Client order IDs the system believes are still open at the broker.
    """

    model_config = ConfigDict(frozen=True)

    cash: Decimal
    positions: Mapping[str, int]
    open_client_order_ids: tuple[str, ...]


# ---------------------------------------------------------------------------
# Reconcile function
# ---------------------------------------------------------------------------


def reconcile(*, internal: InternalState, broker: Broker) -> ReconciliationResult:
    """Compare *internal* state against the live *broker* and return differences.

    Parameters
    ----------
    internal:
        What the system believes to be true.
    broker:
        Live broker adapter (``get_account``, ``list_positions``,
        ``list_open_orders`` are called exactly once each).

    Returns
    -------
    ReconciliationResult
        Immutable result.  ``result.blocked`` is ``True`` iff any differences
        are found.

    Difference ordering
    -------------------
    Cash difference first (at most one), then position differences sorted
    alphabetically by symbol, then open-order differences sorted alphabetically
    by client_order_id.
    """
    diffs: list[ReconciliationDifference] = []

    # --- Fetch broker state (one call each) ---
    account = broker.get_account()
    broker_positions = broker.list_positions()
    broker_open_orders = broker.list_open_orders()

    # ------------------------------------------------------------------
    # 1. Cash
    # ------------------------------------------------------------------
    broker_cash = account.cash
    if internal.cash != broker_cash:
        diffs.append(
            ReconciliationDifference(
                kind="cash",
                symbol=None,
                client_order_id=None,
                expected=str(internal.cash),
                actual=str(broker_cash),
            )
        )

    # ------------------------------------------------------------------
    # 2. Positions (sorted by symbol alphabetically)
    # ------------------------------------------------------------------
    broker_pos_map: dict[str, int] = {p.symbol: p.quantity for p in broker_positions}
    all_symbols: set[str] = set(internal.positions.keys()) | set(broker_pos_map.keys())

    for symbol in sorted(all_symbols):
        internal_qty = internal.positions.get(symbol, 0)
        broker_qty = broker_pos_map.get(symbol, 0)
        if internal_qty != broker_qty:
            diffs.append(
                ReconciliationDifference(
                    kind="position",
                    symbol=symbol,
                    client_order_id=None,
                    expected=str(internal_qty),
                    actual=str(broker_qty),
                )
            )

    # ------------------------------------------------------------------
    # 3. Open orders (sorted by client_order_id alphabetically)
    # ------------------------------------------------------------------
    broker_open_cids: set[str] = {o.client_order_id for o in broker_open_orders}
    internal_open_cids: set[str] = set(internal.open_client_order_ids)
    all_open_cids: set[str] = internal_open_cids | broker_open_cids

    for cid in sorted(all_open_cids):
        internal_has = cid in internal_open_cids
        broker_has = cid in broker_open_cids
        if internal_has != broker_has:
            diffs.append(
                ReconciliationDifference(
                    kind="open_order",
                    symbol=None,
                    client_order_id=cid,
                    expected="open" if internal_has else "absent",
                    actual="open" if broker_has else "absent",
                )
            )

    return ReconciliationResult(differences=tuple(diffs))
