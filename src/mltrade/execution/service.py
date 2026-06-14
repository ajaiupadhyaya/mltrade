"""Safe execution service with reconciliation and idempotent submission.

Public API
----------
- ``IntentOutcome``    — per-intent submission result.
- ``SubmitResult``     — aggregate result of a :meth:`ExecutionService.submit`
                         call.
- ``Preview``          — pre-submission snapshot: intents + reconciliation +
                         risk report.  ``blocked`` = any gate tripped.
- ``ExecutionService`` — orchestrates preview + safe submit.

Timeout handling (fail-closed)
-------------------------------
The service differentiates two timeout flavours:

TIMEOUT_AFTER (broker recorded the order, THEN raised BrokerTimeout):
    - ``get_order_by_client_id`` returns the existing order.
    - Status: ``"submitted"`` — the order is live; do NOT resubmit.

TIMEOUT_BEFORE (broker raised BrokerTimeout before recording the order):
    - ``get_order_by_client_id`` returns ``None``.
    - The service retries **once**.
    - If the retry also times out → status ``"timeout_failed"`` (not submitted).
    - If the retry succeeds → status ``"submitted"``.

This guarantees at-most-once order creation for any given client_order_id.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import date
from decimal import Decimal

from pydantic import BaseModel, ConfigDict

from mltrade.config import Settings
from mltrade.execution.broker import Broker, BrokerTimeout, OrderSide, OrderStatus
from mltrade.execution.intents import ExecutionIntent, build_intent
from mltrade.execution.reconciliation import (
    InternalState,
    ReconciliationResult,
    reconcile,
)
from mltrade.risk.checks import RiskReport
from mltrade.risk.policy import PreTradeContext, evaluate_pre_trade

# ---------------------------------------------------------------------------
# Result value objects
# ---------------------------------------------------------------------------


class IntentOutcome(BaseModel):
    """Result of submitting one :class:`~mltrade.execution.intents.ExecutionIntent`.

    Attributes
    ----------
    client_order_id:
        The deterministic client-order ID of the intent.
    symbol:
        The instrument ticker.
    status:
        One of:
        - ``"submitted"``       — order is live at the broker (new or existing).
        - ``"already_existed"`` — broker already had this order (idempotent check).
        - ``"rejected"``        — broker accepted but immediately rejected.
        - ``"timeout_failed"``  — two consecutive timeouts; order not confirmed.
        - ``"blocked"``         — preview was blocked; no submission attempted.
    """

    model_config = ConfigDict(frozen=True)

    client_order_id: str
    symbol: str
    status: str


class SubmitResult(BaseModel):
    """Aggregate outcome of a :meth:`ExecutionService.submit` call.

    Attributes
    ----------
    submitted:
        Count of intents now confirmed live at the broker (new + already_existed).
    blocked:
        True when the preview was blocked and no submission was attempted.
    outcomes:
        Per-intent :class:`IntentOutcome` objects, in the same order as
        ``preview.intents``.
    """

    model_config = ConfigDict(frozen=True)

    submitted: int
    blocked: bool
    outcomes: tuple[IntentOutcome, ...]


class Preview(BaseModel):
    """Pre-submission snapshot assembled by :meth:`ExecutionService.preview`.

    Attributes
    ----------
    intents:
        Execution intents proposed for submission (after minimum-notional filter).
    reconciliation:
        Result of comparing internal state against the live broker state.
    risk_report:
        Full pre-trade risk gate report.

    Properties
    ----------
    blocked:
        ``True`` when either :attr:`risk_report` or :attr:`reconciliation` is
        blocked.
    """

    model_config = ConfigDict(frozen=True)

    intents: tuple[ExecutionIntent, ...]
    reconciliation: ReconciliationResult
    risk_report: RiskReport

    @property
    def blocked(self) -> bool:
        """True when risk or reconciliation blocked submission."""
        return self.risk_report.blocked or self.reconciliation.blocked


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

# Statuses that count as "the order is live at the broker"
_SUBMITTED_STATUSES: frozenset[OrderStatus] = frozenset(
    {
        OrderStatus.NEW,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.FILLED,
    }
)


class ExecutionService:
    """Orchestrates pre-trade risk gating, reconciliation, and safe submission.

    Parameters
    ----------
    broker:
        Broker adapter satisfying the :class:`~mltrade.execution.broker.Broker`
        Protocol.
    risk_evaluator:
        Callable that accepts a :class:`~mltrade.risk.policy.PreTradeContext`
        and returns a :class:`~mltrade.risk.checks.RiskReport`.  Defaults to
        :func:`~mltrade.risk.policy.evaluate_pre_trade`.
    """

    def __init__(
        self,
        broker: Broker,
        *,
        risk_evaluator: Callable[[PreTradeContext], RiskReport] = evaluate_pre_trade,
    ) -> None:
        self._broker = broker
        self._risk_evaluator = risk_evaluator

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def preview(
        self,
        *,
        target_positions: Mapping[str, int],
        internal_state: InternalState,
        settings: Settings,
        strategy_version: str,
        decision_session: date,
        environment: str,
        prices: Mapping[str, Decimal],
    ) -> Preview:
        """Build a pre-submission preview without placing any orders.

        Steps
        -----
        1. Fetch broker account, positions, and open orders (one call each).
        2. Run reconciliation (internal vs. broker).
        3. Compute delta quantities (target - current broker qty).
        4. Filter out deltas whose |notional| < ``settings.minimum_order_notional``.
        5. Build :class:`~mltrade.execution.intents.ExecutionIntent` objects for
           remaining deltas.
        6. Build :class:`~mltrade.risk.policy.PreTradeContext` and evaluate.
        7. Return :class:`Preview`.

        Parameters
        ----------
        target_positions:
            Desired symbol → share-count after the rebalance.
        internal_state:
            System's belief about current state (used for reconciliation).
        settings:
            Runtime configuration (limits, flags).
        strategy_version:
            Strategy/model version label embedded in each intent.
        decision_session:
            XNYS session date on which decisions were made.
        environment:
            Deployment environment string (e.g. ``"paper"``).
        prices:
            Symbol → current price, used for notional computation.
        """
        # --- 1. Fetch broker state ---
        account = self._broker.get_account()
        broker_positions = self._broker.list_positions()

        # --- 2. Reconcile ---
        reconciliation = reconcile(internal=internal_state, broker=self._broker)

        # --- 3. Compute deltas ---
        broker_pos_map: dict[str, int] = {
            p.symbol: p.quantity for p in broker_positions
        }
        deltas: dict[str, int] = {}
        for symbol, target_qty in target_positions.items():
            current_qty = broker_pos_map.get(symbol, 0)
            delta = target_qty - current_qty
            if delta != 0:
                deltas[symbol] = delta

        # --- 4. Filter by minimum notional ---
        filtered_deltas: dict[str, int] = {}
        for symbol, delta in deltas.items():
            price = prices.get(symbol, Decimal("0"))
            notional = abs(delta) * price
            if notional >= settings.minimum_order_notional:
                filtered_deltas[symbol] = delta

        # --- 5. Build intents ---
        intents: list[ExecutionIntent] = []
        for symbol, delta in sorted(filtered_deltas.items()):
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            intent = build_intent(
                environment=environment,
                strategy_version=strategy_version,
                decision_session=decision_session,
                symbol=symbol,
                side=side,
                target_quantity=Decimal(str(abs(delta))),
            )
            intents.append(intent)

        # --- 6. Build PreTradeContext ---
        equity = account.equity
        # Weights: use notional / equity; treat missing price as 0
        weights: dict[str, Decimal] = {}
        order_notionals: dict[str, Decimal] = {}
        for intent in intents:
            price = prices.get(intent.symbol, Decimal("0"))
            signed_notional = (
                Decimal(str(int(intent.target_quantity))) * price
                if intent.side is OrderSide.BUY
                else -Decimal(str(int(intent.target_quantity))) * price
            )
            order_notionals[intent.client_order_id] = signed_notional

        # Build target weight map from target_positions (not just delta intents)
        if equity > Decimal("0"):
            for symbol, target_qty in target_positions.items():
                price = prices.get(symbol, Decimal("0"))
                weights[symbol] = Decimal(str(target_qty)) * price / equity
        else:
            for symbol in target_positions:
                weights[symbol] = Decimal("0")

        cash_weight: Decimal
        total_invested = sum(abs(w) for w in weights.values())
        if total_invested <= Decimal("1"):
            cash_weight = Decimal("1") - total_invested
        else:
            cash_weight = Decimal("0")

        context = PreTradeContext(
            # Snapshot / session provenance — treat everything as fresh
            snapshot_blocked=False,
            snapshot_last_session=decision_session,
            expected_last_session=decision_session,
            decision_session=decision_session,
            expected_decision_session=decision_session,
            # Model / feature versioning
            model_version=strategy_version,
            expected_model_version=strategy_version,
            feature_version=strategy_version,
            expected_feature_version=strategy_version,
            # Portfolio weights
            weights=weights,
            cash_weight=cash_weight,
            # Order intents
            order_notionals=order_notionals,
            equity=equity,
            intent_client_ids=tuple(i.client_order_id for i in intents),
            # Risk limits
            maximum_position_weight=settings.maximum_position_weight,
            minimum_cash_weight=settings.minimum_cash_weight,
            maximum_order_weight=settings.maximum_order_weight,
            maximum_rebalance_weight=settings.maximum_rebalance_weight,
            minimum_order_notional=settings.minimum_order_notional,
            # Broker / account state
            broker_account_active=(account.status == "ACTIVE"),
            broker_account_blocked=(
                account.account_blocked or account.trading_blocked
            ),
            reconciliation_ok=not reconciliation.blocked,
            # Safety
            live_trading_enabled=settings.live_trading_enabled,
        )

        risk_report = self._risk_evaluator(context)

        return Preview(
            intents=tuple(intents),
            reconciliation=reconciliation,
            risk_report=risk_report,
        )

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------

    def submit(self, preview: Preview) -> SubmitResult:
        """Submit all intents in a blocked-checked, idempotent, fail-closed manner.

        If ``preview.blocked`` is True, no orders are placed and all outcomes
        carry status ``"blocked"``.

        Timeout handling
        ----------------
        After catching :exc:`~mltrade.execution.broker.BrokerTimeout`:

        1. Query ``broker.get_order_by_client_id(intent.client_order_id)``.
        2. If the order exists → **TIMEOUT_AFTER**: count as submitted; do NOT
           resubmit.
        3. If the order does not exist → **TIMEOUT_BEFORE**: retry once.
           - If the retry also times out → ``"timeout_failed"`` (not submitted).
           - If the retry succeeds → ``"submitted"``.

        Parameters
        ----------
        preview:
            The :class:`Preview` produced by :meth:`preview`.

        Returns
        -------
        SubmitResult
        """
        if preview.blocked:
            blocked_outcomes = tuple(
                IntentOutcome(
                    client_order_id=intent.client_order_id,
                    symbol=intent.symbol,
                    status="blocked",
                )
                for intent in preview.intents
            )
            return SubmitResult(submitted=0, blocked=True, outcomes=blocked_outcomes)

        outcomes: list[IntentOutcome] = []
        submitted_count = 0

        for intent in preview.intents:
            # --- Idempotency check: does this order already exist? ---
            existing = self._broker.get_order_by_client_id(intent.client_order_id)
            if existing is not None:
                outcomes.append(
                    IntentOutcome(
                        client_order_id=intent.client_order_id,
                        symbol=intent.symbol,
                        status="already_existed",
                    )
                )
                submitted_count += 1
                continue

            # --- First submission attempt ---
            try:
                order = self._broker.submit(intent)
            except BrokerTimeout:
                # Determine if this was TIMEOUT_BEFORE or TIMEOUT_AFTER
                post_timeout_order = self._broker.get_order_by_client_id(
                    intent.client_order_id
                )
                if post_timeout_order is not None:
                    # TIMEOUT_AFTER: order recorded, do not resubmit
                    outcomes.append(
                        IntentOutcome(
                            client_order_id=intent.client_order_id,
                            symbol=intent.symbol,
                            status="submitted",
                        )
                    )
                    submitted_count += 1
                else:
                    # TIMEOUT_BEFORE: retry once
                    try:
                        order = self._broker.submit(intent)
                    except BrokerTimeout:
                        # Second timeout → fail closed
                        outcomes.append(
                            IntentOutcome(
                                client_order_id=intent.client_order_id,
                                symbol=intent.symbol,
                                status="timeout_failed",
                            )
                        )
                    else:
                        # Retry succeeded
                        status = _order_status_to_outcome(order.status)
                        outcomes.append(
                            IntentOutcome(
                                client_order_id=intent.client_order_id,
                                symbol=intent.symbol,
                                status=status,
                            )
                        )
                        if status == "submitted":
                            submitted_count += 1
                continue

            # --- Normal path (no timeout) ---
            status = _order_status_to_outcome(order.status)
            outcomes.append(
                IntentOutcome(
                    client_order_id=intent.client_order_id,
                    symbol=intent.symbol,
                    status=status,
                )
            )
            if status == "submitted":
                submitted_count += 1

        return SubmitResult(
            submitted=submitted_count,
            blocked=False,
            outcomes=tuple(outcomes),
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _order_status_to_outcome(status: OrderStatus) -> str:
    """Map a broker order status to an IntentOutcome status string."""
    if status in _SUBMITTED_STATUSES:
        return "submitted"
    if status is OrderStatus.REJECTED:
        return "rejected"
    return "submitted"  # fallback for any future statuses
