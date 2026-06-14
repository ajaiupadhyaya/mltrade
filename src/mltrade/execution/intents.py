"""Idempotent execution intents.

Public API
----------
- ``ExecutionIntent`` — frozen value object with a deterministic
  ``client_order_id`` (sha256-based, cross-process stable).
- ``build_intent``    — factory that computes and injects ``client_order_id``.

``client_order_id`` format
--------------------------
    mlt-YYYYMMDD-<24-char lowercase hex>

The hex fragment is the first 24 characters of the SHA-256 digest of a
canonical JSON payload.  This guarantees:
- Same inputs → same ID (idempotency).
- Different inputs → different IDs (collision resistance).
- Stability across Python processes and ``PYTHONHASHSEED`` values
  (uses SHA-256, not builtin ``hash()``).
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from decimal import Decimal

from pydantic import BaseModel, ConfigDict

from mltrade.execution.broker import OrderSide

# ---------------------------------------------------------------------------
# Value object
# ---------------------------------------------------------------------------


class ExecutionIntent(BaseModel):
    """Immutable execution intent with a deterministic client-order-ID.

    Fields
    ------
    environment:
        Deployment environment string (e.g. ``"paper"``).
    strategy_version:
        Model / strategy label (e.g. ``"ridge-trend-v1"``).
    decision_session:
        XNYS session date on which the decision was made.
    symbol:
        Instrument ticker (e.g. ``"SPY"``).
    side:
        :class:`~mltrade.execution.broker.OrderSide` (``BUY`` or ``SELL``).
    target_quantity:
        Signed target quantity as :class:`~decimal.Decimal`.
        The broker adapter converts this to ``int`` when submitting.
    client_order_id:
        Deterministic, idempotent order identifier computed by
        :func:`build_intent`.  Do not set manually.
    """

    model_config = ConfigDict(frozen=True)

    environment: str
    strategy_version: str
    decision_session: date
    symbol: str
    side: OrderSide
    target_quantity: Decimal
    client_order_id: str


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_intent(
    *,
    environment: str,
    strategy_version: str,
    decision_session: date,
    symbol: str,
    side: OrderSide,
    target_quantity: Decimal,
) -> ExecutionIntent:
    """Build an :class:`ExecutionIntent` with a deterministic client order ID.

    The ``client_order_id`` is derived from a canonical JSON payload via
    SHA-256, making it:

    - **Idempotent** — identical inputs always produce the same ID.
    - **Cross-process stable** — not affected by ``PYTHONHASHSEED``.
    - **Collision-resistant** — any input difference produces a different ID.

    Parameters
    ----------
    environment:
        Deployment environment string.
    strategy_version:
        Strategy / model version label.
    decision_session:
        XNYS session date of the decision.
    symbol:
        Instrument ticker.
    side:
        ``BUY`` or ``SELL``.
    target_quantity:
        Target share quantity as :class:`~decimal.Decimal`.

    Returns
    -------
    ExecutionIntent
        Frozen value object with ``client_order_id`` set.
    """
    payload = json.dumps(
        {
            "environment": environment,
            "strategy_version": strategy_version,
            "decision_session": decision_session.isoformat(),
            "symbol": symbol,
            "side": str(side.value),
            "target_quantity": str(target_quantity),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("ascii")).hexdigest()[:24]
    client_order_id = f"mlt-{decision_session:%Y%m%d}-{digest}"
    return ExecutionIntent(
        environment=environment,
        strategy_version=strategy_version,
        decision_session=decision_session,
        symbol=symbol,
        side=side,
        target_quantity=target_quantity,
        client_order_id=client_order_id,
    )
