"""Broker value objects, exceptions, and Protocol.

Public types
------------
- ``OrderSide``     — StrEnum: buy, sell
- ``OrderStatus``   — StrEnum: new, partially_filled, filled, rejected
- ``BrokerAccount`` — frozen: account cash/equity/flags
- ``BrokerPosition``— frozen: symbol + quantity + avg_price
- ``BrokerOrder``   — frozen: order identity + fill status
- ``BrokerFill``    — frozen: a single confirmed fill (UTC timestamp enforced)
- ``BrokerError``   — base exception
- ``BrokerTimeout`` — raised on timeout (before or after order submission)
- ``Broker``        — structural Protocol fulfilled by both SimulatedBroker
                      and the future AlpacaBroker adapter
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict, field_validator

from mltrade.domain.time import require_utc

if TYPE_CHECKING:
    from mltrade.execution.intents import ExecutionIntent


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(StrEnum):
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    REJECTED = "rejected"


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


class BrokerAccount(BaseModel):
    """Snapshot of broker account state."""

    model_config = ConfigDict(frozen=True)

    id: str
    status: str
    cash: Decimal
    equity: Decimal
    account_blocked: bool
    trading_blocked: bool
    pattern_day_trader: bool


class BrokerPosition(BaseModel):
    """Current broker-side position for one symbol."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    quantity: int
    avg_price: Decimal


class BrokerOrder(BaseModel):
    """Broker representation of a submitted order."""

    model_config = ConfigDict(frozen=True)

    id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    status: OrderStatus
    filled_quantity: int
    limit_price: Decimal | None = None


class BrokerFill(BaseModel):
    """A single confirmed execution fill.

    ``timestamp`` is UTC-enforced via :func:`~mltrade.domain.time.require_utc`.
    """

    model_config = ConfigDict(frozen=True)

    order_id: str
    client_order_id: str
    symbol: str
    quantity: int
    price: Decimal
    timestamp: datetime

    @field_validator("timestamp")
    @classmethod
    def _enforce_utc(cls, value: datetime) -> datetime:
        return require_utc(value)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class BrokerError(Exception):
    """Base class for all broker-related errors."""


class BrokerTimeout(BrokerError):
    """Raised when a broker operation times out.

    May be raised *before* the order is recorded (TIMEOUT_BEFORE) or *after*
    the order has been recorded with status NEW (TIMEOUT_AFTER).  Callers must
    handle both cases — see :class:`~mltrade.execution.simulated.SimulatedBroker`.
    """


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class Broker(Protocol):
    """Structural interface fulfilled by any broker adapter."""

    def get_account(self) -> BrokerAccount: ...

    def list_positions(self) -> tuple[BrokerPosition, ...]: ...

    def list_open_orders(self) -> tuple[BrokerOrder, ...]: ...

    def list_orders(self) -> tuple[BrokerOrder, ...]: ...

    def list_recent_fills(self) -> tuple[BrokerFill, ...]: ...

    def get_order_by_client_id(self, client_order_id: str) -> BrokerOrder | None: ...

    def submit(self, intent: ExecutionIntent) -> BrokerOrder: ...
