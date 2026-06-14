"""Alpaca paper-trading broker adapter.

Implements the :class:`~mltrade.execution.broker.Broker` Protocol against the
Alpaca paper trading REST API (``https://paper-api.alpaca.markets``).

Safety invariant
----------------
Every public method validates that the configured ``base_url`` is the Alpaca
**paper** endpoint.  If a live URL is supplied the method raises
:class:`~mltrade.execution.broker.BrokerSafetyError` before any network I/O
takes place.

Secrets policy
--------------
API credentials are accepted as :class:`~pydantic.SecretStr` and are only
unwrapped inside ``_auth_headers()``.  They are **never** logged, included in
exception messages, or exposed in repr output.

Module constants
----------------
- ``ALPACA_PAPER_BASE_URL`` — canonical Alpaca paper endpoint
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from decimal import Decimal
from typing import cast
from urllib.parse import urlparse

import httpx
from pydantic import SecretStr

from mltrade.execution.broker import (
    BrokerAccount,
    BrokerFill,
    BrokerOrder,
    BrokerPosition,
    BrokerSafetyError,
    OrderSide,
    OrderStatus,
)
from mltrade.execution.intents import ExecutionIntent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

ALPACA_PAPER_BASE_URL: str = "https://paper-api.alpaca.markets"

_PAPER_HOST: str = "paper-api.alpaca.markets"

# Convenience URL constants (informational; adapter uses injected base_url)
ALPACA_ACCOUNT_PATH: str = "/v2/account"
ALPACA_POSITIONS_PATH: str = "/v2/positions"
ALPACA_ORDERS_PATH: str = "/v2/orders"
ALPACA_ACTIVITIES_PATH: str = "/v2/account/activities"

_TIMEOUT = httpx.Timeout(30.0)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STATUS_MAP: dict[str, OrderStatus] = {
    "new": OrderStatus.NEW,
    "accepted": OrderStatus.NEW,
    "pending_new": OrderStatus.NEW,
    "accepted_for_bidding": OrderStatus.NEW,
    "stopped": OrderStatus.NEW,
    "held": OrderStatus.NEW,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "filled": OrderStatus.FILLED,
    "canceled": OrderStatus.REJECTED,
    "expired": OrderStatus.REJECTED,
    "rejected": OrderStatus.REJECTED,
    "replaced": OrderStatus.REJECTED,
    "pending_cancel": OrderStatus.NEW,
    "pending_replace": OrderStatus.NEW,
    "done_for_day": OrderStatus.REJECTED,
    "suspended": OrderStatus.REJECTED,
    "calculated": OrderStatus.FILLED,
}


def _map_status(raw: str) -> OrderStatus:
    try:
        return _STATUS_MAP[raw.lower()]
    except KeyError:
        return OrderStatus.NEW


def _map_order(raw: dict[str, object]) -> BrokerOrder:
    return BrokerOrder(
        id=str(raw["id"]),
        client_order_id=str(raw["client_order_id"]),
        symbol=str(raw["symbol"]),
        side=OrderSide(str(raw["side"])),
        quantity=int(Decimal(str(raw["qty"]))),
        status=_map_status(str(raw["status"])),
        filled_quantity=int(Decimal(str(raw.get("filled_qty") or "0"))),
        limit_price=(
            Decimal(str(raw["limit_price"]))
            if raw.get("limit_price") is not None
            else None
        ),
    )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class AlpacaPaperBroker:
    """Alpaca paper-trading broker fulfilling the ``Broker`` protocol.

    Parameters
    ----------
    client:
        Synchronous :class:`httpx.Client`.  Caller manages lifecycle.
    base_url:
        Must be ``https://paper-api.alpaca.markets`` (or a test mock of it).
        Any other value causes :class:`BrokerSafetyError` to be raised on the
        first method call.
    api_key:
        Alpaca API key ID.
    api_secret:
        Alpaca API secret key.
    """

    def __init__(
        self,
        *,
        client: httpx.Client,
        base_url: str,
        api_key: SecretStr,
        api_secret: SecretStr,
    ) -> None:
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._api_secret = api_secret

    # ------------------------------------------------------------------
    # Safety
    # ------------------------------------------------------------------

    def _assert_paper_url(self) -> None:
        """Raise :class:`BrokerSafetyError` if base_url is not the paper endpoint."""
        # Compare the parsed hostname exactly — a substring check would accept
        # a crafted host like ``paper-api.alpaca.markets.evil.com``.
        host = urlparse(self._base_url).hostname
        if host != _PAPER_HOST:
            raise BrokerSafetyError(
                f"base_url must be the Alpaca paper endpoint "
                f"({_PAPER_HOST}), not a live URL: {self._base_url!r}"
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _auth_headers(self) -> dict[str, str]:
        """Return Alpaca auth headers.  Secrets never logged."""
        return {
            "APCA-API-KEY-ID": self._api_key.get_secret_value(),
            "APCA-API-SECRET-KEY": self._api_secret.get_secret_value(),
        }

    def _get(self, path: str, **params: str | None) -> object:
        url = f"{self._base_url}{path}"
        filtered: dict[str, str] = {
            k: v for k, v in params.items() if v is not None
        }
        response = self._client.get(
            url,
            params=filtered,
            headers=self._auth_headers(),
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    def _post(self, path: str, body: dict[str, object]) -> object:
        url = f"{self._base_url}{path}"
        response = self._client.post(
            url,
            json=body,
            headers=self._auth_headers(),
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # Broker Protocol methods
    # ------------------------------------------------------------------

    def get_account(self) -> BrokerAccount:
        """Return a snapshot of the paper account."""
        self._assert_paper_url()
        raw = self._get(ALPACA_ACCOUNT_PATH)
        assert isinstance(raw, dict)
        return BrokerAccount(
            id=str(raw["id"]),
            status=str(raw["status"]),
            cash=Decimal(str(raw["cash"])),
            equity=Decimal(str(raw["equity"])),
            account_blocked=bool(raw["account_blocked"]),
            trading_blocked=bool(raw["trading_blocked"]),
            pattern_day_trader=bool(raw["pattern_day_trader"]),
        )

    def list_positions(self) -> tuple[BrokerPosition, ...]:
        """Return all open positions."""
        self._assert_paper_url()
        raw = self._get(ALPACA_POSITIONS_PATH)
        assert isinstance(raw, list)
        return tuple(
            BrokerPosition(
                symbol=str(item["symbol"]),
                quantity=int(Decimal(str(item["qty"]))),
                avg_price=Decimal(str(item["avg_entry_price"])),
            )
            for item in raw
        )

    def list_open_orders(self) -> tuple[BrokerOrder, ...]:
        """Return all open orders (status=open)."""
        self._assert_paper_url()
        raw = self._get(ALPACA_ORDERS_PATH, status="open")
        assert isinstance(raw, list)
        return tuple(
            _map_order(cast(dict[str, object], item)) for item in raw
        )

    def list_orders(self) -> tuple[BrokerOrder, ...]:
        """Return all orders (all statuses)."""
        self._assert_paper_url()
        raw = self._get(ALPACA_ORDERS_PATH, status="all")
        assert isinstance(raw, list)
        return tuple(
            _map_order(cast(dict[str, object], item)) for item in raw
        )

    def list_recent_fills(self) -> tuple[BrokerFill, ...]:
        """Return recent fill activities from the Alpaca activities endpoint."""
        self._assert_paper_url()
        raw = self._get(ALPACA_ACTIVITIES_PATH, activity_type="FILL")
        assert isinstance(raw, list)
        fills: list[BrokerFill] = []
        for item in raw:
            assert isinstance(item, dict)
            tx_time_raw = str(item.get("transaction_time", ""))
            try:
                parsed = datetime.fromisoformat(tx_time_raw.replace("Z", "+00:00"))
            except ValueError as exc:
                # Fail closed: never substitute wall-clock time into evidence.
                raise ValueError(
                    f"cannot parse fill transaction_time {tx_time_raw!r}"
                ) from exc
            tx_time = (
                parsed.astimezone(UTC)
                if parsed.tzinfo is not None
                else parsed.replace(tzinfo=UTC)
            )

            fills.append(
                BrokerFill(
                    order_id=str(item.get("order_id", "")),
                    client_order_id=str(item.get("client_order_id", "")),
                    symbol=str(item["symbol"]),
                    quantity=int(Decimal(str(item["qty"]))),
                    price=Decimal(str(item["price"])),
                    timestamp=tx_time,
                )
            )
        return tuple(fills)

    def get_order_by_client_id(self, client_order_id: str) -> BrokerOrder | None:
        """Retrieve a specific order by its client-assigned ID."""
        self._assert_paper_url()
        try:
            raw = self._get(
                f"{ALPACA_ORDERS_PATH}:by_client_order_id",
                client_order_id=client_order_id,
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return None
            raise
        assert isinstance(raw, dict)
        return _map_order(raw)

    def submit(self, intent: ExecutionIntent) -> BrokerOrder:
        """Submit an execution intent to the Alpaca paper broker.

        The order body includes the deterministic ``client_order_id`` from
        *intent*, making repeated submissions idempotent.
        """
        self._assert_paper_url()
        body: dict[str, object] = {
            "symbol": intent.symbol,
            "qty": str(int(intent.target_quantity)),
            "side": str(intent.side.value),
            "type": "market",
            "time_in_force": "day",
            "client_order_id": intent.client_order_id,
        }
        logger.debug(
            "Submitting order to Alpaca paper",
            extra={
                "symbol": intent.symbol,
                "side": intent.side.value,
                "qty": str(intent.target_quantity),
                "client_order_id": intent.client_order_id,
            },
        )
        raw = self._post(ALPACA_ORDERS_PATH, body)
        assert isinstance(raw, dict)
        return _map_order(raw)
