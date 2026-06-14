"""Unit tests for AlpacaPaperBroker.

All HTTP calls are mocked via respx — no network I/O.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
import respx
from pydantic import SecretStr

from mltrade.execution.alpaca import ALPACA_PAPER_BASE_URL, AlpacaPaperBroker
from mltrade.execution.broker import BrokerSafetyError, OrderSide
from mltrade.execution.intents import build_intent

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

_FIXTURES = Path(__file__).parent.parent / "fixtures" / "alpaca"


def _load(name: str) -> object:
    return json.loads((_FIXTURES / name).read_text())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LIVE_URL = "https://api.alpaca.markets"

_ACCOUNT_URL = f"{ALPACA_PAPER_BASE_URL}/v2/account"
_POSITIONS_URL = f"{ALPACA_PAPER_BASE_URL}/v2/positions"
_ORDERS_URL = f"{ALPACA_PAPER_BASE_URL}/v2/orders"
_ACTIVITIES_URL = f"{ALPACA_PAPER_BASE_URL}/v2/account/activities"


def _make_broker(base_url: str = ALPACA_PAPER_BASE_URL) -> AlpacaPaperBroker:
    return AlpacaPaperBroker(
        client=httpx.Client(),
        base_url=base_url,
        api_key=SecretStr("test-key"),
        api_secret=SecretStr("test-secret"),
    )


# ---------------------------------------------------------------------------
# Safety tests
# ---------------------------------------------------------------------------


def test_alpaca_broker_rejects_non_paper_account() -> None:
    """Constructing with a live URL does NOT raise; get_account() does."""
    adapter = AlpacaPaperBroker(
        client=httpx.Client(),
        base_url=_LIVE_URL,
        api_key=SecretStr("test-key"),
        api_secret=SecretStr("test-secret"),
    )
    with pytest.raises(BrokerSafetyError, match="paper"):
        adapter.get_account()


def test_alpaca_broker_rejects_live_url_on_list_positions() -> None:
    """list_positions() also raises BrokerSafetyError for non-paper URLs."""
    adapter = AlpacaPaperBroker(
        client=httpx.Client(),
        base_url=_LIVE_URL,
        api_key=SecretStr("test-key"),
        api_secret=SecretStr("test-secret"),
    )
    with pytest.raises(BrokerSafetyError, match="paper"):
        adapter.list_positions()


@respx.mock
def test_alpaca_broker_accepts_paper_url() -> None:
    """Paper URL allows get_account() to proceed."""
    respx.get(_ACCOUNT_URL).mock(
        return_value=httpx.Response(200, json=_load("account.json"))
    )
    broker = _make_broker()
    account = broker.get_account()
    assert account.id == "pa-account-paper-001"
    assert account.status == "ACTIVE"


# ---------------------------------------------------------------------------
# Account
# ---------------------------------------------------------------------------


@respx.mock
def test_alpaca_get_account_maps_correctly() -> None:
    """Account JSON maps to BrokerAccount with correct Decimal fields."""
    respx.get(_ACCOUNT_URL).mock(
        return_value=httpx.Response(200, json=_load("account.json"))
    )
    broker = _make_broker()
    account = broker.get_account()

    assert account.id == "pa-account-paper-001"
    assert account.status == "ACTIVE"
    assert account.cash == Decimal("987654.32")
    assert account.equity == Decimal("1023456.78")
    assert account.account_blocked is False
    assert account.trading_blocked is False
    assert account.pattern_day_trader is False


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------


@respx.mock
def test_alpaca_list_positions_maps_correctly() -> None:
    """Positions JSON maps to BrokerPosition list with int quantity."""
    respx.get(_POSITIONS_URL).mock(
        return_value=httpx.Response(200, json=_load("positions.json"))
    )
    broker = _make_broker()
    positions = broker.list_positions()

    assert len(positions) == 1
    pos = positions[0]
    assert pos.symbol == "SPY"
    assert pos.quantity == 100
    assert pos.avg_price == Decimal("535.20")


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------


@respx.mock
def test_alpaca_list_open_orders_maps_correctly() -> None:
    """Open orders JSON maps to BrokerOrder list correctly."""
    respx.get(_ORDERS_URL).mock(
        return_value=httpx.Response(200, json=_load("orders.json"))
    )
    broker = _make_broker()
    orders = broker.list_open_orders()

    assert len(orders) == 1
    order = orders[0]
    assert order.id == "order-uuid-001"
    assert order.client_order_id == "mlt-20260612-0123456789abcdef01234567"
    assert order.symbol == "SPY"
    assert order.side == OrderSide.BUY
    assert order.quantity == 10
    assert order.filled_quantity == 0


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------


@respx.mock
def test_alpaca_submit_posts_correct_body() -> None:
    """submit() POSTs to /v2/orders with correct symbol, qty, side, type."""
    # The response after submit looks like an order object
    submitted_order = {
        "id": "order-uuid-new",
        "client_order_id": "mlt-20260612-0123456789abcdef01234567",
        "symbol": "SPY",
        "qty": "10",
        "filled_qty": "0",
        "side": "buy",
        "type": "market",
        "status": "new",
        "submitted_at": "2026-06-12T14:30:00Z",
        "filled_at": None,
        "filled_avg_price": None,
        "time_in_force": "day",
    }
    route = respx.post(_ORDERS_URL).mock(
        return_value=httpx.Response(200, json=submitted_order)
    )
    broker = _make_broker()
    intent = build_intent(
        environment="paper",
        strategy_version="test-v1",
        decision_session=date(2026, 6, 12),
        symbol="SPY",
        side=OrderSide.BUY,
        target_quantity=Decimal("10"),
    )
    order = broker.submit(intent)

    assert route.called
    request: httpx.Request = route.calls.last.request
    body = json.loads(request.content)
    assert body["symbol"] == "SPY"
    assert body["qty"] == "10"
    assert body["side"] == "buy"
    assert body["type"] == "market"
    assert body["time_in_force"] == "day"

    assert order.symbol == "SPY"
    assert order.side == OrderSide.BUY


@respx.mock
def test_alpaca_submit_includes_client_order_id() -> None:
    """submit() body includes the deterministic client_order_id from the intent."""
    intent = build_intent(
        environment="paper",
        strategy_version="test-v1",
        decision_session=date(2026, 6, 12),
        symbol="SPY",
        side=OrderSide.BUY,
        target_quantity=Decimal("10"),
    )
    response_json = {
        "id": "order-uuid-new",
        "client_order_id": intent.client_order_id,
        "symbol": "SPY",
        "qty": "10",
        "filled_qty": "0",
        "side": "buy",
        "type": "market",
        "status": "new",
        "submitted_at": "2026-06-12T14:30:00Z",
        "filled_at": None,
        "filled_avg_price": None,
        "time_in_force": "day",
    }
    route = respx.post(_ORDERS_URL).mock(
        return_value=httpx.Response(200, json=response_json)
    )
    broker = _make_broker()
    broker.submit(intent)

    body = json.loads(route.calls.last.request.content)
    assert body["client_order_id"] == intent.client_order_id
    assert body["client_order_id"].startswith("mlt-20260612-")


# ---------------------------------------------------------------------------
# list_recent_fills mapping
# ---------------------------------------------------------------------------


@respx.mock
def test_list_recent_fills_maps_client_order_id_and_utc() -> None:
    """Fills map client_order_id from the Alpaca field (not the activity id)."""
    respx.get(_ACTIVITIES_URL).mock(
        return_value=httpx.Response(200, json=_load("fills.json"))
    )
    fills = _make_broker().list_recent_fills()

    assert len(fills) == 1
    fill = fills[0]
    assert fill.client_order_id == "mlt-20260612-0123456789abcdef01234567"
    assert fill.order_id == "order-uuid-001"
    assert fill.symbol == "SPY"
    assert fill.quantity == 10
    assert fill.price == Decimal("539.10")
    assert fill.timestamp.tzinfo is not None
    assert fill.timestamp.utcoffset() == timedelta(0)


@respx.mock
def test_list_recent_fills_fails_closed_on_bad_timestamp() -> None:
    """A malformed transaction_time raises rather than substituting now()."""
    respx.get(_ACTIVITIES_URL).mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "id": "fill-uuid-002",
                    "order_id": "order-uuid-002",
                    "client_order_id": "mlt-20260612-deadbeefdeadbeefdeadbeef",
                    "symbol": "SPY",
                    "qty": "5",
                    "price": "100.00",
                    "transaction_time": "not-a-timestamp",
                }
            ],
        )
    )
    with pytest.raises(ValueError, match="transaction_time"):
        _make_broker().list_recent_fills()


# ---------------------------------------------------------------------------
# Safety / secret hygiene
# ---------------------------------------------------------------------------


def test_crafted_lookalike_host_is_rejected() -> None:
    """A host that merely contains the paper host as a substring is rejected."""
    adapter = _make_broker(
        base_url="https://paper-api.alpaca.markets.evil.com"
    )
    with pytest.raises(BrokerSafetyError, match="paper"):
        adapter.get_account()


@respx.mock
def test_broker_secrets_not_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """API key/secret never appear in logs across account fetch + submit."""
    import logging

    respx.get(_ACCOUNT_URL).mock(
        return_value=httpx.Response(200, json=_load("account.json"))
    )
    respx.post(_ORDERS_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "order-uuid-new",
                "client_order_id": "mlt-20260612-0123456789abcdef01234567",
                "symbol": "SPY",
                "qty": "10",
                "filled_qty": "0",
                "side": "buy",
                "type": "market",
                "status": "new",
                "submitted_at": "2026-06-12T14:30:00Z",
                "filled_at": None,
                "filled_avg_price": None,
                "time_in_force": "day",
            },
        )
    )
    broker = AlpacaPaperBroker(
        client=httpx.Client(),
        base_url=ALPACA_PAPER_BASE_URL,
        api_key=SecretStr("super-secret-key"),
        api_secret=SecretStr("super-secret-value"),
    )
    intent = build_intent(
        environment="paper",
        strategy_version="ridge-trend-v1",
        decision_session=date(2026, 6, 12),
        symbol="SPY",
        side=OrderSide.BUY,
        target_quantity=Decimal("10"),
    )
    with caplog.at_level(logging.DEBUG):
        broker.get_account()
        broker.submit(intent)

    assert "super-secret-key" not in caplog.text
    assert "super-secret-value" not in caplog.text
