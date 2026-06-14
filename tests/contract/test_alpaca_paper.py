"""Contract tests for the Alpaca paper-trading API.

These tests make real network calls to ``https://paper-api.alpaca.markets``
and ``https://data.alpaca.markets``.  They are skipped unless the environment
variable ``MLTRADE_RUN_ALPACA_CONTRACTS`` is set to ``"true"`` **and** valid
Alpaca paper credentials are present in ``MLTRADE_ALPACA_API_KEY`` /
``MLTRADE_ALPACA_API_SECRET``.

Run selectively:
    MLTRADE_RUN_ALPACA_CONTRACTS=true \\
    MLTRADE_ALPACA_API_KEY=... \\
    MLTRADE_ALPACA_API_SECRET=... \\
    uv run pytest tests/contract/test_alpaca_paper.py -v
"""

from __future__ import annotations

import os
from datetime import date, timedelta

import httpx
import pytest
from pydantic import SecretStr

from mltrade.data.alpaca import AlpacaDataAdapter
from mltrade.execution.alpaca import ALPACA_PAPER_BASE_URL, AlpacaPaperBroker

# ---------------------------------------------------------------------------
# Skip gate
# ---------------------------------------------------------------------------

_SKIP = os.environ.get("MLTRADE_RUN_ALPACA_CONTRACTS") != "true"

pytestmark = [
    pytest.mark.alpaca,
    pytest.mark.skipif(
        _SKIP,
        reason=(
            "Set MLTRADE_RUN_ALPACA_CONTRACTS=true (and supply "
            "MLTRADE_ALPACA_API_KEY / MLTRADE_ALPACA_API_SECRET) "
            "to run against the real Alpaca paper API"
        ),
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _api_key() -> SecretStr:
    raw = os.environ.get("MLTRADE_ALPACA_API_KEY", "")
    if not raw:
        pytest.skip("MLTRADE_ALPACA_API_KEY not set")
    return SecretStr(raw)


def _api_secret() -> SecretStr:
    raw = os.environ.get("MLTRADE_ALPACA_API_SECRET", "")
    if not raw:
        pytest.skip("MLTRADE_ALPACA_API_SECRET not set")
    return SecretStr(raw)


# ---------------------------------------------------------------------------
# Broker contract tests
# ---------------------------------------------------------------------------


def test_alpaca_paper_account_is_active() -> None:
    """Real Alpaca paper account is ACTIVE and unblocked."""
    with httpx.Client() as client:
        broker = AlpacaPaperBroker(
            client=client,
            base_url=ALPACA_PAPER_BASE_URL,
            api_key=_api_key(),
            api_secret=_api_secret(),
        )
        account = broker.get_account()

    assert account.status == "ACTIVE"
    assert not account.account_blocked
    assert not account.trading_blocked
    assert account.equity > 0


def test_alpaca_paper_list_positions_returns_list() -> None:
    """list_positions() returns a tuple (possibly empty) without error."""
    with httpx.Client() as client:
        broker = AlpacaPaperBroker(
            client=client,
            base_url=ALPACA_PAPER_BASE_URL,
            api_key=_api_key(),
            api_secret=_api_secret(),
        )
        positions = broker.list_positions()

    assert isinstance(positions, tuple)


def test_alpaca_paper_list_open_orders_returns_list() -> None:
    """list_open_orders() returns a tuple without error."""
    with httpx.Client() as client:
        broker = AlpacaPaperBroker(
            client=client,
            base_url=ALPACA_PAPER_BASE_URL,
            api_key=_api_key(),
            api_secret=_api_secret(),
        )
        orders = broker.list_open_orders()

    assert isinstance(orders, tuple)


# ---------------------------------------------------------------------------
# Data contract tests
# ---------------------------------------------------------------------------


def test_alpaca_data_fetch_returns_bars() -> None:
    """fetch() against real Alpaca data returns at least one bar for SPY."""
    from datetime import UTC, datetime

    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=5)

    # Use a minimal universe with just SPY to reduce test data volume
    from mltrade.domain.instruments import AssetType, InstrumentId
    from mltrade.universe import Universe

    spy_universe = Universe(
        version="contract-test-v1",
        instruments=(InstrumentId(symbol="SPY", asset_type=AssetType.ETF),),
    )

    with httpx.Client() as client:
        adapter = AlpacaDataAdapter(
            client=client,
            api_key=_api_key(),
            api_secret=_api_secret(),
        )
        bars = adapter.fetch(
            spy_universe,
            start=start,
            end=end,
            ingested_at=datetime.now(UTC),
        )

    assert len(bars) > 0
    spy_bar = bars[0]
    assert spy_bar.instrument.symbol == "SPY"
    assert spy_bar.source == "alpaca"
    assert spy_bar.close > 0
