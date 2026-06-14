"""Unit tests for AlpacaDataAdapter.

All HTTP calls are mocked via respx — no network I/O.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
import respx
from pydantic import SecretStr

from mltrade.data.alpaca import (
    ALPACA_BARS_PATH,
    ALPACA_DATA_BASE_URL,
    AlpacaDataAdapter,
)
from mltrade.universe import MVP_UNIVERSE, Universe

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

_FIXTURES = Path(__file__).parent.parent / "fixtures" / "alpaca"


def _bars_fixture() -> dict[str, object]:
    return json.loads((_FIXTURES / "bars.json").read_text())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BARS_URL = f"{ALPACA_DATA_BASE_URL}{ALPACA_BARS_PATH}"

_INGESTED_AT = datetime(2026, 6, 12, 21, 0, tzinfo=UTC)

_SPY_QQQ_UNIVERSE = Universe(
    version="test-v1",
    instruments=tuple(
        inst for inst in MVP_UNIVERSE.instruments if inst.symbol in ("SPY", "QQQ")
    ),
)


def _make_adapter(client: httpx.Client) -> AlpacaDataAdapter:
    return AlpacaDataAdapter(
        client=client,
        api_key=SecretStr("test-key-id"),
        api_secret=SecretStr("test-secret-key"),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@respx.mock
def test_alpaca_bar_response_maps_to_canonical_bar() -> None:
    """Returned DailyBar has source='alpaca', correct symbol, and price type."""
    respx.get(_BARS_URL).mock(
        return_value=httpx.Response(200, json=_bars_fixture())
    )
    with httpx.Client() as client:
        adapter = _make_adapter(client)
        bars = adapter.fetch(
            _SPY_QQQ_UNIVERSE,
            start=date(2026, 6, 11),
            end=date(2026, 6, 12),
            ingested_at=_INGESTED_AT,
        )

    spy_bars = [b for b in bars if b.instrument.symbol == "SPY"]
    assert len(spy_bars) == 2
    first = spy_bars[0]
    assert first.source == "alpaca"
    assert first.instrument.symbol == "SPY"
    assert first.close == Decimal("539.10")


@respx.mock
def test_alpaca_bar_decimal_precision() -> None:
    """Prices are Decimal, not float — no floating-point imprecision."""
    respx.get(_BARS_URL).mock(
        return_value=httpx.Response(200, json=_bars_fixture())
    )
    with httpx.Client() as client:
        adapter = _make_adapter(client)
        bars = adapter.fetch(
            _SPY_QQQ_UNIVERSE,
            start=date(2026, 6, 11),
            end=date(2026, 6, 12),
            ingested_at=_INGESTED_AT,
        )

    for bar in bars:
        assert isinstance(bar.open, Decimal)
        assert isinstance(bar.high, Decimal)
        assert isinstance(bar.low, Decimal)
        assert isinstance(bar.close, Decimal)
        assert isinstance(bar.vwap, Decimal)


@respx.mock
def test_alpaca_bar_maps_all_symbols() -> None:
    """fetch() returns bars for all symbols present in the fixture."""
    respx.get(_BARS_URL).mock(
        return_value=httpx.Response(200, json=_bars_fixture())
    )
    with httpx.Client() as client:
        adapter = _make_adapter(client)
        bars = adapter.fetch(
            _SPY_QQQ_UNIVERSE,
            start=date(2026, 6, 11),
            end=date(2026, 6, 12),
            ingested_at=_INGESTED_AT,
        )

    symbols = {b.instrument.symbol for b in bars}
    assert "SPY" in symbols
    assert "QQQ" in symbols
    # 2 bars per symbol x 2 symbols = 4 total
    assert len(bars) == 4


@respx.mock
def test_alpaca_bar_malformed_entry_fails_closed() -> None:
    """A bar missing required fields raises ValueError (fail closed)."""
    broken_payload = {
        "bars": {
            "SPY": [
                # Missing "h", "l", "c", "vw", "n"
                {"t": "2026-06-11T04:00:00Z", "o": "537.50", "v": 72345678}
            ]
        },
        "next_page_token": None,
    }
    respx.get(_BARS_URL).mock(
        return_value=httpx.Response(200, json=broken_payload)
    )
    with httpx.Client() as client:
        adapter = _make_adapter(client)
        with pytest.raises(ValueError, match="missing required fields"):
            adapter.fetch(
                _SPY_QQQ_UNIVERSE,
                start=date(2026, 6, 11),
                end=date(2026, 6, 12),
                ingested_at=_INGESTED_AT,
            )


@respx.mock
def test_alpaca_bar_session_is_date() -> None:
    """The session field is a date object (not a datetime or string)."""
    respx.get(_BARS_URL).mock(
        return_value=httpx.Response(200, json=_bars_fixture())
    )
    with httpx.Client() as client:
        adapter = _make_adapter(client)
        bars = adapter.fetch(
            _SPY_QQQ_UNIVERSE,
            start=date(2026, 6, 11),
            end=date(2026, 6, 12),
            ingested_at=_INGESTED_AT,
        )

    for bar in bars:
        assert isinstance(bar.session, date)
        assert not isinstance(bar.session, datetime)

    sessions = {b.session for b in bars}
    assert date(2026, 6, 11) in sessions
    assert date(2026, 6, 12) in sessions


@respx.mock
def test_alpaca_bar_volume_is_int() -> None:
    """volume and trade_count are plain Python ints."""
    respx.get(_BARS_URL).mock(
        return_value=httpx.Response(200, json=_bars_fixture())
    )
    with httpx.Client() as client:
        adapter = _make_adapter(client)
        bars = adapter.fetch(
            _SPY_QQQ_UNIVERSE,
            start=date(2026, 6, 11),
            end=date(2026, 6, 12),
            ingested_at=_INGESTED_AT,
        )

    for bar in bars:
        assert isinstance(bar.volume, int)
        assert isinstance(bar.trade_count, int)

    spy_first = next(
        b for b in bars
        if b.instrument.symbol == "SPY" and b.session == date(2026, 6, 11)
    )
    assert spy_first.volume == 72345678
    assert spy_first.trade_count == 421234


@respx.mock
def test_alpaca_request_has_correct_params() -> None:
    """The outgoing request includes symbols, timeframe, start, end params."""
    route = respx.get(_BARS_URL).mock(
        return_value=httpx.Response(200, json=_bars_fixture())
    )
    with httpx.Client() as client:
        adapter = _make_adapter(client)
        adapter.fetch(
            _SPY_QQQ_UNIVERSE,
            start=date(2026, 6, 11),
            end=date(2026, 6, 12),
            ingested_at=_INGESTED_AT,
        )

    assert route.called
    request: httpx.Request = route.calls.last.request
    query = dict(httpx.QueryParams(request.url.query))
    assert query["timeframe"] == "1Day"
    assert query["start"] == "2026-06-11"
    assert query["end"] == "2026-06-12"
    # Both symbols must appear (order may vary within CSV)
    symbols_param = query["symbols"]
    assert "SPY" in symbols_param
    assert "QQQ" in symbols_param


@respx.mock
def test_alpaca_data_auth_headers_not_logged(caplog: pytest.LogCaptureFixture) -> None:
    """Secret values never appear in log output."""
    respx.get(_BARS_URL).mock(
        return_value=httpx.Response(200, json=_bars_fixture())
    )
    with caplog.at_level(logging.DEBUG, logger="mltrade"):
        with httpx.Client() as client:
            adapter = AlpacaDataAdapter(
                client=client,
                api_key=SecretStr("SUPER_SECRET_KEY_ID"),
                api_secret=SecretStr("SUPER_SECRET_KEY_VALUE"),
            )
            adapter.fetch(
                _SPY_QQQ_UNIVERSE,
                start=date(2026, 6, 11),
                end=date(2026, 6, 12),
                ingested_at=_INGESTED_AT,
            )

    combined_logs = " ".join(caplog.messages)
    assert "SUPER_SECRET_KEY_ID" not in combined_logs
    assert "SUPER_SECRET_KEY_VALUE" not in combined_logs
