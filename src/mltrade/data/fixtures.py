"""Deterministic offline market data fixture for testing and offline workflows.

Generates a fully reproducible grid of OHLCV bars for any date range and
universe by composing deterministic trend, cycle, and seeded-noise components.
No network access is required.
"""

from __future__ import annotations

import hashlib
import math
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import exchange_calendars as xcals  # type: ignore[import-untyped]
import numpy.random as npr
import pandas as pd  # type: ignore[import-untyped]

from mltrade.data.bars import DailyBar
from mltrade.domain.instruments import InstrumentId
from mltrade.universe import Universe

# ---------------------------------------------------------------------------
# Stable per-symbol configuration
# ---------------------------------------------------------------------------

# Initial (reference) prices and liquidity scale factors, keyed by symbol.
# These are deliberately hand-chosen to be realistic and stable across any
# change to the universe ordering.
_INITIAL_PRICES: dict[str, float] = {
    "SPY": 380.0,
    "QQQ": 290.0,
    "IWM": 175.0,
    "EFA": 65.0,
    "EEM": 42.0,
    "TLT": 110.0,
    "IEF": 100.0,
    "GLD": 170.0,
    "DBC": 22.0,
    "VNQ": 80.0,
}

_LIQUIDITY_SCALES: dict[str, float] = {
    "SPY": 1.0,
    "QQQ": 0.9,
    "IWM": 0.7,
    "EFA": 0.5,
    "EEM": 0.5,
    "TLT": 0.6,
    "IEF": 0.5,
    "GLD": 0.4,
    "DBC": 0.3,
    "VNQ": 0.3,
}

_DEFAULT_INITIAL_PRICE: float = 50.0
_DEFAULT_LIQUIDITY_SCALE: float = 0.3


def _sessions_in_range(start: date, end: date) -> list[date]:
    """Return XNYS sessions in [start, end] inclusive, in ascending order."""
    cal: Any = xcals.get_calendar("XNYS")
    index: Any = cal.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
    return [ts.date() for ts in index]


def _stable_symbol_seed(base_seed: int, symbol: str) -> int:
    """Derive a deterministic per-symbol seed from base_seed and symbol text.

    Uses a SHA-256 digest (stable across processes and PYTHONHASHSEED values)
    so that per-symbol streams are independent of universe ordering and
    produce identical results in every Python process.
    """
    digest = hashlib.sha256(symbol.encode()).digest()
    h = int.from_bytes(digest[:8], "little")
    # 6364136223846793005 is the well-known PCG/Knuth LCG multiplier
    return (base_seed * 6364136223846793005 + h) & 0xFFFF_FFFF_FFFF_FFFF


def _generate_symbol_bars(
    instrument: InstrumentId,
    sessions: list[date],
    base_seed: int,
    ingested_at: datetime,
) -> list[DailyBar]:
    """Generate all DailyBar objects for one symbol across all sessions."""
    symbol = instrument.symbol
    sym_seed = _stable_symbol_seed(base_seed, symbol)
    rng = npr.default_rng(sym_seed)

    initial_price = _INITIAL_PRICES.get(symbol, _DEFAULT_INITIAL_PRICE)
    liquidity_scale = _LIQUIDITY_SCALES.get(symbol, _DEFAULT_LIQUIDITY_SCALE)

    # Trend: slow upward drift (0.8% annualised per bar ≈ 0.0032% daily)
    trend_per_bar: float = 0.000032

    n = len(sessions)
    bars: list[DailyBar] = []
    prev_close: float = initial_price

    for i in range(n):
        # Decompose daily log-return into trend + cycle + noise
        cycle = 0.002 * math.sin(2 * math.pi * i / 252.0)
        stress = -0.005 * math.sin(2 * math.pi * i / 1260.0)
        noise = rng.normal(0.0, 0.012)
        daily_return: float = trend_per_bar + cycle + stress + float(noise)

        # Split into overnight and intraday components
        overnight: float = float(rng.normal(0.0, 0.002))
        intraday: float = daily_return - overnight

        open_price: float = prev_close * math.exp(overnight)
        close_price: float = open_price * math.exp(intraday)

        # Ensure both are positive (handles extreme noise edge cases)
        open_price = max(open_price, 0.01)
        close_price = max(close_price, 0.01)

        spread: float = abs(float(rng.normal(0.006, 0.002)))
        high_price: float = max(open_price, close_price) * (1.0 + spread)
        low_price: float = min(open_price, close_price) * (1.0 - spread)

        # Volume and trade count driven by liquidity scale + noise
        base_volume: float = 5_000_000.0 * liquidity_scale
        volume: int = max(0, int(base_volume * abs(float(rng.normal(1.0, 0.3)))))
        trade_count: int = max(0, int(volume // 250 * abs(float(rng.normal(1.0, 0.2)))))

        # VWAP: a weighted average between open and close, biased intraday
        vwap_raw: float = (open_price + close_price) / 2.0
        # Defensive belt-and-suspenders clamp: midpoint of open/close is
        # analytically within [low, high], but guard against any future
        # VWAP formula change that might violate the invariant.
        vwap_raw = max(low_price, min(high_price, vwap_raw))
        vwap: Decimal = Decimal(str(round(vwap_raw, 4)))

        bar = DailyBar(
            instrument=instrument,
            session=sessions[i],
            open=Decimal(str(round(open_price, 4))),
            high=Decimal(str(round(high_price, 4))),
            low=Decimal(str(round(low_price, 4))),
            close=Decimal(str(round(close_price, 4))),
            volume=volume,
            vwap=vwap,
            trade_count=trade_count,
            source="fixture",
            ingested_at=ingested_at,
        )
        bars.append(bar)
        prev_close = close_price

    return bars


class DeterministicBarSource:
    """Offline, deterministic OHLCV bar source for testing and research.

    Repeated calls with identical arguments return identical bars.  Different
    seeds produce different bar streams.  No network access is performed.
    """

    def __init__(self, *, seed: int) -> None:
        self._seed = seed

    def fetch(
        self,
        universe: Universe,
        start: date,
        end: date,
        ingested_at: datetime,
    ) -> tuple[DailyBar, ...]:
        if start > end:
            raise ValueError(
                f"start must be on or before end: {start!s} > {end!s}"
            )

        sessions = _sessions_in_range(start, end)
        if not sessions:
            raise ValueError(
                f"no XNYS sessions in range [{start!s}, {end!s}]"
            )

        all_bars: list[DailyBar] = []
        for instrument in universe.instruments:
            symbol_bars = _generate_symbol_bars(
                instrument,
                sessions,
                self._seed,
                ingested_at,
            )
            all_bars.extend(symbol_bars)

        # Sort by (session, symbol) as required by the spec
        all_bars.sort(key=lambda b: (b.session, b.instrument.symbol))
        return tuple(all_bars)
