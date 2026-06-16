"""Frozen real-market-data snapshot source.

Reads the committed, point-in-time Parquet panel produced by
``scripts/fetch_real_snapshot.py`` and yields immutable :class:`DailyBar`
objects.  The panel is split/dividend-adjusted real daily OHLCV (Yahoo Finance),
frozen into the repository so every downstream result is fully deterministic and
offline — no network access at runtime.

``SnapshotBarSource`` satisfies the :class:`~mltrade.data.bars.DailyBarSource`
protocol, so it is a drop-in replacement for
:class:`~mltrade.data.fixtures.DeterministicBarSource` anywhere a bar source is
accepted (the backtest engine, feature pipeline, and workflows are all generic
over the protocol).
"""

from __future__ import annotations

import json
from datetime import date, datetime
from decimal import Decimal
from functools import cache
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from mltrade.data.bars import DailyBar
from mltrade.domain.instruments import AssetType, InstrumentId
from mltrade.universe import Universe

# Default committed snapshot (frozen point-in-time real data).
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SNAPSHOT_DIR = _PROJECT_ROOT / "data" / "snapshots" / "real"
DEFAULT_AS_OF = "2026-06-13"
DEFAULT_PANEL_PATH = DEFAULT_SNAPSHOT_DIR / f"daily_bars_{DEFAULT_AS_OF}.parquet"
DEFAULT_MANIFEST_PATH = (
    DEFAULT_SNAPSHOT_DIR / f"daily_bars_{DEFAULT_AS_OF}.manifest.json"
)

_REQUIRED_COLUMNS = (
    "session",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "trade_count",
    "source",
)


def load_manifest(manifest_path: Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    """Load the snapshot provenance manifest (source, as-of, hashes, counts)."""
    data: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    return data


def _to_decimal(value: object) -> Decimal:
    # Round-trip through ``str`` so the Decimal matches the 4-dp panel exactly.
    return Decimal(str(value))


def _row_to_bar(row: pd.Series, ingested_at: datetime) -> DailyBar:
    session = row["session"]
    if isinstance(session, str):
        session = date.fromisoformat(session)
    elif isinstance(session, pd.Timestamp):
        session = session.date()
    return DailyBar(
        instrument=InstrumentId(symbol=str(row["symbol"]), asset_type=AssetType.ETF),
        session=session,
        open=_to_decimal(row["open"]),
        high=_to_decimal(row["high"]),
        low=_to_decimal(row["low"]),
        close=_to_decimal(row["close"]),
        volume=int(row["volume"]),
        vwap=_to_decimal(row["vwap"]),
        trade_count=int(row["trade_count"]),
        source=str(row["source"]),
        ingested_at=ingested_at,
    )


@cache
def _read_panel(panel_path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(panel_path)
    missing = [c for c in _REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"snapshot panel missing columns: {missing}")
    return frame


class SnapshotBarSource:
    """Deterministic bar source backed by a frozen real-data Parquet panel."""

    def __init__(self, panel_path: Path = DEFAULT_PANEL_PATH) -> None:
        self._panel_path = panel_path

    def fetch(
        self,
        universe: Universe,
        start: date,
        end: date,
        ingested_at: datetime,
    ) -> tuple[DailyBar, ...]:
        if start > end:
            raise ValueError(f"start must be on or before end: {start!s} > {end!s}")

        frame = _read_panel(self._panel_path)
        symbols = set(universe.symbols)
        sessions = pd.to_datetime(frame["session"]).dt.date

        mask = frame["symbol"].isin(symbols) & (sessions >= start) & (sessions <= end)
        selected = frame.loc[mask]
        if selected.empty:
            raise ValueError(
                f"no snapshot bars for the universe in [{start!s}, {end!s}]"
            )

        bars = [_row_to_bar(row, ingested_at) for _, row in selected.iterrows()]
        bars.sort(key=lambda b: (b.session, b.instrument.symbol))
        return tuple(bars)
