"""Unit tests for :mod:`mltrade.data.snapshot` (frozen real-data panel)."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from mltrade.data.bars import DailyBar
from mltrade.data.snapshot import SnapshotBarSource, load_manifest
from mltrade.universe import MVP_UNIVERSE


def test_fetch_returns_sorted_daily_bars_in_range() -> None:
    source = SnapshotBarSource()
    start = date(2026, 6, 1)
    end = date(2026, 6, 12)
    ingested_at = datetime(2026, 6, 12, tzinfo=UTC)

    bars = source.fetch(MVP_UNIVERSE, start, end, ingested_at)

    assert len(bars) > 0
    assert all(isinstance(b, DailyBar) for b in bars)
    # Sorted by (session, symbol) ascending.
    keys = [(b.session, b.instrument.symbol) for b in bars]
    assert keys == sorted(keys)
    # All bars fall within the requested range and the requested universe.
    universe_symbols = set(MVP_UNIVERSE.symbols)
    for b in bars:
        assert start <= b.session <= end
        assert b.instrument.symbol in universe_symbols
        assert b.ingested_at == ingested_at


def test_fetch_rejects_start_after_end() -> None:
    source = SnapshotBarSource()
    ingested_at = datetime(2026, 6, 12, tzinfo=UTC)
    with pytest.raises(ValueError, match="start must be on or before end"):
        source.fetch(
            MVP_UNIVERSE, date(2026, 6, 12), date(2026, 6, 1), ingested_at
        )


def test_fetch_empty_range_raises() -> None:
    source = SnapshotBarSource()
    ingested_at = datetime(2026, 6, 12, tzinfo=UTC)
    # A range with no sessions in the panel (far before the panel start).
    with pytest.raises(ValueError, match="no snapshot bars"):
        source.fetch(
            MVP_UNIVERSE, date(1990, 1, 1), date(1990, 1, 2), ingested_at
        )


def test_load_manifest_returns_expected_keys() -> None:
    manifest = load_manifest()
    assert {"source", "as_of", "content_sha256", "symbols"} <= set(manifest)
    assert manifest["as_of"] == "2026-06-13"
    assert isinstance(manifest["symbols"], list)
    assert set(manifest["symbols"]) == set(MVP_UNIVERSE.symbols)
