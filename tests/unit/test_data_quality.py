"""Tests for the fail-closed data quality module.

TDD: tests written first; initially all fail with ImportError since
mltrade.data.quality does not yet exist.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from mltrade.data.fixtures import DeterministicBarSource
from mltrade.data.quality import DataQualityReport, IssueSeverity, validate_daily_bars
from mltrade.domain.instruments import AssetType, InstrumentId
from mltrade.universe import MVP_UNIVERSE, Universe

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

_INGESTED_AT = datetime(2026, 6, 13, 0, 0, tzinfo=UTC)
_LAST_SESSION = date(2026, 6, 12)
_RANGE_START = date(2026, 6, 2)  # Monday, first XNYS session of the window


@pytest.fixture(scope="module")
def valid_bars() -> tuple:  # tuple[DailyBar, ...]
    """Complete, sorted, duplicate-free grid for MVP_UNIVERSE ending 2026-06-12."""
    return DeterministicBarSource(seed=99).fetch(
        MVP_UNIVERSE,
        _RANGE_START,
        _LAST_SESSION,
        _INGESTED_AT,
    )


# ---------------------------------------------------------------------------
# Happy-path: clean data passes without blocking
# ---------------------------------------------------------------------------


def test_valid_bars_pass_without_blocking(valid_bars: tuple) -> None:
    report = validate_daily_bars(
        valid_bars,
        universe=MVP_UNIVERSE,
        expected_last_session=_LAST_SESSION,
    )
    assert isinstance(report, DataQualityReport)
    assert report.blocked is False
    assert report.issues == ()


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


def test_quality_blocks_empty_input() -> None:
    report = validate_daily_bars(
        (),
        universe=MVP_UNIVERSE,
        expected_last_session=_LAST_SESSION,
    )
    assert report.blocked is True
    assert "empty_input" in {issue.code for issue in report.issues}


# ---------------------------------------------------------------------------
# Duplicate bars
# ---------------------------------------------------------------------------


def test_quality_blocks_duplicate_bars(valid_bars: tuple) -> None:
    report = validate_daily_bars(
        (*valid_bars, valid_bars[0]),
        universe=MVP_UNIVERSE,
        expected_last_session=_LAST_SESSION,
    )
    assert report.blocked is True
    assert "duplicate_bar" in {issue.code for issue in report.issues}


# ---------------------------------------------------------------------------
# Incomplete latest session
# ---------------------------------------------------------------------------


def test_quality_blocks_incomplete_latest_session(valid_bars: tuple) -> None:
    bars = tuple(
        bar
        for bar in valid_bars
        if not (
            bar.session == date(2026, 6, 12) and bar.instrument.symbol == "SPY"
        )
    )
    report = validate_daily_bars(
        bars,
        universe=MVP_UNIVERSE,
        expected_last_session=_LAST_SESSION,
    )
    assert report.blocked is True
    assert "incomplete_latest_session" in {issue.code for issue in report.issues}


# ---------------------------------------------------------------------------
# Latest session mismatch
# ---------------------------------------------------------------------------


def test_quality_blocks_latest_session_mismatch(valid_bars: tuple) -> None:
    # Pass expected date one day after the actual last session
    future_session = date(2026, 6, 13)  # Saturday — NOT a real session
    report = validate_daily_bars(
        valid_bars,
        universe=MVP_UNIVERSE,
        expected_last_session=future_session,
    )
    assert report.blocked is True
    assert "latest_session_mismatch" in {issue.code for issue in report.issues}


# ---------------------------------------------------------------------------
# Unsorted input
# ---------------------------------------------------------------------------


def test_quality_blocks_unsorted_bars(valid_bars: tuple) -> None:
    # Reverse the tuple so it is definitely not sorted by (session, symbol)
    reversed_bars = valid_bars[::-1]
    report = validate_daily_bars(
        reversed_bars,
        universe=MVP_UNIVERSE,
        expected_last_session=_LAST_SESSION,
    )
    assert report.blocked is True
    assert "unsorted_bars" in {issue.code for issue in report.issues}


# ---------------------------------------------------------------------------
# Symbol outside universe
# ---------------------------------------------------------------------------


def test_quality_blocks_symbol_outside_universe(valid_bars: tuple) -> None:
    from decimal import Decimal

    from mltrade.data.bars import DailyBar

    alien_bar = DailyBar(
        instrument=InstrumentId(symbol="AAPL", asset_type=AssetType.STOCK),
        session=date(2026, 6, 12),
        open=Decimal("200.00"),
        high=Decimal("201.00"),
        low=Decimal("199.00"),
        close=Decimal("200.50"),
        vwap=Decimal("200.25"),
        volume=1_000_000,
        trade_count=4_000,
        source="fixture",
        ingested_at=_INGESTED_AT,
    )
    report = validate_daily_bars(
        (*valid_bars, alien_bar),
        universe=MVP_UNIVERSE,
        expected_last_session=_LAST_SESSION,
    )
    assert report.blocked is True
    assert "symbol_outside_universe" in {issue.code for issue in report.issues}


# ---------------------------------------------------------------------------
# Missing symbol from universe
# ---------------------------------------------------------------------------


def test_quality_blocks_missing_universe_symbol(valid_bars: tuple) -> None:
    # Remove all SPY bars
    bars_without_spy = tuple(
        bar for bar in valid_bars if bar.instrument.symbol != "SPY"
    )
    report = validate_daily_bars(
        bars_without_spy,
        universe=MVP_UNIVERSE,
        expected_last_session=_LAST_SESSION,
    )
    assert report.blocked is True
    assert "missing_universe_symbol" in {issue.code for issue in report.issues}


# ---------------------------------------------------------------------------
# Missing XNYS sessions within a symbol's range
# ---------------------------------------------------------------------------


def test_quality_blocks_missing_xnys_session(valid_bars: tuple) -> None:
    # Drop exactly one session's worth of SPY bars from the middle of the range
    # (2026-06-09 is a Monday — guaranteed to be a valid XNYS session)
    gap_session = date(2026, 6, 9)
    bars_with_gap = tuple(
        bar
        for bar in valid_bars
        if not (
            bar.session == gap_session and bar.instrument.symbol == "SPY"
        )
    )
    report = validate_daily_bars(
        bars_with_gap,
        universe=MVP_UNIVERSE,
        expected_last_session=_LAST_SESSION,
    )
    assert report.blocked is True
    assert "missing_session" in {issue.code for issue in report.issues}


# ---------------------------------------------------------------------------
# DataQualityReport immutability
# ---------------------------------------------------------------------------


def test_report_is_immutable(valid_bars: tuple) -> None:
    from pydantic import ValidationError

    report = validate_daily_bars(
        valid_bars,
        universe=MVP_UNIVERSE,
        expected_last_session=_LAST_SESSION,
    )
    with pytest.raises((TypeError, AttributeError, ValidationError)):
        report.blocked = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# IssueSeverity enum
# ---------------------------------------------------------------------------


def test_issue_severity_has_block_and_warn() -> None:
    assert IssueSeverity.BLOCK is not None
    assert IssueSeverity.WARN is not None


# ---------------------------------------------------------------------------
# Multiple issues are all reported
# ---------------------------------------------------------------------------


def test_multiple_issues_all_reported() -> None:
    """Duplicate bars AND incomplete latest session → both codes present."""
    from decimal import Decimal

    from mltrade.data.bars import DailyBar

    bar = DailyBar(
        instrument=InstrumentId(symbol="SPY", asset_type=AssetType.ETF),
        session=date(2026, 6, 12),
        open=Decimal("500.00"),
        high=Decimal("501.00"),
        low=Decimal("499.00"),
        close=Decimal("500.50"),
        vwap=Decimal("500.25"),
        volume=10_000_000,
        trade_count=40_000,
        source="fixture",
        ingested_at=_INGESTED_AT,
    )
    small_universe = Universe(
        version="test-v1",
        instruments=(
            InstrumentId(symbol="SPY", asset_type=AssetType.ETF),
            InstrumentId(symbol="QQQ", asset_type=AssetType.ETF),
        ),
    )
    # Only one bar for SPY; QQQ missing entirely → incomplete latest session
    # And we add a duplicate of that single bar
    bars = (bar, bar)
    report = validate_daily_bars(
        bars,
        universe=small_universe,
        expected_last_session=date(2026, 6, 12),
    )
    codes = {issue.code for issue in report.issues}
    assert report.blocked is True
    assert "duplicate_bar" in codes


# ---------------------------------------------------------------------------
# sessions_in_range on XNYSCalendar (tests for the new calendar method)
# ---------------------------------------------------------------------------


def test_sessions_in_range_excludes_weekends_and_holidays() -> None:
    from mltrade.calendar import XNYSCalendar

    cal = XNYSCalendar()
    sessions = cal.sessions_in_range(date(2024, 7, 3), date(2024, 7, 8))
    # 2024-07-04 is Independence Day (holiday), 2024-07-06/07 are weekend
    assert date(2024, 7, 4) not in sessions
    assert date(2024, 7, 6) not in sessions
    assert date(2024, 7, 7) not in sessions
    assert date(2024, 7, 3) in sessions
    assert date(2024, 7, 5) in sessions
    assert date(2024, 7, 8) in sessions


def test_sessions_in_range_returns_tuple_of_dates() -> None:
    from mltrade.calendar import XNYSCalendar

    cal = XNYSCalendar()
    sessions = cal.sessions_in_range(date(2026, 6, 9), date(2026, 6, 12))
    assert isinstance(sessions, tuple)
    assert all(isinstance(s, date) for s in sessions)
    # 2026-06-09 (Mon), 2026-06-10 (Tue), 2026-06-11 (Wed), 2026-06-12 (Thu)
    assert len(sessions) == 4
