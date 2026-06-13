from datetime import UTC, date, datetime, timedelta, timezone

import pytest

from mltrade.data.fixtures import DeterministicBarSource
from mltrade.universe import MVP_UNIVERSE


def test_fixture_builds_complete_sorted_mvp_session_grid() -> None:
    start = date(2022, 1, 3)
    end = date(2026, 6, 12)
    bars = DeterministicBarSource(seed=7).fetch(
        MVP_UNIVERSE,
        start,
        end,
        datetime(2026, 6, 13, 1, 0, tzinfo=UTC),
    )

    sessions = {bar.session for bar in bars}
    expected_pairs = {
        (session, symbol)
        for session in sessions
        for symbol in MVP_UNIVERSE.symbols
    }
    actual_pairs = {
        (bar.session, bar.instrument.symbol)
        for bar in bars
    }

    assert len(sessions) >= 1_100
    assert len(bars) == len(sessions) * len(MVP_UNIVERSE.instruments)
    assert actual_pairs == expected_pairs
    assert [(bar.session, bar.instrument.symbol) for bar in bars] == sorted(
        actual_pairs
    )
    assert min(sessions) == start
    assert max(sessions) == end


def test_fixture_is_repeatable_per_fetch_and_seed() -> None:
    source = DeterministicBarSource(seed=11)
    ingested_at = datetime(2026, 6, 13, 1, 0, tzinfo=UTC)
    arguments = (
        MVP_UNIVERSE,
        date(2026, 6, 1),
        date(2026, 6, 12),
        ingested_at,
    )

    first = source.fetch(*arguments)
    second = source.fetch(*arguments)
    same_seed = DeterministicBarSource(seed=11).fetch(*arguments)
    different_seed = DeterministicBarSource(seed=12).fetch(*arguments)

    assert first == second == same_seed
    assert first != different_seed


def test_fixture_bars_satisfy_market_data_invariants() -> None:
    eastern = timezone(-timedelta(hours=4))
    ingested_at = datetime(2026, 6, 12, 21, 0, tzinfo=eastern)
    bars = DeterministicBarSource(seed=23).fetch(
        MVP_UNIVERSE,
        date(2026, 6, 8),
        date(2026, 6, 12),
        ingested_at,
    )

    for bar in bars:
        assert bar.low <= min(bar.open, bar.close)
        assert bar.high >= max(bar.open, bar.close)
        assert bar.open > 0
        assert bar.high > 0
        assert bar.low > 0
        assert bar.close > 0
        assert bar.vwap > 0
        assert type(bar.volume) is int
        assert bar.volume >= 0
        assert type(bar.trade_count) is int
        assert bar.trade_count >= 0
        assert bar.ingested_at == datetime(2026, 6, 13, 1, 0, tzinfo=UTC)
        assert bar.ingested_at.tzinfo is UTC


def test_fixture_uses_inclusive_xnys_session_boundaries() -> None:
    bars = DeterministicBarSource(seed=31).fetch(
        MVP_UNIVERSE,
        date(2024, 7, 3),
        date(2024, 7, 5),
        datetime(2024, 7, 6, tzinfo=UTC),
    )

    assert {bar.session for bar in bars} == {
        date(2024, 7, 3),
        date(2024, 7, 5),
    }


def test_fixture_rejects_reversed_date_range() -> None:
    with pytest.raises(ValueError, match="start must be on or before end"):
        DeterministicBarSource(seed=1).fetch(
            MVP_UNIVERSE,
            date(2026, 6, 13),
            date(2026, 6, 12),
            datetime(2026, 6, 13, tzinfo=UTC),
        )


def test_fixture_rejects_range_without_xnys_sessions() -> None:
    with pytest.raises(ValueError, match="no XNYS sessions"):
        DeterministicBarSource(seed=1).fetch(
            MVP_UNIVERSE,
            date(2026, 6, 13),
            date(2026, 6, 14),
            datetime(2026, 6, 15, tzinfo=UTC),
        )
