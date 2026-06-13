"""Unit tests for the trend-momentum-v1 feature pipeline.

Tests cover:
- Output shape and type invariants.
- Missing-flag logic and sentinel values.
- Feature value correctness (spot-checks against manual computation).
- Label correctness (forward return and label_end_session).
- Incomplete label handling at the end of the data range.
- Deterministic, sorted output.
- horizon != 21 rejection.
- Empty bar input.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime

import pytest

from mltrade.data.fixtures import DeterministicBarSource
from mltrade.features import FeatureRow, build_feature_rows
from mltrade.universe import MVP_UNIVERSE

# ---------------------------------------------------------------------------
# Shared fixture bars
# ---------------------------------------------------------------------------

_INGESTED_AT = datetime(2026, 6, 13, 0, 0, tzinfo=UTC)
_FIXTURE_START = date(2022, 1, 3)
_FIXTURE_END = date(2026, 6, 12)

_FIXTURE_BARS = DeterministicBarSource(seed=7).fetch(
    MVP_UNIVERSE,
    _FIXTURE_START,
    _FIXTURE_END,
    _INGESTED_AT,
)


# ---------------------------------------------------------------------------
# Helper: locate a specific FeatureRow
# ---------------------------------------------------------------------------


def _row(
    rows: tuple[FeatureRow, ...],
    symbol: str,
    decision_session: date,
) -> FeatureRow:
    for r in rows:
        if r.symbol == symbol and r.decision_session == decision_session:
            return r
    raise KeyError(f"No row for ({symbol!r}, {decision_session})")


# ---------------------------------------------------------------------------
# Basic shape and type tests
# ---------------------------------------------------------------------------


def test_build_feature_rows_returns_tuple() -> None:
    rows = build_feature_rows(_FIXTURE_BARS, "fixture-1")
    assert isinstance(rows, tuple)
    assert len(rows) > 0
    assert all(isinstance(r, FeatureRow) for r in rows)


def test_output_count_matches_bar_count() -> None:
    """One FeatureRow per (symbol, session) pair = same count as input bars."""
    rows = build_feature_rows(_FIXTURE_BARS, "fixture-1")
    assert len(rows) == len(_FIXTURE_BARS)


def test_output_sorted_by_decision_session_then_symbol() -> None:
    rows = build_feature_rows(_FIXTURE_BARS, "fixture-1")
    keys = [(r.decision_session, r.symbol) for r in rows]
    assert keys == sorted(keys)


def test_snapshot_id_propagated() -> None:
    rows = build_feature_rows(_FIXTURE_BARS, "my-snap-id")
    assert all(r.snapshot_id == "my-snap-id" for r in rows)


def test_feature_version_is_correct_literal() -> None:
    rows = build_feature_rows(_FIXTURE_BARS, "fixture-1")
    assert all(r.feature_version == "trend-momentum-v1" for r in rows)


def test_feature_rows_are_frozen() -> None:
    rows = build_feature_rows(_FIXTURE_BARS, "fixture-1")
    row = rows[0]
    with pytest.raises((TypeError, ValueError)):
        row.symbol = "CHANGED"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Missing-flag and sentinel tests
# ---------------------------------------------------------------------------

# The minimum bar count for full features is 127 (126-return + 1).
_MIN_BARS = 127


def test_early_rows_are_marked_missing() -> None:
    rows = build_feature_rows(_FIXTURE_BARS, "fixture-1")
    spy_rows = sorted(
        (r for r in rows if r.symbol == "SPY"),
        key=lambda r: r.decision_session,
    )
    # First _MIN_BARS - 1 rows (0-indexed) must be missing
    for r in spy_rows[: _MIN_BARS - 1]:
        assert r.missing is True, (
            f"Expected missing=True for decision_session={r.decision_session}"
        )


def test_row_at_min_bar_count_is_not_missing() -> None:
    rows = build_feature_rows(_FIXTURE_BARS, "fixture-1")
    spy_rows = sorted(
        (r for r in rows if r.symbol == "SPY"),
        key=lambda r: r.decision_session,
    )
    # Row at index _MIN_BARS - 1 (0-indexed) is the first non-missing row
    first_full = spy_rows[_MIN_BARS - 1]
    assert first_full.missing is False


def test_missing_rows_have_sentinel_zero_floats() -> None:
    rows = build_feature_rows(_FIXTURE_BARS, "fixture-1")
    for r in rows:
        if r.missing:
            assert r.return_21 == 0.0
            assert r.return_63 == 0.0
            assert r.return_126 == 0.0
            assert r.realized_volatility_21 == 0.0
            assert r.distance_from_sma_100 == 0.0
            assert r.average_dollar_volume_20 == 0.0


def test_non_missing_feature_floats_are_all_finite() -> None:
    rows = build_feature_rows(_FIXTURE_BARS, "fixture-1")
    for r in rows:
        if not r.missing:
            assert math.isfinite(r.return_21), f"return_21 not finite: {r}"
            assert math.isfinite(r.return_63), f"return_63 not finite: {r}"
            assert math.isfinite(r.return_126), f"return_126 not finite: {r}"
            assert math.isfinite(r.realized_volatility_21), (
                f"realized_volatility_21 not finite: {r}"
            )
            assert math.isfinite(r.distance_from_sma_100), (
                f"distance_from_sma_100 not finite: {r}"
            )
            assert math.isfinite(r.average_dollar_volume_20), (
                f"average_dollar_volume_20 not finite: {r}"
            )


# ---------------------------------------------------------------------------
# Feature correctness spot-checks
# ---------------------------------------------------------------------------


def test_latest_source_session_equals_decision_session() -> None:
    """latest_source_session must always equal decision_session (spec requirement)."""
    rows = build_feature_rows(_FIXTURE_BARS, "fixture-1")
    for r in rows:
        assert r.latest_source_session == r.decision_session, (
            f"latest_source_session mismatch: {r}"
        )


def test_return_21_spot_check() -> None:
    """Manually verify return_21 against raw bar data for SPY."""
    bars = _FIXTURE_BARS
    spy_bars = sorted(
        (b for b in bars if b.instrument.symbol == "SPY"),
        key=lambda b: b.session,
    )
    # Find the first non-missing row (index 126)
    idx = _MIN_BARS - 1  # 126
    target_session = spy_bars[idx].session
    expected_return_21 = (
        float(spy_bars[idx].close) / float(spy_bars[idx - 21].close) - 1.0
    )

    rows = build_feature_rows(bars, "fixture-1")
    row = _row(rows, "SPY", target_session)

    assert not row.missing
    assert abs(row.return_21 - expected_return_21) < 1e-10


def test_return_63_spot_check() -> None:
    bars = _FIXTURE_BARS
    spy_bars = sorted(
        (b for b in bars if b.instrument.symbol == "SPY"),
        key=lambda b: b.session,
    )
    idx = _MIN_BARS - 1  # 126
    target_session = spy_bars[idx].session
    expected = float(spy_bars[idx].close) / float(spy_bars[idx - 63].close) - 1.0

    rows = build_feature_rows(bars, "fixture-1")
    row = _row(rows, "SPY", target_session)
    assert abs(row.return_63 - expected) < 1e-10


def test_return_126_spot_check() -> None:
    bars = _FIXTURE_BARS
    spy_bars = sorted(
        (b for b in bars if b.instrument.symbol == "SPY"),
        key=lambda b: b.session,
    )
    idx = _MIN_BARS - 1  # 126
    target_session = spy_bars[idx].session
    expected = float(spy_bars[idx].close) / float(spy_bars[0].close) - 1.0

    rows = build_feature_rows(bars, "fixture-1")
    row = _row(rows, "SPY", target_session)
    assert abs(row.return_126 - expected) < 1e-10


def test_realized_volatility_21_is_log_return_std_annualised() -> None:
    """Verify vol is std of log-returns * sqrt(252), not std of simple returns."""
    import math as _math

    bars = _FIXTURE_BARS
    spy_bars = sorted(
        (b for b in bars if b.instrument.symbol == "SPY"),
        key=lambda b: b.session,
    )
    # Use a later index with plenty of history
    idx = 200
    target_session = spy_bars[idx].session

    # Compute manually: last 21 log-returns ending at idx
    log_rets = [
        _math.log(float(spy_bars[i].close) / float(spy_bars[i - 1].close))
        for i in range(idx - 20, idx + 1)  # 21 log-returns
    ]
    mean_lr = sum(log_rets) / len(log_rets)
    variance = sum((x - mean_lr) ** 2 for x in log_rets) / (len(log_rets) - 1)
    expected_vol = _math.sqrt(variance) * _math.sqrt(252.0)

    rows = build_feature_rows(bars, "fixture-1")
    row = _row(rows, "SPY", target_session)
    assert not row.missing
    assert abs(row.realized_volatility_21 - expected_vol) < 1e-9


def test_distance_from_sma_100_spot_check() -> None:
    bars = _FIXTURE_BARS
    spy_bars = sorted(
        (b for b in bars if b.instrument.symbol == "SPY"),
        key=lambda b: b.session,
    )
    idx = 200
    target_session = spy_bars[idx].session
    sma_100 = sum(float(spy_bars[i].close) for i in range(idx - 99, idx + 1)) / 100.0
    expected = float(spy_bars[idx].close) / sma_100 - 1.0

    rows = build_feature_rows(bars, "fixture-1")
    row = _row(rows, "SPY", target_session)
    assert not row.missing
    assert abs(row.distance_from_sma_100 - expected) < 1e-9


def test_average_dollar_volume_20_spot_check() -> None:
    bars = _FIXTURE_BARS
    spy_bars = sorted(
        (b for b in bars if b.instrument.symbol == "SPY"),
        key=lambda b: b.session,
    )
    idx = 200
    target_session = spy_bars[idx].session
    dv_vals = [
        float(spy_bars[i].close) * spy_bars[i].volume
        for i in range(idx - 19, idx + 1)
    ]
    expected = sum(dv_vals) / 20.0

    rows = build_feature_rows(bars, "fixture-1")
    row = _row(rows, "SPY", target_session)
    assert not row.missing
    assert abs(row.average_dollar_volume_20 - expected) < 1.0  # large scale, small tol


# ---------------------------------------------------------------------------
# Label tests
# ---------------------------------------------------------------------------


def test_forward_return_21_spot_check() -> None:
    bars = _FIXTURE_BARS
    spy_bars = sorted(
        (b for b in bars if b.instrument.symbol == "SPY"),
        key=lambda b: b.session,
    )
    idx = 200
    target_session = spy_bars[idx].session
    expected_fwd = float(spy_bars[idx + 21].close) / float(spy_bars[idx].close) - 1.0
    expected_end = spy_bars[idx + 21].session

    rows = build_feature_rows(bars, "fixture-1")
    row = _row(rows, "SPY", target_session)
    assert row.forward_return_21 is not None
    assert row.label_end_session is not None
    assert abs(row.forward_return_21 - expected_fwd) < 1e-10
    assert row.label_end_session == expected_end


def test_labels_none_at_end_of_bar_range() -> None:
    """Rows within the last 21 sessions should have None labels."""
    rows = build_feature_rows(_FIXTURE_BARS, "fixture-1")
    spy_rows = sorted(
        (r for r in rows if r.symbol == "SPY"),
        key=lambda r: r.decision_session,
    )
    # Last 21 rows must have None labels
    for r in spy_rows[-21:]:
        assert r.forward_return_21 is None, (
            f"Expected None forward_return_21 for {r.decision_session}"
        )
        assert r.label_end_session is None


def test_label_end_session_is_21_sessions_after_decision() -> None:
    """label_end_session should be 21 trading sessions after decision_session."""
    rows = build_feature_rows(_FIXTURE_BARS, "fixture-1")
    spy_rows = sorted(
        (r for r in rows if r.symbol == "SPY"),
        key=lambda r: r.decision_session,
    )
    spy_sessions = [r.decision_session for r in spy_rows]

    # For rows with labels, verify the offset
    for i, r in enumerate(spy_rows):
        if r.label_end_session is not None:
            assert r.label_end_session == spy_sessions[i + 21]


# ---------------------------------------------------------------------------
# Spec requirement test (from plan)
# ---------------------------------------------------------------------------


def test_feature_values_use_only_available_bars() -> None:
    """Spec test: latest_source_session == decision_session, version correct."""
    rows = build_feature_rows(_FIXTURE_BARS, "fixture-1", horizon=21)
    row = next(
        r
        for r in rows
        if r.symbol == "SPY" and r.decision_session == date(2025, 1, 31)
    )
    assert row.latest_source_session == row.decision_session
    assert row.feature_version == "trend-momentum-v1"


# ---------------------------------------------------------------------------
# Determinism test
# ---------------------------------------------------------------------------


def test_build_feature_rows_is_deterministic() -> None:
    rows1 = build_feature_rows(_FIXTURE_BARS, "fixture-1")
    rows2 = build_feature_rows(_FIXTURE_BARS, "fixture-1")
    assert rows1 == rows2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_build_feature_rows_empty_bars_returns_empty() -> None:
    assert build_feature_rows((), "fixture-1") == ()


def test_build_feature_rows_rejects_nonstandard_horizon() -> None:
    with pytest.raises(ValueError, match="horizon=21"):
        build_feature_rows(_FIXTURE_BARS, "fixture-1", horizon=10)
