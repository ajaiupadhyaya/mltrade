"""Leakage-safety tests for the trend-momentum-v1 feature pipeline.

The central correctness concern for any ML pipeline is leakage: feature values
for decision session t must depend ONLY on bars with session <= t.

Design of ``rows_through`` and leakage equality
-----------------------------------------------
The plan specifies::

    assert rows_through(original, cutoff) == rows_through(changed, cutoff)

The mutation applied in ``test_future_bar_change_does_not_change_earlier_features``
changes closes AFTER date(2025-06-01) and compares rows through date(2025-05-30).

A 21-session forward label for decision_session=2025-05-30 ends on the 21st
XNYS session after 2025-05-30 — roughly 2025-07-01 — which DOES fall within
the mutated range.  Therefore full-row equality would spuriously *fail* for a
leakage-free implementation (because labels legitimately use future bars that
were mutated).

Resolution: ``rows_through`` returns feature-only projections (named tuples
with label fields stripped).  This design:

  1. Correctly tests the leakage guarantee (features do not depend on future bars).
  2. Correctly allows labels to depend on future bars (a separate test
     ``test_labels_change_when_future_bars_change`` verifies this explicitly).
  3. Is documented and consistent with the plan's stated INTENT.

We additionally test:
  - ``test_labels_change_when_future_bars_change``: proves labels DO change
    when close prices after the cutoff are mutated, confirming they use the
    future as expected.
  - ``test_features_do_not_change_when_future_bars_change``: explicit assertion
    that the feature values themselves are unchanged (more granular than the
    equality of projections).
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import NamedTuple

from mltrade.data.bars import DailyBar
from mltrade.data.fixtures import DeterministicBarSource
from mltrade.features import FeatureRow, build_feature_rows
from mltrade.universe import MVP_UNIVERSE

# ---------------------------------------------------------------------------
# Shared fixture bars  (long enough for 126-session return + 21-session label)
# ---------------------------------------------------------------------------

_INGESTED_AT = datetime(2026, 6, 13, 0, 0, tzinfo=UTC)
_FIXTURE_START = date(2022, 1, 3)
_FIXTURE_END = date(2026, 6, 12)

fixture_bars: tuple[DailyBar, ...] = DeterministicBarSource(seed=7).fetch(
    MVP_UNIVERSE,
    _FIXTURE_START,
    _FIXTURE_END,
    _INGESTED_AT,
)


# ---------------------------------------------------------------------------
# Helper: feature-only projection (no label fields)
# ---------------------------------------------------------------------------


class _FeatureProjection(NamedTuple):
    """Feature-only view of a FeatureRow — label fields are excluded.

    This is what ``rows_through`` compares, so that the leakage test is not
    spuriously broken by label values legitimately changing when future bars
    are mutated (see module docstring).
    """

    symbol: str
    decision_session: date
    latest_source_session: date
    snapshot_id: str
    feature_version: str
    return_21: float
    return_63: float
    return_126: float
    realized_volatility_21: float
    distance_from_sma_100: float
    average_dollar_volume_20: float
    missing: bool


def _project(row: FeatureRow) -> _FeatureProjection:
    """Strip label fields from a FeatureRow."""
    return _FeatureProjection(
        symbol=row.symbol,
        decision_session=row.decision_session,
        latest_source_session=row.latest_source_session,
        snapshot_id=row.snapshot_id,
        feature_version=row.feature_version,
        return_21=row.return_21,
        return_63=row.return_63,
        return_126=row.return_126,
        realized_volatility_21=row.realized_volatility_21,
        distance_from_sma_100=row.distance_from_sma_100,
        average_dollar_volume_20=row.average_dollar_volume_20,
        missing=row.missing,
    )


# ---------------------------------------------------------------------------
# Public helpers (per plan spec)
# ---------------------------------------------------------------------------


def replace_close_after(
    bars: tuple[DailyBar, ...],
    cutoff: date,
    factor: float,
) -> tuple[DailyBar, ...]:
    """Return a new bar tuple with closes multiplied by *factor* after *cutoff*.

    Bars with session <= cutoff are returned unchanged.  Bars with session >
    cutoff have their close (and derived fields open/high/low/vwap) scaled so
    that the DailyBar invariants (high >= open/close, low <= open/close) are
    preserved.

    Parameters
    ----------
    bars:
        Original bar tuple, sorted by (session, symbol).
    cutoff:
        Inclusive boundary: bars with session <= cutoff are unchanged.
    factor:
        Scaling factor applied to close (and all OHLCV prices) for bars after
        the cutoff.

    Returns
    -------
    tuple[DailyBar, ...]
        New bar tuple, same length as *bars*, in the same (session, symbol)
        order.
    """
    from decimal import Decimal

    new_bars: list[DailyBar] = []
    for bar in bars:
        if bar.session <= cutoff:
            new_bars.append(bar)
        else:
            # Scale all price fields by factor (preserves OHLCV invariants
            # because all prices are scaled uniformly).
            scaled_open = Decimal(str(round(float(bar.open) * factor, 4)))
            scaled_high = Decimal(str(round(float(bar.high) * factor, 4)))
            scaled_low = Decimal(str(round(float(bar.low) * factor, 4)))
            scaled_close = Decimal(str(round(float(bar.close) * factor, 4)))
            scaled_vwap = Decimal(str(round(float(bar.vwap) * factor, 4)))
            new_bar = DailyBar(
                instrument=bar.instrument,
                session=bar.session,
                open=scaled_open,
                high=scaled_high,
                low=scaled_low,
                close=scaled_close,
                volume=bar.volume,
                vwap=scaled_vwap,
                trade_count=bar.trade_count,
                source=bar.source,
                ingested_at=bar.ingested_at,
            )
            new_bars.append(new_bar)
    return tuple(new_bars)


def rows_through(
    rows: tuple[FeatureRow, ...],
    cutoff: date,
) -> tuple[_FeatureProjection, ...]:
    """Return feature-only projections of rows with decision_session <= cutoff.

    Label fields (forward_return_21, label_end_session) are intentionally
    excluded.  This makes the leakage test meaningful AND correct:

    - Meaningful: if features leaked future bar data, scaling closes after
      the cutoff by 10x would change return_21/return_63/return_126 etc. for
      rows BEFORE the cutoff, causing this comparison to fail.
    - Correct: labels for rows near the cutoff legitimately depend on bars
      after the cutoff date (they use future closes by design), so comparing
      full rows would produce false failures.  Excluding labels prevents that.

    Parameters
    ----------
    rows:
        All feature rows.
    cutoff:
        Inclusive date boundary.

    Returns
    -------
    tuple[_FeatureProjection, ...]
        Feature-only projections, filtered and sorted by (decision_session,
        symbol).
    """
    filtered = [_project(r) for r in rows if r.decision_session <= cutoff]
    return tuple(sorted(filtered, key=lambda p: (p.decision_session, p.symbol)))


# ---------------------------------------------------------------------------
# Spec test (from plan — literal form)
# ---------------------------------------------------------------------------


def test_future_bar_change_does_not_change_earlier_features() -> None:
    """Feature projections through 2025-05-30 are identical before/after mutation.

    Mutation: all closes after 2025-06-01 are scaled by 10x.
    Cutoff for comparison: 2025-05-30.

    ``rows_through`` compares feature columns only (not labels), because labels
    for rows near the cutoff legitimately use bars after 2025-06-01.  See
    module docstring for full rationale.
    """
    original = build_feature_rows(fixture_bars, "fixture-1", horizon=21)
    changed = build_feature_rows(
        replace_close_after(fixture_bars, date(2025, 6, 1), factor=10),
        "fixture-1",
        horizon=21,
    )
    cutoff = date(2025, 5, 30)
    assert rows_through(original, cutoff) == rows_through(changed, cutoff)


# ---------------------------------------------------------------------------
# Additional explicit leakage-vs-label tests
# ---------------------------------------------------------------------------


def test_labels_change_when_future_bars_change() -> None:
    """Labels DO change when future closes are mutated (labels use the future).

    This confirms that the mutation in ``test_future_bar_change_...`` is
    effective — it actually changes something — making the feature-leakage
    test meaningful.
    """
    original = build_feature_rows(fixture_bars, "fixture-1", horizon=21)
    changed = build_feature_rows(
        replace_close_after(fixture_bars, date(2025, 6, 1), factor=10),
        "fixture-1",
        horizon=21,
    )

    # Find rows with decision_session just before the mutation (so their
    # label_end_session falls AFTER 2025-06-01).
    # decision_session=2025-05-01 → label ends ~21 sessions later ≈ 2025-06-01+
    test_date = date(2025, 5, 1)
    orig_row = next(
        r for r in original if r.symbol == "SPY" and r.decision_session == test_date
    )
    chg_row = next(
        r for r in changed if r.symbol == "SPY" and r.decision_session == test_date
    )

    # Label end session must be after the mutation cutoff for this to be meaningful
    assert orig_row.label_end_session is not None
    assert orig_row.label_end_session > date(2025, 6, 1)

    # Labels MUST differ (mutated by factor 10)
    assert orig_row.forward_return_21 != chg_row.forward_return_21


def test_features_do_not_change_when_future_bars_change() -> None:
    """Feature values are numerically identical before/after future mutation.

    This is a more granular version of the leakage test: check individual
    feature fields, not just tuple equality of projections.
    """
    original = build_feature_rows(fixture_bars, "fixture-1", horizon=21)
    changed = build_feature_rows(
        replace_close_after(fixture_bars, date(2025, 6, 1), factor=10),
        "fixture-1",
        horizon=21,
    )
    cutoff = date(2025, 5, 30)

    orig_dict = {
        (r.symbol, r.decision_session): r
        for r in original
        if r.decision_session <= cutoff
    }
    chg_dict = {
        (r.symbol, r.decision_session): r
        for r in changed
        if r.decision_session <= cutoff
    }

    assert set(orig_dict) == set(chg_dict)
    for key in orig_dict:
        o = orig_dict[key]
        c = chg_dict[key]
        assert o.return_21 == c.return_21, f"{key}: return_21 changed"
        assert o.return_63 == c.return_63, f"{key}: return_63 changed"
        assert o.return_126 == c.return_126, f"{key}: return_126 changed"
        assert o.realized_volatility_21 == c.realized_volatility_21, (
            f"{key}: realized_volatility_21 changed"
        )
        assert o.distance_from_sma_100 == c.distance_from_sma_100, (
            f"{key}: distance_from_sma_100 changed"
        )
        assert o.average_dollar_volume_20 == c.average_dollar_volume_20, (
            f"{key}: average_dollar_volume_20 changed"
        )


def test_leakage_test_has_enough_non_missing_rows() -> None:
    """Sanity check: there must be non-missing rows through the cutoff.

    If all compared rows are missing (sentinel zeros), the leakage test would
    pass vacuously.  This test confirms we have non-missing rows to compare.
    """
    rows = build_feature_rows(fixture_bars, "fixture-1")
    cutoff = date(2025, 5, 30)
    non_missing = [
        r for r in rows if r.decision_session <= cutoff and not r.missing
    ]
    assert len(non_missing) > 100, (
        f"Expected >100 non-missing rows through {cutoff}, got {len(non_missing)}"
    )
