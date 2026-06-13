"""Point-in-time feature pipeline: trend-momentum-v1.

Overview
--------
``build_feature_rows`` accepts a tuple of :class:`~mltrade.data.bars.DailyBar`
objects and returns a sorted, immutable tuple of :class:`FeatureRow` objects.

Leakage safety
--------------
All feature columns depend ONLY on bars with session <= decision_session.
The implementation:

1. Groups bars by symbol.
2. Sorts each group by session ascending.
3. Computes rolling window features using only lags (shift > 0) or the
   current bar — never any look-ahead Polars expression.
4. Forward return (the LABEL) uses shift(-21), which does use future bars.
   Label fields are excluded from the leakage-safety guarantee.

The ``build_feature_rows`` function is fully deterministic: same bars →
same output regardless of call order.

Performance note
----------------
We use Polars in eager mode.  For MVP scale (10 symbols x ~1100 sessions),
eager computation is fast enough and keeps the code straightforward.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Literal

import polars as pl

from mltrade.data.bars import DailyBar
from mltrade.features.definitions import FEATURE_VERSION, FeatureRow

# Minimum number of bars per symbol required for a non-missing feature row.
# Dominated by the 126-session return which needs close[t] + close[t-126]
# = 127 distinct session bars.
_MIN_BARS_FOR_FULL_FEATURES: int = 127

_SENTINEL: float = 0.0
_SQRT_252: float = math.sqrt(252.0)


def _bars_to_dataframe(bars: tuple[DailyBar, ...]) -> pl.DataFrame:
    """Convert DailyBar tuple to a Polars DataFrame with float close/volume."""
    records = [
        {
            "symbol": bar.instrument.symbol,
            "session": bar.session,
            "close": float(bar.close),
            "volume": float(bar.volume),
        }
        for bar in bars
    ]
    return pl.DataFrame(
        records,
        schema={
            "symbol": pl.Utf8,
            "session": pl.Date,
            "close": pl.Float64,
            "volume": pl.Float64,
        },
    )


def _compute_symbol_features(
    df: pl.DataFrame,
    snapshot_id: str,
) -> pl.DataFrame:
    """Compute all feature columns for a single-symbol DataFrame.

    Parameters
    ----------
    df:
        Must be sorted by session ascending, single symbol.
    snapshot_id:
        Passed through to the output rows.

    Returns
    -------
    pl.DataFrame
        One row per session with all feature columns populated.
    """
    # ------------------------------------------------------------------
    # Rolling feature definitions (all use only past bars via shift(N>0)
    # or the current row — never look-ahead).
    # ------------------------------------------------------------------

    # Simple N-session returns: close[t] / close[t-N] - 1
    close = pl.col("close")

    df = df.with_columns(
        [
            # Returns (simple)
            (close / close.shift(21) - 1.0).alias("return_21"),
            (close / close.shift(63) - 1.0).alias("return_63"),
            (close / close.shift(126) - 1.0).alias("return_126"),
            # Daily log-returns for volatility: log(close[t]/close[t-1])
            (close / close.shift(1)).log(base=math.e).alias("log_ret"),
            # SMA-100 of close (uses 100 bars ending at t, inclusive)
            close.rolling_mean(window_size=100).alias("sma_100"),
            # Dollar volume per session: close * volume
            (close * pl.col("volume")).alias("dv"),
            # Forward return label: close[t+21] / close[t] - 1
            (close.shift(-21) / close - 1.0).alias("forward_return_21_raw"),
            # Session at t+21 (for label_end_session)
            pl.col("session").shift(-21).alias("label_end_session_raw"),
        ]
    )

    # Realized volatility: std of last 21 log-returns, annualised
    # rolling_std with window_size=21 uses the 21 most recent log-returns
    # (indices t-20 .. t), each of which uses only bars up to t.
    df = df.with_columns(
        [
            (pl.col("log_ret").rolling_std(window_size=21) * _SQRT_252).alias(
                "realized_volatility_21"
            ),
            # Distance from SMA-100
            (close / pl.col("sma_100") - 1.0).alias("distance_from_sma_100"),
            # Average dollar volume over last 20 sessions
            pl.col("dv").rolling_mean(window_size=20).alias("average_dollar_volume_20"),
        ]
    )

    # Missing flag: True when the row index (0-based) < _MIN_BARS_FOR_FULL_FEATURES - 1
    # i.e. when we have fewer than 127 bars (including the current bar).
    n = df.height
    missing_mask = [i < (_MIN_BARS_FOR_FULL_FEATURES - 1) for i in range(n)]
    df = df.with_columns(pl.Series("missing", missing_mask, dtype=pl.Boolean))

    # Add snapshot_id and feature_version columns
    df = df.with_columns(
        [
            pl.lit(snapshot_id).alias("snapshot_id"),
            pl.lit(FEATURE_VERSION).alias("feature_version"),
            # latest_source_session = decision_session (always)
            pl.col("session").alias("latest_source_session"),
        ]
    )

    return df


def _get_fval(
    row: dict[str, object],
    name: str,
    is_missing: bool,
) -> float:
    """Return a finite float feature value, or the sentinel if missing/non-finite.

    Extracted as a module-level function to avoid the B023 lint error that
    arises when a nested function closes over a loop variable.
    """
    if is_missing:
        return _SENTINEL
    v = row[name]
    if v is None or not math.isfinite(float(v)):  # type: ignore[arg-type]
        return _SENTINEL
    return float(v)  # type: ignore[arg-type]


def _row_to_feature_row(
    row: dict[str, object],
    snapshot_id: str,
) -> FeatureRow:
    """Convert one Polars row dict to a FeatureRow.

    Extracted as a module-level function (not a nested closure) so that mypy
    and ruff (B023) are both satisfied.
    """
    is_missing: bool = bool(row["missing"])

    fwd_raw = row["forward_return_21_raw"]
    label_end_raw = row["label_end_session_raw"]

    forward_return_21: float | None
    label_end_session: date | None

    if fwd_raw is None or (
        isinstance(fwd_raw, float) and not math.isfinite(fwd_raw)
    ):
        forward_return_21 = None
        label_end_session = None
    else:
        forward_return_21 = float(fwd_raw)  # type: ignore[arg-type]
        label_end_session = label_end_raw  # type: ignore[assignment]

    feature_version: Literal["trend-momentum-v1"] = "trend-momentum-v1"

    return FeatureRow(
        symbol=str(row["symbol"]),
        decision_session=row["session"],  # type: ignore[arg-type]
        latest_source_session=row["session"],  # type: ignore[arg-type]
        snapshot_id=snapshot_id,
        feature_version=feature_version,
        return_21=_get_fval(row, "return_21", is_missing),
        return_63=_get_fval(row, "return_63", is_missing),
        return_126=_get_fval(row, "return_126", is_missing),
        realized_volatility_21=_get_fval(row, "realized_volatility_21", is_missing),
        distance_from_sma_100=_get_fval(row, "distance_from_sma_100", is_missing),
        average_dollar_volume_20=_get_fval(row, "average_dollar_volume_20", is_missing),
        forward_return_21=forward_return_21,
        label_end_session=label_end_session,
        missing=is_missing,
    )


def build_feature_rows(
    bars: tuple[DailyBar, ...],
    snapshot_id: str,
    horizon: int = 21,
) -> tuple[FeatureRow, ...]:
    """Compute point-in-time trend-momentum features for every session and symbol.

    Parameters
    ----------
    bars:
        All daily bars to process.  Bars for any symbol that appears in the
        tuple are processed; no universe filtering is applied here.
    snapshot_id:
        Opaque identifier for the bar snapshot; stored verbatim in every row.
    horizon:
        Forward-return horizon in sessions.  Currently fixed at 21; the
        parameter is accepted for API compatibility and must be 21.

    Returns
    -------
    tuple[FeatureRow, ...]
        One row per (symbol, session) pair, sorted by (decision_session,
        symbol).  Rows with insufficient history have ``missing=True`` and
        sentinel feature values.

    Raises
    ------
    ValueError
        If *horizon* is not 21 (the only supported value).
    """
    if horizon != 21:
        raise ValueError(f"Only horizon=21 is supported; got {horizon!r}")

    if not bars:
        return ()

    df = _bars_to_dataframe(bars)

    # Process each symbol independently (preserves per-symbol bar ordering)
    symbol_frames: list[pl.DataFrame] = []
    for symbol in df["symbol"].unique(maintain_order=False).sort():
        sym_df = (
            df.filter(pl.col("symbol") == symbol)
            .sort("session")
        )
        sym_df = _compute_symbol_features(sym_df, snapshot_id)
        symbol_frames.append(sym_df)

    all_df = pl.concat(symbol_frames)

    # Sort by (decision_session, symbol) for deterministic output
    all_df = all_df.sort(["session", "symbol"])

    # ------------------------------------------------------------------
    # Convert to FeatureRow objects, applying sentinel substitution and
    # missing-flag handling.
    # ------------------------------------------------------------------
    feature_rows: list[FeatureRow] = []

    for row in all_df.iter_rows(named=True):
        feature_rows.append(_row_to_feature_row(row, snapshot_id))

    return tuple(feature_rows)
