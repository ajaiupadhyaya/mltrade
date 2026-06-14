"""Embargoed walk-forward forecasting pipeline (ridge-trend-v1).

Overview
--------
``generate_forecast_batch`` fits a Ridge regression on past feature rows and
predicts the 21-session forward return for every symbol in the current
cross-section at ``decision_session``.

Embargo / leakage safety
------------------------
The forecast made at decision session ``t`` targets the 21-session forward
return, whose label ends ~21 sessions after ``t``.  To prevent any overlap
between training labels and the forecast horizon we apply a **hard embargo**:

    embargo_start = the XNYS session that is ``embargo_sessions`` (default 21)
                    trading sessions BEFORE ``decision_session`` (inclusive).

Concretely, ``embargo_start`` is chosen so that
``sessions_in_range(embargo_start, decision_session)`` contains exactly
``embargo_sessions`` sessions (i.e. ``embargo_start`` is ``sessions[-embargo_sessions]``
when you take all sessions up through ``decision_session``).

A training row is **usable** iff ALL of:
  1. ``missing=False`` — the feature pipeline had enough history.
  2. ``forward_return_21 is not None`` — the label is available.
  3. ``label_end_session is not None`` — the label boundary is known.
  4. ``label_end_session < embargo_start`` — the training label ends strictly
     before the embargo zone, ensuring no overlap with the forecast horizon.

The plan test asserts exactly::

    max(row.label_end_session for row in split.training) < split.embargo_start

This holds by construction: usable rows are filtered to (4) above, so every
``label_end_session`` in ``split.training`` is strictly less than
``embargo_start``.

Why 21-session embargo?
~~~~~~~~~~~~~~~~~~~~~~~
Forward labels span 21 sessions.  A row at decision session ``s`` has
``label_end_session`` approximately ``s + 21`` sessions.  If we allowed
training rows up to ``label_end_session < t`` (zero embargo), the most recent
training label would end at ``t-1``, whose return computation began at
``t - 21`` sessions — overlapping the very feature window used for the
forecast.  By setting the embargo to 21 sessions, we guarantee that every
training label ends at least 1 session before the embargo zone, which starts
21 sessions before ``t``.  This prevents any contamination of training targets
by information about the forecast period.

Cross-sectional standardisation
--------------------------------
Within each distinct ``decision_session`` group (i.e. within each cross-
section date), each of the 6 feature columns is z-scored across symbols:

    z = (x - mean) / std

where ``std`` is the sample standard deviation (ddof=1, via numpy).  If
``std == 0`` (constant feature within a date), standardised values are set to
``0.0`` rather than producing NaN.  This guard is applied AFTER the non-finite
input check, so a genuine NaN/inf in the raw features raises ``ForecastBlocked``
before standardisation is attempted.

The same per-date statistics (mean + std) computed on the training set are
NOT re-applied to the prediction cross-section.  Instead, the prediction
cross-section (the rows at ``decision_session``) is standardised using its
own cross-sectional mean and std.  This is deliberate: cross-sectional
standardisation is a within-date ranking operation and by definition uses only
the current date's cross-section, so there is no leakage.

Minimum training history
------------------------
At least **504 distinct decision-session dates** must exist among the usable
training rows.  504 ≈ 2 XNYS trading years.  Fewer than 504 raises
``ForecastBlocked("insufficient training history: ...")``.

Non-finite guard
----------------
All 6 raw feature values in any training row must be finite (math.isfinite).
All predicted return values must also be finite.  Either violation raises
``ForecastBlocked("non-finite: ...")``.  (Note: ``missing=False`` rows from
the feature pipeline already store sentinel ``0.0`` rather than NaN in their
feature fields, but the guard is applied defensively for belt-and-suspenders
correctness.)

Determinism
-----------
``Ridge(alpha=1.0)`` with ``sklearn`` is fully deterministic given the same
input matrix — no random state is involved.  Cross-sectional standardisation
uses numpy sample statistics (ddof=1 for std), which are also deterministic.
The feature matrix is constructed in sorted ``(decision_session, symbol)``
order (the canonical output order of ``build_feature_rows``), which is also
deterministic.  Therefore ``generate_forecast_batch`` is fully deterministic:
same inputs → identical ``ForecastBatch``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

import numpy as np
from sklearn.linear_model import Ridge  # type: ignore[import-untyped]

from mltrade.calendar import XNYSCalendar
from mltrade.features.definitions import FeatureRow
from mltrade.models.forecasts import (
    MODEL_VERSION,
    Forecast,
    ForecastBatch,
    ForecastBlocked,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Names of the 6 feature columns (in order) used as model inputs.
_FEATURE_COLS: tuple[str, ...] = (
    "return_21",
    "return_63",
    "return_126",
    "realized_volatility_21",
    "distance_from_sma_100",
    "average_dollar_volume_20",
)

# Minimum distinct decision-session count required in the usable training set.
_MIN_TRAINING_SESSIONS: int = 504

# Default number of sessions to embargo before the decision session.
_DEFAULT_EMBARGO_SESSIONS: int = 21

# Module-level calendar singleton.
_CALENDAR = XNYSCalendar()


# ---------------------------------------------------------------------------
# Training split data structure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingSplit:
    """Result of ``build_training_split``.

    Attributes
    ----------
    training:
        Tuple of usable training ``FeatureRow`` objects (``missing=False``,
        ``forward_return_21`` and ``label_end_session`` are not ``None``,
        ``label_end_session < embargo_start``).  Sorted in the canonical
        ``(decision_session, symbol)`` order inherited from the pipeline.
    embargo_start:
        The first session date of the embargo zone.  Every row in ``training``
        satisfies ``label_end_session < embargo_start`` by construction.
    """

    training: tuple[FeatureRow, ...]
    embargo_start: date


# ---------------------------------------------------------------------------
# Embargo helpers
# ---------------------------------------------------------------------------


def _compute_embargo_start(
    decision_session: date,
    embargo_sessions: int,
) -> date:
    """Return the first session of the embargo zone.

    The embargo zone is the ``embargo_sessions`` XNYS sessions ending at
    (and including) ``decision_session``.  ``embargo_start`` is the first of
    those sessions, i.e. the session such that
    ``sessions_in_range(embargo_start, decision_session)`` contains exactly
    ``embargo_sessions`` elements.

    Parameters
    ----------
    decision_session:
        The forecast date.
    embargo_sessions:
        Number of sessions to embargo (default 21 = one forward-return horizon).

    Returns
    -------
    date
        The first session in the embargo zone.

    Raises
    ------
    ValueError
        If there are fewer than ``embargo_sessions`` XNYS sessions in a
        sufficiently wide lookback window (sanity guard; should not occur for
        realistic inputs).
    """
    # Use a 3x window to ensure we always have enough sessions to look back.
    lookback_days = embargo_sessions * 3 + 10
    window_start = decision_session - timedelta(days=lookback_days)
    sessions = _CALENDAR.sessions_in_range(window_start, decision_session)
    if len(sessions) < embargo_sessions:
        raise ValueError(
            f"Cannot compute embargo_start: only {len(sessions)} XNYS sessions "
            f"in window [{window_start}, {decision_session}] but need "
            f"{embargo_sessions}."
        )
    # sessions[-embargo_sessions] is the session such that there are exactly
    # embargo_sessions sessions from it to decision_session inclusive.
    return sessions[-embargo_sessions]


# ---------------------------------------------------------------------------
# Public: build training split
# ---------------------------------------------------------------------------


def build_training_split(
    feature_rows: tuple[FeatureRow, ...],
    *,
    decision_session: date,
    embargo_sessions: int = _DEFAULT_EMBARGO_SESSIONS,
) -> TrainingSplit:
    """Filter feature rows to the usable training set for a given decision date.

    A row is usable iff:
      1. ``missing=False``
      2. ``forward_return_21 is not None``
      3. ``label_end_session is not None``
      4. ``label_end_session < embargo_start``

    The returned ``TrainingSplit.embargo_start`` is the first XNYS session in
    the embargo zone (see module docstring for the precise definition).

    Parameters
    ----------
    feature_rows:
        All available feature rows (may include rows at or after
        ``decision_session``; those are simply excluded by the filter).
    decision_session:
        The rebalance date.  Rows at this date form the prediction cross-
        section; they are NOT included in the training set.
    embargo_sessions:
        Number of sessions to embargo.  Defaults to 21 (one forward-return
        horizon).

    Returns
    -------
    TrainingSplit
        ``training`` holds the filtered rows; ``embargo_start`` is the
        embargo boundary date.
    """
    embargo_start = _compute_embargo_start(decision_session, embargo_sessions)

    usable: list[FeatureRow] = []
    for row in feature_rows:
        # Must have all required fields
        if row.missing:
            continue
        if row.forward_return_21 is None:
            continue
        if row.label_end_session is None:
            continue
        # Hard embargo: label must end strictly before the embargo zone
        if row.label_end_session >= embargo_start:
            continue
        usable.append(row)

    return TrainingSplit(
        training=tuple(usable),
        embargo_start=embargo_start,
    )


# ---------------------------------------------------------------------------
# Internal: cross-sectional standardisation
# ---------------------------------------------------------------------------


def _standardize_cross_sectionally(
    matrix: np.ndarray,  # shape (n_rows, n_features)
    date_indices: list[tuple[int, int]],
) -> np.ndarray:
    """Z-score each feature column within each decision-date group.

    Parameters
    ----------
    matrix:
        Feature matrix with rows ordered by (decision_session, symbol).
    date_indices:
        List of (start_idx, end_idx) pairs (exclusive end) identifying the
        contiguous row slices belonging to each distinct decision date.
        Must cover all rows in ``matrix`` (non-overlapping, consecutive).

    Returns
    -------
    np.ndarray
        A new array of the same shape with within-date z-scores.  If a
        column's std is 0 within a date, those values are set to 0.0.
    """
    result = matrix.copy()
    for start, end in date_indices:
        group = matrix[start:end, :]  # shape (n_symbols, n_features)
        mean = group.mean(axis=0)  # shape (n_features,)
        std = group.std(axis=0, ddof=1)  # sample std, shape (n_features,)
        # Guard: std==0 → leave as 0.0 (avoids NaN from 0/0)
        safe_std = np.where(std == 0.0, 1.0, std)
        result[start:end, :] = (group - mean) / safe_std
        # Where std was 0, the result is (x - mean) / 1.0 which would be
        # a non-zero value if any x != mean; but if std==0 then ALL x==mean
        # so (x - mean) = 0, giving 0.0 correctly.
    return result


def _build_date_index_map(
    rows: list[FeatureRow],
) -> list[tuple[int, int]]:
    """Identify contiguous slices of rows sharing the same decision_session.

    Parameters
    ----------
    rows:
        Feature rows expected to be sorted by (decision_session, symbol).

    Returns
    -------
    list[tuple[int, int]]
        Each entry ``(start, end)`` means rows[start:end] share a single
        decision_session.
    """
    slices: list[tuple[int, int]] = []
    if not rows:
        return slices
    current_date = rows[0].decision_session
    start = 0
    for i, row in enumerate(rows):
        if row.decision_session != current_date:
            slices.append((start, i))
            current_date = row.decision_session
            start = i
    slices.append((start, len(rows)))
    return slices


# ---------------------------------------------------------------------------
# Internal: feature extraction
# ---------------------------------------------------------------------------


def _extract_features(rows: list[FeatureRow]) -> np.ndarray:
    """Extract the 6 feature columns as a float64 numpy array.

    Parameters
    ----------
    rows:
        Feature rows to convert.

    Returns
    -------
    np.ndarray
        Shape ``(len(rows), 6)``, dtype float64.
    """
    data = [
        [
            row.return_21,
            row.return_63,
            row.return_126,
            row.realized_volatility_21,
            row.distance_from_sma_100,
            row.average_dollar_volume_20,
        ]
        for row in rows
    ]
    return np.array(data, dtype=np.float64)


# ---------------------------------------------------------------------------
# Public: generate forecast batch
# ---------------------------------------------------------------------------


def generate_forecast_batch(
    feature_rows: tuple[FeatureRow, ...],
    decision_session: date,
    *,
    embargo_sessions: int = _DEFAULT_EMBARGO_SESSIONS,
) -> ForecastBatch:
    """Fit a Ridge regression and predict forward returns for one decision date.

    Full pipeline:
      1. Build the embargoed training split via ``build_training_split``.
      2. Verify at least 504 distinct training decision-sessions exist;
         raise ``ForecastBlocked("insufficient training history: ...")`` if not.
      3. Check all 6 raw feature values in training rows are finite;
         raise ``ForecastBlocked("non-finite: ...")`` if any are not.
      4. Build the prediction cross-section (non-missing rows at
         ``decision_session``).
      5. Check all 6 raw feature values in the prediction cross-section are
         finite; raise ``ForecastBlocked("non-finite: ...")`` if any are not.
      6. Standardise training features cross-sectionally (z-score within each
         decision-date group).
      7. Standardise prediction cross-section cross-sectionally (z-score within
         the single decision-date group).
      8. Fit ``Ridge(alpha=1.0)`` on (standardised training features,
         training labels).
      9. Predict on the standardised prediction cross-section.
      10. Verify all predictions are finite; raise ``ForecastBlocked`` if not.
      11. Return a ``ForecastBatch`` with one ``Forecast`` per prediction symbol.

    Parameters
    ----------
    feature_rows:
        All available feature rows.  Should include rows spanning many
        years before ``decision_session`` (for training) and rows at
        ``decision_session`` (for prediction).
    decision_session:
        The rebalance date.
    embargo_sessions:
        Number of XNYS sessions to embargo.  Defaults to 21.

    Returns
    -------
    ForecastBatch
        Contains one ``Forecast`` per symbol in the prediction cross-section,
        plus metadata about the training split used.

    Raises
    ------
    ForecastBlocked
        - ``"non-finite: ..."`` — a feature value or model output is not finite.
        - ``"insufficient training history: ..."`` — fewer than 504 distinct
          training sessions after embargo filtering.
    """
    # ------------------------------------------------------------------
    # Step 1: build embargoed training split
    # ------------------------------------------------------------------
    split = build_training_split(
        feature_rows,
        decision_session=decision_session,
        embargo_sessions=embargo_sessions,
    )
    training_rows = list(split.training)
    embargo_start = split.embargo_start

    # ------------------------------------------------------------------
    # Step 2: check minimum training sessions
    # ------------------------------------------------------------------
    distinct_training_sessions: set[date] = {
        row.decision_session for row in training_rows
    }
    n_training_sessions = len(distinct_training_sessions)
    if n_training_sessions < _MIN_TRAINING_SESSIONS:
        raise ForecastBlocked(
            f"insufficient training history: only {n_training_sessions} distinct "
            f"training sessions available (need >= {_MIN_TRAINING_SESSIONS})"
        )

    # ------------------------------------------------------------------
    # Step 3: check non-finite training features (before standardisation)
    # ------------------------------------------------------------------
    for row in training_rows:
        values: tuple[float, ...] = (
            row.return_21,
            row.return_63,
            row.return_126,
            row.realized_volatility_21,
            row.distance_from_sma_100,
            row.average_dollar_volume_20,
        )
        for col_name, val in zip(_FEATURE_COLS, values, strict=True):
            if not math.isfinite(val):
                raise ForecastBlocked(
                    f"non-finite: training row ({row.symbol}, "
                    f"{row.decision_session}) has {col_name}={val!r}"
                )

    # ------------------------------------------------------------------
    # Step 4: build prediction cross-section
    # ------------------------------------------------------------------
    pred_rows: list[FeatureRow] = [
        row
        for row in feature_rows
        if row.decision_session == decision_session and not row.missing
    ]
    if not pred_rows:
        raise ForecastBlocked(
            f"non-finite: no non-missing rows found at decision_session "
            f"{decision_session} — cannot build prediction cross-section"
        )
    # Sort for determinism: (decision_session, symbol)
    pred_rows.sort(key=lambda r: (r.decision_session, r.symbol))

    # ------------------------------------------------------------------
    # Step 5: check non-finite prediction features (before standardisation)
    # ------------------------------------------------------------------
    for row in pred_rows:
        values = (
            row.return_21,
            row.return_63,
            row.return_126,
            row.realized_volatility_21,
            row.distance_from_sma_100,
            row.average_dollar_volume_20,
        )
        for col_name, val in zip(_FEATURE_COLS, values, strict=True):
            if not math.isfinite(val):
                raise ForecastBlocked(
                    f"non-finite: prediction row ({row.symbol}, "
                    f"{row.decision_session}) has {col_name}={val!r}"
                )

    # ------------------------------------------------------------------
    # Step 6: build training matrix and labels; standardise cross-sectionally
    # ------------------------------------------------------------------
    # training_rows is sorted by (decision_session, symbol) from the pipeline
    # (build_feature_rows output ordering is preserved through the filter).
    # Sort explicitly to be safe.
    training_rows.sort(key=lambda r: (r.decision_session, r.symbol))

    X_train_raw = _extract_features(training_rows)  # (n_train, 6)
    y_train = np.array(
        [row.forward_return_21 for row in training_rows],
        dtype=np.float64,
    )

    train_date_slices = _build_date_index_map(training_rows)
    X_train = _standardize_cross_sectionally(X_train_raw, train_date_slices)

    # ------------------------------------------------------------------
    # Step 7: standardise prediction cross-section
    # ------------------------------------------------------------------
    X_pred_raw = _extract_features(pred_rows)  # (n_pred, 6)
    # Single date group spanning all pred rows
    pred_date_slices = [(0, len(pred_rows))]
    X_pred = _standardize_cross_sectionally(X_pred_raw, pred_date_slices)

    # ------------------------------------------------------------------
    # Step 8: fit Ridge regression
    # ------------------------------------------------------------------
    model: Ridge = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # Step 9: predict
    # ------------------------------------------------------------------
    y_pred: np.ndarray = model.predict(X_pred)

    # ------------------------------------------------------------------
    # Step 10: verify finite predictions
    # ------------------------------------------------------------------
    pred_symbols = [row.symbol for row in pred_rows]
    for sym, pred_val in zip(pred_symbols, y_pred.tolist(), strict=True):
        if not math.isfinite(pred_val):
            raise ForecastBlocked(
                f"non-finite: model prediction for {sym} at "
                f"{decision_session} is {pred_val!r}"
            )

    # ------------------------------------------------------------------
    # Step 11: build result
    # ------------------------------------------------------------------
    forecasts: tuple[Forecast, ...] = tuple(
        Forecast(
            symbol=row.symbol,
            decision_session=decision_session,
            predicted_forward_return=float(pred_val),
            model_version=MODEL_VERSION,
        )
        for row, pred_val in zip(pred_rows, y_pred.tolist(), strict=True)
    )

    training_sessions_sorted = sorted(distinct_training_sessions)
    model_version: Literal["ridge-trend-v1"] = MODEL_VERSION

    return ForecastBatch(
        decision_session=decision_session,
        forecasts=forecasts,
        training_start=training_sessions_sorted[0],
        training_end=training_sessions_sorted[-1],
        embargo_start=embargo_start,
        training_row_count=len(training_rows),
        training_session_count=n_training_sessions,
        model_version=model_version,
    )
