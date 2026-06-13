"""Typed feature row definition for the trend-momentum-v1 feature set.

Design decisions
----------------
Volatility:
    ``realized_volatility_21`` is the sample standard deviation of the last 21
    daily LOG returns (log(close[t] / close[t-1])), annualised by sqrt(252).
    Log returns are used for statistical properties (additive, better
    normality); the choice is documented and tested.

Missing rows:
    When a decision session lacks enough history to compute ALL features
    (e.g. < 127 bars for the 126-session return or < 101 bars for SMA-100),
    the row still appears in the output with ``missing=True`` and sentinel
    values of ``0.0`` for every float feature field.  This design:

    - Never silently stores NaN or inf in a FeatureRow.
    - Lets downstream consumers filter on ``missing`` rather than silently
      training on garbage.
    - Produces a contiguous index from the first available bar onward.

    We define "enough history" as having at least 127 closes in the per-symbol
    history ending at (and including) the decision session, because 126-return
    needs close[t] and close[t-126], i.e. 127 distinct bars.  SMA-100 needs
    101 bars; volatility needs 22 bars; 63-return needs 64 bars; 21-return
    needs 22 bars; dollar-volume needs 20 bars.  All are dominated by 127.

Labels:
    ``forward_return_21`` = close[t+21] / close[t] - 1 (simple 21-session
    forward return).  ``label_end_session`` is the session at t+21.  If fewer
    than 21 bars follow the decision session in the supplied bar data, both are
    ``None``.  Labels legitimately use future bars — the leakage guarantee
    covers only the feature columns.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict

FEATURE_VERSION: Literal["trend-momentum-v1"] = "trend-momentum-v1"

# Sentinel value for float features when ``missing=True``.
_MISSING_FLOAT: float = 0.0


class FeatureRow(BaseModel):
    """Immutable, point-in-time feature row for one symbol on one decision session.

    Attributes
    ----------
    symbol:
        Ticker symbol (e.g. "SPY").
    decision_session:
        The trading session for which features are computed.  All feature
        values depend ONLY on bars with session <= decision_session.
    latest_source_session:
        The most recent bar session consumed by the feature computation.
        Always equals ``decision_session`` when ``missing`` is False (we
        always consume the bar for the decision session itself).  Also equals
        ``decision_session`` for missing rows, because the decision session bar
        is available (we just lack enough *history*).
    snapshot_id:
        Opaque identifier for the bar snapshot that produced this row.
    feature_version:
        Identifies the feature set schema; always ``"trend-momentum-v1"``.
    return_21:
        21-session simple return: close[t] / close[t-21] - 1.
        Sentinel ``0.0`` when ``missing=True``.
    return_63:
        63-session simple return: close[t] / close[t-63] - 1.
        Sentinel ``0.0`` when ``missing=True``.
    return_126:
        126-session simple return: close[t] / close[t-126] - 1.
        Sentinel ``0.0`` when ``missing=True``.
    realized_volatility_21:
        Sample standard deviation of the last 21 daily log-returns,
        annualised by sqrt(252).  Sentinel ``0.0`` when ``missing=True``.
    distance_from_sma_100:
        close[t] / SMA(close, 100)[t] - 1.
        Sentinel ``0.0`` when ``missing=True``.
    average_dollar_volume_20:
        Mean over the last 20 sessions of (close * volume).
        Sentinel ``0.0`` when ``missing=True``.
    forward_return_21:
        Label: close[t+21] / close[t] - 1.  ``None`` if t+21 is beyond
        available bar data.
    label_end_session:
        Session date at t+21.  ``None`` when ``forward_return_21`` is ``None``.
    missing:
        True when any required feature input is unavailable (e.g. not enough
        history).  All float feature fields hold ``0.0`` sentinels.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    decision_session: date
    latest_source_session: date
    snapshot_id: str
    feature_version: Literal["trend-momentum-v1"]

    return_21: float
    return_63: float
    return_126: float
    realized_volatility_21: float
    distance_from_sma_100: float
    average_dollar_volume_20: float

    forward_return_21: float | None
    label_end_session: date | None

    missing: bool
