"""Returns-based factor attribution (Sharpe-style exposure analysis).

Regress the strategy's daily returns on a curated set of liquid macro factors
(equity, duration, gold, commodities, emerging-market risk) to reveal what the
strategy is *actually* exposed to — the question that follows any headline
Sharpe.  Each exposure carries a t-statistic, and the regression R-squared says
how much of the strategy's variance the macro factors explain (the residual is
the idiosyncratic, security-selection part).

Also provides the universe → asset-class map used to summarise the live
portfolio's composition.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

from mltrade.analytics.returns import as_array

ROUND_PLACES = 10

# Universe → asset class (for portfolio composition summaries).
ASSET_CLASS: dict[str, str] = {
    "SPY": "US Equity",
    "QQQ": "US Equity",
    "IWM": "US Equity",
    "EFA": "Intl Equity",
    "EEM": "Intl Equity",
    "TLT": "Rates",
    "IEF": "Rates",
    "GLD": "Gold",
    "DBC": "Commodities",
    "VNQ": "Real Estate",
}

# Curated, reasonably distinct macro factors (ETF proxy → factor name).
MACRO_FACTORS: tuple[tuple[str, str], ...] = (
    ("SPY", "Equity"),
    ("TLT", "Duration"),
    ("GLD", "Gold"),
    ("DBC", "Commodity"),
    ("EEM", "EM Risk"),
)


def _round(x: float) -> float:
    return round(x, ROUND_PLACES)


class FactorExposure(BaseModel):
    """One factor's regression loading."""

    model_config = ConfigDict(frozen=True)

    factor: str
    beta: float
    tstat: float


class AttributionStats(BaseModel):
    """Returns-based factor attribution for the strategy."""

    model_config = ConfigDict(frozen=True)

    exposures: tuple[FactorExposure, ...]
    alpha_annualized: float
    alpha_tstat: float
    r_squared: float
    n_sessions: int


def compute_attribution(
    strategy_returns: Sequence[float],
    factor_returns: Mapping[str, Sequence[float]],
    *,
    factors: Sequence[tuple[str, str]] = MACRO_FACTORS,
) -> AttributionStats:
    """Multivariate OLS of strategy returns on the macro factor returns."""
    y = as_array(strategy_returns)
    n = y.size
    columns: list[np.ndarray] = [np.ones(n)]
    names: list[str] = ["alpha"]
    for symbol, label in factors:
        series = as_array(factor_returns[symbol])
        if series.size != n:
            raise ValueError(f"factor {symbol} not aligned with strategy returns")
        columns.append(series)
        names.append(label)

    design = np.column_stack(columns)
    k = design.shape[1]
    if n <= k:
        raise ValueError("not enough observations for the factor regression")

    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ coef
    dof = n - k
    sigma2 = float(residuals @ residuals) / dof
    xtx_inv = np.linalg.inv(design.T @ design)
    se = np.sqrt(np.diag(sigma2 * xtx_inv))
    tstats = np.divide(coef, se, out=np.zeros_like(coef), where=se > 0.0)

    ss_res = float(residuals @ residuals)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0

    exposures = tuple(
        FactorExposure(
            factor=names[i],
            beta=_round(float(coef[i])),
            tstat=_round(float(tstats[i])),
        )
        for i in range(1, k)
    )
    return AttributionStats(
        exposures=exposures,
        alpha_annualized=_round(float(coef[0]) * 252.0),
        alpha_tstat=_round(float(tstats[0])),
        r_squared=_round(r_squared),
        n_sessions=n,
    )
