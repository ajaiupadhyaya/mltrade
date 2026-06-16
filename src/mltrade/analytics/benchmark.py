"""Benchmark-relative performance statistics.

Given a strategy return series and an aligned benchmark return series (same
sessions), compute the metrics an allocator asks for first: market beta,
annualised Jensen's alpha *with a t-statistic and p-value*, tracking error,
information ratio, up/down capture, correlation and R-squared.

Alpha and beta come from an ordinary least-squares regression
``r_strategy = alpha + beta * r_benchmark + e`` with classical standard errors,
so the alpha t-stat answers the real question — is the excess return
distinguishable from zero?
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy import stats  # type: ignore[import-untyped]

from mltrade.analytics.returns import PERIODS_PER_YEAR, as_array

ROUND_PLACES = 10


def _round(x: float) -> float:
    return round(x, ROUND_PLACES)


class BenchmarkStats(BaseModel):
    """Strategy performance measured against a benchmark return series."""

    model_config = ConfigDict(frozen=True)

    benchmark: str
    beta: float
    alpha_annualized: float
    alpha_tstat: float
    alpha_pvalue: float
    correlation: float
    r_squared: float
    tracking_error: float
    information_ratio: float
    up_capture: float
    down_capture: float
    n_sessions: int


def _ols(
    y: np.ndarray, x: np.ndarray
) -> tuple[float, float, float]:
    """Regress y on [1, x]; return (intercept, slope, intercept_tstat)."""
    n = y.size
    design = np.column_stack([np.ones(n), x])
    beta_hat, *_ = np.linalg.lstsq(design, y, rcond=None)
    intercept = float(beta_hat[0])
    slope = float(beta_hat[1])
    residuals = y - design @ beta_hat
    dof = n - 2
    if dof <= 0:
        return intercept, slope, 0.0
    sigma2 = float(residuals @ residuals) / dof
    # Var-covar of coefficients: sigma2 * (X'X)^-1
    xtx_inv = np.linalg.inv(design.T @ design)
    se_intercept = float(np.sqrt(sigma2 * xtx_inv[0, 0]))
    tstat = intercept / se_intercept if se_intercept > 0.0 else 0.0
    return intercept, slope, tstat


def compute_benchmark_stats(
    strategy_returns: Sequence[float],
    benchmark_returns: Sequence[float],
    *,
    benchmark: str,
) -> BenchmarkStats:
    """Compute benchmark-relative statistics from two aligned return series."""
    r = as_array(strategy_returns)
    b = as_array(benchmark_returns)
    if r.size != b.size:
        raise ValueError("strategy and benchmark returns must be aligned/equal length")
    n = r.size
    if n < 3:
        raise ValueError("need at least 3 aligned sessions")

    intercept, slope, tstat = _ols(r, b)
    dof = n - 2
    pvalue = float(2.0 * stats.t.sf(abs(tstat), dof)) if dof > 0 else 1.0

    corr = float(np.corrcoef(r, b)[0, 1])
    active = r - b
    te = float(np.std(active, ddof=1)) * np.sqrt(PERIODS_PER_YEAR)
    ir = (float(np.mean(active)) * PERIODS_PER_YEAR / te) if te > 0.0 else 0.0

    up = b > 0.0
    down = b < 0.0
    up_capture = (
        float(np.mean(r[up]) / np.mean(b[up]))
        if up.any() and np.mean(b[up]) != 0.0
        else 0.0
    )
    down_capture = (
        float(np.mean(r[down]) / np.mean(b[down]))
        if down.any() and np.mean(b[down]) != 0.0
        else 0.0
    )

    return BenchmarkStats(
        benchmark=benchmark,
        beta=_round(slope),
        alpha_annualized=_round(intercept * PERIODS_PER_YEAR),
        alpha_tstat=_round(tstat),
        alpha_pvalue=_round(pvalue),
        correlation=_round(corr),
        r_squared=_round(corr**2),
        tracking_error=_round(te),
        information_ratio=_round(ir),
        up_capture=_round(up_capture),
        down_capture=_round(down_capture),
        n_sessions=n,
    )
