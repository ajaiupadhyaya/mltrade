"""Backtest-overfitting diagnostics: Deflated Sharpe Ratio and PBO.

The single most important question a serious reviewer asks of any backtest is
whether the reported Sharpe survives the multiple-testing and non-normality that
selection introduces.  Two complementary, literature-standard answers:

- **Deflated Sharpe Ratio (DSR)** — Bailey & López de Prado (2014).  The
  probability that the *true* Sharpe is positive after deflating the observed
  Sharpe for the number of trials, sample length, skewness and kurtosis of the
  returns.  Built on the Probabilistic Sharpe Ratio (PSR).

- **Probability of Backtest Overfitting (PBO)** — Bailey, Borwein, López de
  Prado & Zhu (2017), via Combinatorially Symmetric Cross-Validation (CSCV).
  The fraction of in-sample/out-of-sample splits in which the configuration
  selected as best in-sample underperforms the median out-of-sample.

Both consume the frozen trials matrix (one return series per ridge-alpha
configuration) so the diagnostics are computed deterministically and offline.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import combinations

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict
from scipy import stats  # type: ignore[import-untyped]

from mltrade.analytics.returns import (
    PERIODS_PER_YEAR,
    ReturnsLike,
    as_array,
    per_period_sharpe,
)

ROUND_PLACES = 10
_EULER_MASCHERONI = 0.5772156649015329


def _round(x: float) -> float:
    return round(x, ROUND_PLACES)


class OverfittingStats(BaseModel):
    """Combined DSR + PBO overfitting diagnostics."""

    model_config = ConfigDict(frozen=True)

    observed_sharpe_annualized: float
    n_trials: int
    n_observations: int
    skewness: float
    kurtosis: float
    deflated_threshold_sharpe: float
    deflated_sharpe_ratio: float
    psr_vs_zero: float
    pbo: float
    pbo_n_splits: int
    pbo_n_combinations: int
    logit_median: float
    logits: tuple[float, ...]


def probabilistic_sharpe_ratio(
    returns: ReturnsLike, *, threshold_sr: float = 0.0
) -> float:
    """PSR: probability the per-period Sharpe exceeds ``threshold_sr``.

    Accounts for sample length, skewness and (non-excess) kurtosis.
    """
    arr = as_array(returns)
    n = arr.size
    if n < 3:
        return 0.0
    sr = per_period_sharpe(arr)
    skew = float(stats.skew(arr, bias=False))
    kurt = float(stats.kurtosis(arr, fisher=False, bias=False))  # non-excess
    denom = math.sqrt(max(1e-12, 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr**2))
    z = (sr - threshold_sr) * math.sqrt(n - 1) / denom
    return float(stats.norm.cdf(z))


def expected_max_sharpe(sr_std: float, n_trials: int) -> float:
    """Expected maximum per-period Sharpe of ``n_trials`` under the null (SR=0).

    Bailey & López de Prado's analytic approximation for E[max] of N i.i.d.
    standard-normal-scaled Sharpe estimates.
    """
    if n_trials < 2 or sr_std <= 0.0:
        return 0.0
    gamma = _EULER_MASCHERONI
    z1 = float(stats.norm.ppf(1.0 - 1.0 / n_trials))
    z2 = float(stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e)))
    return sr_std * ((1.0 - gamma) * z1 + gamma * z2)


def deflated_sharpe_ratio(
    selected_returns: ReturnsLike, trial_sharpes: ReturnsLike
) -> tuple[float, float]:
    """Return (DSR, deflated_threshold_SR_per_period).

    ``trial_sharpes`` are the per-period Sharpe estimates of every trial; their
    dispersion sets the deflation threshold (the expected best-of-N under the
    null).
    """
    sr_values = as_array(trial_sharpes)
    sr_std = float(np.std(sr_values, ddof=1)) if sr_values.size > 1 else 0.0
    sr0 = expected_max_sharpe(sr_std, sr_values.size)
    dsr = probabilistic_sharpe_ratio(selected_returns, threshold_sr=sr0)
    return dsr, sr0


def probability_of_backtest_overfitting(
    matrix: npt.NDArray[np.float64], *, n_splits: int = 10
) -> tuple[float, list[float]]:
    """PBO via CSCV.  ``matrix`` is (T observations, N trials).

    Returns (pbo, logits) where ``logits`` is the relative-rank logit for each
    train/test combination (negative ⇒ in-sample winner lagged out-of-sample).
    """
    t_obs, n_trials = matrix.shape
    if n_trials < 2:
        raise ValueError("PBO needs at least 2 trials")
    if n_splits % 2 != 0 or n_splits < 2:
        raise ValueError("n_splits must be a positive even integer")
    if t_obs < n_splits:
        raise ValueError("not enough observations for the requested splits")

    blocks = [
        np.asarray(b, dtype=np.int64)
        for b in np.array_split(np.arange(t_obs), n_splits)
    ]
    logits: list[float] = []
    half = n_splits // 2

    def _sharpes(rows: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
        sub = matrix[rows]
        mean = sub.mean(axis=0)
        sd = sub.std(axis=0, ddof=1)
        out = np.zeros(n_trials, dtype=np.float64)
        nz = sd > 0.0
        out[nz] = mean[nz] / sd[nz]
        return out

    for combo in combinations(range(n_splits), half):
        is_rows = np.concatenate([blocks[i] for i in combo])
        oos_rows = np.concatenate(
            [blocks[i] for i in range(n_splits) if i not in combo]
        )
        is_perf = _sharpes(is_rows)
        oos_perf = _sharpes(oos_rows)
        best = int(np.argmax(is_perf))
        ranks = stats.rankdata(oos_perf)  # 1 = worst
        omega = float(ranks[best] / (n_trials + 1))
        omega = min(max(omega, 1e-9), 1.0 - 1e-9)
        logits.append(math.log(omega / (1.0 - omega)))

    pbo = float(np.mean([1.0 if x <= 0.0 else 0.0 for x in logits]))
    return pbo, logits


def compute_overfitting(
    trials: Mapping[str, Sequence[float]],
    *,
    selected: str,
    n_splits: int = 10,
) -> OverfittingStats:
    """Assemble DSR + PBO diagnostics from the aligned trials matrix."""
    if selected not in trials:
        raise ValueError(f"selected trial {selected!r} not in trials")
    labels = sorted(trials)
    lengths = {len(trials[label]) for label in labels}
    if len(lengths) != 1:
        raise ValueError("all trials must share the same number of observations")

    selected_returns = as_array(trials[selected])
    trial_sharpes = [per_period_sharpe(trials[label]) for label in labels]
    dsr, sr0 = deflated_sharpe_ratio(selected_returns, trial_sharpes)

    matrix = np.column_stack([as_array(trials[label]) for label in labels])
    pbo, logits = probability_of_backtest_overfitting(matrix, n_splits=n_splits)

    skew = float(stats.skew(selected_returns, bias=False))
    kurt = float(stats.kurtosis(selected_returns, fisher=True, bias=False))

    return OverfittingStats(
        observed_sharpe_annualized=_round(
            per_period_sharpe(selected_returns) * math.sqrt(PERIODS_PER_YEAR)
        ),
        n_trials=len(labels),
        n_observations=int(selected_returns.size),
        skewness=_round(skew),
        kurtosis=_round(kurt),
        deflated_threshold_sharpe=_round(sr0 * math.sqrt(PERIODS_PER_YEAR)),
        deflated_sharpe_ratio=_round(dsr),
        psr_vs_zero=_round(probabilistic_sharpe_ratio(selected_returns)),
        pbo=_round(pbo),
        pbo_n_splits=n_splits,
        pbo_n_combinations=len(logits),
        logit_median=_round(float(np.median(logits))),
        logits=tuple(_round(x) for x in logits),
    )
