"""Unit tests for :mod:`mltrade.analytics.overfitting`."""

from __future__ import annotations

import numpy as np
import pytest

from mltrade.analytics.overfitting import (
    OverfittingStats,
    compute_overfitting,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
)


def test_psr_positive_sharpe_above_half() -> None:
    rng = np.random.default_rng(1)
    # Clearly positive drift relative to noise -> PSR(>0) well above 0.5.
    returns = (0.01 + rng.normal(0.0, 0.005, size=400)).tolist()
    assert probabilistic_sharpe_ratio(returns) > 0.5


def test_psr_negative_sharpe_below_half() -> None:
    rng = np.random.default_rng(2)
    returns = (-0.01 + rng.normal(0.0, 0.005, size=400)).tolist()
    assert probabilistic_sharpe_ratio(returns) < 0.5


def test_psr_short_input_is_zero() -> None:
    assert probabilistic_sharpe_ratio([0.01, 0.02]) == 0.0


def test_expected_max_sharpe_monotone_in_n_trials() -> None:
    sr_std = 0.2
    a = expected_max_sharpe(sr_std, 5)
    b = expected_max_sharpe(sr_std, 50)
    c = expected_max_sharpe(sr_std, 500)
    assert a < b < c


def test_expected_max_sharpe_monotone_in_sr_std() -> None:
    n = 20
    assert expected_max_sharpe(0.1, n) < expected_max_sharpe(0.5, n)


def test_expected_max_sharpe_zero_edge_cases() -> None:
    assert expected_max_sharpe(0.0, 100) == 0.0
    assert expected_max_sharpe(0.2, 1) == 0.0
    assert expected_max_sharpe(0.2, 0) == 0.0


def test_deflated_sharpe_ratio_bounds() -> None:
    rng = np.random.default_rng(4)
    selected = (0.008 + rng.normal(0.0, 0.005, size=300)).tolist()
    trial_sharpes = rng.normal(0.0, 0.05, size=10).tolist()
    dsr, sr0 = deflated_sharpe_ratio(selected, trial_sharpes)
    assert 0.0 <= dsr <= 1.0
    assert sr0 >= 0.0


def test_pbo_best_column_near_zero() -> None:
    rng = np.random.default_rng(6)
    t_obs, n_trials = 200, 8
    noise = rng.normal(0.0, 0.01, size=(t_obs, n_trials))
    # Column 0 dominates everywhere both IS and OOS.
    noise[:, 0] += 0.02
    pbo, logits = probability_of_backtest_overfitting(noise, n_splits=10)
    assert len(logits) > 0
    assert pbo == pytest.approx(0.0, abs=0.05)


def test_pbo_iid_noise_does_not_raise() -> None:
    rng = np.random.default_rng(8)
    matrix = rng.normal(0.0, 0.01, size=(120, 6))
    pbo, logits = probability_of_backtest_overfitting(matrix, n_splits=8)
    assert 0.0 <= pbo <= 1.0
    assert len(logits) > 0


def test_pbo_odd_splits_raise() -> None:
    matrix = np.zeros((40, 3)) + np.arange(3)
    with pytest.raises(ValueError, match="even integer"):
        probability_of_backtest_overfitting(matrix, n_splits=5)


def test_pbo_too_few_trials_raise() -> None:
    matrix = np.zeros((40, 1))
    with pytest.raises(ValueError, match="at least 2 trials"):
        probability_of_backtest_overfitting(matrix, n_splits=4)


def test_pbo_too_few_observations_raise() -> None:
    matrix = np.zeros((4, 3)) + np.arange(3)
    with pytest.raises(ValueError, match="not enough observations"):
        probability_of_backtest_overfitting(matrix, n_splits=10)


def _trials(n: int = 300, seed: int = 21) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    trials: dict[str, list[float]] = {}
    for k, drift in enumerate((0.012, 0.006, 0.001, -0.002)):
        trials[f"alpha={k}"] = (drift + rng.normal(0.0, 0.01, size=n)).tolist()
    return trials


def test_compute_overfitting_end_to_end() -> None:
    trials = _trials()
    stats = compute_overfitting(trials, selected="alpha=0", n_splits=10)

    assert isinstance(stats, OverfittingStats)
    assert stats.n_trials == 4
    assert stats.n_observations == 300
    assert 0.0 <= stats.deflated_sharpe_ratio <= 1.0
    assert 0.0 <= stats.pbo <= 1.0
    assert 0.0 <= stats.psr_vs_zero <= 1.0
    assert stats.pbo_n_splits == 10
    assert stats.pbo_n_combinations == len(stats.logits)
    assert stats.pbo_n_combinations > 0


def test_compute_overfitting_missing_selected_raises() -> None:
    trials = _trials()
    with pytest.raises(ValueError, match="not in trials"):
        compute_overfitting(trials, selected="alpha=missing")


def test_compute_overfitting_unequal_lengths_raise() -> None:
    trials = {
        "alpha=0": [0.01] * 50,
        "alpha=1": [0.01] * 49,
    }
    with pytest.raises(ValueError, match="same number of observations"):
        compute_overfitting(trials, selected="alpha=0")
