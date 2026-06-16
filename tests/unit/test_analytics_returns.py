"""Unit tests for :mod:`mltrade.analytics.returns`."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mltrade.analytics.returns import (
    PERIODS_PER_YEAR,
    align_by_session,
    annualized_return,
    annualized_volatility,
    as_array,
    cumulative_growth,
    equity_to_returns,
    per_period_sharpe,
    sharpe_ratio,
)


def test_equity_to_returns_known_answer() -> None:
    # initial=100; curve 110, 121 -> +10% then +10%.
    out = equity_to_returns([110.0, 121.0], initial=100.0)
    assert out == [pytest.approx(0.1), pytest.approx(0.1)]
    # Length matches the equity curve.
    assert len(out) == 2


def test_equity_to_returns_first_measured_against_initial() -> None:
    out = equity_to_returns([90.0], initial=100.0)
    assert out == [pytest.approx(-0.1)]


def test_equity_to_returns_rejects_nonpositive_initial() -> None:
    with pytest.raises(ValueError, match="initial equity must be positive"):
        equity_to_returns([100.0], initial=0.0)
    with pytest.raises(ValueError, match="initial equity must be positive"):
        equity_to_returns([100.0], initial=-5.0)


def test_equity_to_returns_zero_value_then_recovers() -> None:
    # A zero NAV yields a 0.0 return on the following period (prev <= 0 branch).
    out = equity_to_returns([0.0, 50.0], initial=100.0)
    assert out[0] == pytest.approx(-1.0)
    assert out[1] == 0.0


def test_cumulative_growth_known_answer() -> None:
    out = cumulative_growth([0.1, -0.5, 0.0], initial=100.0)
    assert out == [
        pytest.approx(110.0),
        pytest.approx(55.0),
        pytest.approx(55.0),
    ]


def test_cumulative_growth_default_initial_is_one() -> None:
    out = cumulative_growth([0.0, 0.0])
    assert out == [pytest.approx(1.0), pytest.approx(1.0)]


def test_annualized_return_constant_returns_known() -> None:
    # Geometric: growth=(1+r)**n compounded, annualised -> (1+r)**252 - 1.
    r = 0.001
    n = 50
    expected = (1.0 + r) ** PERIODS_PER_YEAR - 1.0
    assert annualized_return([r] * n) == pytest.approx(expected)


def test_annualized_return_empty_is_zero() -> None:
    assert annualized_return([]) == 0.0


def test_annualized_return_total_wipeout_is_minus_one() -> None:
    # A -100% return makes cumulative growth zero -> floor at -1.0.
    assert annualized_return([-1.0, 0.05]) == -1.0


def test_annualized_volatility_constant_is_zero() -> None:
    assert annualized_volatility([0.01] * 10) == 0.0


def test_annualized_volatility_short_input_is_zero() -> None:
    assert annualized_volatility([0.01]) == 0.0
    assert annualized_volatility([]) == 0.0


def test_annualized_volatility_known_answer() -> None:
    returns = [0.01, -0.01, 0.01, -0.01]
    sample_sd = float(np.std(returns, ddof=1))
    expected = sample_sd * math.sqrt(PERIODS_PER_YEAR)
    assert annualized_volatility(returns) == pytest.approx(expected)


def test_sharpe_ratio_zero_dispersion_is_zero() -> None:
    # Exactly-constant (zero std) returns -> Sharpe defined as 0.0 (no risk).
    assert sharpe_ratio([0.0] * 20) == 0.0


def test_sharpe_ratio_short_input_is_zero() -> None:
    assert sharpe_ratio([0.01]) == 0.0
    assert sharpe_ratio([]) == 0.0


def test_sharpe_ratio_known_answer() -> None:
    returns = [0.02, -0.01, 0.03, -0.005, 0.015]
    arr = np.asarray(returns)
    expected = (
        float(np.mean(arr)) / float(np.std(arr, ddof=1)) * math.sqrt(PERIODS_PER_YEAR)
    )
    assert sharpe_ratio(returns) == pytest.approx(expected)


def test_sharpe_ratio_risk_free_shifts_mean() -> None:
    returns = [0.02, -0.01, 0.03, -0.005, 0.015]
    rf = 0.001
    arr = np.asarray(returns) - rf
    expected = (
        float(np.mean(arr)) / float(np.std(arr, ddof=1)) * math.sqrt(PERIODS_PER_YEAR)
    )
    assert sharpe_ratio(returns, risk_free=rf) == pytest.approx(expected)


def test_per_period_sharpe_is_unannualized_sharpe() -> None:
    returns = [0.02, -0.01, 0.03, -0.005, 0.015]
    ann = sharpe_ratio(returns)
    per = per_period_sharpe(returns)
    assert per == pytest.approx(ann / math.sqrt(PERIODS_PER_YEAR))


def test_per_period_sharpe_short_and_zero_vol() -> None:
    assert per_period_sharpe([0.01]) == 0.0
    assert per_period_sharpe([0.01, 0.01, 0.01]) == 0.0


def test_as_array_rejects_non_finite() -> None:
    with pytest.raises(ValueError, match="finite"):
        as_array([0.1, float("nan")])
    with pytest.raises(ValueError, match="finite"):
        as_array([0.1, float("inf")])


def test_as_array_rejects_non_1d() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        as_array([[0.1, 0.2], [0.3, 0.4]])  # type: ignore[list-item]


def test_as_array_returns_float64_copy() -> None:
    out = as_array([1, 2, 3])
    assert out.dtype == np.float64
    assert out.tolist() == [1.0, 2.0, 3.0]


def test_align_by_session_inner_join() -> None:
    a_sessions = ["2024-01-03", "2024-01-02", "2024-01-04"]
    a_returns = [0.01, 0.02, 0.03]
    b_sessions = ["2024-01-02", "2024-01-04", "2024-01-05"]
    b_returns = [0.5, 0.6, 0.7]

    common, a_aligned, b_aligned = align_by_session(
        a_sessions, a_returns, b_sessions, b_returns
    )
    # Inner join keeps only the shared sessions, ascending by string key.
    assert common == ["2024-01-02", "2024-01-04"]
    assert a_aligned == [0.02, 0.03]
    assert b_aligned == [0.5, 0.6]


def test_align_by_session_empty_intersection() -> None:
    common, a_aligned, b_aligned = align_by_session(
        ["a"], [1.0], ["b"], [2.0]
    )
    assert common == []
    assert a_aligned == []
    assert b_aligned == []
