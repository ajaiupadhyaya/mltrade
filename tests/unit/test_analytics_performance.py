"""Unit tests for :mod:`mltrade.analytics.performance`."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pytest
from pydantic import ValidationError

from mltrade.analytics.performance import (
    PerformanceStats,
    compute_performance,
    cornish_fisher_var,
    drawdown_series,
    historical_cvar,
    historical_var,
    max_drawdown,
    max_drawdown_duration,
    monthly_returns,
    return_histogram,
    rolling_sharpe,
    sortino_ratio,
    time_to_recovery,
)


def test_sortino_ratio_known_answer() -> None:
    returns = [0.02, -0.01, 0.03, -0.02, 0.01]
    arr = np.asarray(returns)
    downside = np.minimum(arr, 0.0)
    dd = float(np.sqrt(np.mean(downside**2)))
    expected = float(np.mean(arr)) / dd * math.sqrt(252.0)
    assert sortino_ratio(returns) == pytest.approx(expected)


def test_sortino_ratio_no_downside_is_zero() -> None:
    # All returns >= target=0 -> downside deviation is zero -> 0.0.
    assert sortino_ratio([0.01, 0.02, 0.0]) == 0.0


def test_sortino_ratio_short_input_is_zero() -> None:
    assert sortino_ratio([0.01]) == 0.0


def test_drawdown_series_known_path() -> None:
    dd = drawdown_series([0.1, -0.5, 0.0])
    assert dd == [pytest.approx(0.0), pytest.approx(-0.5), pytest.approx(-0.5)]


def test_drawdown_series_empty() -> None:
    assert drawdown_series([]) == []


def test_drawdown_series_is_nonpositive() -> None:
    dd = drawdown_series([0.05, -0.1, 0.2, -0.3, 0.1])
    assert all(x <= 0.0 for x in dd)


def test_max_drawdown_known_answer() -> None:
    assert max_drawdown([0.1, -0.5, 0.0]) == pytest.approx(-0.5)


def test_max_drawdown_empty_is_zero() -> None:
    assert max_drawdown([]) == 0.0


def test_max_drawdown_all_positive_is_zero() -> None:
    assert max_drawdown([0.01, 0.02, 0.03]) == pytest.approx(0.0)


def test_max_drawdown_duration_known() -> None:
    # equity: 1.1, 0.55, 0.605, 0.66 -> underwater for 3 sessions after peak.
    assert max_drawdown_duration([0.1, -0.5, 0.1, 0.1]) == 3


def test_max_drawdown_duration_no_drawdown() -> None:
    assert max_drawdown_duration([0.01, 0.02]) == 0


def test_max_drawdown_duration_empty() -> None:
    assert max_drawdown_duration([]) == 0


def test_time_to_recovery_recovers() -> None:
    # Drop 50% then climb back above the prior peak.
    # equity: 1.1, 0.55, 0.825, 1.2375 -> trough at idx 1, reclaim at idx 3.
    assert time_to_recovery([0.1, -0.5, 0.5, 0.5]) == 2


def test_time_to_recovery_never_recovers_is_none() -> None:
    assert time_to_recovery([0.1, -0.5, 0.01, 0.01]) is None


def test_time_to_recovery_empty_is_none() -> None:
    assert time_to_recovery([]) is None


def test_historical_var_known_quantile() -> None:
    # 100 returns: -0.05 .. 0.04 by 0.001 steps -> 5% quantile is a known loss.
    returns = [(-50 + i) / 1000.0 for i in range(100)]
    var95 = historical_var(returns, level=0.95)
    expected = -float(np.quantile(np.asarray(returns), 0.05))
    assert var95 == pytest.approx(expected)
    assert var95 > 0.0


def test_historical_var_empty_is_zero() -> None:
    assert historical_var([]) == 0.0


def test_historical_var_all_positive_is_zero() -> None:
    # Positive quantile -> max(0, -q) clamps to 0.
    assert historical_var([0.01, 0.02, 0.03, 0.04], level=0.95) == 0.0


def test_historical_var_99_deeper_than_95() -> None:
    returns = [(-50 + i) / 1000.0 for i in range(100)]
    assert historical_var(returns, level=0.99) >= historical_var(returns, level=0.95)


def test_historical_cvar_is_at_least_var() -> None:
    returns = [(-50 + i) / 1000.0 for i in range(100)]
    var95 = historical_var(returns, level=0.95)
    cvar95 = historical_cvar(returns, level=0.95)
    # Expected shortfall is no smaller than VaR.
    assert cvar95 >= var95 - 1e-9


def test_historical_cvar_empty_is_zero() -> None:
    assert historical_cvar([]) == 0.0


def test_cornish_fisher_var_short_falls_back_to_historical() -> None:
    # < 4 observations -> historical VaR path.
    short = [-0.02, 0.01, 0.03]
    assert cornish_fisher_var(short, level=0.95) == historical_var(short, level=0.95)


def test_cornish_fisher_var_normal_close_to_parametric() -> None:
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0, 0.01, size=2000).tolist()
    cf = cornish_fisher_var(returns, level=0.95)
    # For ~symmetric near-normal data, modified VaR ~ 1.645 * sigma.
    assert cf == pytest.approx(0.01 * 1.645, rel=0.15)


def test_cornish_fisher_var_zero_vol_is_zero() -> None:
    assert cornish_fisher_var([0.01, 0.01, 0.01, 0.01], level=0.95) == 0.0


def test_rolling_sharpe_length_and_none_prefix() -> None:
    returns = [0.01, -0.02, 0.03, -0.01, 0.02, 0.0, 0.01]
    out = rolling_sharpe(returns, window=3)
    assert len(out) == len(returns)
    # First (window-1) entries are None.
    assert out[0] is None
    assert out[1] is None
    assert out[2] is not None
    assert all(v is not None for v in out[2:])


def test_rolling_sharpe_window_larger_than_series_all_none() -> None:
    out = rolling_sharpe([0.01, 0.02], window=5)
    assert out == [None, None]


def test_return_histogram_counts_sum_to_n() -> None:
    rng = np.random.default_rng(3)
    returns = rng.normal(0.0, 0.01, size=200).tolist()
    centres, counts = return_histogram(returns, bins=41)
    assert len(centres) == 41
    assert len(counts) == 41
    assert sum(counts) == 200


def test_return_histogram_empty() -> None:
    centres, counts = return_histogram([])
    assert centres == []
    assert counts == []


def test_monthly_returns_two_months_compound() -> None:
    dates = [
        date(2024, 1, 10),
        date(2024, 1, 20),
        date(2024, 2, 5),
    ]
    returns = [0.1, 0.1, -0.5]
    out = monthly_returns(dates, returns)
    assert len(out) == 2
    # January: (1.1)(1.1) - 1 = 0.21
    assert out[0] == {"year": 2024, "month": 1, "ret": pytest.approx(0.21)}
    # February: single -50% day.
    assert out[1] == {"year": 2024, "month": 2, "ret": pytest.approx(-0.5)}


def test_monthly_returns_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        monthly_returns([date(2024, 1, 1)], [0.1, 0.2])


def test_compute_performance_frozen_and_sane() -> None:
    rng = np.random.default_rng(11)
    returns = (0.0005 + rng.normal(0.0, 0.01, size=300)).tolist()
    stats = compute_performance(returns)

    assert isinstance(stats, PerformanceStats)
    assert stats.n_sessions == 300
    assert stats.max_drawdown <= 0.0
    assert stats.max_drawdown_duration >= 0
    assert -1.0 <= stats.positive_fraction <= 1.0
    assert stats.var_95 >= 0.0
    assert stats.cvar_95 >= 0.0
    assert stats.var_99 >= 0.0
    assert stats.cvar_99 >= 0.0
    assert stats.cornish_fisher_var_95 >= 0.0
    assert stats.best_day >= stats.worst_day
    # Frozen pydantic model: mutation is rejected.
    with pytest.raises(ValidationError):
        stats.sharpe = 0.0  # type: ignore[misc]


def test_compute_performance_calmar_positive_when_drawdown() -> None:
    # Engineered series with a real drawdown and positive annualised return.
    returns = [0.02] * 50 + [-0.1] + [0.02] * 50
    stats = compute_performance(returns)
    assert stats.max_drawdown < 0.0
    assert stats.calmar != 0.0


def test_compute_performance_empty_raises() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        compute_performance([])
