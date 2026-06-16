"""Unit tests for :mod:`mltrade.analytics.benchmark`."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mltrade.analytics.benchmark import BenchmarkStats, compute_benchmark_stats


def _benchmark_series(n: int = 120, seed: int = 5) -> list[float]:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0005, 0.01, size=n).tolist()


def test_strategy_is_two_times_benchmark() -> None:
    bench = _benchmark_series()
    strat = [2.0 * b for b in bench]
    stats = compute_benchmark_stats(strat, bench, benchmark="SPY")

    assert isinstance(stats, BenchmarkStats)
    assert stats.benchmark == "SPY"
    assert stats.beta == pytest.approx(2.0, abs=1e-6)
    assert stats.alpha_annualized == pytest.approx(0.0, abs=1e-6)
    # Perfect linear relationship.
    assert stats.correlation == pytest.approx(1.0, abs=1e-9)
    assert stats.r_squared == pytest.approx(1.0, abs=1e-9)
    assert stats.n_sessions == len(bench)


def test_zero_alpha_when_strategy_equals_benchmark() -> None:
    bench = _benchmark_series(seed=9)
    stats = compute_benchmark_stats(bench, bench, benchmark="SPY")
    assert stats.beta == pytest.approx(1.0, abs=1e-6)
    assert stats.alpha_annualized == pytest.approx(0.0, abs=1e-6)
    # Active return is identically zero -> IR/TE are zero.
    assert stats.tracking_error == pytest.approx(0.0, abs=1e-9)
    assert stats.information_ratio == 0.0


def test_capture_and_ir_are_finite_with_alpha() -> None:
    bench = _benchmark_series(seed=3)
    # Add a constant alpha and noise so active return is nonzero.
    rng = np.random.default_rng(17)
    strat = [
        0.0003 + 1.2 * b + n
        for b, n in zip(bench, rng.normal(0.0, 0.002, size=len(bench)), strict=True)
    ]
    stats = compute_benchmark_stats(strat, bench, benchmark="SPY")

    assert math.isfinite(stats.information_ratio)
    assert math.isfinite(stats.tracking_error)
    assert math.isfinite(stats.up_capture)
    assert math.isfinite(stats.down_capture)
    assert stats.tracking_error > 0.0
    assert 0.0 <= stats.alpha_pvalue <= 1.0
    assert 0.0 <= stats.r_squared <= 1.0


def test_misaligned_lengths_raise() -> None:
    with pytest.raises(ValueError, match="aligned/equal length"):
        compute_benchmark_stats([0.01, 0.02, 0.03], [0.01, 0.02], benchmark="SPY")


def test_too_few_sessions_raise() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        compute_benchmark_stats([0.01, 0.02], [0.01, 0.02], benchmark="SPY")
