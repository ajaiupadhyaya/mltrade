"""Unit tests for :mod:`mltrade.analytics.attribution`."""

from __future__ import annotations

import numpy as np
import pytest

from mltrade.analytics.attribution import (
    ASSET_CLASS,
    MACRO_FACTORS,
    AttributionStats,
    compute_attribution,
)
from mltrade.universe import MVP_UNIVERSE


def _factor_returns(n: int = 200, seed: int = 13) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    return {
        symbol: rng.normal(0.0, 0.01, size=n).tolist() for symbol, _ in MACRO_FACTORS
    }


def test_betas_recoverable_noise_free() -> None:
    factors = _factor_returns()
    n = len(factors["SPY"])
    # strategy = 1.5*SPY + 0.5*TLT (exact, zero residual).
    strategy = [
        1.5 * s + 0.5 * t
        for s, t in zip(factors["SPY"], factors["TLT"], strict=True)
    ]
    attr = compute_attribution(strategy, factors)

    assert isinstance(attr, AttributionStats)
    assert len(attr.exposures) == len(MACRO_FACTORS)
    by_factor = {e.factor: e.beta for e in attr.exposures}
    assert by_factor["Equity"] == pytest.approx(1.5, abs=1e-6)
    assert by_factor["Duration"] == pytest.approx(0.5, abs=1e-6)
    assert by_factor["Gold"] == pytest.approx(0.0, abs=1e-6)
    assert by_factor["Commodity"] == pytest.approx(0.0, abs=1e-6)
    assert by_factor["EM Risk"] == pytest.approx(0.0, abs=1e-6)
    # Perfect linear fit -> R-squared ~ 1, near-zero alpha.
    assert attr.r_squared == pytest.approx(1.0, abs=1e-9)
    assert attr.alpha_annualized == pytest.approx(0.0, abs=1e-6)
    assert attr.n_sessions == n


def test_r_squared_in_unit_interval_with_noise() -> None:
    factors = _factor_returns(seed=27)
    rng = np.random.default_rng(99)
    spy = factors["SPY"]
    noise = rng.normal(0.0, 0.005, size=len(spy))
    strategy = [
        0.0002 + 0.8 * s + n for s, n in zip(spy, noise, strict=True)
    ]
    attr = compute_attribution(strategy, factors)
    assert 0.0 <= attr.r_squared <= 1.0
    assert len(attr.exposures) == len(MACRO_FACTORS)


def test_misaligned_factor_raises() -> None:
    factors = _factor_returns()
    factors["TLT"] = factors["TLT"][:-1]  # one short -> misaligned
    strategy = factors["SPY"]
    with pytest.raises(ValueError, match="not aligned"):
        compute_attribution(strategy, factors)


def test_not_enough_observations_raises() -> None:
    # n <= k where k = 1 (alpha) + 5 factors = 6.
    n = 5
    factors = {symbol: [0.01] * n for symbol, _ in MACRO_FACTORS}
    strategy = [0.01] * n
    with pytest.raises(ValueError, match="not enough observations"):
        compute_attribution(strategy, factors)


def test_asset_class_maps_all_universe_symbols() -> None:
    symbols = set(MVP_UNIVERSE.symbols)
    assert len(symbols) == 10
    assert symbols <= set(ASSET_CLASS)
    # Every universe symbol resolves to a non-empty class label.
    for symbol in symbols:
        assert ASSET_CLASS[symbol]
