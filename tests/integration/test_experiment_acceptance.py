"""Acceptance checks for the tracked example specs and platform invariants."""

from __future__ import annotations

from pathlib import Path

import pytest

from mltrade.config import Settings
from mltrade.experiments.loading import load_experiment_spec

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXPERIMENTS = _REPO_ROOT / "experiments"


def test_committed_baseline_spec_is_valid() -> None:
    loaded = load_experiment_spec(_EXPERIMENTS / "ridge-baseline.toml")

    assert loaded.spec.name == "ridge-baseline"
    assert loaded.spec.search is None
    assert len(loaded.spec_sha256) == 64


def test_committed_balanced_search_spec_defines_a_space() -> None:
    loaded = load_experiment_spec(_EXPERIMENTS / "ridge-balanced-search.toml")
    search = loaded.spec.search

    assert search is not None
    assert search.alpha.low == 0.001
    assert search.alpha.high == 1000.0
    assert search.alpha.log is True
    assert loaded.spec.resources.max_trials >= 10


def test_live_trading_is_structurally_disabled() -> None:
    with pytest.raises(ValueError, match="live trading"):
        Settings(live_trading_enabled=True)
