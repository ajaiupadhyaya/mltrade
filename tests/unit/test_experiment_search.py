from __future__ import annotations

import pytest
from pydantic import ValidationError

pytest.importorskip("optuna")
import optuna

from mltrade.experiments.search import sample_ridge_trial
from mltrade.experiments.specs import (
    DatasetSpec,
    ExperimentSpec,
    FloatSearchSpec,
    RidgeSearchSpace,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _base() -> ExperimentSpec:
    return ExperimentSpec(
        name="ridge-baseline",
        dataset=DatasetSpec(snapshot_id="fixture-2026-06-12"),
        search=RidgeSearchSpace(),
    )


def _sample_once(seed: int) -> ExperimentSpec:
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=seed))
    captured: dict[str, ExperimentSpec] = {}

    def objective(trial: optuna.Trial) -> float:
        spec = sample_ridge_trial(trial, _base(), RidgeSearchSpace())
        captured["spec"] = spec
        return spec.model.alpha

    study.optimize(objective, n_trials=1)
    return captured["spec"]


def test_sampling_is_deterministic_with_seed() -> None:
    first = _sample_once(42)
    second = _sample_once(42)

    assert first.model.alpha == second.model.alpha
    assert (
        first.validation.minimum_training_sessions
        == second.validation.minimum_training_sessions
    )
    assert (
        first.validation.retrain_every_sessions
        == second.validation.retrain_every_sessions
    )


def test_float_search_validates_bounds() -> None:
    with pytest.raises(ValidationError):
        FloatSearchSpec(low=1.0, high=1.0)
    with pytest.raises(ValidationError):
        FloatSearchSpec(low=-1.0, high=1.0, log=True)


def test_sampled_spec_is_revalidated_and_in_bounds() -> None:
    spec = _sample_once(7)

    assert 0.001 <= spec.model.alpha <= 1000.0
    assert spec.validation.minimum_training_sessions in (504, 756, 1008)
    assert spec.validation.retrain_every_sessions in (5, 10, 21, 42)
