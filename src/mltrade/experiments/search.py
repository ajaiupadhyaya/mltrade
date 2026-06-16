"""Typed Optuna search-space sampling for ridge experiments.

``sample_ridge_trial`` maps an Optuna trial onto a fully-revalidated
:class:`ExperimentSpec`, so a sampled candidate can never bypass the spec
contracts.
"""

from __future__ import annotations

from typing import Any

from mltrade.experiments.specs import ExperimentSpec, RidgeSearchSpace


def sample_ridge_trial(
    trial: Any,
    base: ExperimentSpec,
    search: RidgeSearchSpace,
) -> ExperimentSpec:
    alpha = trial.suggest_float(
        "model.alpha", search.alpha.low, search.alpha.high, log=search.alpha.log
    )
    minimum_training_sessions = trial.suggest_categorical(
        "validation.minimum_training_sessions",
        list(search.minimum_training_sessions),
    )
    retrain_every_sessions = trial.suggest_categorical(
        "validation.retrain_every_sessions",
        list(search.retrain_every_sessions),
    )
    candidate = base.model_copy(
        update={
            "model": base.model.model_copy(update={"alpha": alpha}),
            "validation": base.validation.model_copy(
                update={
                    "minimum_training_sessions": minimum_training_sessions,
                    "retrain_every_sessions": retrain_every_sessions,
                }
            ),
        }
    )
    # The one deliberate validated-copy update path; revalidate immediately.
    return ExperimentSpec.model_validate(candidate.model_dump())
