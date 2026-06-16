"""Persistent, resumable Optuna tuning over the ridge research pipeline.

Studies live in a local SQLite database so they survive interruption and
resume.  A study records an immutable *context hash*; resuming with a changed
dataset/objective/search context is rejected rather than silently mixing
incompatible trials.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Protocol

from mltrade.experiments.loading import LoadedExperimentSpec, loaded_from_spec
from mltrade.experiments.records import build_compatibility_key
from mltrade.experiments.runner import (
    ExperimentBlocked,
    ExperimentFailed,
    ExperimentRunResult,
)
from mltrade.experiments.search import sample_ridge_trial
from mltrade.experiments.specs import ExperimentSpec, StrictFrozenModel


class StudyContextMismatch(RuntimeError):
    """Resuming a study with a changed immutable context was rejected."""


class _RunnerLike(Protocol):
    def run(self, loaded: LoadedExperimentSpec) -> ExperimentRunResult: ...


class TuningResult(StrictFrozenModel):
    study_name: str
    storage_path: Path
    completed_trials: int
    pruned_trials: int
    failed_trials: int
    best_run_id: str | None
    best_value: float | None
    elapsed_seconds: float


def build_study_context_hash(spec: ExperimentSpec) -> str:
    """Hash the immutable (non-searched) context that a study must preserve."""
    payload = {
        "schema_version": spec.schema_version,
        "name": spec.name,
        "dataset": spec.dataset.model_dump(mode="json"),
        "model_family": spec.model.family,
        "model_version": spec.model.version,
        "fit_intercept": spec.model.fit_intercept,
        "embargo_sessions": spec.validation.embargo_sessions,
        "costs": spec.costs.model_dump(mode="json"),
        "portfolio": spec.portfolio.model_dump(mode="json"),
        "objective": spec.objective.model_dump(mode="json"),
        "seed": spec.seed,
        "search": (
            spec.search.model_dump(mode="json") if spec.search is not None else None
        ),
    }
    return build_compatibility_key(payload)


def _trial_seed(base_seed: int, trial_number: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{trial_number}".encode()).hexdigest()
    return int(digest, 16) % (2**32)


class OptunaTuner:
    def __init__(self, *, storage_path: Path, runner: _RunnerLike) -> None:
        self._storage_path = Path(storage_path)
        self._runner = runner

    def tune(
        self,
        loaded: LoadedExperimentSpec,
        *,
        study_name: str,
        n_trials: int,
    ) -> TuningResult:
        import optuna
        from optuna.trial import TrialState

        spec = loaded.spec
        if spec.search is None:
            raise ValueError("tuning requires a [search] space in the spec")
        search = spec.search

        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        storage_uri = f"sqlite:///{self._storage_path.resolve()}"
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_uri,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=spec.seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
            load_if_exists=True,
        )

        context_hash = build_study_context_hash(spec)
        stored = study.user_attrs.get("mltrade.context_hash")
        if stored is None:
            study.set_user_attr("mltrade.context_hash", context_hash)
        elif stored != context_hash:
            raise StudyContextMismatch(
                f"study context hash mismatch: stored={stored}, "
                f"current={context_hash}"
            )

        def objective(trial: Any) -> float:
            candidate = sample_ridge_trial(trial, spec, search)
            candidate = candidate.model_copy(
                update={"seed": _trial_seed(spec.seed, trial.number)}
            )
            result = self._runner.run(loaded_from_spec(candidate, path=loaded.path))
            trial.set_user_attr("mltrade.run_id", result.record.run_id)
            if result.record.metrics is None:
                raise ExperimentFailed(
                    f"run {result.record.run_id} produced no metrics"
                )
            return result.record.metrics.robust_sharpe

        start = time.monotonic()
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=spec.resources.timeout_minutes * 60,
            n_jobs=spec.resources.worker_count,
            catch=(ExperimentBlocked, ExperimentFailed),
        )
        elapsed = time.monotonic() - start

        completed = len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))
        pruned = len(study.get_trials(deepcopy=False, states=(TrialState.PRUNED,)))
        failed = len(study.get_trials(deepcopy=False, states=(TrialState.FAIL,)))

        best_run_id: str | None = None
        best_value: float | None = None
        try:
            best = study.best_trial
        except ValueError:
            best = None
        if best is not None:
            best_value = best.value
            run_id = best.user_attrs.get("mltrade.run_id")
            best_run_id = run_id if isinstance(run_id, str) else None

        return TuningResult(
            study_name=study_name,
            storage_path=self._storage_path,
            completed_trials=completed,
            pruned_trials=pruned,
            failed_trials=failed,
            best_run_id=best_run_id,
            best_value=best_value,
            elapsed_seconds=elapsed,
        )
