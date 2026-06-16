from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

import pytest

pytest.importorskip("optuna")
import optuna

from mltrade.experiments.loading import (
    LoadedExperimentSpec,
    loaded_from_spec,
)
from mltrade.experiments.records import (
    ExperimentRunRecord,
    RunMetrics,
    RunProvenance,
)
from mltrade.experiments.runner import ExperimentRunResult
from mltrade.experiments.specs import (
    DatasetSpec,
    ExperimentSpec,
    RidgeSearchSpace,
)
from mltrade.experiments.tuning import (
    OptunaTuner,
    StudyContextMismatch,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

_T = datetime(2026, 6, 14, tzinfo=UTC)
_PROV = RunProvenance(
    git_commit="c" * 40,
    git_dirty=False,
    git_diff_sha256=None,
    python_version="3.13.1",
    platform="test",
    mltrade_version="0.1.0",
    dependencies={},
    command=("mltrade",),
)


class FakeRunner:
    def __init__(self, tmp_path: Path) -> None:
        self._tmp = tmp_path
        self.calls = 0

    def run(self, loaded: LoadedExperimentSpec) -> ExperimentRunResult:
        self.calls += 1
        alpha = loaded.spec.model.alpha
        # Smooth objective with an optimum near alpha = 10.
        robust = round(-((math.log10(alpha) - 1.0) ** 2), 10)
        record = ExperimentRunRecord(
            run_id="run-" + loaded.spec_sha256[:20],
            experiment_name=loaded.spec.name,
            status="complete",
            spec_sha256=loaded.spec_sha256,
            dataset_sha256="b" * 64,
            dataset_snapshot_id=loaded.spec.dataset.snapshot_id,
            compatibility_key="d" * 64,
            seed=loaded.spec.seed,
            started_at=_T,
            finished_at=_T,
            provenance=_PROV,
            parameters={"model.alpha": alpha},
            metrics=RunMetrics(
                annualized_return=0.1,
                annualized_volatility=0.1,
                sharpe=1.0,
                max_drawdown=-0.2,
                turnover=0.3,
                total_costs=10.0,
                hit_rate=0.5,
                equal_weight_return=0.05,
                cash_return=0.0,
                robust_sharpe=robust,
                window_sharpe_std=0.0,
            ),
            artifacts=(),
        )
        return ExperimentRunResult(
            record=record,
            run_directory=self._tmp,
            report_markdown=self._tmp / "report.md",
            report_json=self._tmp / "report.json",
        )


def _loaded(
    tmp_path: Path, *, snapshot: str, with_search: bool = True
) -> LoadedExperimentSpec:
    spec = ExperimentSpec(
        name="ridge-baseline",
        dataset=DatasetSpec(snapshot_id=snapshot),
        search=RidgeSearchSpace() if with_search else None,
    )
    return loaded_from_spec(spec, path=tmp_path / "spec.toml")


def test_study_resumes_completed_trials(tmp_path: Path) -> None:
    loaded = _loaded(tmp_path, snapshot="fixture-2026-06-12")
    tuner = OptunaTuner(
        storage_path=tmp_path / "studies.db", runner=FakeRunner(tmp_path)
    )

    first = tuner.tune(loaded, study_name="ridge-test", n_trials=2)
    second = tuner.tune(loaded, study_name="ridge-test", n_trials=1)

    assert first.completed_trials == 2
    assert second.completed_trials == 3
    assert second.best_run_id is not None
    assert second.best_value is not None


def test_resume_rejects_immutable_context_drift(tmp_path: Path) -> None:
    base = _loaded(tmp_path, snapshot="fixture-2026-06-12")
    tuner = OptunaTuner(
        storage_path=tmp_path / "studies.db", runner=FakeRunner(tmp_path)
    )
    tuner.tune(base, study_name="ridge-test", n_trials=1)

    changed = _loaded(tmp_path, snapshot="fixture-2026-06-11")
    with pytest.raises(StudyContextMismatch, match="context hash"):
        tuner.tune(changed, study_name="ridge-test", n_trials=1)


def test_tune_requires_search_space(tmp_path: Path) -> None:
    loaded = _loaded(tmp_path, snapshot="fixture-2026-06-12", with_search=False)
    tuner = OptunaTuner(
        storage_path=tmp_path / "studies.db", runner=FakeRunner(tmp_path)
    )
    with pytest.raises(ValueError, match="search"):
        tuner.tune(loaded, study_name="ridge-test", n_trials=1)
