"""Local research experiment platform: specs, runs, tracking, tuning."""

from mltrade.experiments.comparison import (
    ComparisonResult,
    RankedRun,
    compare_runs,
)
from mltrade.experiments.examples import EXAMPLE_SPECS, write_example_specs
from mltrade.experiments.loading import (
    ExperimentSpecError,
    LoadedExperimentSpec,
    canonical_json_for_spec,
    load_experiment_spec,
    loaded_from_spec,
)
from mltrade.experiments.provenance import capture_provenance
from mltrade.experiments.records import (
    ArtifactRecord,
    ExperimentRunRecord,
    FailureRecord,
    RunIdentityContext,
    RunMetrics,
    RunProvenance,
    build_compatibility_key,
    build_run_id,
)
from mltrade.experiments.reporting import build_report_json, build_report_markdown
from mltrade.experiments.runner import (
    ExperimentBlocked,
    ExperimentFailed,
    ExperimentRunner,
    ExperimentRunResult,
    ExperimentTrackingError,
    build_run_metrics,
)
from mltrade.experiments.search import sample_ridge_trial
from mltrade.experiments.specs import (
    CostSpec,
    DatasetSpec,
    ExperimentSpec,
    FloatSearchSpec,
    ObjectiveSpec,
    PortfolioSpec,
    ResourceBudget,
    RidgeModelSpec,
    RidgeSearchSpace,
    ValidationSpec,
)
from mltrade.experiments.storage import RunStorageError, RunStore
from mltrade.experiments.tracking import (
    MlflowRunTracker,
    NullRunTracker,
    RunTracker,
)
from mltrade.experiments.tuning import (
    OptunaTuner,
    StudyContextMismatch,
    TuningResult,
    build_study_context_hash,
)

__all__ = [
    "EXAMPLE_SPECS",
    "ArtifactRecord",
    "ComparisonResult",
    "CostSpec",
    "DatasetSpec",
    "ExperimentBlocked",
    "ExperimentFailed",
    "ExperimentRunRecord",
    "ExperimentRunResult",
    "ExperimentRunner",
    "ExperimentSpec",
    "ExperimentSpecError",
    "ExperimentTrackingError",
    "FailureRecord",
    "FloatSearchSpec",
    "LoadedExperimentSpec",
    "MlflowRunTracker",
    "NullRunTracker",
    "ObjectiveSpec",
    "OptunaTuner",
    "PortfolioSpec",
    "RankedRun",
    "ResourceBudget",
    "RidgeModelSpec",
    "RidgeSearchSpace",
    "RunIdentityContext",
    "RunMetrics",
    "RunProvenance",
    "RunStorageError",
    "RunStore",
    "RunTracker",
    "StudyContextMismatch",
    "TuningResult",
    "ValidationSpec",
    "build_compatibility_key",
    "build_report_json",
    "build_report_markdown",
    "build_run_id",
    "build_run_metrics",
    "build_study_context_hash",
    "canonical_json_for_spec",
    "capture_provenance",
    "compare_runs",
    "load_experiment_spec",
    "loaded_from_spec",
    "sample_ridge_trial",
    "write_example_specs",
]
