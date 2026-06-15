"""Deterministic single-run experiment orchestration.

``ExperimentRunner.run`` turns a loaded experiment spec into a canonical,
content-addressed run record plus durable reports.  Re-running identical
code/spec/data returns the first completed record idempotently — timestamps are
evidence, not identity.
"""

from __future__ import annotations

import hashlib
import math
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from mltrade.backtest.engine import BacktestConfig
from mltrade.backtest.reporting import BacktestResult
from mltrade.config import Settings
from mltrade.experiments.loading import LoadedExperimentSpec
from mltrade.experiments.provenance import capture_provenance
from mltrade.experiments.records import (
    ArtifactRecord,
    ExperimentRunRecord,
    FailureRecord,
    JsonValue,
    RunIdentityContext,
    RunMetrics,
    TerminalStatus,
    build_compatibility_key,
    build_run_id,
)
from mltrade.experiments.reporting import build_report_json, build_report_markdown
from mltrade.experiments.specs import ExperimentSpec
from mltrade.experiments.storage import RunStore
from mltrade.experiments.tracking import NullRunTracker, RunTracker
from mltrade.models.forecasts import RidgeForecastConfig
from mltrade.storage.snapshots import SnapshotStore
from mltrade.workflows.research import run_research

_FORECAST_HORIZON = 21


class ExperimentBlocked(RuntimeError):
    """The run could not be evaluated (bad context, quality, or optimizer)."""


class ExperimentFailed(RuntimeError):
    """The run produced an invalid (e.g. non-finite) result."""


class ExperimentTrackingError(RuntimeError):
    """Canonical evidence was saved but external tracking degraded."""


@dataclass(frozen=True, slots=True)
class ExperimentRunResult:
    record: ExperimentRunRecord
    run_directory: Path
    report_markdown: Path
    report_json: Path


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _utc_now() -> datetime:
    return datetime.now(UTC)


def build_run_metrics(backtest: BacktestResult) -> RunMetrics:
    """Derive canonical run metrics, including the robust-Sharpe objective.

    ``robust_sharpe = min(Sharpe@5bps, Sharpe@10bps) - stdev(window Sharpes)``.
    """
    sharpe_5 = backtest.cost_sensitivity.get(Decimal("5"))
    sharpe_10 = backtest.cost_sensitivity.get(Decimal("10"))
    s5 = sharpe_5.sharpe if sharpe_5 is not None else backtest.sharpe
    s10 = sharpe_10.sharpe if sharpe_10 is not None else backtest.sharpe
    window_sharpes = [w.sharpe for w in backtest.evaluation_windows]
    window_std = statistics.pstdev(window_sharpes) if window_sharpes else 0.0
    return RunMetrics(
        annualized_return=backtest.annualized_return,
        annualized_volatility=backtest.annualized_volatility,
        sharpe=backtest.sharpe,
        max_drawdown=backtest.max_drawdown,
        turnover=backtest.turnover,
        total_costs=backtest.total_costs,
        hit_rate=backtest.hit_rate,
        equal_weight_return=backtest.equal_weight_return,
        cash_return=backtest.cash_return,
        robust_sharpe=round(min(s5, s10) - window_std, 10),
        window_sharpe_std=round(window_std, 10),
    )


class ExperimentRunner:
    def __init__(
        self,
        *,
        settings: Settings,
        tracker: RunTracker | None = None,
        repo_root: Path | None = None,
    ) -> None:
        self._settings = settings
        self._tracker: RunTracker = tracker or NullRunTracker()
        experiment_root = settings.experiment_root
        assert experiment_root is not None
        self._store = RunStore(experiment_root)
        self._repo_root = repo_root or _default_repo_root()

    # -- public ---------------------------------------------------------------

    def run(self, loaded: LoadedExperimentSpec) -> ExperimentRunResult:
        spec = loaded.spec
        snapshots = SnapshotStore(self._settings.data_root)
        try:
            manifest = snapshots.load_manifest(
                spec.dataset.name, spec.dataset.snapshot_id
            )
        except (ValueError, OSError) as exc:
            raise ExperimentBlocked(
                f"snapshot {spec.dataset.snapshot_id!r} is unavailable"
            ) from exc

        self._verify_context(spec, manifest.dataset, manifest.metadata)

        provenance = capture_provenance(
            self._repo_root, command=("mltrade", "experiment", "run")
        )
        identity = RunIdentityContext(
            spec_sha256=loaded.spec_sha256,
            dataset_sha256=manifest.content_sha256,
            git_commit=provenance.git_commit,
            git_diff_sha256=provenance.git_diff_sha256,
        )
        run_id = build_run_id(identity)

        if self._store.exists(run_id):
            return self._result_for(self._store.load(run_id))

        settings_for_run = self._settings.model_copy(
            update={
                "reference_equity": spec.portfolio.reference_equity,
                "maximum_position_weight": spec.portfolio.maximum_position_weight,
                "minimum_cash_weight": spec.portfolio.minimum_cash_weight,
                "target_annual_volatility": spec.portfolio.target_annual_volatility,
            }
        )

        started = _utc_now()
        research = run_research(
            settings_for_run, manifest, backtest_config=self._backtest_config(spec)
        )
        finished = _utc_now()

        if research.quality.blocked:
            raise ExperimentBlocked("data quality gate blocked the snapshot")
        if research.target.blocked:
            raise ExperimentBlocked("portfolio optimizer blocked the target")

        metrics = self._build_metrics(research.backtest)
        status, failure = self._evaluate_objective(spec, research.backtest, metrics)

        compatibility_key = build_compatibility_key(
            self._compatibility_payload(
                spec, manifest.content_sha256, manifest.snapshot_id
            )
        )
        base_record = ExperimentRunRecord(
            run_id=run_id,
            experiment_name=spec.name,
            status=status,
            spec_sha256=loaded.spec_sha256,
            dataset_sha256=manifest.content_sha256,
            dataset_snapshot_id=manifest.snapshot_id,
            compatibility_key=compatibility_key,
            seed=spec.seed,
            started_at=started,
            finished_at=finished,
            provenance=provenance,
            parameters=self._parameters(spec),
            metrics=metrics,
            artifacts=(),
            failure=failure,
        )

        markdown = build_report_markdown(base_record, backtest=research.backtest)
        report_json = build_report_json(base_record)
        artifacts = (
            ArtifactRecord(
                relative_path="report.md",
                sha256=_sha256(markdown),
                media_type="text/markdown",
                size_bytes=len(markdown.encode("utf-8")),
            ),
            ArtifactRecord(
                relative_path="report.json",
                sha256=_sha256(report_json),
                media_type="application/json",
                size_bytes=len(report_json.encode("utf-8")),
            ),
            ArtifactRecord(
                relative_path="spec.json",
                sha256=_sha256(loaded.canonical_json),
                media_type="application/json",
                size_bytes=len(loaded.canonical_json.encode("utf-8")),
            ),
        )
        record = base_record.model_copy(update={"artifacts": artifacts})

        self._store.save(
            record,
            artifacts={
                "report.md": markdown.encode("utf-8"),
                "report.json": report_json.encode("utf-8"),
                "spec.json": loaded.canonical_json.encode("utf-8"),
            },
        )

        self._track(record)
        return self._result_for(self._store.load(run_id))

    # -- helpers --------------------------------------------------------------

    def _verify_context(
        self,
        spec: ExperimentSpec,
        manifest_dataset: str,
        metadata: dict[str, str],
    ) -> None:
        if manifest_dataset != spec.dataset.name:
            raise ExperimentBlocked(
                f"dataset mismatch: spec expects {spec.dataset.name!r}, "
                f"manifest is {manifest_dataset!r}"
            )
        manifest_universe = metadata.get("universe_version")
        if manifest_universe != spec.dataset.universe_version:
            raise ExperimentBlocked(
                f"universe_version mismatch: spec expects "
                f"{spec.dataset.universe_version!r}, manifest is "
                f"{manifest_universe!r}"
            )

    def _backtest_config(self, spec: ExperimentSpec) -> BacktestConfig:
        return BacktestConfig(
            forecast=RidgeForecastConfig(
                alpha=spec.model.alpha,
                fit_intercept=spec.model.fit_intercept,
                minimum_training_sessions=spec.validation.minimum_training_sessions,
                embargo_sessions=spec.validation.embargo_sessions,
            ),
            retrain_every_sessions=spec.validation.retrain_every_sessions,
            cost_bps=spec.costs.headline_bps,
            cost_sensitivity_bps=spec.costs.sensitivity_bps,
        )

    def _build_metrics(self, backtest: BacktestResult) -> RunMetrics:
        metrics = build_run_metrics(backtest)
        for name, value in metrics.model_dump().items():
            if not math.isfinite(value):
                raise ExperimentFailed(f"non-finite metric: {name}={value}")
        return metrics

    def _evaluate_objective(
        self,
        spec: ExperimentSpec,
        backtest: BacktestResult,
        metrics: RunMetrics,
    ) -> tuple[TerminalStatus, FailureRecord | None]:
        violations: list[str] = []
        if backtest.max_drawdown < spec.objective.maximum_drawdown:
            violations.append(
                f"max_drawdown {backtest.max_drawdown:.4f} is worse than the "
                f"objective floor {spec.objective.maximum_drawdown:.4f}"
            )
        if backtest.turnover > spec.objective.maximum_turnover:
            violations.append(
                f"turnover {backtest.turnover:.4f} exceeds the objective cap "
                f"{spec.objective.maximum_turnover:.4f}"
            )
        if violations:
            return "blocked", FailureRecord(
                category="objective", message="; ".join(violations)
            )
        return "complete", None

    def _compatibility_payload(
        self, spec: ExperimentSpec, dataset_sha256: str, snapshot_id: str
    ) -> dict[str, object]:
        return {
            "dataset_sha256": dataset_sha256,
            "snapshot_id": snapshot_id,
            "universe_version": spec.dataset.universe_version,
            "feature_version": spec.dataset.feature_version,
            "forecast_horizon": _FORECAST_HORIZON,
            "minimum_training_sessions": spec.validation.minimum_training_sessions,
            "embargo_sessions": spec.validation.embargo_sessions,
            "retrain_every_sessions": spec.validation.retrain_every_sessions,
            "portfolio": spec.portfolio.model_dump(mode="json"),
            "headline_cost_bps": str(spec.costs.headline_bps),
            "objective": spec.objective.model_dump(mode="json"),
        }

    def _parameters(self, spec: ExperimentSpec) -> dict[str, JsonValue]:
        return {
            "model.alpha": spec.model.alpha,
            "model.fit_intercept": spec.model.fit_intercept,
            "validation.minimum_training_sessions": (
                spec.validation.minimum_training_sessions
            ),
            "validation.embargo_sessions": spec.validation.embargo_sessions,
            "validation.retrain_every_sessions": (
                spec.validation.retrain_every_sessions
            ),
            "costs.headline_bps": str(spec.costs.headline_bps),
            "objective.maximum_drawdown": spec.objective.maximum_drawdown,
            "objective.maximum_turnover": spec.objective.maximum_turnover,
            "seed": spec.seed,
        }

    def _track(self, record: ExperimentRunRecord) -> None:
        run_dir = self._store.run_directory(record.run_id)
        try:
            external_id = self._tracker.log(record, run_dir)
        except Exception as exc:  # degrade, never lose canonical evidence
            degraded = record.model_copy(
                update={
                    "tracking_status": "degraded",
                    "failure": FailureRecord(category="tracking", message=str(exc)),
                }
            )
            self._store.replace_tracking_state(degraded)
            raise ExperimentTrackingError(degraded.run_id) from exc
        if external_id:
            self._store.replace_tracking_state(
                record.model_copy(update={"tracking_status": "logged"})
            )

    def _result_for(self, record: ExperimentRunRecord) -> ExperimentRunResult:
        run_dir = self._store.run_directory(record.run_id)
        return ExperimentRunResult(
            record=record,
            run_directory=run_dir,
            report_markdown=run_dir / "report.md",
            report_json=run_dir / "report.json",
        )
