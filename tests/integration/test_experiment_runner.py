from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from mltrade.config import Environment, Settings
from mltrade.experiments.loading import LoadedExperimentSpec, load_experiment_spec
from mltrade.experiments.runner import ExperimentBlocked, ExperimentRunner
from mltrade.experiments.tracking import NullRunTracker
from mltrade.storage.manifests import DatasetManifest
from mltrade.storage.snapshots import SnapshotStore
from mltrade.workflows.demo import run_demo

_CLOCK = datetime(2026, 6, 13, tzinfo=UTC)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        environment=Environment.TEST,
        data_root=tmp_path,
        database_url=f"sqlite+pysqlite:///{tmp_path / 'ops.db'}",
    )


def _baseline_toml(snapshot_id: str) -> str:
    return (
        "schema_version = 1\n"
        'name = "ridge-baseline"\n'
        'description = "test baseline"\n'
        "seed = 42\n\n"
        "[dataset]\n"
        'name = "daily_bars"\n'
        f'snapshot_id = "{snapshot_id}"\n'
        'universe_version = "mvp-etf-v1"\n'
        'feature_version = "trend-momentum-v1"\n'
    )


def _published(
    tmp_path: Path,
) -> tuple[Settings, DatasetManifest, LoadedExperimentSpec]:
    settings = _settings(tmp_path)
    demo = run_demo(settings, clock=_CLOCK)
    manifest = SnapshotStore(settings.data_root).load_manifest(
        "daily_bars", demo.snapshot_id
    )
    spec_path = tmp_path / "baseline.toml"
    spec_path.write_text(_baseline_toml(manifest.snapshot_id), encoding="utf-8")
    return settings, manifest, load_experiment_spec(spec_path)


def test_runner_produces_deterministic_canonical_record(tmp_path: Path) -> None:
    settings, manifest, loaded = _published(tmp_path)
    runner = ExperimentRunner(settings=settings, tracker=NullRunTracker())

    first = runner.run(loaded)
    second = runner.run(loaded)

    assert first.record.run_id == second.record.run_id
    assert first.record.metrics == second.record.metrics
    assert first.record.dataset_sha256 == manifest.content_sha256
    assert first.report_markdown.read_text() == second.report_markdown.read_text()

    markdown = first.report_markdown.read_text()
    assert "# MLTrade Experiment Report" in markdown
    assert "Dataset snapshot" in markdown
    assert "Robust Sharpe" in markdown
    assert "Not a promotion decision" in markdown
    assert first.report_json.exists()


def test_runner_records_metrics_and_terminal_status(tmp_path: Path) -> None:
    settings, _, loaded = _published(tmp_path)
    result = ExperimentRunner(settings=settings).run(loaded)

    assert result.record.metrics is not None
    assert result.record.status in {"complete", "blocked"}
    if result.record.status == "blocked":
        assert result.record.failure is not None
        assert result.record.failure.category == "objective"


def test_runner_blocks_manifest_context_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, manifest, loaded = _published(tmp_path)
    bad_manifest = DatasetManifest.model_validate(
        {
            **manifest.model_dump(),
            "metadata": {**manifest.metadata, "universe_version": "unsupported-v2"},
        }
    )
    monkeypatch.setattr(
        SnapshotStore,
        "load_manifest",
        lambda self, dataset, snapshot_id: bad_manifest,
    )
    runner = ExperimentRunner(settings=settings, tracker=NullRunTracker())

    with pytest.raises(ExperimentBlocked, match="universe_version"):
        runner.run(loaded)
