from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mltrade.cli import app
from mltrade.config import Settings
from mltrade.experiments.records import (
    ExperimentRunRecord,
    RunMetrics,
    RunProvenance,
)
from mltrade.experiments.storage import RunStore
from mltrade.workflows.demo import run_demo

runner = CliRunner()
_CLOCK = datetime(2026, 6, 13, tzinfo=UTC)


def _env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLTRADE_ENVIRONMENT", "test")
    monkeypatch.setenv("MLTRADE_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("MLTRADE_EXPERIMENT_ROOT", str(tmp_path / "experiments"))
    monkeypatch.setenv(
        "MLTRADE_DATABASE_URL", f"sqlite+pysqlite:///{tmp_path / 'ops.db'}"
    )


def _baseline_toml(snapshot_id: str) -> str:
    return (
        "schema_version = 1\n"
        'name = "ridge-baseline"\n'
        'description = "cli test"\n'
        "seed = 42\n\n"
        "[dataset]\n"
        'name = "daily_bars"\n'
        f'snapshot_id = "{snapshot_id}"\n'
        'universe_version = "mvp-etf-v1"\n'
        'feature_version = "trend-momentum-v1"\n'
    )


def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _env(tmp_path, monkeypatch)
    demo = run_demo(Settings(), clock=_CLOCK)
    spec_path = tmp_path / "baseline.toml"
    spec_path.write_text(_baseline_toml(demo.snapshot_id), encoding="utf-8")
    return spec_path


def _run_id_from(stdout: str) -> str:
    for line in stdout.splitlines():
        if line.startswith("run id:"):
            return line.split("run id:")[1].strip()
    raise AssertionError(f"no run id in output:\n{stdout}")


def test_experiment_init_writes_examples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _env(tmp_path, monkeypatch)
    result = runner.invoke(app, ["experiment", "init", str(tmp_path / "specs")])

    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "specs" / "ridge-baseline.toml").exists()
    assert (tmp_path / "specs" / "ridge-balanced-search.toml").exists()


def test_experiment_doctor_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _env(tmp_path, monkeypatch)
    result = runner.invoke(app, ["experiment", "doctor"])

    assert result.exit_code == 0, result.stdout
    assert "experiment doctor: ok" in result.stdout


def test_experiment_validate_prints_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec_path = _setup(tmp_path, monkeypatch)
    result = runner.invoke(app, ["experiment", "validate", str(spec_path)])

    assert result.exit_code == 0, result.stdout
    assert "spec: valid" in result.stdout
    assert "snapshot:" in result.stdout
    assert "spec sha256:" in result.stdout


def test_experiment_run_inspect_report_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec_path = _setup(tmp_path, monkeypatch)

    run_result = runner.invoke(app, ["experiment", "run", str(spec_path)])
    assert run_result.exit_code == 0, run_result.stdout
    assert "run id:" in run_result.stdout
    assert "report:" in run_result.stdout
    run_id = _run_id_from(run_result.stdout)

    inspect = runner.invoke(app, ["experiment", "inspect", run_id, "--json"])
    assert inspect.exit_code == 0, inspect.stdout
    assert run_id in inspect.stdout

    listing = runner.invoke(app, ["experiment", "list"])
    assert listing.exit_code == 0, listing.stdout
    assert run_id in listing.stdout

    report = runner.invoke(app, ["experiment", "report", run_id])
    assert report.exit_code == 0, report.stdout
    assert "report:" in report.stdout


def test_experiment_compare_incompatible_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _env(tmp_path, monkeypatch)
    store = RunStore(tmp_path / "experiments")

    def _record(run_id: str, key: str, dataset_sha: str) -> ExperimentRunRecord:
        return ExperimentRunRecord(
            run_id=run_id,
            experiment_name="ridge-baseline",
            status="complete",
            spec_sha256="a" * 64,
            dataset_sha256=dataset_sha,
            dataset_snapshot_id="fixture-2026-06-12",
            compatibility_key=key,
            seed=42,
            started_at=_CLOCK,
            finished_at=_CLOCK,
            provenance=RunProvenance(
                git_commit="c" * 40,
                git_dirty=False,
                git_diff_sha256=None,
                python_version="3.13.1",
                platform="test",
                mltrade_version="0.1.0",
                dependencies={},
                command=("mltrade",),
            ),
            parameters={"model.alpha": 1.0},
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
                robust_sharpe=1.0,
                window_sharpe_std=0.1,
            ),
            artifacts=(),
        )

    run_a = "run-" + "a" * 20
    run_b = "run-" + "b" * 20
    store.save(_record(run_a, "d" * 64, "b" * 64))
    store.save(_record(run_b, "f" * 64, "e" * 64))

    result = runner.invoke(app, ["experiment", "compare", run_a, run_b])

    assert result.exit_code != 0
    assert "incompatible" in result.stdout.lower()
