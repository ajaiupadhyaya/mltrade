import hashlib
import json
from pathlib import Path

import pytest

from mltrade.experiments.loading import (
    ExperimentSpecError,
    load_experiment_spec,
)

BASELINE_TOML = """
schema_version = 1
name = "ridge_baseline"
description = "Deterministic ridge baseline"
seed = 42

[dataset]
name = "daily_bars"
snapshot_id = "daily-bars:2026-06-12"
universe_version = "mvp-etf-v1"
feature_version = "trend-momentum-v1"

[model]
family = "ridge"
version = "ridge-trend-v1"
alpha = 1.0
fit_intercept = true

[validation]
minimum_training_sessions = 504
embargo_sessions = 21
retrain_every_sessions = 21

[costs]
headline_bps = "5"
sensitivity_bps = ["2", "5", "10"]

[portfolio]
reference_equity = "1000000"
maximum_position_weight = "0.25"
minimum_cash_weight = "0.05"
target_annual_volatility = "0.15"

[objective]
name = "robust_sharpe"
maximum_drawdown = -0.35
maximum_turnover = 1.0

[resources]
max_trials = 40
timeout_minutes = 60
worker_count = 1
""".lstrip()


def write_baseline(tmp_path: Path) -> Path:
    path = tmp_path / "baseline.toml"
    path.write_text(BASELINE_TOML, encoding="utf-8")
    return path


def test_loading_builds_deterministic_canonical_json_and_hash(
    tmp_path: Path,
) -> None:
    path = write_baseline(tmp_path)

    first = load_experiment_spec(path)
    second = load_experiment_spec(path)
    expected_json = (
        json.dumps(
            first.spec.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )

    assert first.path == path.resolve()
    assert first == second
    assert first.canonical_json == expected_json
    assert first.spec_sha256 == hashlib.sha256(
        expected_json.encode("utf-8")
    ).hexdigest()


def test_loading_rejects_malformed_toml_with_source_path(
    tmp_path: Path,
) -> None:
    path = tmp_path / "malformed.toml"
    path.write_text('name = "unterminated\n', encoding="utf-8")

    with pytest.raises(ExperimentSpecError) as exc_info:
        load_experiment_spec(path)

    assert str(path.resolve()) in str(exc_info.value)
    assert "TOML" in str(exc_info.value)


def test_loading_rejects_missing_file_with_resolved_source_path(
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing.toml"

    with pytest.raises(ExperimentSpecError) as exc_info:
        load_experiment_spec(path)

    assert str(path.resolve()) in str(exc_info.value)
    assert "not found" in str(exc_info.value).lower()


def test_loading_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "unknown.toml"
    path.write_text(BASELINE_TOML + "\nunexpected = true\n", encoding="utf-8")

    with pytest.raises(ExperimentSpecError) as exc_info:
        load_experiment_spec(path)

    message = str(exc_info.value)
    assert str(path.resolve()) in message
    assert "unexpected" in message


def test_loading_wraps_decoding_errors_with_source_path(tmp_path: Path) -> None:
    path = tmp_path / "invalid-utf8.toml"
    path.write_bytes(b"\xff\xfe")

    with pytest.raises(ExperimentSpecError) as exc_info:
        load_experiment_spec(path)

    assert str(path.resolve()) in str(exc_info.value)
