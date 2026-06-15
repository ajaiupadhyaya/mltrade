import hashlib
import json
from decimal import localcontext
from pathlib import Path

import pytest

import mltrade.experiments.loading as experiment_loading
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

EXPECTED_BASELINE_CANONICAL_JSON = (
    '{"costs":{"headline_bps":"5","sensitivity_bps":["2","5","10"]},'
    '"dataset":{"feature_version":"trend-momentum-v1","name":"daily_bars",'
    '"snapshot_id":"daily-bars:2026-06-12",'
    '"universe_version":"mvp-etf-v1"},'
    '"description":"Deterministic ridge baseline",'
    '"model":{"alpha":1.0,"family":"ridge","fit_intercept":true,'
    '"version":"ridge-trend-v1"},'
    '"name":"ridge_baseline",'
    '"objective":{"maximum_drawdown":-0.35,"maximum_turnover":1.0,'
    '"name":"robust_sharpe"},'
    '"portfolio":{"maximum_position_weight":"0.25",'
    '"minimum_cash_weight":"0.05","reference_equity":"1000000",'
    '"target_annual_volatility":"0.15"},'
    '"resources":{"max_trials":40,"timeout_minutes":60,"worker_count":1},'
    '"schema_version":1,"seed":42,'
    '"validation":{"embargo_sessions":21,"minimum_training_sessions":504,'
    '"retrain_every_sessions":21}}\n'
)


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
    assert first.canonical_json == EXPECTED_BASELINE_CANONICAL_JSON
    assert first.spec_sha256 == hashlib.sha256(
        expected_json.encode("utf-8")
    ).hexdigest()


def test_semantically_equal_decimals_have_identical_canonical_artifacts(
    tmp_path: Path,
) -> None:
    variants = (
        BASELINE_TOML,
        BASELINE_TOML.replace(
            'headline_bps = "5"',
            'headline_bps = "5.0"',
        ).replace(
            'sensitivity_bps = ["2", "5", "10"]',
            'sensitivity_bps = ["2.0", "5.00", "10.000"]',
        ),
        BASELINE_TOML.replace(
            'headline_bps = "5"',
            'headline_bps = "5.00"',
        ),
    )
    loaded = []
    for index, content in enumerate(variants):
        path = tmp_path / f"variant-{index}.toml"
        path.write_text(content, encoding="utf-8")
        loaded.append(load_experiment_spec(path))

    assert {item.canonical_json for item in loaded} == {
        EXPECTED_BASELINE_CANONICAL_JSON
    }
    assert len({item.spec_sha256 for item in loaded}) == 1


def test_decimal_canonicalization_is_independent_of_context_precision(
    tmp_path: Path,
) -> None:
    precise_equity = "12345678901234567890.1234567890123456789000"
    path = tmp_path / "high-precision.toml"
    path.write_text(
        BASELINE_TOML.replace(
            'reference_equity = "1000000"',
            f'reference_equity = "{precise_equity}"',
        ),
        encoding="utf-8",
    )

    loaded = []
    for precision in (10, 28, 50):
        with localcontext() as context:
            context.prec = precision
            loaded.append(load_experiment_spec(path))

    canonical_json = loaded[0].canonical_json
    assert (
        '"reference_equity":'
        '"12345678901234567890.1234567890123456789"'
        in canonical_json
    )
    assert len({item.canonical_json for item in loaded}) == 1
    assert len({item.spec_sha256 for item in loaded}) == 1


@pytest.mark.parametrize("non_finite", ("inf", "nan"))
def test_loading_rejects_non_finite_objective_floats(
    tmp_path: Path,
    non_finite: str,
) -> None:
    path = tmp_path / f"{non_finite}.toml"
    path.write_text(
        BASELINE_TOML.replace(
            "maximum_turnover = 1.0",
            f"maximum_turnover = {non_finite}",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ExperimentSpecError, match="maximum_turnover"):
        load_experiment_spec(path)


def test_signed_zero_floats_have_identical_canonical_artifacts(
    tmp_path: Path,
) -> None:
    loaded = []
    for index, zero in enumerate(("0.0", "-0.0")):
        path = tmp_path / f"zero-{index}.toml"
        path.write_text(
            BASELINE_TOML.replace(
                "maximum_turnover = 1.0",
                f"maximum_turnover = {zero}",
            ),
            encoding="utf-8",
        )
        loaded.append(load_experiment_spec(path))

    assert '"maximum_turnover":0.0' in loaded[0].canonical_json
    assert loaded[0].canonical_json == loaded[1].canonical_json
    assert loaded[0].spec_sha256 == loaded[1].spec_sha256


@pytest.mark.parametrize(
    ("old", "new", "field"),
    (
        ("worker_count = 1", "worker_count = true", "worker_count"),
        ("seed = 42", "seed = true", "seed"),
        ("schema_version = 1", "schema_version = true", "schema_version"),
        ("fit_intercept = true", "fit_intercept = 1", "fit_intercept"),
        ("alpha = 1.0", 'alpha = "1.5"', "alpha"),
    ),
)
def test_loading_rejects_coercive_scalar_types(
    tmp_path: Path,
    old: str,
    new: str,
    field: str,
) -> None:
    path = tmp_path / f"invalid-{field}.toml"
    path.write_text(BASELINE_TOML.replace(old, new), encoding="utf-8")

    with pytest.raises(ExperimentSpecError, match=field):
        load_experiment_spec(path)


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


def test_loading_expands_tilde_from_controlled_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    experiment_dir = home / "experiments"
    experiment_dir.mkdir(parents=True)
    path = experiment_dir / "baseline.toml"
    path.write_text(BASELINE_TOML, encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))

    loaded = load_experiment_spec(Path("~/experiments/baseline.toml"))

    assert loaded.path == path.resolve()


def test_loading_wraps_path_normalization_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Path("~/secret.toml")

    def fail_expanduser(self: Path) -> Path:
        raise OSError(5, "normalization failed", "private-value")

    monkeypatch.setattr(Path, "expanduser", fail_expanduser)

    with pytest.raises(ExperimentSpecError) as exc_info:
        load_experiment_spec(source)

    message = str(exc_info.value)
    assert str(source) in message
    assert "normalization failed" in message
    assert "private-value" not in message


def test_loading_wraps_generic_os_errors_without_file_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "baseline.toml"

    def fail_open(self: Path, mode: str) -> object:
        raise OSError(13, "permission denied", "private-file-contents")

    monkeypatch.setattr(experiment_loading.Path, "open", fail_open)

    with pytest.raises(ExperimentSpecError) as exc_info:
        load_experiment_spec(path)

    message = str(exc_info.value)
    assert str(path.resolve()) in message
    assert "permission denied" in message
    assert "private-file-contents" not in message
