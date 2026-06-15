import hashlib
import json
import tomllib
from pathlib import Path
from typing import NamedTuple

from pydantic import ValidationError

from mltrade.experiments.specs import ExperimentSpec


class ExperimentSpecError(ValueError):
    pass


class LoadedExperimentSpec(NamedTuple):
    path: Path
    spec: ExperimentSpec
    canonical_json: str
    spec_sha256: str


def _validation_summary(exc: ValidationError) -> str:
    details = []
    for error in exc.errors(
        include_url=False,
        include_context=False,
        include_input=False,
    ):
        location = ".".join(str(part) for part in error["loc"])
        details.append(f"{location}: {error['msg']}")
    return "; ".join(details)


def _safe_cause_summary(exc: OSError | RuntimeError) -> str:
    if isinstance(exc, OSError):
        return exc.strerror or type(exc).__name__
    return "home directory could not be resolved"


def load_experiment_spec(path: Path) -> LoadedExperimentSpec:
    original_path = str(path)
    try:
        source_path = path.expanduser().resolve()
    except (OSError, RuntimeError) as exc:
        raise ExperimentSpecError(
            f"Failed to normalize experiment spec path {original_path}: "
            f"{_safe_cause_summary(exc)}"
        ) from exc

    try:
        with source_path.open("rb") as source:
            raw_spec = tomllib.load(source)
        spec = ExperimentSpec.model_validate(raw_spec)
    except FileNotFoundError as exc:
        raise ExperimentSpecError(
            f"Failed to load experiment spec {source_path}: file not found"
        ) from exc
    except tomllib.TOMLDecodeError as exc:
        raise ExperimentSpecError(
            f"Failed to load experiment spec {source_path}: malformed TOML: {exc}"
        ) from exc
    except UnicodeDecodeError as exc:
        raise ExperimentSpecError(
            f"Failed to load experiment spec {source_path}: "
            "TOML is not valid UTF-8"
        ) from exc
    except ValidationError as exc:
        raise ExperimentSpecError(
            f"Failed to load experiment spec {source_path}: "
            f"validation failed: {_validation_summary(exc)}"
        ) from exc
    except OSError as exc:
        raise ExperimentSpecError(
            f"Failed to load experiment spec {source_path}: "
            f"{_safe_cause_summary(exc)}"
        ) from exc

    canonical_json = (
        json.dumps(
            spec.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    spec_sha256 = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    return LoadedExperimentSpec(
        path=source_path,
        spec=spec,
        canonical_json=canonical_json,
        spec_sha256=spec_sha256,
    )
