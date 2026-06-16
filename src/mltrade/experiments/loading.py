import hashlib
import json
import math
import tomllib
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any, NamedTuple

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


def _canonicalize(value: Any) -> Any:
    if isinstance(value, Decimal):
        sign, digits, exponent = value.as_tuple()
        if not isinstance(exponent, int):
            raise ValueError("canonical JSON does not support non-finite Decimal")
        if not any(digits):
            return "0"

        normalized_digits = list(digits)
        while normalized_digits[-1] == 0:
            normalized_digits.pop()
            exponent += 1

        digit_string = "".join(str(digit) for digit in normalized_digits)
        adjusted_exponent = exponent + len(normalized_digits) - 1
        if -6 <= adjusted_exponent <= 20:
            decimal_index = len(digit_string) + exponent
            if decimal_index <= 0:
                formatted = (
                    "0."
                    + ("0" * -decimal_index)
                    + digit_string
                )
            elif decimal_index >= len(digit_string):
                formatted = (
                    digit_string
                    + ("0" * (decimal_index - len(digit_string)))
                )
            else:
                formatted = (
                    digit_string[:decimal_index]
                    + "."
                    + digit_string[decimal_index:]
                )
        else:
            coefficient = digit_string[0]
            if len(digit_string) > 1:
                coefficient += "." + digit_string[1:]
            formatted = f"{coefficient}e{adjusted_exponent:+d}"

        return f"-{formatted}" if sign else formatted
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON does not support non-finite float")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        return {
            key: _canonicalize(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_canonicalize(item) for item in value]
    return value


def canonical_json_for_spec(spec: ExperimentSpec) -> tuple[str, str]:
    """Return the (canonical_json, sha256) pair for an in-memory spec."""
    canonical_json = (
        json.dumps(
            _canonicalize(spec.model_dump(mode="python")),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    spec_sha256 = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    return canonical_json, spec_sha256


def loaded_from_spec(spec: ExperimentSpec, *, path: Path) -> LoadedExperimentSpec:
    """Build a :class:`LoadedExperimentSpec` from an in-memory spec.

    Used when a spec is derived in process (e.g. an Optuna trial candidate)
    rather than read from disk.
    """
    canonical_json, spec_sha256 = canonical_json_for_spec(spec)
    return LoadedExperimentSpec(
        path=path,
        spec=spec,
        canonical_json=canonical_json,
        spec_sha256=spec_sha256,
    )


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

    canonical_json, spec_sha256 = canonical_json_for_spec(spec)
    return LoadedExperimentSpec(
        path=source_path,
        spec=spec,
        canonical_json=canonical_json,
        spec_sha256=spec_sha256,
    )
