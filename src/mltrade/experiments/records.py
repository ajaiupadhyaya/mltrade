"""Canonical experiment run records and content-addressed identity.

The :class:`ExperimentRunRecord` is the durable, immutable artifact produced by
a completed (or terminally blocked/failed/pruned) experiment run.  Run identity
is *content addressed*: the same spec, dataset, and code (including any dirty
worktree diff) always yield the same ``run_id``.  Timestamps are recorded as
evidence but are deliberately excluded from identity so that re-running
identical work is idempotent rather than producing a second record.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Literal

from pydantic import Field

from mltrade.experiments.specs import StrictFrozenModel

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list[JsonScalar] | dict[str, JsonScalar]

TerminalStatus = Literal["complete", "blocked", "pruned", "failed"]
TrackingStatus = Literal["pending", "logged", "degraded"]


class ArtifactRecord(StrictFrozenModel):
    relative_path: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    media_type: str
    size_bytes: int = Field(ge=0)


class FailureRecord(StrictFrozenModel):
    category: str
    message: str


class RunIdentityContext(StrictFrozenModel):
    spec_sha256: str
    dataset_sha256: str
    git_commit: str
    git_diff_sha256: str | None


class RunMetrics(StrictFrozenModel):
    annualized_return: float
    annualized_volatility: float
    sharpe: float
    max_drawdown: float
    turnover: float
    total_costs: float
    hit_rate: float
    equal_weight_return: float
    cash_return: float
    robust_sharpe: float
    window_sharpe_std: float


class RunProvenance(StrictFrozenModel):
    git_commit: str
    git_dirty: bool
    git_diff_sha256: str | None
    python_version: str
    platform: str
    mltrade_version: str
    dependencies: dict[str, str]
    command: tuple[str, ...]


class ExperimentRunRecord(StrictFrozenModel):
    schema_version: Literal[1] = 1
    run_id: str
    experiment_name: str
    status: TerminalStatus
    spec_sha256: str
    dataset_sha256: str
    dataset_snapshot_id: str
    compatibility_key: str
    seed: int
    started_at: datetime
    finished_at: datetime
    provenance: RunProvenance
    parameters: dict[str, JsonValue]
    metrics: RunMetrics | None
    artifacts: tuple[ArtifactRecord, ...]
    failure: FailureRecord | None = None
    study_name: str | None = None
    trial_number: int | None = None
    tracking_status: TrackingStatus = "pending"


def _canonical_payload(payload: dict[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def build_run_id(context: RunIdentityContext) -> str:
    """Return the deterministic, content-addressed run id for *context*."""
    digest = hashlib.sha256(
        _canonical_payload(context.model_dump(mode="json"))
    ).hexdigest()
    return f"run-{digest[:20]}"


def build_compatibility_key(payload: dict[str, object]) -> str:
    """Hash a methodological-compatibility payload into a stable key."""
    return hashlib.sha256(_canonical_payload(payload)).hexdigest()
