"""Compatibility-aware comparison and deterministic ranking of runs.

Only runs that share a methodological *compatibility key* may be ranked against
one another.  Blocked, failed, pruned, degraded-tracking, and dirty-worktree
runs stay visible but are excluded from the ranking by default — incompatible
sets never produce a winner.
"""

from __future__ import annotations

from mltrade.experiments.records import ExperimentRunRecord, RunMetrics
from mltrade.experiments.specs import StrictFrozenModel

_DIFF_FIELDS = (
    "compatibility_key",
    "dataset_sha256",
    "dataset_snapshot_id",
    "spec_sha256",
)


class RankedRun(StrictFrozenModel):
    rank: int
    run_id: str
    robust_sharpe: float
    sharpe: float
    max_drawdown: float
    turnover: float


class ComparisonResult(StrictFrozenModel):
    run_ids: tuple[str, ...]
    compatible: bool
    compatibility_key: str | None
    differences: dict[str, tuple[str, ...]]
    ranking: tuple[RankedRun, ...]
    excluded_run_ids: tuple[str, ...]


def _is_eligible(record: ExperimentRunRecord, *, include_dirty: bool) -> bool:
    return (
        record.status == "complete"
        and record.metrics is not None
        and record.tracking_status != "degraded"
        and (include_dirty or not record.provenance.git_dirty)
    )


def _sort_key(item: tuple[ExperimentRunRecord, RunMetrics]) -> tuple[object, ...]:
    record, metrics = item
    return (
        -metrics.robust_sharpe,
        -metrics.sharpe,
        -metrics.max_drawdown,
        metrics.turnover,
        record.run_id,
    )


def compare_runs(
    records: tuple[ExperimentRunRecord, ...],
    *,
    include_dirty: bool = False,
) -> ComparisonResult:
    run_ids = tuple(record.run_id for record in records)

    differences: dict[str, tuple[str, ...]] = {}
    for field in _DIFF_FIELDS:
        values = sorted({getattr(record, field) for record in records})
        if len(values) > 1:
            differences[field] = tuple(values)

    keys = {record.compatibility_key for record in records}
    compatible = len(keys) <= 1
    if not compatible:
        return ComparisonResult(
            run_ids=run_ids,
            compatible=False,
            compatibility_key=None,
            differences=differences,
            ranking=(),
            excluded_run_ids=run_ids,
        )

    eligible: list[tuple[ExperimentRunRecord, RunMetrics]] = []
    excluded: list[str] = []
    for record in records:
        if _is_eligible(record, include_dirty=include_dirty) and record.metrics:
            eligible.append((record, record.metrics))
        else:
            excluded.append(record.run_id)

    ranking = tuple(
        RankedRun(
            rank=index + 1,
            run_id=record.run_id,
            robust_sharpe=metrics.robust_sharpe,
            sharpe=metrics.sharpe,
            max_drawdown=metrics.max_drawdown,
            turnover=metrics.turnover,
        )
        for index, (record, metrics) in enumerate(sorted(eligible, key=_sort_key))
    )

    return ComparisonResult(
        run_ids=run_ids,
        compatible=True,
        compatibility_key=next(iter(keys)) if keys else None,
        differences=differences,
        ranking=ranking,
        excluded_run_ids=tuple(excluded),
    )
