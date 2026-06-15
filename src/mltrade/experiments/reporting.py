"""Durable Markdown + JSON reports for a canonical experiment run.

The JSON report is the canonical record itself.  The Markdown report is a
human-readable summary built from the record plus the backtest detail.  Neither
contains credentials, environment variable values, or database URLs, and every
report states plainly that it is **not a promotion decision**.
"""

from __future__ import annotations

from mltrade.backtest.reporting import BacktestResult
from mltrade.experiments.records import ExperimentRunRecord


def build_report_json(record: ExperimentRunRecord) -> str:
    return record.model_dump_json(indent=2) + "\n"


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def build_report_markdown(
    record: ExperimentRunRecord,
    *,
    backtest: BacktestResult,
) -> str:
    lines: list[str] = []
    add = lines.append

    add("# MLTrade Experiment Report")
    add("")
    add(f"- Experiment: `{record.experiment_name}`")
    add(f"- Run id: `{record.run_id}`")
    add(f"- Status: **{record.status}**")
    add(
        f"- Dataset snapshot: `{record.dataset_snapshot_id}` "
        f"(content sha256 `{record.dataset_sha256[:16]}…`)"
    )
    add(f"- Compatibility key: `{record.compatibility_key[:16]}…`")
    add(f"- Git commit: `{record.provenance.git_commit}`")
    add(f"- Dirty worktree: {'yes' if record.provenance.git_dirty else 'no'}")
    add(f"- Started: {record.started_at.isoformat()}")
    add(f"- Finished: {record.finished_at.isoformat()}")
    if record.provenance.git_dirty:
        add("")
        add(
            "> ⚠️ **Dirty worktree** — uncommitted code changed this run's "
            "identity; results are not reproducible from a clean commit."
        )

    add("")
    add("## Resolved parameters")
    for key in sorted(record.parameters):
        add(f"- `{key}` = {record.parameters[key]}")

    add("")
    add("## Headline metrics")
    metrics = record.metrics
    if metrics is not None:
        add(f"- Annualized return: {_fmt(metrics.annualized_return)}")
        add(f"- Annualized volatility: {_fmt(metrics.annualized_volatility)}")
        add(f"- Sharpe: {_fmt(metrics.sharpe)}")
        add(f"- Max drawdown: {_fmt(metrics.max_drawdown)}")
        add(f"- Turnover: {_fmt(metrics.turnover)}")
        add(f"- Hit rate: {_fmt(metrics.hit_rate)}")
        add(
            f"- **Robust Sharpe**: {_fmt(metrics.robust_sharpe)} "
            f"(min(Sharpe@5, Sharpe@10) minus window Sharpe stdev "
            f"{_fmt(metrics.window_sharpe_std)})"
        )
    else:
        add("- (no metrics: run did not produce a result)")

    add("")
    add("## Cost sensitivity (Sharpe)")
    for bps in sorted(backtest.cost_sensitivity):
        add(f"- {bps} bps: {_fmt(backtest.cost_sensitivity[bps].sharpe)}")

    add("")
    add("## Baselines")
    add(f"- Equal-weight annualized return: {_fmt(backtest.equal_weight_return)}")
    add(f"- Cash annualized return: {_fmt(backtest.cash_return)}")

    add("")
    add("## Per-symbol contribution (buy & hold)")
    for symbol in sorted(backtest.per_symbol_contribution):
        add(f"- {symbol}: {_fmt(backtest.per_symbol_contribution[symbol])}")

    add("")
    add("## Evaluation windows")
    if backtest.evaluation_windows:
        for window in backtest.evaluation_windows:
            add(
                f"- {window.start_session} → {window.end_session} "
                f"({window.sessions} sessions): Sharpe {_fmt(window.sharpe)}, "
                f"max drawdown {_fmt(window.max_drawdown)}"
            )
    else:
        add("- (insufficient history for a full evaluation window)")

    if record.failure is not None:
        add("")
        add("## Blocked / failure reason")
        add(f"- {record.failure.category}: {record.failure.message}")

    add("")
    add("---")
    add(
        "_This report is a research artifact. **Not a promotion decision** — "
        "no model is approved for live trading by this run._"
    )
    return "\n".join(lines) + "\n"
