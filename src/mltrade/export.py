"""Dashboard JSON export (schema v2 — real-market-data research terminal).

Assembles a single deterministic JSON payload from the frozen real-data research
pipeline (:func:`mltrade.workflows.research_real.run_real_research`) and the
institutional analytics layer:

- headline KPIs (Sharpe, alpha + t-stat, IR, Deflated Sharpe, PBO);
- performance & risk (Sortino, Calmar, drawdown depth/duration, VaR/CVaR,
  return distribution) plus the chart series (equity vs benchmark, drawdown,
  rolling Sharpe, monthly/yearly returns, histogram);
- benchmark-relative statistics, returns-based factor attribution, and the
  backtest-overfitting diagnostics;
- the current decision (portfolio target, 17 pre-trade risk gates, execution
  preview, forecast) and the research-experiment leaderboard.

Local-first and reproducible: every number derives from the committed snapshot;
no network, no secrets, live trading structurally disabled.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from mltrade.analytics import (
    ASSET_CLASS,
    MACRO_FACTORS,
    compute_attribution,
    compute_benchmark_stats,
    compute_overfitting,
    compute_performance,
)
from mltrade.analytics.performance import (
    drawdown_series,
    monthly_returns,
    return_histogram,
    rolling_sharpe,
)
from mltrade.analytics.returns import cumulative_growth
from mltrade.calendar import XNYSCalendar
from mltrade.config import Settings
from mltrade.data.snapshot import DEFAULT_AS_OF, DEFAULT_SNAPSHOT_DIR
from mltrade.features.definitions import FEATURE_VERSION
from mltrade.models.forecasts import MODEL_VERSION
from mltrade.workflows.research_real import RealResearchResult, run_real_research

SCHEMA_VERSION = 2

_INITIAL_EQUITY = 1_000_000.0
_MAX_CURVE_POINTS = 320
_ROLLING_WINDOW = 126
_SELECTED_TRIAL = "alpha=1"
_DEFAULT_OUTPUT = Path("web/public/data/dashboard.json")


def _f(value: Any) -> float:
    return float(value)


def _r(value: float, places: int = 6) -> float:
    return round(value, places)


def _downsample_indices(n: int, max_points: int) -> list[int]:
    if n <= max_points:
        return list(range(n))
    step = math.ceil(n / max_points)
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return idx


def _yearly(sessions: Sequence[date], returns: Sequence[float]) -> dict[int, float]:
    by_year: dict[int, float] = {}
    for d, r in zip(sessions, returns, strict=True):
        by_year[d.year] = (1.0 + by_year.get(d.year, 0.0)) * (1.0 + r) - 1.0
    return by_year


def _load_trials(as_of: str = DEFAULT_AS_OF) -> dict[str, list[float]] | None:
    path = DEFAULT_SNAPSHOT_DIR / f"trials_{as_of}.parquet"
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    trials: dict[str, list[float]] = {}
    for alpha, group in frame.groupby("alpha"):
        ordered = group.sort_values("session")
        trials[f"alpha={alpha:g}"] = [float(x) for x in ordered["ret"]]
    return trials


def _meta_payload(
    result: RealResearchResult, settings: Settings, generated_at: datetime
) -> dict[str, Any]:
    manifest = result.manifest
    universe = sorted({b.instrument.symbol for b in result.bars})
    return {
        "platform": "MLTrade",
        "snapshot_id": result.snapshot_id,
        "as_of": manifest["as_of"],
        "environment": settings.environment.value,
        "live_trading_enabled": settings.live_trading_enabled,
        "data_mode": "real-frozen-snapshot",
        "synthetic": False,
        "source": manifest["source"],
        "adjustment": manifest["adjustment"],
        "benchmark": manifest["benchmark"],
        "universe": universe,
        "universe_version": "mvp-etf-v1",
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "reference_equity": _f(settings.reference_equity),
        "oos_start": result.aligned_sessions[0].isoformat(),
        "oos_end": result.last_session.isoformat(),
        "n_sessions": len(result.aligned_sessions),
        "n_symbols": len(universe),
        "generated_at": generated_at.isoformat(),
    }


def _headline_payload(
    perf: Any, bench: Any, overfitting: Any, final_equity: float
) -> dict[str, Any]:
    block = {
        "sharpe": perf.sharpe,
        "annualized_return": perf.annualized_return,
        "annualized_volatility": perf.annualized_volatility,
        "max_drawdown": perf.max_drawdown,
        "sortino": perf.sortino,
        "calmar": perf.calmar,
        "beta": bench.beta,
        "alpha_annualized": bench.alpha_annualized,
        "alpha_tstat": bench.alpha_tstat,
        "alpha_pvalue": bench.alpha_pvalue,
        "information_ratio": bench.information_ratio,
        "final_equity": _r(final_equity, 2),
        "total_return_multiple": _r(final_equity / _INITIAL_EQUITY, 4),
    }
    if overfitting is not None:
        block["deflated_sharpe_ratio"] = overfitting.deflated_sharpe_ratio
        block["pbo"] = overfitting.pbo
    return block


def _performance_payload(result: RealResearchResult, perf: Any) -> dict[str, Any]:
    sessions = result.aligned_sessions
    returns = result.strategy_returns
    equity_by_session = dict(result.equity_curve)

    strat_equity = [equity_by_session[s] for s in sessions]
    bench_equity = cumulative_growth(result.benchmark_returns, initial=_INITIAL_EQUITY)
    dd = drawdown_series(returns)
    roll = rolling_sharpe(returns, window=_ROLLING_WINDOW)

    idx = _downsample_indices(len(sessions), _MAX_CURVE_POINTS)
    equity_curve = [
        {
            "date": sessions[i].isoformat(),
            "strategy": _r(strat_equity[i], 2),
            "benchmark": _r(bench_equity[i], 2),
        }
        for i in idx
    ]
    drawdown_curve = [
        {"date": sessions[i].isoformat(), "dd": _r(dd[i], 6)} for i in idx
    ]
    rolling: list[dict[str, Any]] = []
    for i in idx:
        value = roll[i]
        if value is not None:
            rolling.append({"date": sessions[i].isoformat(), "value": _r(value, 4)})

    centres, counts = return_histogram(returns, bins=41)
    monthly = monthly_returns(sessions, returns)
    strat_year = _yearly(sessions, returns)
    bench_year = _yearly(sessions, result.benchmark_returns)
    yearly = [
        {
            "year": year,
            "strategy": _r(strat_year[year], 4),
            "benchmark": _r(bench_year.get(year, 0.0), 4),
        }
        for year in sorted(strat_year)
    ]

    bt = result.backtest
    cost_sensitivity = [
        {
            "bps": int(bps),
            "annualized_return": cs.annualized_return,
            "annualized_volatility": cs.annualized_volatility,
            "sharpe": cs.sharpe,
            "max_drawdown": cs.max_drawdown,
        }
        for bps, cs in sorted(bt.cost_sensitivity.items(), key=lambda kv: float(kv[0]))
    ]
    evaluation_windows = [
        {
            "start": w.start_session.isoformat(),
            "end": w.end_session.isoformat(),
            "sharpe": w.sharpe,
            "annualized_return": w.annualized_return,
            "max_drawdown": w.max_drawdown,
        }
        for w in bt.evaluation_windows
    ]

    return {
        **perf.model_dump(),
        "headline_cost_bps": 5,
        "turnover": bt.turnover,
        "hit_rate": bt.hit_rate,
        "total_costs": bt.total_costs,
        "equal_weight_return": bt.equal_weight_return,
        "cash_return": bt.cash_return,
        "equity_curve": equity_curve,
        "drawdown": drawdown_curve,
        "rolling_sharpe": rolling,
        "histogram": {"centres": centres, "counts": counts},
        "monthly": monthly,
        "yearly": yearly,
        "cost_sensitivity": cost_sensitivity,
        "evaluation_windows": evaluation_windows,
        "per_symbol_contribution": [
            {"symbol": sym, "return": val}
            for sym, val in sorted(
                bt.per_symbol_contribution.items(), key=lambda kv: kv[1], reverse=True
            )
        ],
    }


def _attribution_payload(result: RealResearchResult) -> dict[str, Any]:
    factors = {sym: result.factor_returns[sym] for sym, _ in MACRO_FACTORS}
    attr = compute_attribution(result.strategy_returns, factors)
    return {
        "exposures": [e.model_dump() for e in attr.exposures],
        "alpha_annualized": attr.alpha_annualized,
        "alpha_tstat": attr.alpha_tstat,
        "r_squared": attr.r_squared,
        "n_sessions": attr.n_sessions,
    }


def _portfolio_payload(result: RealResearchResult) -> dict[str, Any]:
    target = result.target
    weights = [
        {"symbol": sym, "weight": _f(w), "asset_class": ASSET_CLASS.get(sym, "Other")}
        for sym, w in sorted(target.weights.items(), key=lambda kv: kv[1], reverse=True)
    ]
    by_class: dict[str, float] = defaultdict(float)
    for sym, w in target.weights.items():
        by_class[ASSET_CLASS.get(sym, "Other")] += _f(w)
    classes = [
        {"asset_class": name, "weight": _r(weight, 6)}
        for name, weight in sorted(by_class.items(), key=lambda kv: kv[1], reverse=True)
    ]
    invested = sum(_f(w) for w in target.weights.values())
    return {
        "blocked": target.blocked,
        "cash_weight": _f(target.cash_weight),
        "invested_weight": _r(invested, 10),
        "weights": weights,
        "asset_classes": classes,
    }


def _risk_payload(result: RealResearchResult) -> dict[str, Any]:
    report = result.preview.risk_report
    summary = {"pass": 0, "warn": 0, "block": 0}
    checks = []
    for check in report.checks:
        status = check.status.value
        summary[status] = summary.get(status, 0) + 1
        checks.append({"code": check.code, "status": status, "message": check.message})
    return {"blocked": report.blocked, "summary": summary, "checks": checks}


def _execution_payload(result: RealResearchResult) -> dict[str, Any]:
    recon = result.preview.reconciliation
    intents = [
        {
            "side": intent.side.value,
            "symbol": intent.symbol,
            "quantity": int(intent.target_quantity),
            "notional": _r(
                int(intent.target_quantity) * _f(result.prices.get(intent.symbol, 0)), 2
            ),
            "client_order_id": intent.client_order_id,
        }
        for intent in result.preview.intents
    ]
    return {
        "preview_only": True,
        "broker": "SimulatedBroker",
        "count": len(intents),
        "reconciliation_blocked": recon.blocked,
        "intents": intents,
    }


def _forecast_payload(result: RealResearchResult) -> dict[str, Any]:
    fb = result.forecast_batch
    return {
        "decision_session": fb.decision_session.isoformat(),
        "model_version": fb.model_version,
        "count": len(fb.forecasts),
        "training_row_count": fb.training_row_count,
        "training_session_count": fb.training_session_count,
        "forecasts": [
            {"symbol": f.symbol, "predicted_forward_return": f.predicted_forward_return}
            for f in sorted(
                fb.forecasts, key=lambda f: f.predicted_forward_return, reverse=True
            )
        ],
    }


def _quality_payload(result: RealResearchResult) -> dict[str, Any]:
    manifest = result.manifest
    start = date.fromisoformat(manifest["start_session"])
    end = date.fromisoformat(manifest["end_session"])
    expected = len(XNYSCalendar().sessions_in_range(start, end))
    panel = int(manifest["session_count"])
    return {
        "source": manifest["source"],
        "adjustment": manifest["adjustment"],
        "start_session": manifest["start_session"],
        "end_session": manifest["end_session"],
        "panel_sessions": panel,
        "expected_xnys_sessions": expected,
        "completeness": _r(panel / expected, 4) if expected else 1.0,
        "excluded_sessions": max(0, expected - panel),
        "n_symbols": len(manifest["symbols"]),
        "row_count": int(manifest["row_count"]),
        "content_sha256": manifest["content_sha256"],
    }


def _overfitting_payload(overfitting: Any) -> dict[str, Any] | None:
    if overfitting is None:
        return None
    data: dict[str, Any] = overfitting.model_dump()
    # Downsample the logit cloud for the payload; keep summary stats intact.
    logits = list(data.pop("logits"))
    data["logit_histogram"] = _logit_histogram(logits)
    return data


def _logit_histogram(logits: list[float]) -> dict[str, list[float] | list[int]]:
    if not logits:
        return {"centres": [], "counts": []}
    counts, edges = np.histogram(logits, bins=21)
    centres = [
        round((float(edges[i]) + float(edges[i + 1])) / 2.0, 4)
        for i in range(len(counts))
    ]
    return {"centres": centres, "counts": [int(x) for x in counts]}


def _experiments_payload(settings: Settings) -> dict[str, Any]:
    placeholder = {
        "available": False,
        "ranked_by": "robust_sharpe",
        "runs": [],
        "note": (
            "No experiment runs yet. Run `mltrade experiment run "
            "experiments/ridge-baseline.toml` to populate this leaderboard."
        ),
    }
    root = settings.experiment_root
    if root is None:
        return placeholder
    try:
        from mltrade.experiments import RunStore, compare_runs

        records = RunStore(root).list_records()
    except Exception:
        return placeholder
    if not records:
        return placeholder

    by_id = {record.run_id: record for record in records}
    result = compare_runs(tuple(records))

    def alpha_of(record: Any) -> str:
        value = record.parameters.get("model.alpha")
        return f"{value:g}" if isinstance(value, int | float) else "—"

    def fmt(value: float | None) -> str:
        return f"{value:.2f}" if value is not None else "—"

    runs: list[dict[str, Any]] = []
    for ranked in result.ranking:
        record = by_id.get(ranked.run_id)
        runs.append(
            {
                "rank": ranked.rank,
                "run_id": ranked.run_id,
                "alpha": alpha_of(record) if record is not None else "—",
                "robust_sharpe": fmt(ranked.robust_sharpe),
                "sharpe": fmt(ranked.sharpe),
                "max_drawdown": fmt(ranked.max_drawdown),
                "turnover": fmt(ranked.turnover),
                "status": record.status if record is not None else "complete",
            }
        )
    for run_id in result.excluded_run_ids:
        record = by_id.get(run_id)
        if record is None:
            continue
        metrics = record.metrics
        runs.append(
            {
                "rank": "—",
                "run_id": run_id,
                "alpha": alpha_of(record),
                "robust_sharpe": fmt(metrics.robust_sharpe if metrics else None),
                "sharpe": fmt(metrics.sharpe if metrics else None),
                "max_drawdown": fmt(metrics.max_drawdown if metrics else None),
                "turnover": fmt(metrics.turnover if metrics else None),
                "status": record.status,
            }
        )
    return {
        "available": True,
        "ranked_by": "robust_sharpe",
        "runs": runs,
        "compatible": result.compatible,
    }


def build_dashboard_payload(
    settings: Settings,
    *,
    clock: datetime | None = None,
) -> dict[str, Any]:
    """Run the real-data research pipeline and assemble the v2 dashboard JSON."""
    generated_at = clock if clock is not None else datetime(2026, 6, 13, tzinfo=UTC)
    result = run_real_research(settings)

    perf = compute_performance(result.strategy_returns)
    bench = compute_benchmark_stats(
        result.strategy_returns,
        result.benchmark_returns,
        benchmark=result.manifest["benchmark"],
    )
    trials = _load_trials(result.manifest["as_of"])
    overfitting = (
        compute_overfitting(trials, selected=_SELECTED_TRIAL, n_splits=10)
        if trials is not None and _SELECTED_TRIAL in trials
        else None
    )
    final_equity = result.equity_curve[-1][1]

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at.isoformat(),
        "meta": _meta_payload(result, settings, generated_at),
        "headline": _headline_payload(perf, bench, overfitting, final_equity),
        "performance": _performance_payload(result, perf),
        "benchmark": bench.model_dump(),
        "attribution": _attribution_payload(result),
        "overfitting": _overfitting_payload(overfitting),
        "portfolio": _portfolio_payload(result),
        "risk": _risk_payload(result),
        "execution": _execution_payload(result),
        "forecast": _forecast_payload(result),
        "quality": _quality_payload(result),
        "experiments": _experiments_payload(settings),
    }


def write_dashboard_json(
    settings: Settings,
    output_path: Path,
    *,
    clock: datetime | None = None,
) -> Path:
    """Build the payload and write it to ``output_path`` (parents created)."""
    payload = build_dashboard_payload(settings, clock=clock)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    return output_path
