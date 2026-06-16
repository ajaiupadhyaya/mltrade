"""Dashboard JSON export.

Serialises the offline demo pipeline (data quality, walk-forward backtest with
its real per-session equity curve, portfolio target, pre-trade risk gates,
execution preview, and the research-experiment registry when available) into a
single deterministic JSON payload that the local web dashboard reads.

Local-first by design: no network, no secrets, fully reproducible.  The payload
is derived entirely from deterministic fixture data, so the same code, settings,
and clock always produce byte-identical JSON.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from mltrade.backtest.engine import compute_equity_curve
from mltrade.calendar import XNYSCalendar
from mltrade.config import Settings
from mltrade.data.fixtures import DeterministicBarSource
from mltrade.features.definitions import FEATURE_VERSION
from mltrade.models.forecasts import MODEL_VERSION
from mltrade.portfolio.targets import PortfolioLimits
from mltrade.universe import MVP_UNIVERSE
from mltrade.workflows.demo import DemoResult, run_demo

SCHEMA_VERSION = 1

# Mirror workflows.demo defaults so the regenerated bars (used for the equity
# curve) match the snapshot run_demo produces exactly.
_CLOCK = datetime(2026, 6, 13, tzinfo=UTC)
_FIXTURE_START = date(2019, 1, 2)
_UNIVERSE_VERSION = "mvp-etf-v1"

# Equity curve is downsampled for a compact, fast-loading payload.
_MAX_CURVE_POINTS = 260

_DEFAULT_OUTPUT = Path("web/public/data/dashboard.json")


def _f(value: Any) -> float:
    """Coerce a Decimal/float to a JSON float."""
    return float(value)


def _limits(settings: Settings) -> PortfolioLimits:
    return PortfolioLimits(
        maximum_position_weight=settings.maximum_position_weight,
        minimum_cash_weight=settings.minimum_cash_weight,
        target_annual_volatility=settings.target_annual_volatility,
    )


def _downsample(points: list[tuple[date, float]]) -> list[tuple[date, float]]:
    """Thin a dense curve to at most ``_MAX_CURVE_POINTS`` points, keeping the last."""
    n = len(points)
    if n <= _MAX_CURVE_POINTS:
        return points
    step = math.ceil(n / _MAX_CURVE_POINTS)
    sampled = points[::step]
    if sampled[-1] != points[-1]:
        sampled.append(points[-1])
    return sampled


def _backtest_payload(
    result: DemoResult,
    equity_curve: list[tuple[date, float]],
) -> dict[str, Any]:
    bt = result.backtest
    sensitivity = [
        {
            "bps": int(bps),
            "annualized_return": cs.annualized_return,
            "annualized_volatility": cs.annualized_volatility,
            "sharpe": cs.sharpe,
            "max_drawdown": cs.max_drawdown,
        }
        for bps, cs in sorted(bt.cost_sensitivity.items(), key=lambda kv: float(kv[0]))
    ]
    return {
        "sessions": bt.sessions,
        "headline_cost_bps": 5,
        "annualized_return": bt.annualized_return,
        "annualized_volatility": bt.annualized_volatility,
        "sharpe": bt.sharpe,
        "max_drawdown": bt.max_drawdown,
        "turnover": bt.turnover,
        "total_costs": bt.total_costs,
        "hit_rate": bt.hit_rate,
        "equal_weight_return": bt.equal_weight_return,
        "cash_return": bt.cash_return,
        "cost_sensitivity": sensitivity,
        "per_symbol_contribution": [
            {"symbol": sym, "return": val}
            for sym, val in sorted(
                bt.per_symbol_contribution.items(),
                key=lambda kv: kv[1],
                reverse=True,
            )
        ],
        "equity_curve": [
            {"date": d.isoformat(), "equity": round(eq, 2)}
            for d, eq in _downsample(equity_curve)
        ],
    }


def _portfolio_payload(result: DemoResult) -> dict[str, Any]:
    target = result.target
    weights = [
        {"symbol": sym, "weight": _f(w)}
        for sym, w in sorted(
            target.weights.items(), key=lambda kv: kv[1], reverse=True
        )
    ]
    invested = sum(_f(w) for w in target.weights.values())
    return {
        "blocked": target.blocked,
        "cash_weight": _f(target.cash_weight),
        "invested_weight": round(invested, 10),
        "weights": weights,
    }


def _risk_payload(result: DemoResult) -> dict[str, Any]:
    report = result.preview.risk_report
    summary = {"pass": 0, "warn": 0, "block": 0}
    checks = []
    for check in report.checks:
        status = check.status.value
        summary[status] = summary.get(status, 0) + 1
        checks.append(
            {
                "code": check.code,
                "status": status,
                "message": check.message,
            }
        )
    return {
        "blocked": report.blocked,
        "summary": summary,
        "checks": checks,
    }


def _execution_payload(result: DemoResult) -> dict[str, Any]:
    recon = result.preview.reconciliation
    intents = [
        {
            "side": intent.side.value,
            "symbol": intent.symbol,
            "quantity": int(intent.target_quantity),
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


def _quality_payload(result: DemoResult) -> dict[str, Any]:
    quality = result.quality
    fb = result.forecast_batch
    return {
        "blocked": quality.blocked,
        "issues_count": len(quality.issues),
        "issues": [
            {"code": i.code, "severity": i.severity.value, "message": i.message}
            for i in quality.issues
        ],
        "training_rows": fb.training_row_count,
        "training_sessions": fb.training_session_count,
    }


def _forecast_payload(result: DemoResult) -> dict[str, Any]:
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


def _experiments_payload(settings: Settings) -> dict[str, Any]:
    """Read the local research-experiment leaderboard if any runs exist.

    Compatible complete runs are ranked by robust Sharpe; blocked/failed/pruned
    runs stay visible but unranked. When the registry is empty (a fresh
    checkout), return an honest empty state the dashboard renders as a
    call-to-action rather than fabricated runs.
    """
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
    """Run the offline demo pipeline and assemble the dashboard JSON payload."""
    effective_clock = clock if clock is not None else _CLOCK
    last_session = XNYSCalendar().last_completed_session(effective_clock)

    result = run_demo(settings, clock=effective_clock)

    # Real per-session equity curve, computed with the same walk-forward
    # decisions and accounting as the headline backtest.
    bars = DeterministicBarSource(seed=42).fetch(
        MVP_UNIVERSE, _FIXTURE_START, last_session, effective_clock
    )
    equity_curve = compute_equity_curve(bars, limits=_limits(settings))

    universe = sorted({f.symbol for f in result.forecast_batch.forecasts})

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": effective_clock.isoformat(),
        "meta": {
            "platform": "MLTrade",
            "snapshot_id": result.snapshot_id,
            "environment": settings.environment.value,
            "live_trading_enabled": settings.live_trading_enabled,
            "last_session": last_session.isoformat(),
            "reference_equity": _f(settings.reference_equity),
            "model_version": MODEL_VERSION,
            "feature_version": FEATURE_VERSION,
            "universe_version": _UNIVERSE_VERSION,
            "universe": universe,
            "data_mode": "offline-fixture",
            "synthetic": True,
        },
        "backtest": _backtest_payload(result, equity_curve),
        "portfolio": _portfolio_payload(result),
        "risk": _risk_payload(result),
        "execution": _execution_payload(result),
        "quality": _quality_payload(result),
        "forecast": _forecast_payload(result),
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
