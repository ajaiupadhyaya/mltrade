"""Precompute the backtest-overfitting trials matrix (provenance/ops tool).

Runs the real walk-forward backtest across a grid of ridge regularisation
strengths (the only tuned hyper-parameter) and freezes the per-trial,
per-session strategy returns into a committed Parquet artifact.  The analytics
layer reads this frozen matrix to compute the Deflated Sharpe Ratio and the
Probability of Backtest Overfitting (PBO via CSCV) instantly and
deterministically, without re-running ten backtests on every export.

Run:  uv run python scripts/compute_trials.py
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from mltrade.backtest.engine import BacktestConfig, compute_equity_curve
from mltrade.data.snapshot import DEFAULT_AS_OF, SnapshotBarSource
from mltrade.models.forecasts import RidgeForecastConfig
from mltrade.universe import MVP_UNIVERSE

# Ridge alpha grid — the trials whose selection risk PBO/DSR quantify.  The
# headline strategy uses alpha=1.0 (RidgeForecastConfig default).
ALPHA_GRID: tuple[float, ...] = (
    0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0,
)
SELECTED_ALPHA = 1.0
_INITIAL_EQUITY = 1_000_000.0

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "snapshots" / "real"
TRIALS_PATH = OUT_DIR / f"trials_{DEFAULT_AS_OF}.parquet"
MANIFEST_PATH = OUT_DIR / f"trials_{DEFAULT_AS_OF}.manifest.json"


def _trial_returns(alpha: float) -> list[dict[str, object]]:
    """Run one walk-forward backtest at ``alpha`` → per-session returns."""
    src = SnapshotBarSource()
    bars = src.fetch(
        MVP_UNIVERSE, date(2007, 1, 3), date(2026, 6, 12),
        datetime(2026, 6, 13, tzinfo=UTC),
    )
    config = BacktestConfig(forecast=RidgeForecastConfig(alpha=alpha))
    curve = compute_equity_curve(bars, config=config)
    out: list[dict[str, object]] = []
    prev = _INITIAL_EQUITY
    for session, equity in curve:
        ret = equity / prev - 1.0 if prev > 0 else 0.0
        out.append({"alpha": alpha, "session": session, "ret": round(ret, 12)})
        prev = equity
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=min(10, len(ALPHA_GRID))) as pool:
        results = list(pool.map(_trial_returns, ALPHA_GRID))

    rows = [row for trial in results for row in trial]
    matrix = (
        pd.DataFrame(rows)
        .sort_values(["alpha", "session"])
        .reset_index(drop=True)
    )
    matrix.to_parquet(TRIALS_PATH, index=False)

    sha = hashlib.sha256(TRIALS_PATH.read_bytes()).hexdigest()
    sessions = sorted(matrix["session"].unique())
    manifest = {
        "dataset": "overfitting_trials",
        "as_of": DEFAULT_AS_OF,
        "computed_at": datetime.now(UTC).isoformat(),
        "tuned_parameter": "model.alpha (ridge regularisation)",
        "alpha_grid": list(ALPHA_GRID),
        "selected_alpha": SELECTED_ALPHA,
        "n_trials": len(ALPHA_GRID),
        "n_sessions": len(sessions),
        "start_session": str(sessions[0]),
        "end_session": str(sessions[-1]),
        "content_sha256": sha,
        "trials_file": TRIALS_PATH.name,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {TRIALS_PATH.relative_to(REPO_ROOT)}  ({len(matrix):,} rows)")
    print(f"  trials: {len(ALPHA_GRID)}  sessions: {len(sessions):,}")
    print(f"  sha256: {sha}")


if __name__ == "__main__":
    main()
