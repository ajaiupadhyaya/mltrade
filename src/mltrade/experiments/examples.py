"""Packaged example experiment specifications.

``mltrade experiment init`` writes these into a target directory.  The same
text is committed under ``experiments/`` for convenience.
"""

from __future__ import annotations

from pathlib import Path

_DEFAULT_SNAPSHOT = "fixture-2026-06-12"

RIDGE_BASELINE_TOML = """schema_version = 1
name = "ridge-baseline"
description = "Untuned ridge-trend-v1 research baseline"
seed = 42

[dataset]
name = "daily_bars"
snapshot_id = "fixture-2026-06-12"
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

[objective]
name = "robust_sharpe"
maximum_drawdown = -0.35
maximum_turnover = 1.0

[resources]
max_trials = 1
timeout_minutes = 10
worker_count = 1
"""

RIDGE_BALANCED_SEARCH_TOML = """schema_version = 1
name = "ridge-balanced-search"
description = "Balanced Optuna search over ridge-trend-v1"
seed = 42

[dataset]
name = "daily_bars"
snapshot_id = "fixture-2026-06-12"
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

[objective]
name = "robust_sharpe"
maximum_drawdown = -0.35
maximum_turnover = 1.0

[resources]
max_trials = 12
timeout_minutes = 60
worker_count = 1

[search]
minimum_training_sessions = [504, 756, 1008]
retrain_every_sessions = [5, 10, 21, 42]

[search.alpha]
low = 0.001
high = 1000.0
log = true
"""

EXAMPLE_SPECS: dict[str, str] = {
    "ridge-baseline.toml": RIDGE_BASELINE_TOML,
    "ridge-balanced-search.toml": RIDGE_BALANCED_SEARCH_TOML,
}


def write_example_specs(
    directory: Path,
    *,
    snapshot_id: str | None = None,
) -> list[Path]:
    """Write the packaged example specs into *directory* without overwriting."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, text in EXAMPLE_SPECS.items():
        if snapshot_id is not None:
            text = text.replace(
                f'snapshot_id = "{_DEFAULT_SNAPSHOT}"',
                f'snapshot_id = "{snapshot_id}"',
            )
        target = directory / name
        if target.exists():
            continue
        target.write_text(text, encoding="utf-8")
        written.append(target)
    return written
