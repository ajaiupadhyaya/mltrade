# Research Experiments Runbook

The local research experiment platform runs and tunes the existing
`ridge-trend-v1` research pipeline through versioned, immutable TOML
specifications, with content-addressed run records, durable Markdown/JSON
reports, local MLflow tracking, and resumable Optuna studies.

> **Boundaries.** Every report is a research artifact and is **not a promotion
> decision**. Live trading stays structurally disabled. No model is approved
> for live money by any command here.

## Prerequisites

```bash
uv sync --frozen --extra dev
```

An experiment runs against an **exact, immutable snapshot**. Publish the
deterministic fixture snapshot first:

```bash
uv run mltrade data ingest    # publishes daily_bars/fixture-YYYY-MM-DD
```

The tracked example `experiments/ridge-baseline.toml` references
`snapshot_id = "fixture-2026-06-12"`. If your local `data ingest` produces a
different last-session date, regenerate the examples for that snapshot:

```bash
uv run mltrade experiment init experiments --snapshot-id fixture-YYYY-MM-DD
```

## Commands

```bash
uv run mltrade experiment doctor        # check dirs, deps, example snapshot
uv run mltrade experiment validate experiments/ridge-baseline.toml
uv run mltrade experiment run experiments/ridge-baseline.toml
uv run mltrade experiment list
uv run mltrade experiment inspect <run-id> --json
uv run mltrade experiment report <run-id>
uv run mltrade experiment compare <run-id-a> <run-id-b>
uv run mltrade experiment tune experiments/ridge-balanced-search.toml --trials 12
uv run mltrade experiment resume <study-name> --spec experiments/ridge-balanced-search.toml
```

### Single run

`experiment run` loads the exact manifest, verifies the immutable context
(dataset, content hash, universe version), captures git + runtime provenance,
computes a **content-addressed `run-...` id**, runs the research pipeline, and
saves a canonical record plus `report.md` / `report.json` / `spec.json` under
`$MLTRADE_EXPERIMENT_ROOT/runs/<run-id>/`.

Re-running identical code + spec + data returns the **same run id** and the same
canonical metrics — timestamps are evidence, not identity, so re-runs are
idempotent rather than producing a second record.

### Objective and ranking

The objective is `robust_sharpe = min(Sharpe@5bps, Sharpe@10bps) − stdev(window
Sharpes)`. A run whose backtest violates the spec's `maximum_drawdown` floor or
`maximum_turnover` cap is recorded with status **blocked** and excluded from
ranking (its metrics are still computed and reported). `experiment compare`
ranks only runs that share a methodological compatibility key; blocked, failed,
pruned, degraded-tracking, and **dirty-worktree** runs stay visible but are
excluded by default, and incompatible sets never produce a winner.

> The deterministic demo fixture has a deep structural drawdown (~−0.72) that
> exceeds the default `maximum_drawdown = −0.35`, so the untuned baseline is
> honestly reported as **blocked** on that objective. This reflects the
> synthetic fixture, not a platform defect — see the roadmap for the
> honest-fixture work item.

### Tuning and resume

`experiment tune` drives a persistent Optuna study stored in SQLite at
`$MLTRADE_EXPERIMENT_ROOT/optuna/studies.db`. The study records an immutable
**context hash**; resuming with a changed dataset/objective/search context is
rejected (`study context mismatch`) rather than mixing incompatible trials.
Each trial materializes through the normal runner, so every trial leaves a
canonical record and reports.

The balanced example uses `max_trials = 12` (each trial is a full walk-forward
backtest). Override with `--trials N` for a quick smoke. Interrupt safely with
`Ctrl-C`; re-run `experiment tune` (or `experiment resume <study>`) to continue
from the persisted trials.

## Canonical records vs MLflow

The **canonical record** (`run.json`) is the source of truth and is written
atomically before any tracker runs. Local MLflow tracking (`--track`) is a
convenience mirror: if it fails, the run's `tracking_status` is marked
`degraded`, the canonical evidence is preserved, and the command exits nonzero.
Reports render fully **without** MLflow.

## Local state and cleanup

All runtime state lives under `$MLTRADE_EXPERIMENT_ROOT` (default
`data/experiments/`): `runs/`, `mlflow/`, and `optuna/`. It is gitignored via
`data/`. To reset:

```bash
rm -rf data/experiments
```

The container image (`docker build -t mltrade .`) includes the example specs
but no local experiment state, MLflow store, Optuna database, `.env`, or
`.claude/`.

## Failure diagnosis

- `snapshot ... unavailable` — run `uv run mltrade data ingest`, or point the
  spec at a published `snapshot_id`.
- `universe_version mismatch` — the snapshot was built for a different universe.
- `study context mismatch` — you changed an immutable field; start a new study.
- `tracking degraded` — MLflow failed; the canonical record is intact, inspect
  it with `experiment inspect <run-id>`.
