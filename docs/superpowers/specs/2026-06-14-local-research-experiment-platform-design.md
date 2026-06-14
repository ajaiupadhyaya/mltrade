# Local Research Experiment Platform Design

Date: 2026-06-14
Status: Approved design

## 1. Objective

Build a local-first, CLI-driven experiment platform around MLTrade's existing
research pipeline. The platform will make ridge-model research reproducible,
resumable, comparable, and auditable without rewriting the trusted data,
feature, walk-forward, portfolio, risk, or backtest implementations.

The first release optimizes experiment velocity. A normal tuning study should
finish in approximately 30 to 60 minutes on the user's Mac while retaining
deterministic execution and defensible validation boundaries.

Live trading remains disabled. Experiment results are research evidence and
cannot directly promote a model into paper or live execution.

## 2. Scope

### 2.1 Included

- immutable, versioned experiment specifications;
- deterministic single-run experiments;
- bounded Optuna tuning for the existing ridge workflow;
- resumable local studies backed by SQLite;
- local MLflow tracking and artifact storage;
- run comparison with compatibility checks;
- Markdown and JSON research reports;
- complete data, code, parameter, seed, and environment provenance;
- CLI workflows for initialization, validation, execution, tuning, comparison,
  inspection, and reporting; and
- focused refactoring of existing research code where needed to expose stable
  experiment interfaces.

### 2.2 Deferred

- new model families;
- new strategy sleeves;
- automatic model promotion;
- distributed or cloud execution;
- remote MLflow services;
- web dashboards and notebooks as primary interfaces;
- automated scheduling;
- SEC, FRED, alternative-data, and individual-equity expansion; and
- any live-money execution capability.

## 3. Architecture

Add `mltrade.experiments` as an orchestration layer. It calls the existing
snapshot, feature, model, backtest, and reporting components through explicit
interfaces. It does not copy trading calculations into experiment-specific
implementations.

```text
src/mltrade/experiments/
├── specs.py          # Immutable TOML specification models and validation
├── loading.py        # Safe specification loading and canonical serialization
├── provenance.py     # Git, dataset, dependency, seed, and runtime identity
├── runner.py         # Deterministic single-run orchestration
├── search.py         # Bounded Optuna search-space contracts
├── tuning.py         # Resumable study orchestration and trial lifecycle
├── tracking.py       # RunTracker protocol and local MLflow implementation
├── comparison.py     # Compatibility checks and run comparison
├── reporting.py      # Markdown and JSON report generation
└── records.py        # Immutable run, trial, metric, and artifact contracts

experiments/
├── ridge-baseline.toml
└── ridge-balanced-search.toml

data/experiments/
├── mlflow/
├── optuna/
├── reports/
└── runs/
```

Tracked example specifications live under `experiments/`. Mutable experiment
state lives under `data/experiments/` and remains excluded from git.

### 3.1 Primary Components

`ExperimentSpec`
: Immutable validated configuration covering dataset identity, features,
model parameters, walk-forward validation, portfolio limits, transaction
costs, optimization objective, seeds, and resource budget.

`ExperimentRunner`
: Resolves a verified immutable snapshot, runs one experiment through existing
MLTrade components, calculates the approved metrics, persists a durable run
record, and sends the same record to the tracker.

`SearchSpace`
: Defines named, typed, bounded parameters for the ridge workflow. Invalid or
computationally unreasonable ranges are rejected before a study starts.

`OptunaTuner`
: Creates or resumes a study, derives deterministic trial seeds, runs trials,
records failures without losing study state, and materializes the selected
trial as a normal experiment run.

`RunTracker`
: A narrow interface for run metadata, parameters, metrics, tags, and
artifacts. The first implementation uses local MLflow file storage.

`ComparisonService`
: Loads durable run records, verifies that comparisons are methodologically
compatible, and produces deterministic tables and rankings.

`ReportBuilder`
: Writes human-readable Markdown plus machine-readable JSON without requiring
the MLflow UI.

## 4. Experiment Specification

TOML is the canonical user-facing format because it is readable, typed enough
for review, and already familiar in this Python repository.

Every specification contains:

- `schema_version`;
- stable experiment name and optional description;
- dataset name and exact snapshot ID;
- universe and feature-set versions;
- model family, model version, and ridge parameters;
- walk-forward training, retraining, horizon, and embargo settings;
- portfolio and cost assumptions;
- random seeds;
- objective metric and constraint metrics;
- tuning search space when applicable; and
- resource limits such as maximum trials, timeout, and worker count.

The default balanced tuning specification uses:

- a maximum duration of 60 minutes;
- a bounded trial count selected to fit that duration;
- one or two local workers, chosen conservatively;
- deterministic Optuna sampler seeds;
- no GPU assumptions; and
- pruning only after enough walk-forward evidence exists to avoid favoring
  trials with incomplete evaluation.

Unknown keys are rejected. Paths are resolved relative to the specification
file only where explicitly allowed. Dataset selection never silently falls
back to "latest" during a recorded experiment.

## 5. Execution and Data Flow

### 5.1 Single Run

1. Load and validate the TOML specification.
2. Resolve the exact immutable dataset manifest.
3. Verify manifest hashes and dataset compatibility.
4. Capture code, dependency, host, and seed provenance.
5. Build point-in-time features using the existing feature pipeline.
6. Run the configured embargoed walk-forward ridge evaluation.
7. Construct portfolios and run the existing backtester at configured costs.
8. Calculate performance, risk, turnover, stability, and baseline metrics.
9. Evaluate experiment health and methodological guardrails.
10. Persist the canonical run record atomically.
11. Log parameters, metrics, tags, and artifacts to local MLflow.
12. Generate Markdown and JSON reports.

The durable run record is authoritative. MLflow is an index and exploration
surface, not the only copy of research evidence.

### 5.2 Tuning Study

1. Load and validate a tuning specification.
2. Create or resume the named Optuna study using SQLite.
3. Reject resume attempts when immutable study context differs, including
   dataset snapshot, feature version, validation design, or objective.
4. Generate bounded trial parameters and a deterministic trial seed.
5. Execute each trial through the same `ExperimentRunner` calculation path.
6. Record successful, pruned, blocked, and failed trial outcomes.
7. Apply objective and constraint rules only to complete valid trials.
8. Materialize the winning trial as a first-class experiment run.
9. Produce study diagnostics, parameter importance, stability summaries, and
   a comparison against the untuned ridge baseline.

Interrupted studies retain completed trial state and can be resumed by name.

## 6. Provenance and Reproducibility

Every run records:

- experiment and schema versions;
- dataset manifest and content hashes;
- universe, feature, and model versions;
- canonical resolved specification and its SHA-256 hash;
- git commit and dirty-worktree status;
- Python version and relevant locked dependency versions;
- platform and CPU information available without privileged access;
- all random seeds;
- start and finish timestamps in UTC;
- command invocation and MLTrade version;
- trial and study identity when applicable;
- generated artifact hashes; and
- terminal status: `complete`, `blocked`, `pruned`, or `failed`.

A dirty worktree is allowed for local exploration but is prominently tagged.
Comparisons can exclude dirty runs by default. Reports never claim exact
reproducibility when the code state cannot be identified.

Run directories are written through a temporary staging directory and renamed
only after the canonical record and required artifacts are complete.

## 7. Metrics and Comparison Rules

The initial objective is configurable, but the default is a conservative
risk-adjusted score derived from out-of-sample results. It must not optimize
raw return alone.

Each run records at least:

- annualized return and volatility;
- Sharpe ratio;
- maximum drawdown;
- turnover;
- transaction costs;
- hit rate;
- cash, gross, and net exposure;
- equal-weight and cash baseline deltas;
- performance at 2, 5, and 10 basis-point costs;
- per-instrument contribution;
- fold or evaluation-window dispersion; and
- blocked or invalid evaluation counts.

Runs are directly rank-comparable only when these fields match:

- dataset snapshot;
- universe version;
- feature version;
- forecast horizon;
- walk-forward and embargo design;
- portfolio limits;
- transaction-cost scenario; and
- objective definition.

Incompatible runs may be displayed side by side, but the CLI labels them
incompatible and does not produce a winner.

No single metric constitutes a paper-trading promotion decision.

## 8. CLI

Add an `experiment` command group:

```text
mltrade experiment init
mltrade experiment validate SPEC
mltrade experiment run SPEC
mltrade experiment tune SPEC
mltrade experiment resume STUDY
mltrade experiment list
mltrade experiment inspect RUN_ID
mltrade experiment compare RUN_ID...
mltrade experiment report RUN_ID
mltrade experiment doctor
```

Key behavior:

- `init` creates tracked-ready example specifications only when requested and
  does not overwrite existing files.
- `validate` performs schema, dataset, compatibility, and resource checks
  without running a model.
- `run` emits a stable run ID and report paths.
- `tune` prints study progress and a concise final result without flooding the
  terminal with per-fold details.
- `resume` refuses context drift.
- `compare` exits nonzero when asked to rank incompatible runs.
- `doctor` checks writable experiment directories, MLflow storage, Optuna
  storage, dataset availability, and optional dependency readiness.

Commands support structured JSON output for automation in addition to concise
human-readable output.

## 9. Failure Handling

The subsystem fails closed on:

- missing, mutable, or hash-invalid snapshots;
- feature leakage or version mismatch;
- non-finite inputs, forecasts, weights, or metrics;
- invalid walk-forward windows;
- optimizer or backtest failure;
- incompatible study resume context;
- duplicate run identity with different content;
- artifact publication failure; and
- objective calculation from incomplete evidence.

A failed trial does not terminate a study unless the failure indicates a
study-wide invariant violation. Authentication, broker, and order submission
code are outside the experiment runner.

MLflow logging failure does not erase a completed canonical run. The run is
marked `tracking_degraded`, the local record remains available, and the CLI
exits nonzero so the operator can repair tracking and replay metadata.

## 10. Testing

### 10.1 Unit Tests

- specification parsing, defaults, and unknown-key rejection;
- canonical serialization and hashing;
- search-space bounds and conditional parameters;
- deterministic seed derivation;
- provenance capture and secret redaction;
- compatibility classification;
- objective and constraint calculations;
- atomic run publication; and
- MLflow adapter behavior through a temporary local tracking directory.

### 10.2 Integration Tests

- deterministic single-run experiment from fixture snapshot to reports;
- repeated identical run producing identical core metrics and identity;
- bounded multi-trial Optuna study and resume;
- interrupted-study recovery;
- incompatible comparison rejection;
- dirty-worktree provenance behavior;
- tracking degradation with preserved canonical evidence; and
- CLI validation, execution, inspection, comparison, and reporting.

### 10.3 Acceptance

- Ruff and strict mypy pass;
- unit and integration branch coverage remains at least 90%;
- existing 430 unit/integration tests remain green;
- PostgreSQL contracts remain green;
- the offline demo remains deterministic;
- two identical experiment runs produce matching canonical metrics;
- a tuning study can be interrupted and resumed;
- the balanced example completes within the intended 30-to-60-minute budget on
  the target Mac under normal load;
- local MLflow can inspect the run and its artifacts;
- reports can be read without MLflow; and
- no secrets, experiment databases, MLflow state, or generated reports are
  tracked by git.

## 11. Delivery Sequence

1. Introduce dependencies, storage layout, and immutable specification models.
2. Extract stable configurable boundaries from the current ridge and
   walk-forward implementation without changing default behavior.
3. Implement canonical run records, provenance, and atomic publication.
4. Build deterministic single-run orchestration and reports.
5. Add local MLflow tracking behind `RunTracker`.
6. Add bounded Optuna search, persistent studies, and resume protection.
7. Add comparison and inspection services.
8. Expose the CLI and example specifications.
9. Run regression, reproducibility, performance-budget, and operator
   acceptance checks.
10. Refresh README, runbooks, project status, and stale MVP completion notes.

## 12. Repository Hygiene

The implementation will:

- add `data/experiments/` and local MLflow/Optuna files to `.gitignore`;
- preserve `.claude/` as local tooling state unless separately cleaned up;
- remove stale branch/worktree claims from `MVP_PROGRESS.md`;
- avoid committing runtime databases, datasets, reports, credentials, or
  broker responses; and
- keep `live_trading_enabled=False` as an enforced configuration contract.

