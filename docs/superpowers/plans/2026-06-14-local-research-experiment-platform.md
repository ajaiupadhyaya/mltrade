# Local Research Experiment Platform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible, local-first experiment registry that runs and tunes the existing ridge research pipeline through versioned TOML specifications, persistent Optuna studies, local MLflow tracking, durable reports, and a CLI.

**Architecture:** Introduce `mltrade.experiments` as an orchestration layer over the existing immutable snapshot, feature, ridge, portfolio, and backtest code. First expose typed model/backtest configuration without changing current defaults, then add canonical experiment records and atomic local storage, and finally layer tracking, tuning, comparison, reports, and CLI commands over those records.

**Tech Stack:** Python 3.13, Pydantic v2, standard-library `tomllib`, Optuna with SQLite RDB storage, MLflow local file tracking, scikit-learn Ridge, Typer, Pytest, Ruff, strict mypy.

**Approved design:** `docs/superpowers/specs/2026-06-14-local-research-experiment-platform-design.md`

---

## Scope and Delivery Order

This plan deliberately keeps the first release to the existing
`ridge-trend-v1` model family:

1. dependencies and repository hygiene;
2. immutable TOML experiment contracts;
3. configurable ridge and backtest boundaries with unchanged defaults;
4. canonical run records, provenance, and atomic storage;
5. deterministic single-run orchestration and reports;
6. local MLflow tracking;
7. persistent Optuna tuning and resume protection;
8. compatible run comparison;
9. CLI and tracked example specifications; and
10. full acceptance, documentation, and stale-status cleanup.

New model families, strategy sleeves, notebooks, remote tracking, scheduling,
and model promotion remain excluded.

## File Structure

```text
src/mltrade/
├── backtest/
│   ├── engine.py                    # Accept BacktestConfig and emit windows
│   └── reporting.py                 # WindowSummary and robust metrics
├── models/
│   ├── forecasts.py                 # Ridge model version/config metadata
│   └── walk_forward.py              # Accept RidgeForecastConfig
├── experiments/
│   ├── __init__.py                  # Public experiment API
│   ├── specs.py                     # Immutable experiment/TOML contracts
│   ├── loading.py                   # TOML loading and canonical hashing
│   ├── records.py                   # Run/trial/metric/artifact value objects
│   ├── provenance.py                # Git/runtime/dependency provenance
│   ├── storage.py                   # Atomic canonical run persistence
│   ├── reporting.py                 # Markdown and JSON reports
│   ├── runner.py                    # Single-run orchestration
│   ├── tracking.py                  # RunTracker and local MLflow adapter
│   ├── search.py                    # Typed Optuna search-space sampling
│   ├── tuning.py                    # Persistent study lifecycle
│   └── comparison.py                # Compatibility and ranking
├── cli.py                           # `experiment` Typer command group
└── config.py                        # Experiment-root derived settings

experiments/
├── ridge-baseline.toml
└── ridge-balanced-search.toml

tests/
├── unit/
│   ├── test_experiment_specs.py
│   ├── test_experiment_loading.py
│   ├── test_experiment_records.py
│   ├── test_experiment_provenance.py
│   ├── test_experiment_storage.py
│   ├── test_experiment_reporting.py
│   ├── test_experiment_tracking.py
│   ├── test_experiment_search.py
│   ├── test_experiment_tuning.py
│   └── test_experiment_comparison.py
└── integration/
    ├── test_experiment_runner.py
    └── test_experiment_cli.py
```

## Task 1: Add Experiment Dependencies, Paths, and Hygiene

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `.gitignore`
- Modify: `.env.example`
- Modify: `src/mltrade/config.py`
- Modify: `tests/conftest.py`
- Modify: `tests/unit/test_config.py`
- Modify: `tests/integration/test_repository_hygiene.py`
- Create: `src/mltrade/experiments/__init__.py`

- [ ] **Step 1: Write failing configuration and hygiene tests**

Add:

```python
def test_experiment_paths_default_under_data_root(tmp_path: Path) -> None:
    settings = Settings(data_root=tmp_path)

    assert settings.experiment_root == tmp_path / "experiments"
    assert settings.mlflow_tracking_root == tmp_path / "experiments" / "mlflow"
    assert settings.optuna_storage_path == tmp_path / "experiments" / "optuna" / "studies.db"


def test_explicit_experiment_root_is_absolute(tmp_path: Path) -> None:
    settings = Settings(data_root=tmp_path, experiment_root=tmp_path / "research")

    assert settings.experiment_root == (tmp_path / "research").resolve()
```

Extend the repository hygiene assertion to require:

```python
for entry in ("data/", "artifacts/", "mlruns/", "*.db", ".claude/"):
    assert entry in ignored
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/unit/test_config.py \
  tests/integration/test_repository_hygiene.py -v
```

Expected: failures for missing experiment settings and missing `.claude/`
ignore entry.

- [ ] **Step 3: Add dependencies and derived paths**

Add runtime dependencies:

```toml
"mlflow>=3.1,<4",
"optuna>=4.4,<5",
```

Add to `Settings`:

```python
experiment_root: Path | None = None

@model_validator(mode="after")
def validate_safety_constraints(self) -> "Settings":
    if self.live_trading_enabled:
        raise ValueError("live trading is not available in this release")
    if (
        self.environment is Environment.PAPER
        and self.alpaca_base_url.rstrip("/")
        != "https://paper-api.alpaca.markets"
    ):
        raise ValueError(
            "paper environment requires "
            "https://paper-api.alpaca.markets as alpaca_base_url"
        )
    if self.maximum_position_weight > Decimal("1") - self.minimum_cash_weight:
        raise ValueError(
            "maximum_position_weight cannot exceed 1 - minimum_cash_weight"
        )
    if self.experiment_root is None:
        object.__setattr__(
            self,
            "experiment_root",
            (self.data_root / "experiments").resolve(),
        )
    else:
        object.__setattr__(
            self,
            "experiment_root",
            self.experiment_root.expanduser().resolve(),
        )
    return self

@property
def mlflow_tracking_root(self) -> Path:
    assert self.experiment_root is not None
    return self.experiment_root / "mlflow"

@property
def optuna_storage_path(self) -> Path:
    assert self.experiment_root is not None
    return self.experiment_root / "optuna" / "studies.db"
```

Keep `live_trading_enabled=False` and all current paper URL checks unchanged.
Add `MLTRADE_EXPERIMENT_ROOT=` to `.env.example` and clear it in the autouse
test fixture.

Add `.claude/` to `.gitignore`. Existing untracked agent worktrees must not be
deleted as part of this task.

- [ ] **Step 4: Lock and verify**

Run:

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv lock
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/unit/test_config.py \
  tests/integration/test_repository_hygiene.py -v
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run ruff check .
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run mypy src
```

Expected: all commands pass.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock .gitignore .env.example \
  src/mltrade/config.py src/mltrade/experiments/__init__.py \
  tests/conftest.py tests/unit/test_config.py \
  tests/integration/test_repository_hygiene.py
git commit -m "build: add local experiment platform dependencies"
```

## Task 2: Define Immutable Experiment Specifications

**Files:**
- Create: `src/mltrade/experiments/specs.py`
- Create: `src/mltrade/experiments/loading.py`
- Create: `tests/unit/test_experiment_specs.py`
- Create: `tests/unit/test_experiment_loading.py`

- [ ] **Step 1: Write failing specification tests**

Create tests covering this exact baseline:

```python
def test_baseline_spec_is_immutable_and_explicit() -> None:
    spec = ExperimentSpec(
        name="ridge-baseline",
        dataset=DatasetSpec(
            name="daily_bars",
            snapshot_id="fixture-2026-06-12",
            universe_version="mvp-etf-v1",
            feature_version="trend-momentum-v1",
        ),
    )

    assert spec.schema_version == 1
    assert spec.model.family == "ridge"
    assert spec.model.alpha == 1.0
    assert spec.validation.minimum_training_sessions == 504
    assert spec.validation.embargo_sessions == 21
    assert spec.validation.retrain_every_sessions == 21
    assert spec.costs.headline_bps == Decimal("5")
    assert spec.objective.name == "robust_sharpe"

    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate(
            {**spec.model_dump(), "unexpected": True}
        )
```

Add validation tests:

```python
def test_spec_requires_exact_snapshot_id() -> None:
    with pytest.raises(ValidationError, match="snapshot_id"):
        DatasetSpec(
            name="daily_bars",
            snapshot_id="latest",
            universe_version="mvp-etf-v1",
            feature_version="trend-momentum-v1",
        )


def test_balanced_budget_rejects_unsafe_parallelism() -> None:
    with pytest.raises(ValidationError, match="worker_count"):
        ResourceBudget(max_trials=40, timeout_minutes=60, worker_count=8)
```

- [ ] **Step 2: Write failing TOML loading tests**

Use a temporary file:

```python
def test_load_spec_canonicalizes_and_hashes(tmp_path: Path) -> None:
    path = tmp_path / "baseline.toml"
    path.write_text(BASELINE_TOML, encoding="utf-8")

    loaded = load_experiment_spec(path)

    assert loaded.spec.name == "ridge-baseline"
    assert len(loaded.spec_sha256) == 64
    assert loaded.canonical_json.endswith("\n")
    assert load_experiment_spec(path).spec_sha256 == loaded.spec_sha256
```

Also assert malformed TOML, unknown keys, and missing files raise
`ExperimentSpecError` with the source path in the message.

- [ ] **Step 3: Run tests and verify import failures**

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/unit/test_experiment_specs.py \
  tests/unit/test_experiment_loading.py -v
```

Expected: import failures for the new modules.

- [ ] **Step 4: Implement the contracts**

Use frozen Pydantic models with `extra="forbid"`:

```python
class StrictFrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DatasetSpec(StrictFrozenModel):
    name: Literal["daily_bars"] = "daily_bars"
    snapshot_id: str = Field(pattern=r"^(?!latest$)[A-Za-z0-9_.:-]+$")
    universe_version: Literal["mvp-etf-v1"] = "mvp-etf-v1"
    feature_version: Literal["trend-momentum-v1"] = "trend-momentum-v1"


class RidgeModelSpec(StrictFrozenModel):
    family: Literal["ridge"] = "ridge"
    version: Literal["ridge-trend-v1"] = "ridge-trend-v1"
    alpha: float = Field(default=1.0, gt=0.0, le=10_000.0)
    fit_intercept: bool = True


class ValidationSpec(StrictFrozenModel):
    minimum_training_sessions: int = Field(default=504, ge=252, le=2520)
    embargo_sessions: int = Field(default=21, ge=1, le=126)
    retrain_every_sessions: int = Field(default=21, ge=1, le=126)


class CostSpec(StrictFrozenModel):
    headline_bps: Decimal = Field(default=Decimal("5"), ge=0, le=100)
    sensitivity_bps: tuple[Decimal, ...] = (
        Decimal("2"),
        Decimal("5"),
        Decimal("10"),
    )


class ObjectiveSpec(StrictFrozenModel):
    name: Literal["robust_sharpe"] = "robust_sharpe"
    maximum_drawdown: float = Field(default=-0.35, ge=-1.0, le=0.0)
    maximum_turnover: float = Field(default=1.0, ge=0.0)


class ResourceBudget(StrictFrozenModel):
    max_trials: int = Field(default=40, ge=1, le=500)
    timeout_minutes: int = Field(default=60, ge=1, le=720)
    worker_count: int = Field(default=1, ge=1, le=2)


class ExperimentSpec(StrictFrozenModel):
    schema_version: Literal[1] = 1
    name: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    description: str = ""
    dataset: DatasetSpec
    model: RidgeModelSpec = RidgeModelSpec()
    validation: ValidationSpec = ValidationSpec()
    costs: CostSpec = CostSpec()
    objective: ObjectiveSpec = ObjectiveSpec()
    resources: ResourceBudget = ResourceBudget()
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
```

Add portfolio fields mirroring `Settings` through a separate
`PortfolioSpec`; do not embed `Settings` or permit broker configuration.

Implement `load_experiment_spec()` using `tomllib.load()`, sorted compact JSON,
and SHA-256:

```python
class ExperimentSpecError(ValueError):
    pass


class LoadedExperimentSpec(NamedTuple):
    path: Path
    spec: ExperimentSpec
    canonical_json: str
    spec_sha256: str


canonical_json = json.dumps(
    spec.model_dump(mode="json"),
    sort_keys=True,
    separators=(",", ":"),
) + "\n"
```

- [ ] **Step 5: Verify and commit**

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/unit/test_experiment_specs.py \
  tests/unit/test_experiment_loading.py -v
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run mypy src
git add src/mltrade/experiments/specs.py \
  src/mltrade/experiments/loading.py \
  tests/unit/test_experiment_specs.py \
  tests/unit/test_experiment_loading.py
git commit -m "feat: define immutable experiment specifications"
```

## Task 3: Make Ridge and Backtest Configuration Explicit

**Files:**
- Modify: `src/mltrade/models/forecasts.py`
- Modify: `src/mltrade/models/walk_forward.py`
- Modify: `src/mltrade/models/__init__.py`
- Modify: `src/mltrade/backtest/engine.py`
- Modify: `src/mltrade/backtest/reporting.py`
- Modify: `src/mltrade/backtest/__init__.py`
- Modify: `src/mltrade/workflows/research.py`
- Test: `tests/unit/test_walk_forward.py`
- Test: `tests/integration/test_walk_forward_backtest.py`
- Test: `tests/integration/test_research_workflow.py`

- [ ] **Step 1: Write failing ridge-configuration tests**

Add:

```python
def test_ridge_config_defaults_preserve_existing_behavior() -> None:
    default_batch = generate_forecast_batch(_feature_rows, _DECISION_SESSION)
    explicit_batch = generate_forecast_batch(
        _feature_rows,
        _DECISION_SESSION,
        config=RidgeForecastConfig(),
    )

    assert explicit_batch == default_batch


def test_alpha_changes_forecasts_but_remains_deterministic() -> None:
    low = generate_forecast_batch(
        _feature_rows,
        _DECISION_SESSION,
        config=RidgeForecastConfig(alpha=0.01),
    )
    high = generate_forecast_batch(
        _feature_rows,
        _DECISION_SESSION,
        config=RidgeForecastConfig(alpha=100.0),
    )

    assert low != high
    assert high == generate_forecast_batch(
        _feature_rows,
        _DECISION_SESSION,
        config=RidgeForecastConfig(alpha=100.0),
    )
```

- [ ] **Step 2: Write failing backtest-configuration tests**

Add:

```python
def test_backtest_config_defaults_preserve_existing_result(bars: tuple[DailyBar, ...]) -> None:
    original = run_backtest(bars, limits=LIMITS)
    explicit = run_backtest(
        bars,
        limits=LIMITS,
        config=BacktestConfig(),
    )

    assert explicit == original


def test_backtest_emits_deterministic_evaluation_windows(
    bars: tuple[DailyBar, ...],
) -> None:
    result = run_backtest(bars, limits=LIMITS, config=BacktestConfig())

    assert result.evaluation_windows
    assert result.evaluation_windows == run_backtest(
        bars,
        limits=LIMITS,
        config=BacktestConfig(),
    ).evaluation_windows
```

- [ ] **Step 3: Run focused tests and verify failure**

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/unit/test_walk_forward.py \
  tests/integration/test_walk_forward_backtest.py \
  tests/integration/test_research_workflow.py -q
```

Expected: failures for missing config types and evaluation windows.

- [ ] **Step 4: Implement typed configuration**

Add:

```python
class RidgeForecastConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    alpha: float = Field(default=1.0, gt=0.0)
    fit_intercept: bool = True
    minimum_training_sessions: int = Field(default=504, ge=1)
    embargo_sessions: int = Field(default=21, ge=1)


class BacktestConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    forecast: RidgeForecastConfig = RidgeForecastConfig()
    retrain_every_sessions: int = Field(default=21, ge=1)
    cost_bps: Decimal = Field(default=Decimal("5"), ge=0)
    cost_sensitivity_bps: tuple[Decimal, ...] = (
        Decimal("2"),
        Decimal("5"),
        Decimal("10"),
    )
    evaluation_window_sessions: int = Field(default=252, ge=63)
```

Change the ridge construction to:

```python
model = Ridge(
    alpha=config.alpha,
    fit_intercept=config.fit_intercept,
)
```

Use config values for minimum history, embargo, retraining cadence, headline
cost, and sensitivity levels. Keep old keyword arguments temporarily only
where required for compatibility, and route them into `BacktestConfig`.

Add:

```python
class EvaluationWindow(BaseModel):
    model_config = ConfigDict(frozen=True)

    start_session: date
    end_session: date
    sessions: int
    annualized_return: float
    annualized_volatility: float
    sharpe: float
    max_drawdown: float
```

Build non-overlapping 252-session windows from the headline equity curve.
Append `evaluation_windows: tuple[EvaluationWindow, ...]` to `BacktestResult`.

Update `run_research()` to accept an optional `backtest_config` and pass it
through. Existing calls must remain byte-for-byte behaviorally equivalent.

- [ ] **Step 5: Run regression checks and commit**

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/unit/test_walk_forward.py \
  tests/integration/test_walk_forward_backtest.py \
  tests/integration/test_research_workflow.py -q
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/unit tests/integration -q
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run mypy src
git add src/mltrade/models src/mltrade/backtest \
  src/mltrade/workflows/research.py \
  tests/unit/test_walk_forward.py \
  tests/integration/test_walk_forward_backtest.py \
  tests/integration/test_research_workflow.py
git commit -m "refactor: expose configurable ridge research boundaries"
```

Expected: all 430 existing tests plus new tests pass.

## Task 4: Add Canonical Run Records, Provenance, and Atomic Storage

**Files:**
- Create: `src/mltrade/experiments/records.py`
- Create: `src/mltrade/experiments/provenance.py`
- Create: `src/mltrade/experiments/storage.py`
- Create: `tests/unit/test_experiment_records.py`
- Create: `tests/unit/test_experiment_provenance.py`
- Create: `tests/unit/test_experiment_storage.py`

- [ ] **Step 1: Write failing record and identity tests**

```python
def test_run_identity_is_content_addressed() -> None:
    context = RunIdentityContext(
        spec_sha256="a" * 64,
        dataset_sha256="b" * 64,
        git_commit="c" * 40,
        git_diff_sha256=None,
    )

    assert build_run_id(context) == build_run_id(context)
    assert build_run_id(context).startswith("run-")


def test_terminal_status_is_closed_set() -> None:
    with pytest.raises(ValidationError):
        ExperimentRunRecord(
            **BASE_RECORD,
            status="running",
        )
```

Use statuses `complete`, `blocked`, `pruned`, and `failed`; transient running
state belongs only in Optuna/MLflow, not canonical completed records.

- [ ] **Step 2: Write failing provenance tests**

```python
def test_capture_provenance_records_dirty_diff_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provenance, "_git_commit", lambda _: "a" * 40)
    monkeypatch.setattr(provenance, "_git_diff", lambda _: "changed\n")

    result = capture_provenance(tmp_path, command=("mltrade", "experiment", "run"))

    assert result.git_dirty is True
    assert result.git_diff_sha256 == hashlib.sha256(b"changed\n").hexdigest()
    assert "python_version" in result.model_dump()
```

Dependency capture must include only package names and versions, never
environment variables or credentials.

- [ ] **Step 3: Write failing atomic-storage tests**

```python
def test_run_store_publishes_atomically(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    path = store.save(RECORD)

    assert path == tmp_path / "runs" / RECORD.run_id / "run.json"
    assert store.load(RECORD.run_id) == RECORD
    assert not list((tmp_path / "runs").glob(".*.tmp-*"))


def test_same_run_id_with_different_content_is_rejected(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    store.save(RECORD)

    with pytest.raises(RunStorageError, match="different content"):
        store.save(RECORD.model_copy(update={"metrics": OTHER_METRICS}))
```

- [ ] **Step 4: Implement the record model**

Define:

```python
JsonScalar = str | int | float | bool | None


class ArtifactRecord(StrictFrozenModel):
    relative_path: str
    sha256: str
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
    status: Literal["complete", "blocked", "pruned", "failed"]
    spec_sha256: str
    dataset_sha256: str
    dataset_snapshot_id: str
    compatibility_key: str
    seed: int
    started_at: datetime
    finished_at: datetime
    provenance: RunProvenance
    parameters: dict[str, JsonScalar | list[JsonScalar] | dict[str, JsonScalar]]
    metrics: RunMetrics | None
    artifacts: tuple[ArtifactRecord, ...]
    failure: FailureRecord | None = None
    study_name: str | None = None
    trial_number: int | None = None
    tracking_status: Literal["pending", "logged", "degraded"] = "pending"


def build_run_id(context: RunIdentityContext) -> str:
    payload = json.dumps(
        context.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return f"run-{hashlib.sha256(payload).hexdigest()[:20]}"
```

Use canonical JSON SHA-256 for the identity and compatibility key. Include the
dirty diff hash in run identity so uncommitted code changes cannot collide.

- [ ] **Step 5: Implement safe provenance and atomic storage**

Use non-shell `subprocess.run()` calls:

```python
subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=repo_root,
    check=True,
    capture_output=True,
    text=True,
)
```

Capture `git diff --binary HEAD` for the dirty hash but never store the diff
body in the run record.

Write `run.json`, `spec.json`, and artifacts under a sibling temporary
directory, `fsync` files, then publish with `Path.replace()`. If an identical
record already exists, return it idempotently. Reject symlinked run directories
and differing content at an existing run ID.

- [ ] **Step 6: Verify and commit**

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/unit/test_experiment_records.py \
  tests/unit/test_experiment_provenance.py \
  tests/unit/test_experiment_storage.py -v
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run mypy src
git add src/mltrade/experiments/records.py \
  src/mltrade/experiments/provenance.py \
  src/mltrade/experiments/storage.py \
  tests/unit/test_experiment_records.py \
  tests/unit/test_experiment_provenance.py \
  tests/unit/test_experiment_storage.py
git commit -m "feat: persist canonical experiment run records"
```

## Task 5: Build the Deterministic Experiment Runner and Reports

**Files:**
- Create: `src/mltrade/experiments/runner.py`
- Create: `src/mltrade/experiments/reporting.py`
- Create: `tests/unit/test_experiment_reporting.py`
- Create: `tests/integration/test_experiment_runner.py`

- [ ] **Step 1: Write a failing end-to-end runner test**

Publish fixture data through the existing demo, then run:

```python
def test_runner_produces_deterministic_canonical_record(tmp_path: Path) -> None:
    settings, manifest = published_fixture(tmp_path)
    loaded = load_experiment_spec(write_baseline_spec(tmp_path, manifest.snapshot_id))
    runner = ExperimentRunner(settings=settings, tracker=NullRunTracker())

    first = runner.run(loaded)
    second = runner.run(loaded)

    assert first.record.run_id == second.record.run_id
    assert first.record.metrics == second.record.metrics
    assert first.record.dataset_sha256 == manifest.content_sha256
    assert first.report_markdown.read_text() == second.report_markdown.read_text()
```

Also test:

```python
def test_runner_blocks_manifest_context_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, manifest = published_fixture(tmp_path)
    path = write_baseline_spec(tmp_path, manifest.snapshot_id)
    loaded = load_experiment_spec(path)
    bad_manifest = DatasetManifest.model_validate(
        {
            **manifest.model_dump(),
            "metadata": {
                **manifest.metadata,
                "universe_version": "unsupported-v2",
            },
        }
    )
    monkeypatch.setattr(
        SnapshotStore,
        "load_manifest",
        lambda self, dataset, snapshot_id: bad_manifest,
    )
    runner = ExperimentRunner(settings=settings, tracker=NullRunTracker())

    with pytest.raises(ExperimentBlocked, match="universe_version"):
        runner.run(loaded)
```

- [ ] **Step 2: Write failing objective/report tests**

Define the default objective exactly:

```python
robust_sharpe = min(sharpe_at_5_bps, sharpe_at_10_bps) - window_sharpe_std
```

Constraint violations set status `blocked` and exclude the run from ranking:

```python
def test_objective_uses_cost_and_window_stability() -> None:
    metrics = build_run_metrics(BACKTEST_RESULT)

    expected = min(
        BACKTEST_RESULT.cost_sensitivity[Decimal("5")].sharpe,
        BACKTEST_RESULT.cost_sensitivity[Decimal("10")].sharpe,
    ) - statistics.pstdev(
        window.sharpe for window in BACKTEST_RESULT.evaluation_windows
    )
    assert metrics.robust_sharpe == round(expected, 10)
```

Report assertions:

```python
assert "# MLTrade Experiment Report" in markdown
assert "Dataset snapshot" in markdown
assert "Dirty worktree" in markdown
assert "Robust Sharpe" in markdown
assert "Not a promotion decision" in markdown
```

- [ ] **Step 3: Implement runner orchestration**

`ExperimentRunner.run()` must:

1. load the exact manifest with `SnapshotStore.load_manifest()`;
2. verify dataset, content hash, universe version, and feature version;
3. capture provenance and build the deterministic run ID;
4. return the existing canonical run idempotently when that exact run ID is
   already complete;
5. map `ExperimentSpec` into `PortfolioLimits`, `RidgeForecastConfig`, and
   `BacktestConfig`;
6. call `run_research(settings, manifest, backtest_config=config)`;
7. reject blocked quality, optimizer, non-finite metrics, or objective
   constraint violations;
8. calculate `RunMetrics` and compatibility key;
9. save the canonical record atomically;
10. generate `report.md` and `report.json`; and
11. call the tracker only after canonical storage succeeds.

The idempotent load in step 4 is required because timestamps are evidence in
the canonical record but are not part of run identity. Re-running identical
code/spec/data returns the first completed record instead of attempting to
publish a second record with different timestamps.

Define the runner result and domain errors explicitly:

```python
class ExperimentBlocked(RuntimeError):
    pass


class ExperimentFailed(RuntimeError):
    pass


class ExperimentTrackingError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ExperimentRunResult:
    record: ExperimentRunRecord
    run_directory: Path
    report_markdown: Path
    report_json: Path
```

Use this compatibility payload:

```python
{
    "dataset_sha256": manifest.content_sha256,
    "snapshot_id": manifest.snapshot_id,
    "universe_version": spec.dataset.universe_version,
    "feature_version": spec.dataset.feature_version,
    "forecast_horizon": 21,
    "minimum_training_sessions": spec.validation.minimum_training_sessions,
    "embargo_sessions": spec.validation.embargo_sessions,
    "retrain_every_sessions": spec.validation.retrain_every_sessions,
    "portfolio": spec.portfolio.model_dump(mode="json"),
    "headline_cost_bps": str(spec.costs.headline_bps),
    "objective": spec.objective.model_dump(mode="json"),
}
```

- [ ] **Step 4: Implement durable reports**

`ReportBuilder` writes JSON from the canonical record and Markdown with:

- identity and terminal status;
- exact dataset and code provenance;
- resolved parameters;
- headline, sensitivity, baseline, contribution, and window metrics;
- methodological compatibility key;
- dirty-worktree warning;
- blocked/failure reason; and
- the explicit statement that the report is not a promotion decision.

Do not include credentials, environment variable values, or database URLs.

- [ ] **Step 5: Verify and commit**

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/unit/test_experiment_reporting.py \
  tests/integration/test_experiment_runner.py -v
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/unit tests/integration -q
git add src/mltrade/experiments/runner.py \
  src/mltrade/experiments/reporting.py \
  tests/unit/test_experiment_reporting.py \
  tests/integration/test_experiment_runner.py
git commit -m "feat: run and report deterministic ridge experiments"
```

## Task 6: Add Local MLflow Tracking Behind an Interface

**Files:**
- Create: `src/mltrade/experiments/tracking.py`
- Create: `tests/unit/test_experiment_tracking.py`
- Modify: `src/mltrade/experiments/runner.py`
- Modify: `tests/integration/test_experiment_runner.py`

- [ ] **Step 1: Write failing tracker tests**

```python
def test_local_mlflow_tracker_logs_record_and_artifacts(tmp_path: Path) -> None:
    tracker = MlflowRunTracker(tmp_path / "mlflow")
    tracker.log(RECORD, artifact_dir=ARTIFACT_DIR)

    client = MlflowClient(tracking_uri=(tmp_path / "mlflow").resolve().as_uri())
    runs = client.search_runs(experiment_ids=["0"])

    assert any(run.data.tags["mltrade.run_id"] == RECORD.run_id for run in runs)
```

Also test that `NullRunTracker.log()` is a no-op and that a tracker exception
returns a copied canonical record with `tracking_status="degraded"` while
leaving the original run files present.

- [ ] **Step 2: Implement the tracker protocol**

```python
class RunTracker(Protocol):
    def log(self, record: ExperimentRunRecord, artifact_dir: Path) -> str:
        """Return the external tracking run ID."""


class NullRunTracker:
    def log(self, record: ExperimentRunRecord, artifact_dir: Path) -> str:
        return ""
```

Configure local MLflow using current supported APIs:

```python
tracking_uri = tracking_root.resolve().as_uri()
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(record.experiment_name)
with mlflow.start_run(run_name=record.run_id) as active:
    mlflow.log_params(flatten_parameters(record.parameters))
    mlflow.log_metrics(record.metrics.model_dump() if record.metrics else {})
    mlflow.set_tags({
        "mltrade.run_id": record.run_id,
        "mltrade.snapshot_id": record.dataset_snapshot_id,
        "mltrade.compatibility_key": record.compatibility_key,
        "mltrade.git_dirty": str(record.provenance.git_dirty).lower(),
    })
    mlflow.log_artifacts(str(artifact_dir), artifact_path="research")
```

Limit parameter values to MLflow-compatible scalar strings and cap long
serialized values by logging their hash plus the full resolved spec artifact.

- [ ] **Step 3: Integrate degradation handling**

After canonical save:

```python
try:
    tracking_run_id = self._tracker.log(record, artifact_dir)
except Exception as exc:
    degraded = record.model_copy(
        update={
            "tracking_status": "degraded",
            "failure": FailureRecord(
                category="tracking",
                message=str(exc),
            ),
        }
    )
    self._store.replace_tracking_state(degraded)
    raise ExperimentTrackingError(degraded.run_id) from exc
```

Do not catch `BaseException`. Do not delete canonical evidence.

- [ ] **Step 4: Verify and commit**

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/unit/test_experiment_tracking.py \
  tests/integration/test_experiment_runner.py -v
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run mypy src
git add src/mltrade/experiments/tracking.py \
  src/mltrade/experiments/runner.py \
  tests/unit/test_experiment_tracking.py \
  tests/integration/test_experiment_runner.py
git commit -m "feat: track local experiments with mlflow"
```

## Task 7: Add Persistent Optuna Search and Resume Protection

**Files:**
- Modify: `src/mltrade/experiments/specs.py`
- Create: `src/mltrade/experiments/search.py`
- Create: `src/mltrade/experiments/tuning.py`
- Create: `tests/unit/test_experiment_search.py`
- Create: `tests/unit/test_experiment_tuning.py`
- Modify: `tests/integration/test_experiment_runner.py`

- [ ] **Step 1: Write failing search-space tests**

Add typed search fields:

```python
class FloatSearchSpec(StrictFrozenModel):
    low: float
    high: float
    log: bool = False

    @model_validator(mode="after")
    def validate_bounds(self) -> "FloatSearchSpec":
        if self.low >= self.high:
            raise ValueError("low must be less than high")
        if self.log and self.low <= 0:
            raise ValueError("log search requires low > 0")
        return self


class RidgeSearchSpace(StrictFrozenModel):
    alpha: FloatSearchSpec = FloatSearchSpec(
        low=0.001,
        high=1000.0,
        log=True,
    )
    minimum_training_sessions: tuple[int, ...] = (504, 756, 1008)
    retrain_every_sessions: tuple[int, ...] = (5, 10, 21, 42)
```

Test deterministic sampling with `TPESampler(seed=42)` and a fixed one-trial
study.

Add `search: RidgeSearchSpace | None = None` to `ExperimentSpec`. Baseline
specifications leave it unset; tuning commands require it.

- [ ] **Step 2: Write failing persistent-study tests**

```python
def test_study_resumes_completed_trials(tmp_path: Path) -> None:
    tuner = OptunaTuner(storage_path=tmp_path / "studies.db", runner=FAKE_RUNNER)
    first = tuner.tune(LOADED_SPEC, study_name="ridge-test", n_trials=2)
    second = tuner.tune(LOADED_SPEC, study_name="ridge-test", n_trials=1)

    assert first.completed_trials == 2
    assert second.completed_trials == 3


def test_resume_rejects_immutable_context_drift(tmp_path: Path) -> None:
    tuner.tune(BASELINE, study_name="ridge-test", n_trials=1)

    with pytest.raises(StudyContextMismatch, match="context hash"):
        tuner.tune(CHANGED_SNAPSHOT, study_name="ridge-test", n_trials=1)
```

- [ ] **Step 3: Implement search sampling**

```python
def sample_ridge_trial(
    trial: optuna.Trial,
    base: ExperimentSpec,
    search: RidgeSearchSpace,
) -> ExperimentSpec:
    alpha = trial.suggest_float(
        "model.alpha",
        search.alpha.low,
        search.alpha.high,
        log=search.alpha.log,
    )
    minimum_training_sessions = trial.suggest_categorical(
        "validation.minimum_training_sessions",
        list(search.minimum_training_sessions),
    )
    retrain_every_sessions = trial.suggest_categorical(
        "validation.retrain_every_sessions",
        list(search.retrain_every_sessions),
    )
    return base.model_copy(
        update={
            "model": base.model.model_copy(update={"alpha": alpha}),
            "validation": base.validation.model_copy(
                update={
                    "minimum_training_sessions": minimum_training_sessions,
                    "retrain_every_sessions": retrain_every_sessions,
                }
            ),
        }
    )
```

This is the one deliberate use of validated immutable-copy updates in the
experiment layer; immediately revalidate with
`ExperimentSpec.model_validate(candidate.model_dump())`.

- [ ] **Step 4: Implement persistent tuning**

Use:

```python
storage_uri = f"sqlite:///{storage_path.resolve()}"
study = optuna.create_study(
    study_name=study_name,
    storage=storage_uri,
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=spec.seed),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    load_if_exists=True,
)
```

Set and verify:

```python
context_hash = build_study_context_hash(spec)
existing = study.user_attrs.get("mltrade.context_hash")
if existing is None:
    study.set_user_attr("mltrade.context_hash", context_hash)
elif existing != context_hash:
    raise StudyContextMismatch(
        f"study context hash mismatch: stored={existing}, current={context_hash}"
    )
```

Run:

```python
study.optimize(
    objective,
    n_trials=remaining_trials,
    timeout=spec.resources.timeout_minutes * 60,
    n_jobs=spec.resources.worker_count,
    catch=(ExperimentBlocked, ExperimentFailed),
)
```

For SQLite, default example specs use `worker_count=1`. Permit `2` only as an
explicit local override. Each trial seed is
`sha256(f"{spec.seed}:{trial.number}") mod 2**32`.

Return a `TuningResult` containing counts by Optuna trial state, best run ID,
best value, study name, storage path, and elapsed seconds. Materialize the
best trial through the normal runner and generate a study report.

Use:

```python
class StudyContextMismatch(RuntimeError):
    pass


class TuningResult(StrictFrozenModel):
    study_name: str
    storage_path: Path
    completed_trials: int
    pruned_trials: int
    failed_trials: int
    best_run_id: str | None
    best_value: float | None
    elapsed_seconds: float
```

- [ ] **Step 5: Verify and commit**

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/unit/test_experiment_search.py \
  tests/unit/test_experiment_tuning.py \
  tests/integration/test_experiment_runner.py -v
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run mypy src
git add src/mltrade/experiments/specs.py \
  src/mltrade/experiments/search.py \
  src/mltrade/experiments/tuning.py \
  tests/unit/test_experiment_search.py \
  tests/unit/test_experiment_tuning.py \
  tests/integration/test_experiment_runner.py
git commit -m "feat: tune ridge experiments with resumable optuna studies"
```

## Task 8: Add Compatibility-Aware Run Comparison

**Files:**
- Create: `src/mltrade/experiments/comparison.py`
- Create: `tests/unit/test_experiment_comparison.py`

- [ ] **Step 1: Write failing comparison tests**

```python
def test_compatible_runs_rank_by_robust_sharpe() -> None:
    result = compare_runs((LOW_SCORE, HIGH_SCORE))

    assert result.compatible is True
    assert result.ranking[0].run_id == HIGH_SCORE.run_id


def test_incompatible_runs_have_no_winner() -> None:
    result = compare_runs((BASELINE, DIFFERENT_DATASET))

    assert result.compatible is False
    assert result.ranking == ()
    assert "dataset_sha256" in result.differences


def test_dirty_runs_are_excluded_by_default() -> None:
    result = compare_runs((CLEAN, DIRTY))

    assert [item.run_id for item in result.ranking] == [CLEAN.run_id]
    assert DIRTY.run_id in result.excluded_run_ids
```

- [ ] **Step 2: Implement deterministic comparison**

Define:

```python
class RankedRun(StrictFrozenModel):
    rank: int = Field(ge=1)
    run_id: str
    robust_sharpe: float
    sharpe_10_bps: float
    max_drawdown: float
    turnover: float


class ComparisonResult(StrictFrozenModel):
    run_ids: tuple[str, ...]
    compatible: bool
    compatibility_key: str | None
    differences: dict[str, tuple[str, ...]]
    ranking: tuple[RankedRun, ...]
    excluded_run_ids: tuple[str, ...]
```

Sort eligible complete runs by:

1. descending `robust_sharpe`;
2. descending 10 bps Sharpe;
3. less-negative maximum drawdown;
4. lower turnover; and
5. run ID for deterministic ties.

Blocked, failed, pruned, degraded-tracking, or dirty runs remain visible but
are excluded by default. Flags may include them, but incompatible runs never
produce a winner.

- [ ] **Step 3: Verify and commit**

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/unit/test_experiment_comparison.py -v
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run mypy src
git add src/mltrade/experiments/comparison.py \
  tests/unit/test_experiment_comparison.py
git commit -m "feat: compare compatible experiment runs"
```

## Task 9: Expose the Experiment CLI and Example Specs

**Files:**
- Modify: `src/mltrade/cli.py`
- Modify: `src/mltrade/experiments/__init__.py`
- Create: `experiments/ridge-baseline.toml`
- Create: `experiments/ridge-balanced-search.toml`
- Create: `tests/integration/test_experiment_cli.py`

- [ ] **Step 1: Write failing CLI tests**

```python
def test_experiment_validate_prints_hash(
    published_spec: Path,
) -> None:
    result = runner.invoke(app, ["experiment", "validate", str(published_spec)])

    assert result.exit_code == 0, result.stdout
    assert "spec: valid" in result.stdout
    assert "snapshot:" in result.stdout
    assert "spec sha256:" in result.stdout


def test_experiment_run_prints_run_and_reports(
    published_spec: Path,
) -> None:
    result = runner.invoke(app, ["experiment", "run", str(published_spec)])

    assert result.exit_code == 0, result.stdout
    assert "run id:" in result.stdout
    assert "report:" in result.stdout


def test_compare_incompatible_runs_exits_nonzero(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["experiment", "compare", RUN_A, RUN_B],
    )

    assert result.exit_code != 0
    assert "incompatible" in result.stdout.lower()
```

Cover `init`, `validate`, `run`, `tune` with a 1-trial override, `resume`,
`list`, `inspect`, `compare`, `report`, `doctor`, and `--json`.

- [ ] **Step 2: Add the Typer group**

Register:

```python
experiment_app = typer.Typer(
    no_args_is_help=True,
    help="Run, tune, inspect, and compare reproducible research experiments.",
)
app.add_typer(experiment_app, name="experiment")
```

Keep CLI functions thin. Construct settings, stores, tracker, runner, or tuner,
then delegate.

Required command behavior:

- `init [DIRECTORY]`: write the two packaged examples without overwriting;
- `validate SPEC`: parse, load manifest, and verify immutable context only;
- `run SPEC`: run once and print run/report IDs;
- `tune SPEC [--trials N] [--timeout-minutes N]`: bounded override only;
- `resume STUDY --spec SPEC`: require the original spec and context hash;
- `list`: sorted newest-finished first;
- `inspect RUN_ID`: summary or canonical JSON;
- `compare RUN_ID...`: nonzero on incompatible ranking request;
- `report RUN_ID`: regenerate reports from the canonical record;
- `doctor`: verify directories, exact example snapshot availability, MLflow
  file URI, Optuna SQLite parent, and dependency imports.

Catch domain exceptions and convert them to concise nonzero CLI exits. Do not
print tracebacks or secrets in normal operation.

- [ ] **Step 3: Add tracked example specifications**

`ridge-baseline.toml` uses exact safe defaults and the deterministic fixture
snapshot ID. `experiment init` replaces it only when the caller supplies
`--snapshot-id`. The tracked file itself uses:

```toml
schema_version = 1
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
```

The balanced search uses 40 trials, 60 minutes, one worker, and the ridge
search space from Task 7:

```toml
[search]
minimum_training_sessions = [504, 756, 1008]
retrain_every_sessions = [5, 10, 21, 42]

[search.alpha]
low = 0.001
high = 1000.0
log = true
```

- [ ] **Step 4: Verify and commit**

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/integration/test_experiment_cli.py -v
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run mltrade experiment --help
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run ruff check .
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run mypy src
git add src/mltrade/cli.py src/mltrade/experiments/__init__.py \
  experiments tests/integration/test_experiment_cli.py
git commit -m "feat: expose local experiment CLI"
```

## Task 10: Acceptance, Performance Budget, and Documentation

**Files:**
- Modify: `README.md`
- Modify: `MVP_PROGRESS.md`
- Create: `docs/runbooks/research-experiments.md`
- Modify: `docs/runbooks/paper-trading-mvp.md`
- Modify: `tests/integration/test_repository_hygiene.py`
- Modify: `Dockerfile`

- [ ] **Step 1: Add final acceptance tests**

Add integration assertions that:

- two baseline runs have the same run ID and canonical metrics;
- a two-trial study resumes to three completed trials;
- the generated report is readable without MLflow;
- local tracking state is ignored by git;
- the Docker build contains example specs but no local experiment state; and
- live trading remains rejected.

- [ ] **Step 2: Run the complete local quality gate**

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv sync --frozen --extra dev
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run ruff check .
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run mypy src
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/unit tests/integration \
  --cov=mltrade --cov-report=term-missing -q
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run mltrade doctor
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run mltrade demo run
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run mltrade experiment doctor
```

Expected: lint/type checks pass, coverage remains at least 90%, all prior tests
remain green, and both doctor commands plus the demo exit 0.

- [ ] **Step 3: Verify PostgreSQL contracts**

```bash
docker compose up -d --wait postgres
MLTRADE_TEST_DATABASE_URL='postgresql+psycopg://mltrade:mltrade@127.0.0.1:5432/mltrade' \
  UV_CACHE_DIR=/private/tmp/mltrade-uv-cache \
  uv run pytest tests/contract -m "contract and not alpaca" -q
```

Expected: three PostgreSQL contracts pass. Alpaca remains opt-in and unrelated
to experiment acceptance.

- [ ] **Step 4: Run the balanced Mac benchmark**

First publish or reuse the exact fixture snapshot, then run:

```bash
/usr/bin/time -p \
  env UV_CACHE_DIR=/private/tmp/mltrade-uv-cache \
  uv run mltrade experiment tune \
  experiments/ridge-balanced-search.toml
```

Acceptance:

- exits 0;
- finishes within 60 minutes under normal foreground load;
- completes at least 10 valid trials;
- persists the study for resume;
- materializes a best run;
- generates study and run reports; and
- does not exceed two workers.

If the median trial duration makes 40 trials unrealistic, reduce the example's
`max_trials` to the largest empirically supported count that leaves a 10%
time margin. Do not weaken validation windows or leakage controls to hit the
budget.

- [ ] **Step 5: Verify container behavior**

```bash
docker build -t mltrade:experiments .
docker run --rm mltrade:experiments demo run
docker run --rm mltrade:experiments experiment --help
```

Expected: demo succeeds and experiment CLI is available. The image must not
contain host `data/experiments`, MLflow state, Optuna databases, `.env`, or
`.claude/`.

- [ ] **Step 6: Update documentation**

Document:

- exact snapshot requirement;
- baseline run and balanced tune commands;
- study interruption and resume;
- canonical records versus MLflow;
- compatibility rules;
- dirty-worktree behavior;
- report locations;
- the measured Mac benchmark;
- local state cleanup;
- failure diagnosis; and
- explicit non-promotion and live-trading boundaries.

Rewrite stale `MVP_PROGRESS.md` branch/worktree claims to state that the paper
MVP is merged on `main`, and add the research experiment platform as the active
phase. Correct the stale paper-submit readiness statement without claiming the
stub is operational.

- [ ] **Step 7: Final repository check and commit**

```bash
git status --short
git diff --check
git ls-files | rg '(^data/|\\.db$|^mlruns/|^\\.claude/)'
```

Expected: only intended source/docs/tests are tracked; the hygiene search
returns no runtime state.

```bash
git add README.md MVP_PROGRESS.md Dockerfile \
  docs/runbooks/research-experiments.md \
  docs/runbooks/paper-trading-mvp.md \
  tests/integration/test_repository_hygiene.py
git commit -m "docs: complete local experiment platform acceptance"
```

## Final Acceptance Checklist

- [ ] Specifications are immutable, versioned, strict, and exact-snapshot only.
- [ ] Existing ridge/backtest defaults remain behaviorally unchanged.
- [ ] Identical clean-code runs produce identical IDs and canonical metrics.
- [ ] Dirty code changes alter run identity and are prominently reported.
- [ ] Canonical records survive MLflow degradation.
- [ ] Optuna studies persist and reject incompatible resume attempts.
- [ ] Compatible runs rank deterministically; incompatible runs have no winner.
- [ ] CLI supports validate, run, tune, resume, list, inspect, compare, report,
  doctor, and JSON output.
- [ ] Balanced tuning fits the measured Mac budget without weakening research
  controls.
- [ ] Existing demo, paper safety defaults, PostgreSQL contracts, lint, typing,
  and coverage gates remain green.
- [ ] No credentials, runtime databases, generated reports, MLflow state,
  Optuna state, datasets, or `.claude/` worktrees are tracked.
