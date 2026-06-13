# End-to-End Paper-Trading MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a locally runnable, deterministic data-to-order-preview pipeline with optional Alpaca paper submission, shared research/production logic, fail-closed risk controls, and complete operational evidence.

**Architecture:** Extend the existing Python 3.13 foundation with focused packages for market data, features, modeling, portfolio construction, risk, backtesting, execution, and workflows. Keep domain calculations pure and immutable; place filesystem, SQL, HTTP, and broker behavior behind adapters. Build the offline fixture path first so every later layer is testable without network access, then add opt-in Alpaca and PostgreSQL contracts.

**Tech Stack:** Python 3.13, uv, Pydantic v2, Polars, PyArrow/Parquet, NumPy, scikit-learn, CVXPY, SQLAlchemy 2, PostgreSQL, HTTPX, Typer, structlog, Docker Compose, Pytest, Ruff, strict mypy.

**Approved design:** `docs/superpowers/specs/2026-06-13-end-to-end-paper-trading-mvp-design.md`

---

## Scope and Delivery Order

This plan implements one vertical slice in dependency order:

1. configuration and dependencies;
2. canonical market data and deterministic fixtures;
3. immutable Parquet publication and data quality;
4. point-in-time features and labels;
5. walk-forward forecasts;
6. constrained portfolios and deterministic risk checks;
7. shared backtesting;
8. broker contracts, reconciliation, and idempotent intents;
9. operational persistence;
10. offline orchestration and CLI;
11. Alpaca adapters;
12. documentation, containers, and final acceptance.

Live-money execution, short selling, individual equities, automated scheduling,
and automated model promotion remain excluded.

## File Structure

```text
src/mltrade/
├── cli.py                              # Root Typer application and subcommands
├── config.py                           # MVP settings and safety validation
├── universe.py                         # Versioned ETF universe
├── data/
│   ├── bars.py                         # Canonical DailyBar and source protocol
│   ├── fixtures.py                     # Deterministic offline bar generator
│   ├── quality.py                      # Fail-closed bar quality report
│   ├── publication.py                  # Parquet + immutable manifest publication
│   └── alpaca.py                       # Alpaca historical bars adapter
├── features/
│   ├── definitions.py                  # Feature/label schemas and versions
│   └── pipeline.py                     # Point-in-time feature computation
├── models/
│   ├── forecasts.py                    # Forecast contracts
│   └── walk_forward.py                 # Embargoed ridge walk-forward runner
├── portfolio/
│   ├── targets.py                      # Portfolio target contracts
│   └── optimizer.py                    # CVXPY long-only optimizer
├── risk/
│   ├── checks.py                       # Structured pass/warn/block checks
│   └── policy.py                       # Pre-trade policy evaluation
├── backtest/
│   ├── accounting.py                   # Positions, cash, fills, and costs
│   ├── engine.py                       # Next-session event loop
│   └── reporting.py                    # Metrics, attribution, sensitivity
├── execution/
│   ├── broker.py                       # Broker protocol and value objects
│   ├── simulated.py                    # Deterministic in-memory broker
│   ├── alpaca.py                       # Alpaca paper broker adapter
│   ├── intents.py                      # Stable client order identities/deltas
│   ├── reconciliation.py               # Broker/internal state comparison
│   └── service.py                      # Preview/submit/retry workflow
├── operations/
│   ├── models.py                       # Existing audit plus MVP SQL models
│   └── repositories.py                 # Transactional operational persistence
└── workflows/
    ├── demo.py                         # Complete deterministic offline run
    ├── research.py                     # Snapshot-to-backtest workflow
    └── paper.py                        # Snapshot-to-preview/submit workflow
tests/
├── fixtures/alpaca/                    # Sanitized response contracts
├── unit/                               # Pure domain and adapter tests
├── integration/                        # Offline vertical-slice tests
└── contract/                           # PostgreSQL and opt-in Alpaca contracts
```

## Task 1: Add MVP Dependencies and Safe Settings

**Files:**
- Modify: `pyproject.toml`
- Modify: `.env.example`
- Modify: `src/mltrade/config.py`
- Modify: `tests/conftest.py`
- Modify: `tests/unit/test_config.py`

- [ ] **Step 1: Write failing settings tests**

Add tests that instantiate these exact fields:

```python
def test_mvp_settings_have_safe_defaults(tmp_path: Path) -> None:
    settings = Settings(data_root=tmp_path)

    assert settings.reference_equity == Decimal("1000000")
    assert settings.maximum_position_weight == Decimal("0.25")
    assert settings.minimum_cash_weight == Decimal("0.05")
    assert settings.target_annual_volatility == Decimal("0.15")
    assert settings.maximum_order_weight == Decimal("0.10")
    assert settings.maximum_rebalance_weight == Decimal("0.50")
    assert settings.minimum_order_notional == Decimal("500")
    assert settings.transaction_cost_bps == Decimal("5")


def test_paper_environment_requires_paper_url() -> None:
    with pytest.raises(ValidationError, match="paper-api.alpaca.markets"):
        Settings(
            environment=Environment.PAPER,
            alpaca_base_url="https://api.alpaca.markets",
        )
```

- [ ] **Step 2: Run the tests and verify failure**

Run:

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/unit/test_config.py -v
```

Expected: failures for missing MVP settings.

- [ ] **Step 3: Add dependencies and settings**

Add runtime dependencies:

```toml
"cvxpy>=1.6",
"httpx>=0.28",
"numpy>=2.2",
"pyarrow>=19.0",
"scikit-learn>=1.6",
```

Add the recorded HTTP contract helper to the development dependencies:

```toml
"respx>=0.22",
```

Add positive `Decimal` settings for every limit in Step 1 and validators that
enforce:

```python
@model_validator(mode="after")
def validate_paper_boundary(self) -> "Settings":
    if self.live_trading_enabled:
        raise ValueError("live trading is not available in this release")
    if (
        self.environment is Environment.PAPER
        and self.alpaca_base_url.rstrip("/")
        != "https://paper-api.alpaca.markets"
    ):
        raise ValueError("paper mode requires paper-api.alpaca.markets")
    if self.maximum_position_weight > 1 - self.minimum_cash_weight:
        raise ValueError("position limit conflicts with cash reserve")
    return self
```

Add matching `MLTRADE_REFERENCE_EQUITY`,
`MLTRADE_MAXIMUM_POSITION_WEIGHT`, `MLTRADE_MINIMUM_CASH_WEIGHT`,
`MLTRADE_TARGET_ANNUAL_VOLATILITY`, `MLTRADE_MAXIMUM_ORDER_WEIGHT`,
`MLTRADE_MAXIMUM_REBALANCE_WEIGHT`, `MLTRADE_MINIMUM_ORDER_NOTIONAL`, and
`MLTRADE_TRANSACTION_COST_BPS` entries to `.env.example` and remove them in
the autouse environment fixture.

- [ ] **Step 4: Lock and verify**

Run:

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv lock
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/unit/test_config.py -v
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run mypy src
```

Expected: settings tests and mypy pass.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock .env.example src/mltrade/config.py \
  tests/conftest.py tests/unit/test_config.py
git commit -m "build: add paper trading MVP dependencies"
```

## Task 2: Define the Versioned ETF Universe and Canonical Bars

**Files:**
- Create: `src/mltrade/universe.py`
- Create: `src/mltrade/data/__init__.py`
- Create: `src/mltrade/data/bars.py`
- Create: `tests/unit/test_universe.py`
- Create: `tests/unit/test_bars.py`

- [ ] **Step 1: Write failing universe and bar tests**

```python
def test_mvp_universe_is_fixed_and_versioned() -> None:
    assert MVP_UNIVERSE.version == "mvp-etf-v1"
    assert MVP_UNIVERSE.symbols == (
        "SPY", "QQQ", "IWM", "EFA", "EEM",
        "TLT", "IEF", "GLD", "DBC", "VNQ",
    )


def test_daily_bar_rejects_invalid_ohlc() -> None:
    with pytest.raises(ValidationError, match="high"):
        DailyBar(
            instrument=InstrumentId(symbol="SPY", asset_type=AssetType.ETF),
            session=date(2026, 6, 12),
            open=Decimal("100"),
            high=Decimal("99"),
            low=Decimal("98"),
            close=Decimal("100"),
            volume=1_000,
            vwap=Decimal("99.5"),
            trade_count=100,
            source="fixture",
            ingested_at=datetime(2026, 6, 13, tzinfo=UTC),
        )
```

- [ ] **Step 2: Verify failure**

Run:

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/unit/test_universe.py tests/unit/test_bars.py -v
```

Expected: import failures for the new modules.

- [ ] **Step 3: Implement immutable contracts**

Implement:

```python
class Universe(BaseModel):
    model_config = ConfigDict(frozen=True)
    version: str
    instruments: tuple[InstrumentId, ...]

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(item.symbol for item in self.instruments)


class DailyBar(BaseModel):
    model_config = ConfigDict(frozen=True)
    instrument: InstrumentId
    session: date
    open: Decimal = Field(gt=0)
    high: Decimal = Field(gt=0)
    low: Decimal = Field(gt=0)
    close: Decimal = Field(gt=0)
    volume: int = Field(ge=0)
    vwap: Decimal = Field(gt=0)
    trade_count: int = Field(ge=0)
    source: str = Field(min_length=1)
    ingested_at: datetime

    @model_validator(mode="after")
    def validate_ohlc(self) -> "DailyBar":
        if self.high < max(self.open, self.low, self.close):
            raise ValueError("high must be at least open, low, and close")
        if self.low > min(self.open, self.high, self.close):
            raise ValueError("low must be no greater than open, high, and close")
        return self


class DailyBarSource(Protocol):
    def fetch(
        self,
        universe: Universe,
        start: date,
        end: date,
        ingested_at: datetime,
    ) -> tuple[DailyBar, ...]:
        pass
```

Normalize `ingested_at` with `require_utc`.

- [ ] **Step 4: Run tests and static checks**

Run:

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/unit/test_universe.py tests/unit/test_bars.py -v
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run ruff check .
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run mypy src
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/mltrade/universe.py src/mltrade/data \
  tests/unit/test_universe.py tests/unit/test_bars.py
git commit -m "feat: define ETF universe and daily bar contracts"
```

## Task 3: Build Deterministic Offline Market Data

**Files:**
- Create: `src/mltrade/data/fixtures.py`
- Create: `tests/unit/test_data_fixtures.py`

- [ ] **Step 1: Write failing deterministic-fixture tests**

```python
def test_fixture_is_deterministic_and_session_complete() -> None:
    source = DeterministicBarSource(seed=7)
    first = source.fetch(
        MVP_UNIVERSE,
        date(2022, 1, 3),
        date(2026, 6, 12),
        datetime(2026, 6, 13, tzinfo=UTC),
    )
    second = source.fetch(
        MVP_UNIVERSE,
        date(2022, 1, 3),
        date(2026, 6, 12),
        datetime(2026, 6, 13, tzinfo=UTC),
    )

    assert first == second
    assert {bar.instrument.symbol for bar in first} == set(MVP_UNIVERSE.symbols)
    assert len({bar.session for bar in first}) >= 1_100
```

- [ ] **Step 2: Verify failure**

Run:

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/unit/test_data_fixtures.py -v
```

Expected: `DeterministicBarSource` import failure.

- [ ] **Step 3: Implement the fixture generator**

Use `numpy.random.default_rng(seed)` and XNYS sessions. Generate log returns
from deterministic trend, cycle, stress, and seeded noise components. Build
positive OHLCV values from each close:

```python
overnight = rng.normal(0.0, 0.002)
intraday = daily_return - overnight
open_price = previous_close * exp(overnight)
close_price = open_price * exp(intraday)
spread = abs(rng.normal(0.006, 0.002))
high = max(open_price, close_price) * (1 + spread)
low = min(open_price, close_price) * (1 - spread)
```

Use stable per-symbol initial prices and liquidity scales. Return bars sorted by
`(session, symbol)`.

- [ ] **Step 4: Verify determinism**

Run:

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/unit/test_data_fixtures.py -v
```

Expected: all fixture tests pass without network access.

- [ ] **Step 5: Commit**

```bash
git add src/mltrade/data/fixtures.py tests/unit/test_data_fixtures.py
git commit -m "feat: add deterministic market data fixture"
```

## Task 4: Add Fail-Closed Data Quality

**Files:**
- Create: `src/mltrade/data/quality.py`
- Create: `tests/unit/test_data_quality.py`

- [ ] **Step 1: Write failing quality tests**

```python
def test_quality_blocks_duplicate_bars(valid_bars: tuple[DailyBar, ...]) -> None:
    report = validate_daily_bars(
        valid_bars + (valid_bars[0],),
        universe=MVP_UNIVERSE,
        expected_last_session=date(2026, 6, 12),
    )

    assert report.blocked is True
    assert "duplicate_bar" in {issue.code for issue in report.issues}


def test_quality_blocks_incomplete_latest_session(
    valid_bars: tuple[DailyBar, ...],
) -> None:
    bars = tuple(
        bar for bar in valid_bars
        if not (
            bar.session == date(2026, 6, 12)
            and bar.instrument.symbol == "SPY"
        )
    )

    report = validate_daily_bars(
        bars,
        universe=MVP_UNIVERSE,
        expected_last_session=date(2026, 6, 12),
    )

    assert report.blocked is True
    assert "incomplete_latest_session" in {
        issue.code for issue in report.issues
    }
```

- [ ] **Step 2: Verify failure**

Run:

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/unit/test_data_quality.py -v
```

Expected: quality module import failure.

- [ ] **Step 3: Implement quality value objects and rules**

Implement immutable `QualityIssue` and `DataQualityReport`. Evaluate:

- empty input;
- symbols outside or missing from the universe;
- duplicate `(symbol, session)`;
- unsorted source output;
- non-finite numeric values;
- latest session mismatch;
- incomplete latest session;
- missing XNYS sessions per symbol inside the observed range.

`DataQualityReport.blocked` must be true whenever any issue has severity
`block`.

- [ ] **Step 4: Run quality and calendar tests**

Run:

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/unit/test_data_quality.py tests/unit/test_calendar.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/mltrade/data/quality.py tests/unit/test_data_quality.py
git commit -m "feat: add fail-closed market data quality"
```

## Task 5: Publish and Verify Immutable Parquet Snapshots

**Files:**
- Create: `src/mltrade/data/publication.py`
- Modify: `src/mltrade/storage/manifests.py`
- Create: `tests/unit/test_data_publication.py`

- [ ] **Step 1: Write failing publication tests**

```python
def test_publish_round_trip_verifies_content(
    tmp_path: Path,
    valid_bars: tuple[DailyBar, ...],
) -> None:
    publisher = DailyBarPublisher(SnapshotStore(tmp_path))
    published = publisher.publish(
        bars=valid_bars,
        quality=passing_report(valid_bars),
        snapshot_id="fixture-20260612",
        created_at=datetime(2026, 6, 13, tzinfo=UTC),
    )

    loaded = publisher.load_verified(published.manifest)

    assert loaded == valid_bars
    assert published.manifest.metadata["universe_version"] == "mvp-etf-v1"


def test_tampered_parquet_is_rejected(
    tmp_path: Path,
    valid_bars: tuple[DailyBar, ...],
) -> None:
    publisher = DailyBarPublisher(SnapshotStore(tmp_path))
    published = publisher.publish(
        bars=valid_bars,
        quality=passing_report(valid_bars),
        snapshot_id="fixture-20260612",
        created_at=datetime(2026, 6, 13, tzinfo=UTC),
    )
    parquet_path = published.data_files[0]
    parquet_path.write_bytes(parquet_path.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="content hash"):
        publisher.load_verified(published.manifest)
    parquet_path.write_bytes(parquet_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="content hash"):
        publisher.load_verified(published.manifest)
```

- [ ] **Step 2: Verify failure**

Run:

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/unit/test_data_publication.py -v
```

Expected: publisher import failure.

- [ ] **Step 3: Extend manifests and implement publication**

Add immutable `metadata: dict[str, str]` to `DatasetManifest`. Publication must:

1. reject blocked quality reports;
2. serialize sorted bars to `daily-bars.parquet` using a fixed PyArrow schema;
3. fsync the file;
4. compute SHA-256 from the exact file bytes;
5. save a manifest with dataset `daily_bars`;
6. include universe, schema, and quality versions in metadata; and
7. verify hash, row count, schema, and manifest identity on load.

Use a temporary file and an exclusive hard-link publication pattern consistent
with `SnapshotStore`.

- [ ] **Step 4: Run publication and snapshot tests**

Run:

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/unit/test_data_publication.py tests/unit/test_snapshots.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/mltrade/data/publication.py src/mltrade/storage/manifests.py \
  tests/unit/test_data_publication.py tests/unit/test_snapshots.py
git commit -m "feat: publish verified daily bar snapshots"
```

## Task 6: Compute Point-in-Time Features and Labels

**Files:**
- Create: `src/mltrade/features/__init__.py`
- Create: `src/mltrade/features/definitions.py`
- Create: `src/mltrade/features/pipeline.py`
- Create: `tests/unit/test_features.py`
- Create: `tests/unit/test_feature_leakage.py`

- [ ] **Step 1: Write failing feature and leakage tests**

```python
def test_feature_values_use_only_available_bars() -> None:
    rows = build_feature_rows(
        bars=fixture_bars,
        snapshot_id="fixture-1",
        horizon=21,
    )
    row = next(
        item for item in rows
        if item.symbol == "SPY" and item.decision_session == date(2025, 1, 31)
    )

    assert row.latest_source_session == row.decision_session
    assert row.feature_version == "trend-momentum-v1"


def test_future_bar_change_does_not_change_earlier_features() -> None:
    original = build_feature_rows(fixture_bars, "fixture-1", horizon=21)
    changed = build_feature_rows(
        replace_close_after(fixture_bars, date(2025, 6, 1), factor=10),
        "fixture-1",
        horizon=21,
    )

    cutoff = date(2025, 5, 30)
    assert rows_through(original, cutoff) == rows_through(changed, cutoff)
```

- [ ] **Step 2: Verify failure**

Run:

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/unit/test_features.py tests/unit/test_feature_leakage.py -v
```

Expected: feature modules are missing.

- [ ] **Step 3: Implement typed feature rows**

`FeatureRow` contains:

```python
symbol: str
decision_session: date
latest_source_session: date
snapshot_id: str
feature_version: Literal["trend-momentum-v1"]
return_21: float
return_63: float
return_126: float
realized_volatility_21: float
distance_from_sma_100: float
average_dollar_volume_20: float
forward_return_21: float | None
label_end_session: date | None
missing: bool
```

Compute features per symbol with Polars, using closes through the decision
session. Annualize volatility with `sqrt(252)`. Labels may use later bars but
must record `label_end_session`; incomplete labels are `None`.

- [ ] **Step 4: Verify values and leakage**

Run:

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/unit/test_features.py tests/unit/test_feature_leakage.py -v
```

Expected: all pass, including future-data mutation tests.

- [ ] **Step 5: Commit**

```bash
git add src/mltrade/features tests/unit/test_features.py \
  tests/unit/test_feature_leakage.py
git commit -m "feat: compute point-in-time momentum features"
```

## Task 7: Add Embargoed Walk-Forward Forecasting

**Files:**
- Create: `src/mltrade/models/__init__.py`
- Create: `src/mltrade/models/forecasts.py`
- Create: `src/mltrade/models/walk_forward.py`
- Create: `tests/unit/test_walk_forward.py`

- [ ] **Step 1: Write failing walk-forward tests**

```python
def test_training_rows_end_before_embargo() -> None:
    split = build_training_split(
        feature_rows,
        decision_session=date(2026, 1, 30),
        embargo_sessions=21,
    )

    assert max(row.label_end_session for row in split.training) < (
        split.embargo_start
    )


def test_non_finite_model_input_blocks_forecast() -> None:
    with pytest.raises(ForecastBlocked, match="non-finite"):
        generate_forecast_batch(rows_with_nan, decision_session)
```

- [ ] **Step 2: Verify failure**

Run:

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/unit/test_walk_forward.py -v
```

Expected: model modules are missing.

- [ ] **Step 3: Implement forecast contracts and ridge runner**

Define immutable `Forecast` and `ForecastBatch` with model version
`ridge-trend-v1`. For each rebalance:

1. require at least 504 distinct training sessions;
2. retain rows with known labels ending before the 21-session embargo;
3. standardize each feature cross-sectionally by decision session;
4. fit `sklearn.linear_model.Ridge(alpha=1.0)`;
5. predict the current cross-section;
6. reject non-finite inputs and outputs; and
7. persist training start/end, embargo start, and row count in batch metadata.

- [ ] **Step 4: Run model and leakage tests**

Run:

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/unit/test_walk_forward.py tests/unit/test_feature_leakage.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/mltrade/models tests/unit/test_walk_forward.py
git commit -m "feat: add embargoed walk-forward forecasts"
```

## Task 8: Build Constrained Portfolio Targets

**Files:**
- Create: `src/mltrade/portfolio/__init__.py`
- Create: `src/mltrade/portfolio/targets.py`
- Create: `src/mltrade/portfolio/optimizer.py`
- Create: `tests/unit/test_portfolio_optimizer.py`

- [ ] **Step 1: Write failing optimizer tests**

```python
def test_optimizer_respects_hard_constraints() -> None:
    result = build_target(
        forecasts=positive_forecasts,
        trailing_volatility=volatility,
        limits=PortfolioLimits(
            maximum_position_weight=Decimal("0.25"),
            minimum_cash_weight=Decimal("0.05"),
            target_annual_volatility=Decimal("0.15"),
        ),
    )

    assert result.blocked is False
    assert sum(result.weights.values()) <= Decimal("0.95")
    assert max(result.weights.values()) <= Decimal("0.25")
    assert result.cash_weight >= Decimal("0.05")


def test_solver_failure_returns_blocked_all_cash(monkeypatch) -> None:
    monkeypatch.setattr(cp.Problem, "solve", raising_solver_error)
    result = build_target(
        forecasts=positive_forecasts,
        trailing_volatility=volatility,
        limits=PortfolioLimits(
            maximum_position_weight=Decimal("0.25"),
            minimum_cash_weight=Decimal("0.05"),
            target_annual_volatility=Decimal("0.15"),
        ),
    )

    assert result.blocked is True
    assert result.weights == {}
    assert result.cash_weight == Decimal("1")
```

- [ ] **Step 2: Verify failure**

Run:

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/unit/test_portfolio_optimizer.py -v
```

Expected: portfolio modules are missing.

- [ ] **Step 3: Implement deterministic CVXPY optimization**

Create `PortfolioLimits`, `PortfolioTarget`, and `OptimizationResult`. Rank
positive forecasts, divide conviction by trailing volatility, normalize the
desired score, then solve:

```python
objective = cp.Minimize(cp.sum_squares(weights - desired))
constraints = [
    weights >= 0,
    weights <= maximum_position_weight,
    cp.sum(weights) <= 1 - minimum_cash_weight,
    cp.quad_form(weights, covariance) <= target_variance,
]
```

Use a fixed symbol order and an installed deterministic convex solver. Accept
only `OPTIMAL` or `OPTIMAL_INACCURATE`, and validate the returned values again
in Python before constructing the immutable target.

- [ ] **Step 4: Verify constraint and failure cases**

Run:

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/unit/test_portfolio_optimizer.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/mltrade/portfolio tests/unit/test_portfolio_optimizer.py
git commit -m "feat: add constrained portfolio optimizer"
```

## Task 9: Add Structured Pre-Trade Risk Policy

**Files:**
- Create: `src/mltrade/risk/__init__.py`
- Create: `src/mltrade/risk/checks.py`
- Create: `src/mltrade/risk/policy.py`
- Create: `tests/unit/test_risk_policy.py`

- [ ] **Step 1: Write failing policy tests**

```python
def test_stale_snapshot_blocks_submission() -> None:
    report = evaluate_pre_trade(
        context=valid_context.model_copy(
            update={"snapshot_last_session": date(2026, 6, 11)}
        )
    )

    assert report.blocked is True
    assert report.by_code("snapshot_freshness").status is CheckStatus.BLOCK


def test_all_checks_pass_for_valid_preview() -> None:
    report = evaluate_pre_trade(context=valid_context)

    assert report.blocked is False
    assert all(check.status is not CheckStatus.BLOCK for check in report.checks)
```

- [ ] **Step 2: Verify failure**

Run:

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/unit/test_risk_policy.py -v
```

Expected: risk modules are missing.

- [ ] **Step 3: Implement complete risk evaluation**

Define `CheckStatus`, `RiskCheck`, `RiskReport`, and `PreTradeContext`. Always
emit checks for:

- snapshot health and identity;
- decision-session freshness;
- feature/model version agreement;
- finite prices, forecasts, weights, quantities, and notionals;
- position, gross, net, and cash limits;
- per-order and total rebalance notional;
- minimum order notional filtering;
- duplicate intent IDs;
- paper account status;
- cash, positions, and open-order reconciliation; and
- live trading disabled.

`RiskReport.blocked` is true if any check is `BLOCK`.

- [ ] **Step 4: Verify policy**

Run:

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/unit/test_risk_policy.py tests/unit/test_portfolio_optimizer.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/mltrade/risk tests/unit/test_risk_policy.py
git commit -m "feat: add deterministic pre-trade risk policy"
```

## Task 10: Implement Shared Backtest Accounting and Reports

**Files:**
- Create: `src/mltrade/backtest/__init__.py`
- Create: `src/mltrade/backtest/accounting.py`
- Create: `src/mltrade/backtest/engine.py`
- Create: `src/mltrade/backtest/reporting.py`
- Create: `tests/unit/test_backtest_accounting.py`
- Create: `tests/integration/test_walk_forward_backtest.py`

- [ ] **Step 1: Write failing accounting and integration tests**

```python
def test_trade_cost_reduces_cash_and_equity() -> None:
    state = PortfolioState.initial(Decimal("1000000"))
    filled = apply_fill(
        state,
        Fill(symbol="SPY", quantity=100, price=Decimal("500")),
        transaction_cost_bps=Decimal("5"),
    )

    assert filled.cash == Decimal("949975")
    assert filled.total_costs == Decimal("25")


def test_walk_forward_backtest_is_deterministic(fixture_snapshot) -> None:
    first = run_backtest(fixture_snapshot, cost_bps=Decimal("5"))
    second = run_backtest(fixture_snapshot, cost_bps=Decimal("5"))

    assert first == second
    assert first.sessions > 250
    assert set(first.cost_sensitivity) == {
        Decimal("2"), Decimal("5"), Decimal("10")
    }
```

- [ ] **Step 2: Verify failure**

Run:

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/unit/test_backtest_accounting.py \
  tests/integration/test_walk_forward_backtest.py -v
```

Expected: backtest modules are missing.

- [ ] **Step 3: Implement accounting and event loop**

The engine must:

1. iterate completed decision sessions;
2. retrain every 21 sessions;
3. build forecasts and production portfolio targets;
4. calculate target quantities using next-session open prices;
5. apply fills and costs;
6. mark positions to each close;
7. store equity, cash, exposure, turnover, costs, and contributions; and
8. repeat at 2, 5, and 10 bps for sensitivity.

Calculate annualized return/volatility, zero-rate Sharpe, drawdown, hit rate,
turnover, total costs, equal-weight baseline, cash baseline, and per-symbol
contribution.

- [ ] **Step 4: Verify deterministic reports**

Run:

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/unit/test_backtest_accounting.py \
  tests/integration/test_walk_forward_backtest.py -v
```

Expected: all pass with finite metrics.

- [ ] **Step 5: Commit**

```bash
git add src/mltrade/backtest tests/unit/test_backtest_accounting.py \
  tests/integration/test_walk_forward_backtest.py
git commit -m "feat: add shared walk-forward backtester"
```

## Task 11: Define Broker Contracts and Simulated Execution

**Files:**
- Create: `src/mltrade/execution/__init__.py`
- Create: `src/mltrade/execution/broker.py`
- Create: `src/mltrade/execution/simulated.py`
- Create: `src/mltrade/execution/intents.py`
- Create: `tests/unit/test_execution_intents.py`
- Create: `tests/unit/test_simulated_broker.py`

- [ ] **Step 1: Write failing identity and broker tests**

```python
def test_execution_identity_is_stable() -> None:
    first = build_intent(
        environment="paper",
        strategy_version="ridge-trend-v1",
        decision_session=date(2026, 6, 12),
        symbol="SPY",
        side=OrderSide.BUY,
        target_quantity=Decimal("10"),
    )
    second = build_intent(
        environment="paper",
        strategy_version="ridge-trend-v1",
        decision_session=date(2026, 6, 12),
        symbol="SPY",
        side=OrderSide.BUY,
        target_quantity=Decimal("10"),
    )

    assert first.client_order_id == second.client_order_id
    assert len(first.client_order_id) <= 48


def test_simulated_broker_deduplicates_client_order_id() -> None:
    broker = SimulatedBroker(account=paper_account())
    first = broker.submit(intent)
    second = broker.submit(intent)

    assert first.id == second.id
    assert len(broker.list_orders()) == 1
```

- [ ] **Step 2: Verify failure**

Run:

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/unit/test_execution_intents.py tests/unit/test_simulated_broker.py -v
```

Expected: execution modules are missing.

- [ ] **Step 3: Implement broker protocol and stable intents**

Define immutable account, position, order, fill, and intent value objects.
`Broker` exposes:

```python
class Broker(Protocol):
    def get_account(self) -> BrokerAccount:
        pass

    def list_positions(self) -> tuple[BrokerPosition, ...]:
        pass

    def list_open_orders(self) -> tuple[BrokerOrder, ...]:
        pass

    def list_recent_fills(self) -> tuple[BrokerFill, ...]:
        pass

    def get_order_by_client_id(
        self,
        client_order_id: str,
    ) -> BrokerOrder | None:
        pass

    def submit(self, intent: ExecutionIntent) -> BrokerOrder:
        pass
```

Build client IDs from canonical JSON and SHA-256:

```python
digest = sha256(payload.encode("ascii")).hexdigest()[:24]
client_order_id = f"mlt-{decision_session:%Y%m%d}-{digest}"
```

The simulated broker supports complete, partial, rejected, timeout-before-
acceptance, and timeout-after-acceptance outcomes.

- [ ] **Step 4: Run execution tests**

Run:

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/unit/test_execution_intents.py tests/unit/test_simulated_broker.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/mltrade/execution tests/unit/test_execution_intents.py \
  tests/unit/test_simulated_broker.py
git commit -m "feat: add idempotent broker execution contracts"
```

## Task 12: Add Reconciliation and Safe Submission Service

**Files:**
- Create: `src/mltrade/execution/reconciliation.py`
- Create: `src/mltrade/execution/service.py`
- Create: `tests/unit/test_reconciliation.py`
- Create: `tests/integration/test_execution_service.py`

- [ ] **Step 1: Write failing reconciliation and retry tests**

```python
def test_position_difference_blocks_submission() -> None:
    result = reconcile(
        internal=internal_state,
        broker=broker_state_with_extra_spy,
    )

    assert result.blocked is True
    assert result.differences[0].kind == "position"


def test_timeout_after_acceptance_does_not_duplicate() -> None:
    broker = SimulatedBroker(timeout_after_acceptance=True)
    service = ExecutionService(broker)

    result = service.submit(approved_preview)

    assert result.submitted == 1
    assert len(broker.list_orders()) == 1
```

- [ ] **Step 2: Verify failure**

Run:

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/unit/test_reconciliation.py tests/integration/test_execution_service.py -v
```

Expected: new modules are missing.

- [ ] **Step 3: Implement reconciliation and submission**

`ExecutionService.preview` must:

1. fetch broker state;
2. reconcile it with internal state;
3. calculate quantity deltas;
4. drop trades below `$500`;
5. build stable intents;
6. evaluate the complete risk policy; and
7. return a preview without broker mutation.

`submit` must reject blocked previews. For each approved intent, check
`get_order_by_client_id` before submission. On timeout, query that client ID;
only retry once when no order exists.

- [ ] **Step 4: Verify failure simulations**

Run:

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/unit/test_reconciliation.py tests/integration/test_execution_service.py -v
```

Expected: all complete, partial, rejection, duplicate, and timeout cases pass.

- [ ] **Step 5: Commit**

```bash
git add src/mltrade/execution/reconciliation.py \
  src/mltrade/execution/service.py tests/unit/test_reconciliation.py \
  tests/integration/test_execution_service.py
git commit -m "feat: add reconciliation and safe paper submission"
```

## Task 13: Persist Operational Evidence

**Files:**
- Modify: `src/mltrade/operations/models.py`
- Create: `src/mltrade/operations/repositories.py`
- Create: `tests/unit/test_operational_repositories.py`
- Modify: `tests/contract/test_postgres_audit.py`
- Create: `tests/contract/test_postgres_mvp_state.py`

- [ ] **Step 1: Write failing persistence tests**

```python
def test_repository_persists_preview_and_checks(sqlite_session) -> None:
    repository = OperationsRepository(sqlite_session)
    preview_id = repository.save_preview(preview)

    loaded = repository.load_preview(preview_id)
    assert loaded == preview


def test_execution_intent_client_id_is_unique(sqlite_session) -> None:
    repository = OperationsRepository(sqlite_session)
    repository.save_intent(intent)
    repository.save_intent(intent)

    assert repository.count_intents(intent.client_order_id) == 1
```

- [ ] **Step 2: Verify failure**

Run:

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/unit/test_operational_repositories.py -v
```

Expected: repository module is missing.

- [ ] **Step 3: Add SQL models and transactional repository**

Create tables for:

- dataset snapshots and quality reports;
- model/forecast batches;
- backtest runs;
- portfolio targets;
- risk reports/checks;
- execution previews/intents;
- broker orders/fills; and
- reconciliation runs/differences.

Use UUID primary keys, UTC timestamps, JSON payloads for immutable evidence,
indexes on correlation/decision session, and a unique constraint on
`client_order_id`. Repository methods must flush inside the caller's existing
transaction and never commit independently.

- [ ] **Step 4: Verify SQLite and PostgreSQL**

Run:

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/unit/test_operational_repositories.py -v
docker compose up -d --wait postgres
MLTRADE_TEST_DATABASE_URL=postgresql+psycopg://mltrade:mltrade@localhost:5432/mltrade \
  UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/contract/test_postgres_audit.py \
  tests/contract/test_postgres_mvp_state.py -v
```

Expected: SQLite and PostgreSQL persistence tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/mltrade/operations/models.py \
  src/mltrade/operations/repositories.py \
  tests/unit/test_operational_repositories.py tests/contract
git commit -m "feat: persist paper trading operational evidence"
```

## Task 14: Create Offline, Research, and Paper Workflows

**Files:**
- Create: `src/mltrade/workflows/__init__.py`
- Create: `src/mltrade/workflows/demo.py`
- Create: `src/mltrade/workflows/research.py`
- Create: `src/mltrade/workflows/paper.py`
- Create: `tests/integration/test_demo_workflow.py`
- Create: `tests/integration/test_paper_workflow.py`

- [ ] **Step 1: Write failing vertical-slice tests**

```python
def test_demo_runs_end_to_end_without_network(tmp_path: Path) -> None:
    result = run_demo(
        Settings(
            environment=Environment.TEST,
            data_root=tmp_path,
            database_url=f"sqlite+pysqlite:///{tmp_path / 'ops.db'}",
        )
    )

    assert result.quality.blocked is False
    assert result.backtest.sessions > 250
    assert result.target.blocked is False
    assert result.preview.risk_report.blocked is False
    assert result.preview.intents


def test_replaying_demo_reuses_execution_intents(tmp_path: Path) -> None:
    first = run_demo(settings_for(tmp_path))
    second = run_demo(settings_for(tmp_path))

    assert intent_ids(first) == intent_ids(second)
```

- [ ] **Step 2: Verify failure**

Run:

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/integration/test_demo_workflow.py \
  tests/integration/test_paper_workflow.py -v
```

Expected: workflow modules are missing.

- [ ] **Step 3: Implement orchestration**

`run_demo` uses:

- deterministic fixture source;
- last completed XNYS session fixed by injected clock;
- immutable snapshot publication;
- verified snapshot reload;
- features and walk-forward backtest;
- current forecast and constrained target;
- simulated paper account;
- reconciliation and preview; and
- one SQL transaction per persisted workflow stage.

`run_research` starts from a verified snapshot. `run_paper` refuses research
sources, blocked risk reports, and submission without explicit `submit=True`.

- [ ] **Step 4: Verify vertical slice and replay**

Run:

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/integration/test_demo_workflow.py \
  tests/integration/test_paper_workflow.py -v
```

Expected: all pass without credentials or network access.

- [ ] **Step 5: Commit**

```bash
git add src/mltrade/workflows tests/integration/test_demo_workflow.py \
  tests/integration/test_paper_workflow.py
git commit -m "feat: orchestrate offline and paper workflows"
```

## Task 15: Expose the MVP CLI and Status

**Files:**
- Modify: `src/mltrade/cli.py`
- Modify: `tests/integration/test_doctor.py`
- Create: `tests/integration/test_cli_mvp.py`

- [ ] **Step 1: Write failing CLI tests**

```python
def test_demo_run_prints_acceptance_summary(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MLTRADE_DATA_ROOT", str(tmp_path))
    result = runner.invoke(app, ["demo", "run"])

    assert result.exit_code == 0
    assert "snapshot: ok" in result.stdout
    assert "data quality: pass" in result.stdout
    assert "backtest: complete" in result.stdout
    assert "risk: pass" in result.stdout
    assert "paper orders: preview only" in result.stdout


def test_paper_submit_requires_explicit_flag() -> None:
    result = runner.invoke(app, ["paper", "submit"])
    assert result.exit_code != 0
    assert "--submit" in result.stdout
```

- [ ] **Step 2: Verify failure**

Run:

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/integration/test_cli_mvp.py -v
```

Expected: missing commands.

- [ ] **Step 3: Implement Typer command groups**

Add command groups and exact commands:

```text
demo run
data ingest
data validate
research backtest
portfolio build
paper preview
paper submit --submit
paper reconcile
status
doctor
```

Use dependency functions for settings, clock, source, database, and broker so
tests can replace adapters. Print concise summaries and return nonzero exit
codes for blocked workflows.

- [ ] **Step 4: Verify CLI**

Run:

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/integration/test_doctor.py tests/integration/test_cli_mvp.py -v
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run mltrade demo run
```

Expected: tests pass and the offline demo exits zero.

- [ ] **Step 5: Commit**

```bash
git add src/mltrade/cli.py tests/integration/test_doctor.py \
  tests/integration/test_cli_mvp.py
git commit -m "feat: expose MVP operator CLI"
```

## Task 16: Add Sanitized Alpaca Data and Paper Adapters

**Files:**
- Create: `src/mltrade/data/alpaca.py`
- Create: `src/mltrade/execution/alpaca.py`
- Create: `tests/fixtures/alpaca/bars.json`
- Create: `tests/fixtures/alpaca/account.json`
- Create: `tests/fixtures/alpaca/positions.json`
- Create: `tests/fixtures/alpaca/orders.json`
- Create: `tests/unit/test_alpaca_data_adapter.py`
- Create: `tests/unit/test_alpaca_broker_adapter.py`
- Create: `tests/contract/test_alpaca_paper.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write failing response-contract tests**

```python
def test_alpaca_bar_response_maps_to_canonical_bar(respx_mock) -> None:
    respx_mock.get(ALPACA_BARS_URL).mock(
        return_value=httpx.Response(200, json=load_fixture("bars.json"))
    )
    bars = adapter.fetch(
        universe=MVP_UNIVERSE,
        start=date(2026, 6, 11),
        end=date(2026, 6, 12),
        ingested_at=datetime(2026, 6, 13, tzinfo=UTC),
    )

    assert bars[0].instrument.symbol == "SPY"
    assert bars[0].source == "alpaca"


def test_alpaca_broker_rejects_non_paper_account(respx_mock) -> None:
    respx_mock.get(ALPACA_ACCOUNT_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "account-1",
                "status": "ACTIVE",
                "cash": "1000000",
                "equity": "1000000",
                "account_blocked": False,
                "trading_blocked": False,
                "pattern_day_trader": False,
            },
        )
    )
    adapter = AlpacaPaperBroker(
        client=httpx.Client(),
        base_url="https://api.alpaca.markets",
        api_key=SecretStr("test-key"),
        api_secret=SecretStr("test-secret"),
    )

    with pytest.raises(BrokerSafetyError, match="paper"):
        adapter.get_account()
```

- [ ] **Step 2: Verify failure**

Run:

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/unit/test_alpaca_data_adapter.py \
  tests/unit/test_alpaca_broker_adapter.py -v
```

Expected: Alpaca adapters are missing.

- [ ] **Step 3: Implement HTTPX adapters**

Use an injected `httpx.Client`, explicit timeouts, Alpaca authentication
headers, and no logging of headers or raw secrets. Validate the base URL in
both settings and adapter constructors. Map sanitized response fields into
canonical domain objects.

Submission must use:

```json
{
  "symbol": "SPY",
  "qty": "10",
  "side": "buy",
  "type": "market",
  "time_in_force": "day",
  "client_order_id": "mlt-20260612-0123456789abcdef01234567"
}
```

Register marker:

```toml
"alpaca: requires explicit Alpaca paper credentials and network access",
```

The opt-in contract test skips unless
`MLTRADE_RUN_ALPACA_CONTRACTS=true`.

- [ ] **Step 4: Verify recorded and optional contracts**

Run:

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/unit/test_alpaca_data_adapter.py \
  tests/unit/test_alpaca_broker_adapter.py -v
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/contract/test_alpaca_paper.py -v
```

Expected: recorded tests pass; live contract skips without explicit opt-in.

- [ ] **Step 5: Commit**

```bash
git add src/mltrade/data/alpaca.py src/mltrade/execution/alpaca.py \
  tests/fixtures/alpaca tests/unit/test_alpaca_data_adapter.py \
  tests/unit/test_alpaca_broker_adapter.py \
  tests/contract/test_alpaca_paper.py pyproject.toml
git commit -m "feat: add Alpaca paper adapters"
```

## Task 17: Document and Containerize the Complete Demo

**Files:**
- Modify: `README.md`
- Create: `docs/runbooks/paper-trading-mvp.md`
- Modify: `Dockerfile`
- Modify: `.dockerignore`
- Modify: `.gitignore`
- Create: `tests/integration/test_repository_hygiene.py`

- [ ] **Step 1: Write failing hygiene test**

```python
def test_runtime_artifacts_are_gitignored() -> None:
    ignored = Path(".gitignore").read_text(encoding="utf-8")
    for entry in ("data/", "artifacts/", "mlruns/", "*.db"):
        assert entry in ignored
```

- [ ] **Step 2: Verify failure**

Run:

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/integration/test_repository_hygiene.py -v
```

Expected: failure for any missing exclusion.

- [ ] **Step 3: Write operator documentation**

Document:

- `uv sync --frozen --extra dev`;
- offline demo and expected summary;
- snapshot and SQLite locations;
- research ingestion;
- paper preview;
- explicit paper submission prerequisites;
- reconciliation and blocked-run diagnosis;
- PostgreSQL startup and contracts;
- optional Alpaca contract invocation;
- secret handling;
- replay/idempotency behavior; and
- the explicit statement that live trading is unavailable.

Change the container default to:

```dockerfile
CMD ["demo", "run"]
```

Include an application-owned writable data directory and run as a non-root
user.

- [ ] **Step 4: Verify docs, image, and hygiene**

Run:

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/integration/test_repository_hygiene.py -v
docker build -t mltrade:mvp .
docker run --rm mltrade:mvp demo run
git status --short --ignored
```

Expected: hygiene passes, image builds, demo exits zero, and runtime artifacts
are ignored.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/runbooks/paper-trading-mvp.md Dockerfile \
  .dockerignore .gitignore tests/integration/test_repository_hygiene.py
git commit -m "docs: add paper trading MVP runbook"
```

## Task 18: Final Acceptance and Evidence

**Files:**
- Review: all MVP source and test files
- Modify: `docs/runbooks/paper-trading-mvp.md` only if verification reveals an
  incorrect command or expected output

- [ ] **Step 1: Run locked static and non-contract verification**

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv sync --frozen --extra dev
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run ruff check .
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run mypy src
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/unit tests/integration \
  --cov=mltrade --cov-report=term-missing
```

Expected: Ruff and mypy pass; all tests pass; branch coverage is at least 90%.

- [ ] **Step 2: Run offline acceptance twice**

```bash
MLTRADE_DATA_ROOT=/private/tmp/mltrade-mvp-acceptance \
  UV_CACHE_DIR=/private/tmp/mltrade-uv-cache \
  uv run mltrade demo run
MLTRADE_DATA_ROOT=/private/tmp/mltrade-mvp-acceptance \
  UV_CACHE_DIR=/private/tmp/mltrade-uv-cache \
  uv run mltrade demo run
```

Expected: both runs pass and the second run reports reused snapshot/execution
identities rather than duplicates.

- [ ] **Step 3: Run PostgreSQL contracts**

```bash
docker compose up -d --wait postgres
MLTRADE_TEST_DATABASE_URL=postgresql+psycopg://mltrade:mltrade@localhost:5432/mltrade \
  UV_CACHE_DIR=/private/tmp/mltrade-uv-cache \
  uv run pytest tests/contract -m "contract and not alpaca" -v
```

Expected: all PostgreSQL contracts pass.

- [ ] **Step 4: Verify the application image**

```bash
docker build -t mltrade:mvp .
docker run --rm mltrade:mvp demo run
```

Expected: build succeeds and the offline demo exits zero.

- [ ] **Step 5: Verify safety failures**

Run focused tests:

```bash
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest \
  tests/unit/test_feature_leakage.py \
  tests/unit/test_portfolio_optimizer.py \
  tests/unit/test_risk_policy.py \
  tests/unit/test_reconciliation.py \
  tests/integration/test_execution_service.py -v
```

Expected: stale data, leakage, optimizer failure, reconciliation differences,
duplicates, and ambiguous timeouts all fail closed.

- [ ] **Step 6: Inspect repository state**

```bash
git status --short
git ls-files | rg '(^data/|\.db$|broker|secret|credential)'
git log --oneline --decorate -20
```

Expected: clean status; no runtime data, credentials, or unsanitized broker
responses are tracked.

- [ ] **Step 7: Commit any verification-only documentation correction**

Only when Step 1-6 revealed an inaccurate runbook:

```bash
git add docs/runbooks/paper-trading-mvp.md
git commit -m "docs: correct MVP acceptance runbook"
```

- [ ] **Step 8: Report exact evidence**

The handoff must include:

- unit/integration test count and branch coverage;
- Ruff and mypy results;
- PostgreSQL contract count;
- optional Alpaca contract status;
- both offline demo results;
- container build/run result;
- safety-focused test result;
- final commit hash; and
- any remaining external dependency, especially missing Alpaca credentials.
