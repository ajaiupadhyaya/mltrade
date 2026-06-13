# End-to-End Paper-Trading MVP Design

Date: 2026-06-13
Status: Approved design

## 1. Objective

Build the first runnable vertical slice of MLTrade:

1. ingest completed daily market bars;
2. validate and publish an immutable dataset snapshot;
3. compute point-in-time features;
4. train and walk-forward evaluate a transparent baseline model;
5. convert forecasts into a constrained target portfolio;
6. run deterministic pre-trade risk checks;
7. generate idempotent Alpaca paper orders;
8. reconcile broker and internal state; and
9. preserve a complete audit trail.

The MVP must run locally on macOS with Python 3.13 and Docker Compose. It must
also support a deterministic offline demonstration that does not require
network access or broker credentials.

This is a paper-trading system. Live-money order submission remains
structurally disabled.

## 2. Scope Decisions

### 2.1 Initial Universe

The MVP uses this fixed, versioned ETF universe:

- `SPY`: US large-cap equities
- `QQQ`: US growth and technology equities
- `IWM`: US small-cap equities
- `EFA`: developed international equities
- `EEM`: emerging-market equities
- `TLT`: long-duration US Treasuries
- `IEF`: intermediate US Treasuries
- `GLD`: gold
- `DBC`: broad commodities
- `VNQ`: US real estate

All instruments are highly liquid, US-listed, unleveraged ETFs. The fixed
universe avoids pretending that the MVP has solved historical S&P 500
membership, fundamental availability, delistings, or short-borrow history.

The MVP is long-only. Individual stocks, short selling, borrow modeling,
market-neutral construction, SEC fundamentals, FRED features, and the
mean-reversion sleeve remain later milestones.

### 2.2 Trading Schedule

- Frequency: daily.
- Signal timestamp: after a completed XNYS session.
- Execution assumption: next eligible session.
- Paper order type: market order submitted only during an explicit paper
  execution command.
- Default operation: preview only; no broker mutation.
- Rebalance threshold: skip target changes below the configured minimum
  weight or notional change.

The system never uses an incomplete current-session bar.

### 2.3 Capital and Risk

- Reference portfolio value: `$1,000,000`.
- Maximum gross exposure: `1.0x` for the long-only MVP.
- Net exposure range: `0%` to `100%`.
- Maximum position weight: `25%`.
- Minimum cash reserve: `5%`.
- Target annualized volatility: `15%`.
- Maximum one-day order notional per instrument: `10%` of portfolio equity.
- Maximum total rebalance notional: `50%` of portfolio equity.
- Minimum order notional: `$500`.

These are hard limits. Model output cannot override them.

## 3. Operating Modes

The same pipeline supports three explicit modes.

### 3.1 Offline Demo

Uses a deterministic bundled fixture with multiple market regimes and known
data-quality properties. It runs ingestion, features, walk-forward evaluation,
portfolio construction, risk checks, and order preview without network access.

This is the acceptance path for CI and a new developer checkout.

### 3.2 Research

Fetches historical daily bars from Alpaca when credentials are available,
publishes immutable snapshots, and runs backtests. Research mode has no broker
write capability.

### 3.3 Paper

Reads an already published and validated snapshot, builds the next target
portfolio, previews orders, and submits them only when the operator adds an
explicit `--submit` flag.

Paper submission requires:

- `MLTRADE_ENVIRONMENT=paper`;
- valid Alpaca paper credentials;
- an Alpaca paper base URL;
- a healthy data snapshot;
- a completed decision session;
- a passing risk report;
- successful pre-submit broker reconciliation; and
- confirmation that live trading remains disabled.

## 4. Architecture

The MVP adds six bounded subsystems to the existing foundation.

### 4.1 Market Data

Responsibilities:

- fetch Alpaca daily bars through an adapter;
- accept deterministic fixture bars through the same interface;
- normalize bars into one canonical schema;
- enforce instrument, timestamp, price, volume, and uniqueness contracts;
- reject missing required sessions and incomplete latest-session data;
- write Parquet data files;
- compute content hashes; and
- publish an immutable `DatasetManifest` through `SnapshotStore`.

Canonical daily-bar fields:

- `instrument`
- `session`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `vwap`
- `trade_count`
- `source`
- `ingested_at`

Prices must be finite and positive. `high` must be at least `open`, `low`, and
`close`; `low` must be no greater than those values. Volume and trade count
must be non-negative. `(instrument, session)` must be unique.

### 4.2 Feature and Forecast Pipeline

The first model is a transparent cross-asset trend and momentum baseline.

Per instrument, the feature set contains:

- 21-session return;
- 63-session return;
- 126-session return;
- 21-session realized volatility;
- distance from the 100-session moving average; and
- 20-session average dollar volume.

Every feature row records:

- decision session;
- latest source session;
- dataset snapshot ID;
- feature-set version; and
- missing-value status.

Features for decision session `t` may use only bars from sessions at or before
`t`. A forecast for execution on session `t+1` is generated after session `t`
is complete.

The baseline forecast is a regularized linear regression trained to predict
the next 21-session return. Features are cross-sectionally standardized inside
each decision date. Training uses only observations whose labels are fully
known at the training cutoff.

If the training window is too short, a required feature is non-finite, or the
model produces a non-finite forecast, the pipeline fails closed.

### 4.3 Walk-Forward Backtester

The backtester:

- uses an expanding training window with a minimum of 504 sessions;
- retrains every 21 sessions;
- applies a 21-session embargo for overlapping forward-return labels;
- generates forecasts after each completed decision session;
- constructs targets using the production portfolio and risk code;
- applies targets at the next session's open;
- charges configurable spread and slippage costs;
- tracks cash, holdings, turnover, and portfolio value; and
- produces performance, exposure, turnover, and attribution reports.

The default cost model charges `5` basis points per traded notional. The value
is configurable and sensitivity reports run at `2`, `5`, and `10` basis
points.

The backtest reports:

- annualized return and volatility;
- Sharpe ratio with zero risk-free rate;
- maximum drawdown;
- turnover;
- total estimated costs;
- hit rate;
- exposure through time;
- per-instrument contribution; and
- comparison with equal-weight and cash baselines.

No performance threshold automatically approves paper submission. Reports are
evidence, not a promotion mechanism.

### 4.4 Portfolio and Risk

The portfolio builder converts forecasts to non-negative desired weights:

1. discard forecasts with failed data or model health;
2. rank valid positive forecasts;
3. scale conviction by inverse trailing volatility;
4. solve for weights nearest those scores while respecting hard constraints;
5. reserve at least `5%` cash; and
6. return cash when no positive, valid forecast exists.

The optimizer uses CVXPY and must return a deterministic result. If the solver
fails, returns an invalid status, or produces non-finite weights, the system
returns an all-cash target and a blocked risk report. It does not silently use
an unconstrained fallback.

Pre-trade checks validate:

- snapshot identity and health;
- decision-session freshness;
- model and feature versions;
- finite prices, forecasts, weights, and notionals;
- gross, net, cash, and position limits;
- order count and notional limits;
- minimum trade thresholds;
- duplicate execution intent;
- broker account status; and
- broker/internal cash, position, and open-order agreement.

Every check emits a structured result with `pass`, `warn`, or `block`.
Submission is allowed only when no result is `block`.

### 4.5 Execution and Reconciliation

The broker boundary is defined by a protocol implemented by:

- an in-memory simulated broker for tests and offline demos; and
- an Alpaca paper adapter for paper operation.

An execution intent has a stable identity derived from:

- account environment;
- strategy version;
- decision session;
- instrument;
- side; and
- target quantity.

The identity becomes the broker client order ID. Re-running the same decision
must find or reuse the existing intent and must not create a duplicate order.

The execution workflow is:

1. fetch broker account, positions, open orders, and recent fills;
2. reconcile them with internal state;
3. calculate delta orders from current to target positions;
4. persist the proposed execution intents;
5. run all pre-trade checks;
6. print a human-readable preview;
7. stop unless `--submit` is present;
8. submit approved intents individually;
9. persist broker order IDs and responses;
10. fetch broker state again; and
11. record reconciliation differences.

Any ambiguous timeout is resolved by querying the client order ID before a
retry. A timeout never causes a blind resubmission.

### 4.6 Operational State and CLI

PostgreSQL stores:

- dataset records and quality results;
- model runs and forecast batches;
- backtest runs and summary metrics;
- portfolio targets;
- risk check results;
- execution intents;
- broker orders and fills;
- reconciliation runs and differences; and
- audit events.

SQLite remains supported for offline demos and unit tests.

The CLI exposes:

- `mltrade demo run`
- `mltrade data ingest`
- `mltrade data validate`
- `mltrade research backtest`
- `mltrade portfolio build`
- `mltrade paper preview`
- `mltrade paper submit`
- `mltrade paper reconcile`
- `mltrade status`
- `mltrade doctor`

Commands emit structured logs and concise terminal summaries. Mutating paper
commands record an audit event before and after each broker interaction.

## 5. Data and State Flow

```text
fixture or Alpaca bars
        |
        v
normalize and validate
        |
        v
immutable Parquet snapshot + manifest
        |
        v
point-in-time features
        |
        v
walk-forward model and forecasts
        |
        v
constrained target portfolio
        |
        v
deterministic risk report
        |
        +------------------> backtest simulation
        |
        v
paper order preview
        |
   explicit --submit
        |
        v
Alpaca paper orders
        |
        v
reconciliation + audit
```

Research and paper operation share feature, forecast, portfolio, and risk
implementations. They differ only at data-source and broker adapter boundaries.

## 6. Failure Handling

The pipeline fails closed for:

- missing or stale required bars;
- duplicate or invalid market data;
- an incomplete latest session;
- snapshot hash or schema mismatch;
- insufficient model history;
- leakage or availability violations;
- non-finite features, forecasts, prices, or weights;
- optimizer failure;
- a breached risk limit;
- duplicate execution identity;
- unavailable or non-paper broker account;
- broker/internal state mismatch;
- ambiguous broker response that cannot be resolved by client order ID; or
- database or audit persistence failure.

A blocked run preserves its inputs, quality report, risk report, correlation
ID, and error details. No blocked run submits new orders.

## 7. Security and Safety Boundaries

- Credentials are accepted only through settings and environment variables.
- Secrets and database credentials are redacted from logs and serialization.
- Raw broker responses are sanitized before persistence.
- Paper and live broker URLs are validated as distinct environments.
- `live_trading_enabled` remains rejected by configuration.
- There is no live broker adapter or live-submit code path.
- `paper submit` requires the explicit command and `--submit`; scheduled jobs
  may call preview and reconciliation but not submission in the MVP.
- Dataset and artifact paths use the existing hardened path handling.

## 8. Testing Strategy

### 8.1 Unit Tests

- bar normalization and quality rules;
- exchange-session completeness;
- feature values and availability timestamps;
- label cutoff and embargo behavior;
- model input and output validation;
- cost and portfolio accounting;
- optimizer constraints and failure behavior;
- risk checks;
- execution identity and order deltas; and
- broker response sanitization.

### 8.2 Leakage Tests

Tests alter future bars and assert that earlier features, forecasts, and
portfolio targets remain unchanged. Tests also inject labels that overlap the
training cutoff and assert that they are excluded.

### 8.3 Simulation Tests

The simulated broker covers:

- complete fills;
- partial fills;
- rejected orders;
- timeouts before and after broker acceptance;
- restart and replay;
- duplicate submission attempts;
- stale broker positions; and
- reconciliation differences.

### 8.4 Integration and Contract Tests

- offline fixture to order-preview workflow;
- Parquet and manifest round trip;
- SQLite operational-state workflow;
- PostgreSQL persistence contract;
- Alpaca market-data response contract using recorded sanitized fixtures;
- Alpaca paper broker contract behind an opt-in environment marker; and
- Docker image execution of `mltrade demo run`.

The default test suite does not require network access or credentials.

## 9. Acceptance Criteria

The MVP is complete only when:

1. a clean checkout can run `uv run mltrade demo run`;
2. the demo publishes a validated immutable snapshot;
3. the demo creates point-in-time features and walk-forward forecasts;
4. the demo produces a backtest report with baseline comparisons and cost
   sensitivity;
5. the demo builds a portfolio satisfying every hard constraint;
6. the demo produces a deterministic paper-order preview;
7. replaying the same decision produces no duplicate execution intent;
8. injected stale data, future leakage, optimizer failure, and reconciliation
   mismatches each block submission;
9. PostgreSQL contract tests pass;
10. the application container runs the offline demo successfully;
11. Ruff and strict mypy pass;
12. unit and integration branch coverage remains at least `90%`;
13. optional Alpaca paper contract tests pass when credentials are supplied;
14. all operator workflows and safety defaults are documented; and
15. git contains no credentials, generated datasets, broker responses, or
   runtime state.

## 10. Deferred Work

- individual-stock and point-in-time index membership;
- SEC and FRED ingestion;
- short selling and borrow availability;
- mean-reversion and regime sleeves;
- multi-model capital allocation;
- MLflow and Optuna integration;
- automated scheduling;
- FastAPI and dashboards;
- alerts and remote deployment;
- automatic model promotion;
- live-money execution; and
- claims of production or live-trading readiness.

These items require separate designs after the vertical slice is operational
and its evidence can guide the next investments.
