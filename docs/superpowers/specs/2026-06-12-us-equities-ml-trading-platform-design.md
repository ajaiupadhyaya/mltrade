# US Equities ML Trading Platform Design

Date: 2026-06-12
Status: Approved design

## 1. Objective

Build a research, risk, and execution platform for systematic long/short trading
in liquid US stocks and ETFs. The platform will use machine learning,
statistics, mathematics, algorithms, and economic data to:

- train and evaluate strategies on historical point-in-time data;
- generate adaptive daily trading decisions;
- run realistic, reproducible backtests;
- deploy approved strategies to Alpaca paper trading;
- monitor model, portfolio, execution, and operational risk;
- support controlled retraining and model promotion; and
- keep live trading disabled until explicit readiness gates are satisfied.

The first paper portfolio has $1,000,000 in capital and an aggressive risk
profile. The system supports both market-neutral stock selection and a
directional regime overlay, but activates components progressively.

This system is decision-support and trading infrastructure, not a guarantee of
profit. Its primary design goals are correctness, reproducibility, risk
containment, and honest out-of-sample evaluation.

## 2. Initial Trading Mandate

### 2.1 Instruments

- Current constituents of the S&P 500, subject to data and liquidity checks.
- A curated set of 10 to 20 highly liquid US-listed ETFs representing broad
  equities, sectors, rates, credit, commodities, real estate, and international
  markets.
- Common stocks and standard ETFs only in the first release.
- No options, futures, leveraged or inverse ETFs, OTC securities, or crypto.

Historical research must not assume that today's S&P 500 membership existed in
the past. Until reliable point-in-time membership is available, experiments
using current constituents must be labeled as survivorship-biased and cannot
pass the live-readiness gate.

### 2.2 Horizon and Schedule

- Signals use daily data.
- Expected holding periods range from several days to several weeks.
- Risk, data health, positions, and broker state are checked every trading day.
- Trading is adaptive: orders are generated only when expected benefit exceeds
  estimated costs and configured signal-change thresholds.
- The default decision point is after a completed US market session, with
  orders intended for the next eligible session.

The research calendar must use an exchange-aware trading calendar. Partial
sessions, holidays, late data, and incomplete current-day bars must not be
treated as ordinary completed sessions.

### 2.3 Direction and Exposure

- Long and short individual stocks.
- Market-neutral stock-selection sleeve.
- Directional ETF and equity overlay driven by the regime model.
- Initial risk ceiling: 2.0x gross exposure and net exposure between -50% and
  +50%.
- Target annualized volatility: 15%, subject to model confidence and risk
  controls.
- Maximum drawdown objective: 20%; this is a control threshold, not a promise.

## 3. Architecture

The system consists of seven independently testable layers.

### 3.1 Data Platform

Responsibilities:

- ingest daily OHLCV data, distributions, splits, and other corporate actions;
- ingest SEC company facts and filing metadata;
- ingest FRED macroeconomic series with release and revision timestamps when
  available;
- ingest ETF and benchmark reference data;
- preserve raw source responses and normalized point-in-time datasets;
- run schema, freshness, completeness, duplicate, range, and continuity checks;
  and
- publish immutable, versioned data snapshots for research and production.

Initial storage:

- Parquet for immutable analytical datasets;
- DuckDB and Polars for local analytical queries and transformations; and
- PostgreSQL for operational state, dataset manifests, lineage, and audit data.

Low-cost data is acceptable for prototyping. A dataset may support
live-readiness only if its licensing, adjustment methodology, timestamp
semantics, revision behavior, and survivorship properties are documented.

### 3.2 Feature Platform

Feature families:

- momentum and trend;
- valuation;
- profitability and quality;
- volatility and downside risk;
- liquidity and turnover;
- short-term reversal and mean reversion;
- market, sector, and style exposures;
- macroeconomic conditions;
- cross-sectional ranks and robust z-scores; and
- regime and stress indicators.

Every feature records:

- observation timestamp;
- earliest availability timestamp;
- source dataset version;
- transformation version;
- universe version; and
- missing-value policy.

Feature computation must prevent future data, revised data, post-event
fundamentals, and future universe membership from leaking into training.

### 3.3 Model Laboratory

The initial modular ensemble contains:

1. A cross-sectional factor model that ranks stocks by expected relative
   return.
2. A statistical mean-reversion model that identifies temporary dislocations.
3. A regime model that estimates risk conditions and adjusts net exposure,
   gross exposure, and ensemble weights within hard limits.

Initial model ladder:

- transparent rules and economic baselines;
- linear and logistic models;
- regularized cross-sectional models;
- statistical time-series models;
- tree-based models such as LightGBM; and
- more complex models only after demonstrating incremental out-of-sample value.

Experiments use walk-forward evaluation. Where labels overlap, validation uses
purging and embargoes. Hyperparameter selection occurs only within training
windows. The final test period remains untouched until a candidate is frozen.

MLflow stores experiment metadata, metrics, artifacts, dataset versions,
feature versions, code revisions, and model status. Optuna may tune bounded
search spaces, but tuning budget and trial history must be retained.

### 3.4 Strategy Ensemble

Each model produces a standardized forecast containing:

- instrument;
- forecast horizon;
- expected return or relative score;
- confidence or uncertainty estimate;
- model version; and
- generation timestamp.

The ensemble combines forecasts using constrained, versioned weights. No model
may allocate capital merely because it exists in the ensemble.

Activation sequence:

1. Validate and paper-trade the factor sleeve independently.
2. Validate and paper-trade the mean-reversion sleeve independently.
3. Run the regime model in shadow mode.
4. Allow the regime model to adjust exposure after it passes its own gates.
5. Enable combined capital allocation only after attribution confirms that the
   interaction remains stable after estimated costs.

The initial combination method is a transparent weighted blend. A learned
mixture-of-experts is outside the first implementation scope.

### 3.5 Portfolio and Risk Engine

The portfolio optimizer converts forecasts into target positions using CVXPY.
It considers expected return, risk, turnover, estimated costs, and constraints.

Initial constraints:

- gross exposure no greater than 2.0x;
- net exposure between -50% and +50%;
- 15% annualized volatility target;
- configurable single-name, ETF, sector, industry, and factor-exposure limits;
- beta limits for the market-neutral sleeve;
- liquidity limits based on conservative participation in average daily
  dollar volume;
- turnover and transaction-cost penalties;
- minimum trade thresholds;
- no short order without modeled borrow availability and fees; and
- no trade using stale, incomplete, or quarantined data.

Exact position, sector, beta, participation, and turnover limits will be set in
configuration during implementation and verified with scenario tests before
paper deployment.

Drawdown controls reduce risk progressively:

- warning tier: stop increasing risk and flag review;
- reduction tier: lower target volatility and gross exposure;
- critical tier: cancel open orders, block new risk, and require explicit
  operator approval to resume.

Hard risk controls are deterministic and cannot be overridden by model output.

### 3.6 Backtest and Execution Engine

Backtests and paper trading share strategy, portfolio, and risk code. Adapters
provide historical or live data and simulated or broker execution.

The backtester models:

- commissions and regulatory fees;
- bid-ask spread;
- market impact and slippage;
- next-session availability of signals;
- delayed, partial, and rejected fills;
- dividends, splits, and delistings when source data supports them;
- borrow fees and unavailable shorts;
- cash, margin, and financing assumptions; and
- order cancellation and replacement.

Alpaca is the first broker adapter. Paper fills are not treated as proof of
live execution quality because paper environments do not fully reproduce
latency, queue priority, market impact, borrow constraints, or all fees.

Execution is idempotent. Client order identifiers, reconciliation, and durable
state prevent duplicate orders after retries or restarts.

### 3.7 Operations and Reporting

The initial operational interface consists of FastAPI endpoints, command-line
workflows, structured logs, and generated reports. A dedicated dashboard is a
later milestone.

Operations must expose:

- current positions, cash, exposure, and risk;
- proposed, approved, submitted, rejected, canceled, and filled orders;
- model forecasts, confidence, versions, and health;
- daily performance and strategy attribution;
- data freshness and quality incidents;
- reconciliation differences;
- drift and retraining status;
- active warnings, blocks, and kill switches; and
- an immutable decision and operator audit trail.

## 4. End-to-End Data Flow

1. Scheduled workflows ingest market, corporate-action, SEC, and FRED data.
2. Data contracts validate timestamps, schemas, ranges, completeness, and
   freshness.
3. Validated source data is frozen into a versioned snapshot.
4. Feature jobs create point-in-time feature snapshots with lineage.
5. Training workflows perform walk-forward fitting and candidate evaluation.
6. Approved models generate forecasts with confidence estimates.
7. The ensemble combines forecasts under its active configuration.
8. The optimizer generates target positions.
9. Deterministic pre-trade checks approve, resize, or reject orders.
10. The execution adapter submits approved orders to Alpaca paper trading.
11. Reconciliation compares broker orders, fills, positions, and cash with
    internal state.
12. Monitoring records performance, attribution, drift, and incidents.
13. Retraining creates a challenger model; it never silently replaces the
    champion.

## 5. Retraining and Model Governance

Retraining is scheduled and event-aware, not continuously self-modifying.

- Routine retraining cadence is configurable and begins monthly.
- Drift or data-regime alerts may trigger an evaluation, not automatic
  promotion.
- Every candidate is compared with the active champion on identical frozen
  datasets.
- Candidates run in shadow mode before receiving paper capital.
- Promotion and rollback are explicit, versioned operator actions.
- Historical model artifacts and decisions remain reproducible.

Promotion requires:

- improvement over simple economic and statistical baselines;
- positive net performance after conservative costs;
- acceptable drawdown and tail behavior;
- stability across walk-forward periods and market regimes;
- no material leakage or data-quality findings;
- sensible feature and strategy attribution;
- capacity consistent with the paper portfolio;
- successful shadow operation; and
- successful operational and failure-recovery tests.

## 6. Safety and Failure Handling

Kill switches block trading when any critical condition occurs:

- stale or missing required data;
- failed or incomplete feature generation;
- model artifact or schema mismatch;
- non-finite, extreme, or structurally invalid forecasts;
- breached exposure, leverage, liquidity, or drawdown limits;
- abnormal order count, size, price, or notional;
- broker disconnect or repeated rejection;
- position, cash, or fill reconciliation mismatch;
- duplicate execution intent; or
- corrupted or unavailable operational state.

On critical failure, the default action is to cancel eligible open orders,
refuse new risk, preserve evidence, and require operator review. Automated
liquidation is not the default because it can amplify failures; emergency
de-risking behavior must be separately configured and tested.

## 7. Testing Strategy

### 7.1 Unit Tests

- feature calculations and availability timestamps;
- cost, borrow, and financing calculations;
- portfolio constraints and optimizer fallbacks;
- risk-limit and kill-switch behavior;
- order state transitions; and
- performance and attribution calculations.

### 7.2 Data and Leakage Tests

- point-in-time joins;
- corporate-action adjustments;
- lag and release-date enforcement;
- universe membership timing;
- feature reproducibility;
- missing-session and stale-data detection; and
- explicit tests that inject future information and expect failure.

### 7.3 Simulation and Integration Tests

- delayed, partial, canceled, and rejected fills;
- broker timeouts and retry idempotency;
- restart and state recovery;
- reconciliation mismatches;
- missing and revised source data;
- optimizer infeasibility;
- extreme volatility and gap scenarios; and
- full historical-to-order workflow tests.

### 7.4 Statistical Validation

- walk-forward out-of-sample results;
- benchmark and naive-baseline comparisons;
- turnover and cost sensitivity;
- parameter and universe sensitivity;
- subperiod and regime analysis;
- bootstrap confidence intervals where appropriate;
- concentration of returns by date, symbol, and sleeve; and
- multiple-testing awareness in experiment review.

## 8. Technology Choices

- Python 3.13 for research, models, risk, workflows, and execution.
- Polars, DuckDB, and Parquet for analytical data.
- PostgreSQL for operational and audit state.
- scikit-learn, LightGBM, and statsmodels for initial modeling.
- Optuna for bounded hyperparameter searches.
- MLflow for experiment and model tracking.
- CVXPY for constrained optimization.
- FastAPI for control and monitoring APIs.
- Prefect for ingestion, training, reconciliation, and scheduled workflows.
- Alpaca for initial market data integration and paper execution.
- Docker for reproducible local and deployed services.
- Pytest for unit, data, simulation, integration, and contract tests.

The core trading system is backend-first. A web dashboard and a specific cloud
deployment platform are intentionally deferred until the research-to-paper
pipeline is reliable.

## 9. Delivery Phases

### Phase 1: Foundation

- repository and Python project structure;
- configuration and secrets boundaries;
- trading calendar and instrument model;
- Parquet, DuckDB, and PostgreSQL storage contracts;
- structured logging and audit primitives; and
- deterministic test fixtures.

### Phase 2: Data and Features

- daily market and corporate-action ingestion;
- SEC and FRED adapters;
- dataset manifests and quality gates;
- point-in-time feature computation; and
- survivorship and availability labeling.

### Phase 3: Research and Backtesting

- event-driven daily backtester;
- transaction, borrow, and slippage models;
- walk-forward experiment runner;
- baseline strategies; and
- performance, risk, and attribution reports.

### Phase 4: Independent Strategy Sleeves

- factor ranking sleeve;
- mean-reversion sleeve;
- regime model in shadow mode;
- experiment tracking and candidate registry; and
- sleeve-specific promotion gates.

### Phase 5: Portfolio and Risk

- forecast normalization and ensemble;
- constrained portfolio optimizer;
- exposure, liquidity, drawdown, and kill-switch controls;
- scenario and stress tests; and
- adaptive trade thresholds.

### Phase 6: Alpaca Paper Trading

- market data and broker adapters;
- idempotent order management;
- daily scheduling;
- fill and position reconciliation;
- operational reports and alerts; and
- shadow-to-paper promotion workflow.

### Phase 7: Readiness Evaluation

- paper-versus-backtest execution analysis;
- model and data drift review;
- operational reliability metrics;
- incident and recovery exercises;
- documented live-readiness review; and
- live trading remains disabled unless separately designed and approved.

## 10. Initial Success Criteria

The first major milestone is complete when the platform can reproducibly:

1. create a versioned historical dataset for the approved universe;
2. prove feature availability and reject look-ahead leakage;
3. train and walk-forward evaluate each strategy sleeve;
4. compare every model with transparent baselines after conservative costs;
5. construct a constrained $1,000,000 target portfolio;
6. simulate adverse fills, borrow failures, outages, and restarts;
7. generate and reconcile Alpaca paper orders without duplication;
8. explain daily P&L by instrument, sleeve, factor, and cost; and
9. block trading automatically when a critical control fails.

No Sharpe ratio, return target, or calendar deadline alone qualifies a strategy
for live trading. Readiness requires statistical evidence, execution evidence,
operational reliability, and explicit human approval.

## 11. Deferred Scope

- live-money order submission;
- options, futures, crypto, and non-US markets;
- intraday or tick-level trading;
- high-frequency execution;
- automated model promotion;
- learned mixture-of-experts allocation;
- alternative data, news, and language-model signals;
- tax optimization;
- multi-broker smart order routing; and
- a production web dashboard.
