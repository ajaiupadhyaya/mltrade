# Institutional-Grade Upgrade — Real Data, Rigor, and a Research Terminal

Date: 2026-06-16
Status: implemented on branch `feat/institutional-grade`

## Goal

Elevate MLTrade from a polished demo on engineered synthetic fixtures to a
credible, institutionally-defensible research artifact — "Jane Street / JPMorgan
level." Two forks were chosen by the owner:

1. **Both rigor and a research terminal** (not one or the other).
2. **A real-market-data path** (not synthetic fixtures).

## Principle: honesty over a headline

The synthetic demo's Sharpe ≈ 2.25 was synthetic-by-construction — an instant
tell to any quant reviewer. The credible move is real out-of-sample results plus
the statistics that prove the methodology is disciplined, even when the honest
number is modest. The real OOS Sharpe is **0.72**, alpha vs SPY is **not
significant (t≈0.49)**, ~70% of variance is explained by static macro-factor
beta, and PBO ≈ 0.21. The dashboard states this plainly: MLTrade is a
disciplined, low-turnover **diversified risk-premia** allocation and a
demonstrated research *framework* — not an alpha engine.

## Architecture

### Real data, frozen for determinism
- `scripts/fetch_real_snapshot.py` — one-off fetcher (yfinance, `--with` only;
  never a runtime dep) freezes a rectangular, split/dividend-adjusted daily panel
  for the 10-ETF cross-asset universe (2007→2026) into
  `data/snapshots/real/daily_bars_<as_of>.parquet` + a provenance manifest
  (source, adjustment, sessions, row counts, SHA-256).
- `src/mltrade/data/snapshot.py` — `SnapshotBarSource` reads the frozen panel and
  yields `DailyBar`s; satisfies the `DailyBarSource` protocol, so it is a drop-in
  for `DeterministicBarSource`. The backtest engine is unchanged.
- `scripts/compute_trials.py` — precomputes the overfitting trials matrix (10
  ridge-α backtests) into `trials_<as_of>.parquet`, so DSR/PBO are instant and
  deterministic at export time.

### Analytics layer (`src/mltrade/analytics/`, additive, pure, deterministic)
- `returns.py` — return-series primitives.
- `performance.py` — Sortino, Calmar, drawdown depth/duration/recovery,
  distribution moments, historical & Cornish-Fisher VaR/CVaR, and chart series
  (drawdown, rolling Sharpe, histogram, monthly returns).
- `benchmark.py` — beta, Jensen's alpha **with t-stat/p-value** (OLS), tracking
  error, information ratio, capture ratios.
- `overfitting.py` — Deflated Sharpe Ratio + PSR (Bailey & López de Prado) and
  PBO via CSCV (Bailey, Borwein, López de Prado & Zhu).
- `attribution.py` — returns-based macro-factor exposure regression.

### Export v2 + real pipeline
- `workflows/research_real.py` — runs the backtest + analytics + the current
  decision (forecast → target → 17 pre-trade gates) on the real snapshot, no DB.
- `export.py` (schema v2) — assembles one deterministic JSON payload.

### Research terminal (`web/`)
Rebuilt into a seven-view terminal (Overview, Performance, Risk, Attribution,
Integrity, Portfolio, Experiments) with hand-rolled interactive SVG charts
(crosshair line charts, drawdown, rolling Sharpe, monthly heatmap, distribution,
factor bars, allocation donut) on the warm "Redwood" design system.

## Verification
- ruff clean; strict mypy clean (72 source files).
- Analytics + export test suite added (~95 new assertions); full suite green at
  ≥90% branch coverage.
- `mltrade export` deterministic; web app type-checks and builds; all seven views
  verified rendering via Playwright.

## Honest limitations (stated in the UI)
- DSR/PBO deflate for selection over the ridge-α grid only — not the full
  research process (universe, features, model choice).
- The strategy does not out-return SPY; its merit is risk-adjusted
  diversification and drawdown control.
- Live trading remains structurally disabled; the execution preview is simulated.
