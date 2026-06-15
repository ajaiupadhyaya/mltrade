# MLTrade — Completion & Improvement Plan

_Last updated: 2026-06-15._

This document is the governing roadmap for finishing MLTrade and raising its
quality. It records the current state, the dashboard that was added, and a
prioritized backlog of improvements with effort/impact and rationale.

## 1. Current state

| Workstream | Branch | State |
| --- | --- | --- |
| Paper-trading MVP | `main` | Complete & merged. Research → quality gate → walk-forward ridge backtest → portfolio optimize → 17 pre-trade risk gates → execution preview. Offline/deterministic fixtures. Live trading structurally disabled. ~435 tests, ≥90% branch coverage, ruff + strict mypy clean. |
| Dashboard + export | `feat/dashboard` | Complete (this session). `mltrade export` → deterministic JSON; Vite + React + Tailwind "Redwood" dashboard reading it. Real per-session equity curve exposed additively from the backtest engine. Full suite green; verified live via Playwright. |
| Research-experiment platform | `codex/research-experiment-platform` | In progress. TOML specs + Optuna tuning + MLflow tracking + run comparison + `mltrade experiment` CLI. Tasks 1–3 committed (were red; being greened), Tasks 4–10 being implemented. |

### Integration plan (merge order + expected conflicts)

1. Merge `feat/dashboard` → `main` first (done + green, low risk).
2. Merge `codex/research-experiment-platform` → `main`. **Expected conflicts:**
   - `src/mltrade/backtest/engine.py` — both branches restructure `run_backtest`
     (dashboard extracted `_prepare_backtest`/`compute_equity_curve`; the
     experiment branch added `BacktestConfig`, evaluation windows, and strict
     cost-sensitivity reuse). Resolve by keeping the `_prepare_backtest`
     extraction and threading `BacktestConfig` through it.
   - `src/mltrade/cli.py` — dashboard added the top-level `export` command; the
     experiment branch added the `experiment` Typer sub-app. Likely
     non-overlapping; keep both.
   - `src/mltrade/export.py` — flip the experiment leaderboard from the empty
     placeholder to reading the real registry (`_experiments_payload`).
3. After integration, re-run `mltrade export` and commit the refreshed sample
   JSON so the dashboard shows live experiment runs.

## 2. Dashboard design ("Redwood")

Warm 70s California national-park aesthetic meets a precise financial terminal.
Source design in `design/mltrade-dashboard.pen` (Pencil) + `design/UXDCl.png`.

- **Palette:** warm oat canvas, cream surfaces, forest/pine/sage greens
  (positive), terracotta clay / redwood (blocks), mustard ochre (warnings).
- **Type:** Fraunces (display), Hanken Grotesk (UI), IBM Plex Mono (data).
- **Panels:** backtest hero (equity curve + KPIs + cost sensitivity +
  honest annualized baselines), portfolio allocation donut, the 17 risk gates,
  execution preview, experiment leaderboard, data-quality footer.
- **Honesty:** the dashboard is explicitly labeled "offline fixture · synthetic
  data · no live trading", and shows equal-weight outperforming the strategy on
  raw return (the strategy's edge is its Sharpe via low volatility), rather than
  cherry-picking.

## 3. Prioritized improvement backlog

Effort: S/M/L. Impact: low/med/high. Items are independent unless noted.

### Tier 1 — high impact

1. **Wire `mltrade paper submit` to drive `run_paper`** _(S, exec)._
   `AlpacaPaperBroker.submit()` and `run_paper()` are fully implemented and
   idempotent, but the CLI `paper submit` is a dead stub. Wire it (fail-closed,
   paper-only). _Needs real Alpaca paper credentials to verify end-to-end, so
   land it behind the existing env/cred guards with a contract test._
2. **Reconcile the risk policy/optimizer with the stated mandate** _(S, risk)._
   The design mandates long/short, 2.0x gross, market-neutral sleeve, and "no
   short without modeled borrow", but the implementation is hardcoded long-only
   (gross ≤ 1.0, net ∈ [0,1]). Either implement the long/short machinery or
   formally narrow the spec so design and code agree.
3. **Real Alpaca market-data ingestion into snapshots** _(M, data)._
   `AlpacaDataAdapter.fetch()` exists and is contract-tested but nothing calls
   it; `data ingest` only uses the synthetic `DeterministicBarSource`. Until
   real OHLCV flows into a versioned snapshot, every backtest/forecast runs on a
   synthetic uptrend. Wire a guarded real-data ingest path.
4. **Replace the signal-engineered fixture with an honest synthetic generator**
   _(M, testing)._ `fixtures.py` is tuned (+20%/+50% per-symbol drift) to give
   the ridge model "learnable signal" and guarantee ≥1 intent — the demo is
   green by construction. Add a neutral generator plus null/random-signal
   baselines so the backtest can demonstrate *no* edge when none exists.
5. **Drawdown-tier kill switch + portfolio kill-switch state machine**
   _(M, risk)._ Design §3.5/§6 require progressive drawdown controls
   (warn → reduce vol/gross → critical: cancel orders, block new risk, require
   operator approval) and a durable kill switch. Today risk is a stateless
   per-rebalance gate.
6. **Read-only observability surface** _(M, observability)._ Expose positions,
   cash, exposure, proposed/submitted/filled orders, risk checks, reconciliation
   diffs, and the audit trail over the operations DB (status JSON + small
   FastAPI). This also unlocks a live-data mode for the dashboard.
7. **Finish the research-experiment platform** _(L, modeling)._ In progress on
   `codex/research-experiment-platform` (runner, provenance, storage, MLflow,
   Optuna, comparison, CLI). Land it green and wire it into `mltrade export` so
   the dashboard leaderboard populates.

### Tier 2 — medium impact

8. **First-principles initial-allocation mode** _(S, risk)._ Replace the
   cold-start `model_copy` cap-relaxation hack (rebalance→1.0, order→0.25) with
   an explicit, tested initial-allocation policy.
9. **Operator end-to-end paper runbook + smoke command** _(S, ops)._ The live
   `run_paper → AlpacaPaperBroker → submit → reconcile` path has never been run
   end-to-end (only recorded-response contracts). Add a guarded smoke command.
10. **Extend the dashboard** _(M, UX)._ Add exposures, a risk-gate pass/warn/
    block timeline across sessions, and a reconciliation panel once the
    observability surface (item 6) exists.
11. **Backtest cost/fill realism** _(M, modeling)._ Model spread, slippage, and
    market impact, plus a turnover/cost penalty in the optimizer objective —
    today it is a flat bps cost on traded notional with 100% next-open fills.
12. **Daily P&L + attribution** _(M, observability)._ Decompose realized P&L by
    instrument, sleeve, factor, and cost (design §9.8) instead of the
    buy-and-hold per-symbol proxy.
13. **Walk-forward statistical validation** _(M, modeling)._ Bootstrap CIs,
    subperiod/regime breakdown, return concentration, and multiple-testing
    notes (design §7.4) — the backtest currently reports point estimates only.
14. **Scheduling for the daily decision/reconcile cadence** _(M, ops)._ Design
    §2.2/§8 specify a daily post-session decision point; everything is manual
    CLI today.
15. **Generalize the model ladder** _(M, modeling)._ Add transparent rule
    baselines, linear/logistic, regularized cross-sectional and time-series
    models, a mean-reversion sleeve, and a regime overlay, all compared to
    baselines.
16. **Point-in-time universe / survivorship handling** _(M, data)._ Enforce
    point-in-time membership before any individual-stock expansion (design §2.1);
    the fixed 10-ETF universe sidesteps this today.
17. **Corporate-actions adjustment + continuity test** _(M, data)._ Add splits/
    dividends/distributions adjustment with continuity checks (design §3.1/§7.2).

### Tier 3 — low impact / hygiene

18. **Repo/branch state cleanup** _(S, docs)._ Fix stale `MVP_PROGRESS.md`
    claims, prune leftover branches and `.worktrees/`, and remove the dangling
    agent-worktree gitlink.

## 4. Suggested sequencing

1. Land the experiment platform; integrate both feature branches to `main`.
2. Honesty pass: items 4 (honest fixture/baselines) and 18 (hygiene).
3. Real-data + observability spine: items 3, 6 — unblock a live dashboard mode.
4. Risk hardening: items 5, 8, 2.
5. Research depth: items 11, 13, 15, 12.
6. Operationalization: items 1, 9, 14.
7. Universe expansion prerequisites: items 16, 17.

Each item should follow the project's spec → plan → TDD → review cycle and keep
ruff/strict-mypy/coverage gates green. Live trading remains structurally
disabled until the live-readiness gates (real data, survivorship, kill switch,
attribution, statistical validation) are all met.
