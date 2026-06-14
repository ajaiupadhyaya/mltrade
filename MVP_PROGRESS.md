# MLTrade Paper-Trading MVP — Progress & Handoff

_Last updated: 2026-06-13. Branch: `codex/paper-trading-mvp` (in worktree
`.worktrees/paper-trading-mvp`). Not yet merged to `main`._

## Where things stand

The MVP is being built by executing the approved 18-task plan
(`docs/superpowers/plans/2026-06-13-end-to-end-paper-trading-mvp.md`) against the
design (`docs/superpowers/specs/2026-06-13-end-to-end-paper-trading-mvp-design.md`)
using TDD + per-task adversarial review.

**State at this checkpoint:** clean working tree, everything committed.
At HEAD `6aece24`: `ruff` clean, strict `mypy` clean, **284 unit+integration
tests passing**. This is a safe stopping point.

### Tasks 1–8: DONE and review-approved
| Task | What | Key commits |
|------|------|-------------|
| 1 | MVP deps + safe settings | `1b02fc3`, `67f3056` |
| 2 | ETF universe + canonical `DailyBar` | `e135ee1`, `c843c77`, `ce20d0f` |
| 3 | Deterministic offline market data (`data/fixtures.py`) | `a562464`, `ecda367` (fixed a CRITICAL cross-process determinism bug: was using salted builtin `hash()`) |
| 4 | Fail-closed data quality (`data/quality.py`) | `6a79f33`, `e25812e` |
| 5 | Immutable Parquet snapshot publication (`data/publication.py`) | `4df6133`, `5768d12` |
| 6 | Point-in-time features + labels (`features/`) | `0c5b231` (leakage-safe; verified) |
| 7 | Embargoed walk-forward ridge forecasts (`models/`) | `08cf9e6`, `5dd1514` |
| 8 | Constrained CVXPY portfolio optimizer (`portfolio/`) | `09fcae6`, `116979b` |

### Task 9: IMPLEMENTED, committed, self-verified — review NOT yet completed
- `risk/checks.py`, `risk/policy.py`, `tests/unit/test_risk_policy.py` (commit `6aece24`).
- 17 pre-trade checks, `evaluate_pre_trade(context) -> RiskReport`, fail-closed.
- 66 new tests pass; ruff + mypy clean.
- **RESUME HERE:** the adversarial spec+quality review of Task 9 was interrupted
  before completion. Re-run a review of `git diff 116979b..6aece24` focused on:
  (1) every check is *always* emitted exactly once (no skip path → `by_code`
  can't raise); (2) `evaluate_pre_trade` cannot RAISE on non-finite Decimal
  inputs (NaN comparisons raise `InvalidOperation` → would be fail-OPEN; the
  impl claims early finiteness guards — verify); (3) no inverted threshold
  comparisons; (4) `valid_context` genuinely passes vs. dodges a check.

## Remaining work (Tasks 10–18)

Each is a full TDD task in the plan (read the plan section for exact spec/tests):

- **Task 10** — Shared walk-forward backtester (`backtest/accounting.py`,
  `engine.py`, `reporting.py`). Deterministic next-session loop, cost
  sensitivity at 2/5/10 bps, full metrics + equal-weight/cash baselines.
- **Task 11** — Broker contracts + simulated execution (`execution/broker.py`,
  `simulated.py`, `intents.py`). Stable `client_order_id` (sha256), dedup,
  fill/partial/reject/timeout outcomes.
- **Task 12** — Reconciliation + safe submission (`execution/reconciliation.py`,
  `service.py`). Preview/submit, fail-closed on diffs, idempotent timeout
  handling (never blind resubmit).
- **Task 13** — Persist operational evidence (`operations/models.py` +
  `repositories.py`). SQL tables, UUID PKs, unique `client_order_id`; SQLite +
  Postgres contract tests.
- **Task 14** — Offline/research/paper workflows (`workflows/demo.py`,
  `research.py`, `paper.py`). `run_demo` end-to-end offline + replay idempotency.
- **Task 15** — MVP operator CLI (extend `cli.py`): `demo run`, `data
  ingest/validate`, `research backtest`, `portfolio build`, `paper
  preview/submit --submit/reconcile`, `status`, `doctor`. Dependency injection
  for settings/clock/source/db/broker.
- **Task 16** — Alpaca data + paper adapters (`data/alpaca.py`,
  `execution/alpaca.py`). HTTPX, sanitized response contracts (respx), opt-in
  live contract gated by `MLTRADE_RUN_ALPACA_CONTRACTS=true`.
- **Task 17** — Document + containerize (`README`, `docs/runbooks/`,
  `Dockerfile` `CMD ["demo","run"]`, hygiene test, gitignore runtime artifacts).
- **Task 18** — Final acceptance: ruff/mypy/coverage ≥90, run `demo run` twice
  (replay reuses identities), Postgres contracts, container build/run, safety
  failure tests, repo hygiene, evidence report.

"Complete" = the 15 acceptance criteria in §9 of the design doc.

## How to resume

```bash
cd "/Users/ajaiupadhyaya/Documents/machine learning/.worktrees/paper-trading-mvp"
# sanity check the checkpoint
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest tests/unit tests/integration -q
# then finish reviewing Task 9, then continue at Task 10 in the plan
```

Conventions in force: Python 3.13, `uv` for everything, frozen Pydantic models
with `model_copy`/`copy` overrides that reject updates on *persisted* artifacts
(transient input contexts like `PreTradeContext` intentionally allow
copy-with-update), strict mypy, ruff (E,F,I,B,UP,RUF), fail-closed everywhere,
deterministic outputs, no secrets/runtime data in git. Verify with
`UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run <ruff|mypy|pytest>`.

Live trading is structurally disabled and must stay that way.
