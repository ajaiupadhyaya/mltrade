# MLTrade — Progress & Handoff

_Last updated: 2026-06-15._

## Paper-trading MVP: MERGED to `main`

The end-to-end paper-trading MVP (18 tasks) is complete and **merged to
`main`**. The active phase is the **local research experiment platform** (TOML
specs, content-addressed run records, MLflow tracking, resumable Optuna tuning,
run comparison, and the `mltrade experiment` CLI), built on
`codex/research-experiment-platform` and ready to merge: full quality gate green
(ruff + strict mypy clean, branch coverage ≥90%), `mltrade experiment doctor`
and an end-to-end run + 2-trial tune verified. See
`docs/runbooks/research-experiments.md`.

## Paper MVP status: ALL 18 TASKS IMPLEMENTED + REVIEWED

The full vertical slice from the approved plan
(`docs/superpowers/plans/2026-06-13-end-to-end-paper-trading-mvp.md`) is built,
each task TDD'd and passed through an adversarial spec+quality review.

**Local acceptance (verified on this dev box — NOT the M4/Docker host):**
- `ruff check .` — clean
- `mypy src` (strict) — clean, 50 source files
- `pytest tests/unit tests/integration` — **430 passed**, branch coverage
  **92.77%** (gate ≥90%)
- Contract tests (Postgres + Alpaca) — **skip cleanly** without their env vars
- `mltrade demo run` — exits 0, prints the five acceptance lines, **10 order
  intents**; running it twice reuses the snapshot + identical intent
  client_order_ids (replay/idempotency proven)
- Safety-failure suite (leakage / optimizer / risk / reconciliation /
  execution / paper) — 129 passed; stale snapshot, optimizer failure, and
  reconciliation mismatch each block submission
- Repo hygiene — no datasets, `*.db`, secrets, or broker responses tracked
  (previously-committed `data/operations.db` + fixture parquet were un-tracked)

## Docker acceptance steps — NOW PASSED (2026-06-14, Docker Desktop on this box)

1. **PostgreSQL contract tests** (design §9.9): `docker compose up -d --wait
   postgres` then `MLTRADE_TEST_DATABASE_URL=postgresql+psycopg://mltrade:mltrade@localhost:5432/mltrade
   uv run pytest tests/contract -m "contract and not alpaca"` → **3 passed**.
2. **Container build + in-image demo** (design §9.10): `docker build -t
   mltrade:mvp .` then `docker run --rm mltrade:mvp` (CMD `demo run`) → **exit 0**,
   prints the five acceptance lines + 10 intents. Non-root `uv run` hardening
   confirmed working.

Optional/UNRUN: live Alpaca paper contract (needs real paper creds + network):
`MLTRADE_RUN_ALPACA_CONTRACTS=true ... uv run pytest tests/contract -m alpaca`.

## Acceptance criteria (design §9) status
**1–12, 14, 15 ✅ all verified** (incl. §9.9 Postgres + §9.10 container).
§9.13 (optional live Alpaca contract) is the only unrun item — needs real paper
credentials. The MVP meets its acceptance bar.

## Known design notes / deferred (intentional)
- **Cold-start rebalance allowance:** going all-cash → ~95% invested in one
  rebalance exceeds the steady-state `maximum_rebalance_weight=0.50` cap, so the
  `demo run` / `paper preview` CLI commands relax it (rebalance≤1.0, order≤0.25)
  via `settings.model_copy` for the initial deployment. Steady-state `run_paper`
  keeps the strict 0.50/0.10 caps. This is documented in code + runbook.
- **Execution-service risk provenance:** `ExecutionService.preview` takes
  snapshot/version provenance params (default self-matched); the workflows pass
  REAL values so the snapshot_freshness / version gates fire (stale snapshot
  blocks — tested). Direct `preview()` callers that omit them get the lenient
  defaults.
- **Live Alpaca path not wired into a live workflow.** Adapters exist + are
  contract-tested with recorded responses; `run_paper` is fail-closed and
  paper-only. The reconcile/preview double-fetch of broker state (a TOCTOU note
  for the live adapter) is acceptable for the sim path; revisit before live use.
- Deferred per design §10: individual stocks, SEC/FRED, shorting, mean-reversion
  sleeve, MLflow/Optuna, scheduling, dashboards, live-money execution.

## How to resume / ship
```bash
cd "/Users/ajaiupadhyaya/Documents/machine learning/.worktrees/paper-trading-mvp"
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run pytest tests/unit tests/integration -q   # 430 pass
UV_CACHE_DIR=/private/tmp/mltrade-uv-cache uv run mltrade demo run                          # exit 0, 10 intents
```
Then on a Docker host run the two pending steps above. After that, the branch
is ready to merge to `main` (see `superpowers:finishing-a-development-branch`).

Conventions: Python 3.13, `uv` for everything, frozen Pydantic with
update-rejecting `model_copy` on persisted artifacts (transient input contexts
allow copy-with-update), strict mypy, ruff (E,F,I,B,UP,RUF), fail-closed
everywhere, deterministic outputs, no secrets/runtime data in git. Live trading
is structurally disabled and must stay that way.
