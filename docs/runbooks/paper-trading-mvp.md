# MLTrade Paper-Trading MVP — Operator Runbook

This document covers the day-to-day operator workflow for the MLTrade MVP.
Live trading is intentionally and structurally disabled; `MLTRADE_LIVE_TRADING_ENABLED=true`
is rejected at startup with a hard error.

---

## 1. Setup

```bash
uv sync --frozen --extra dev   # installs runtime + dev dependencies
cp .env.example .env           # fill in Alpaca paper credentials if needed
uv run mltrade doctor          # verify configuration and calendar
```

Environment variables use the `MLTRADE_` prefix and can be placed in `.env`
or exported in the shell. Credentials must **never** be committed to source
control; they are redacted from all log output.

---

## 2. Offline demo (cold-start)

```bash
uv run mltrade demo run
```

This runs the complete offline pipeline without any network access:

1. Generates deterministic fixture bars (seed 42, 2019-01-02 to present).
2. Validates data quality (fail-closed gate).
3. Publishes a Parquet snapshot under `$MLTRADE_DATA_ROOT/daily_bars/`.
4. Builds features, runs a walk-forward backtest, generates forecasts.
5. Optimises a portfolio target and runs an execution preview.
6. Persists all evidence to the SQLite operations DB.

**Expected output** (five lines):

```
snapshot: ok  (id=fixture-YYYY-MM-DD)
data quality: pass
backtest: complete  (N sessions)
risk: pass
paper orders: preview only  (N intents)
```

**Cold-start allowance:** `demo run` overrides `maximum_rebalance_weight` to
`1.0` and `maximum_order_weight` to `0.25` so the initial full-portfolio
deployment from all-cash is not blocked by the steady-state 50 %/10 % caps.
Steady-state production uses the default limits.

**Idempotency:** Re-running the demo reuses the existing snapshot if it already
exists on disk (detected by `FileExistsError` from the publisher). Execution
intent `client_order_id` values are deterministic; replaying never creates
duplicates.

**Artifact locations:**

| Artifact | Default path |
|---|---|
| Parquet snapshots | `$MLTRADE_DATA_ROOT/daily_bars/<snapshot-id>/` |
| Operations SQLite DB | `$MLTRADE_DATA_ROOT/operations.db` |

`MLTRADE_DATA_ROOT` defaults to `data/` relative to the working directory;
override with an absolute path (e.g. `/var/mltrade/data`) for production.

---

## 3. Research pipeline

```bash
# Ingest fixture data and publish a named snapshot
uv run mltrade data ingest

# Validate the latest snapshot quality gate (exits nonzero if blocked)
uv run mltrade data validate

# Run the walk-forward backtest from the latest snapshot
uv run mltrade research backtest

# Print target portfolio weights
uv run mltrade portfolio build
```

---

## 4. Paper execution preview

```bash
uv run mltrade paper preview
```

Runs a complete offline execution preview using `SimulatedBroker`. No orders
are placed. Applies the cold-start allowance (same as `demo run`). Prints each
intended trade with symbol, side, and share count. Exits nonzero if the risk
gate blocks.

---

## 5. Paper order submission

### Prerequisites

All of the following must hold before submitting paper orders:

| Prerequisite | How to satisfy |
|---|---|
| `MLTRADE_ENVIRONMENT=paper` | Set in environment or `.env` |
| Valid Alpaca paper API key | `MLTRADE_ALPACA_API_KEY=...` |
| Valid Alpaca paper API secret | `MLTRADE_ALPACA_API_SECRET=...` |
| Paper base URL | `MLTRADE_ALPACA_BASE_URL=https://paper-api.alpaca.markets` |
| Healthy snapshot | `mltrade data validate` exits 0 |
| Completed decision session | `mltrade research backtest` or `demo run` ran successfully |
| Passing risk report | `mltrade paper preview` exits 0 with `risk: pass` |
| Successful reconciliation | `mltrade paper reconcile` exits 0 with `reconciliation: pass` |
| Live trading disabled | `MLTRADE_LIVE_TRADING_ENABLED` must NOT be set (hard error if set) |

### Submitting

```bash
export MLTRADE_ENVIRONMENT=paper
export MLTRADE_ALPACA_API_KEY=<your-paper-key>
export MLTRADE_ALPACA_API_SECRET=<your-paper-secret>

# Dry-run first (no --submit flag)
uv run mltrade paper submit

# Actual submission — requires the explicit flag
uv run mltrade paper submit --submit
```

Without `--submit`, the command exits nonzero with an explanatory message.
This guards against accidental execution.

---

## 6. Reconciliation

```bash
uv run mltrade paper reconcile
```

Compares the internal position state (from the operations DB) against the
simulated broker. Exits nonzero and prints differences if reconciliation is
blocked.

**Diagnosing a blocked run:**

- Check the risk report's blocked check codes: any `status=block` entry in the
  JSON `checks` array must be resolved.
- Inspect reconciliation differences in the output: quantity mismatches indicate
  that positions have drifted since the last submission.
- Re-run `mltrade data validate` to confirm the snapshot is still healthy.

---

## 7. PostgreSQL contract tests

Bring up Postgres with Compose:

```bash
docker compose up -d --wait postgres
```

Run contract tests (excludes the Alpaca network contract):

```bash
MLTRADE_TEST_DATABASE_URL=postgresql+psycopg://mltrade:mltrade@localhost:5432/mltrade \
  uv run pytest tests/contract -m "contract and not alpaca"
```

### Optional Alpaca contract invocation

Requires live Alpaca paper credentials and network access:

```bash
MLTRADE_RUN_ALPACA_CONTRACTS=true \
MLTRADE_ALPACA_API_KEY=<key> \
MLTRADE_ALPACA_API_SECRET=<secret> \
  uv run pytest tests/contract -m alpaca
```

---

## 8. Container workflow

The Dockerfile runs as a non-root user (`mltrade`) and writes all runtime
artifacts to `/app/data` (owned by that user).

```bash
# Build the image
docker build -t mltrade:mvp .

# Run the offline demo
docker run --rm mltrade:mvp demo run

# Persist data across container restarts
docker run --rm -v mltrade-data:/app/data mltrade:mvp demo run

# Override data root
docker run --rm -e MLTRADE_DATA_ROOT=/mnt/data -v /host/data:/mnt/data \
  mltrade:mvp demo run
```

The container `ENTRYPOINT` is `uv run mltrade`; any CLI sub-command can be
passed as `CMD` arguments.

---

## 9. Secret handling

- Credentials are loaded via `Settings` from environment variables
  (`MLTRADE_ALPACA_API_KEY`, `MLTRADE_ALPACA_API_SECRET`).
- They are never committed to source control; `.gitignore` excludes `.env`.
- Pydantic `SecretStr` fields redact credentials from all log lines and
  `repr()` output. The `database_url` field is also redacted in serialised
  settings.
- Do not pass secrets as Docker build arguments; use `--env-file` or
  `-e KEY=value` at `docker run` time.

---

## 10. Live trading

**Live trading is unavailable and structurally disabled.**

Setting `MLTRADE_LIVE_TRADING_ENABLED=true` raises a hard `ValueError` at
`Settings` instantiation. The config validator also enforces that
`alpaca_base_url` points to the paper endpoint when `environment=paper`. There
is no code path to submit orders to a live brokerage.
