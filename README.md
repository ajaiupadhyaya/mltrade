# MLTrade

MLTrade is a research, risk, and execution platform for systematic long/short
trading in liquid US stocks and ETFs. Live trading is intentionally disabled.

## Requirements

- Python 3.13
- uv
- Docker with Compose (for contract tests)

## Setup

```bash
uv sync --frozen --extra dev
cp .env.example .env
uv run mltrade doctor
```

## Quick start — offline demo

```bash
uv run mltrade demo run
```

Expected output:

```
snapshot: ok  (id=fixture-YYYY-MM-DD)
data quality: pass
backtest: complete  (N sessions)
risk: pass
paper orders: preview only  (N intents)
```

No network access required. The demo runs the full pipeline (data ingestion,
quality gate, backtest, portfolio optimisation, risk check, execution preview)
using deterministic fixture data. Re-running reuses the existing snapshot and
produces no duplicate orders.

## Verification

```bash
# Lint and type-check
uv run ruff check .
uv run mypy src

# Unit + integration tests
uv run pytest tests/unit tests/integration -q

# Contract tests (requires Postgres)
docker compose up -d --wait postgres
MLTRADE_TEST_DATABASE_URL=postgresql+psycopg://mltrade:mltrade@localhost:5432/mltrade \
  uv run pytest tests/contract -m "contract and not alpaca"
```

## Container

```bash
docker build -t mltrade:mvp .
docker run --rm mltrade:mvp demo run
```

## Operator runbook

Full workflow documentation (setup, demo, research, paper submission,
reconciliation, PostgreSQL contract tests, secret handling, container usage):

`docs/runbooks/paper-trading-mvp.md`

## Design

- Platform design:
  `docs/superpowers/specs/2026-06-12-us-equities-ml-trading-platform-design.md`
- Foundation implementation plan:
  `docs/superpowers/plans/2026-06-12-trading-platform-foundation.md`
