# MLTrade

MLTrade is a research, risk, and execution platform for systematic long/short
trading in liquid US stocks and ETFs. Live trading is intentionally disabled.

## Requirements

- Python 3.13
- uv
- Docker with Compose

## Setup

```bash
uv sync --extra dev
cp .env.example .env
uv run mltrade doctor
```

## Verification

```bash
uv run ruff check .
uv run mypy src
uv run pytest tests/unit tests/integration --cov=mltrade --cov-report=term-missing
docker compose up -d --wait postgres
MLTRADE_TEST_DATABASE_URL=postgresql+psycopg://mltrade:mltrade@localhost:5432/mltrade \
  uv run pytest tests/contract -v
```

## Design

- Platform design:
  `docs/superpowers/specs/2026-06-12-us-equities-ml-trading-platform-design.md`
- Foundation implementation plan:
  `docs/superpowers/plans/2026-06-12-trading-platform-foundation.md`
