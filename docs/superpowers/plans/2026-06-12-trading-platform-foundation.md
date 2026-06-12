# Trading Platform Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the tested repository foundation for the US equities ML trading platform: packaging, configuration, core domain contracts, NYSE session logic, immutable dataset manifests, operational audit persistence, structured logging, and local PostgreSQL infrastructure.

**Architecture:** Use a Python 3.13 `src/` package named `mltrade`. Pure domain modules contain immutable value objects and protocols; infrastructure modules adapt calendars, filesystems, logging, and SQL databases. Tests use temporary directories and SQLite where possible, while Docker Compose provides PostgreSQL for contract tests and future services.

**Tech Stack:** Python 3.13, uv, Pydantic v2, pydantic-settings, exchange-calendars, SQLAlchemy 2, psycopg 3, structlog, Typer, DuckDB, Polars, Pytest, Ruff, mypy, Docker Compose.

**Approved design:** `docs/superpowers/specs/2026-06-12-us-equities-ml-trading-platform-design.md`

---

## Scope

This is Phase 1 of the approved design. It deliberately does not ingest market
data, train models, backtest strategies, optimize portfolios, or submit broker
orders. Those capabilities require separate implementation plans:

1. Data ingestion and point-in-time features.
2. Research engine and event-driven backtesting.
3. Independent alpha sleeves and model governance.
4. Portfolio optimization and deterministic risk controls.
5. Alpaca paper execution and reconciliation.
6. Operational reporting and readiness evaluation.

Phase 1 is complete only when later phases can depend on stable, tested
contracts instead of inventing their own configuration, time, identity,
storage, and audit semantics.

## File Structure

```text
.
├── .env.example                         # Non-secret local configuration template
├── .gitignore                           # Python, environment, data, and tool exclusions
├── Dockerfile                           # Reproducible Python 3.13 runtime
├── README.md                            # Local setup and verification commands
├── compose.yaml                         # Local PostgreSQL service
├── pyproject.toml                       # Package metadata, dependencies, and tool config
├── src/mltrade/
│   ├── __init__.py                      # Package version
│   ├── cli.py                           # `mltrade doctor` command
│   ├── config.py                        # Validated environment configuration
│   ├── calendar.py                      # Trading-session abstraction and XNYS adapter
│   ├── domain/
│   │   ├── __init__.py
│   │   ├── instruments.py               # Symbol and instrument identity
│   │   └── time.py                      # UTC normalization helpers
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── manifests.py                 # Immutable dataset manifest model
│   │   └── snapshots.py                 # Atomic manifest persistence and lookup
│   └── operations/
│       ├── __init__.py
│       ├── audit.py                     # Append-only audit service
│       ├── database.py                  # SQLAlchemy engine/session helpers
│       ├── logging.py                   # JSON logging configuration
│       └── models.py                    # Operational SQLAlchemy models
└── tests/
    ├── conftest.py                      # Deterministic environment and clock fixtures
    ├── contract/
    │   └── test_postgres_audit.py       # PostgreSQL integration contract
    ├── integration/
    │   └── test_doctor.py               # CLI foundation health check
    └── unit/
        ├── test_audit.py
        ├── test_calendar.py
        ├── test_config.py
        ├── test_instruments.py
        ├── test_snapshots.py
        └── test_time.py
```

## Task 1: Bootstrap the Python Package

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`
- Create: `src/mltrade/__init__.py`
- Create: `src/mltrade/domain/__init__.py`
- Create: `src/mltrade/storage/__init__.py`
- Create: `src/mltrade/operations/__init__.py`
- Create: `tests/unit/test_package.py`

- [ ] **Step 1: Write the package smoke test**

```python
# tests/unit/test_package.py
import mltrade


def test_package_exposes_version() -> None:
    assert mltrade.__version__ == "0.1.0"
```

- [ ] **Step 2: Create project metadata and tool configuration**

```toml
# pyproject.toml
[build-system]
requires = ["hatchling>=1.27"]
build-backend = "hatchling.build"

[project]
name = "mltrade"
version = "0.1.0"
description = "Research, risk, and execution platform for systematic US equities trading"
readme = "README.md"
requires-python = ">=3.13,<3.14"
dependencies = [
  "duckdb>=1.2",
  "exchange-calendars>=4.8",
  "pandas>=2.2",
  "polars>=1.26",
  "psycopg[binary]>=3.2",
  "pydantic>=2.11",
  "pydantic-settings>=2.8",
  "sqlalchemy>=2.0",
  "structlog>=25.1",
  "typer>=0.15",
]

[project.optional-dependencies]
dev = [
  "mypy>=1.15",
  "pytest>=8.3",
  "pytest-cov>=6.0",
  "ruff>=0.11",
]

[project.scripts]
mltrade = "mltrade.cli:app"

[tool.hatch.build.targets.wheel]
packages = ["src/mltrade"]

[tool.pytest.ini_options]
addopts = "-ra --strict-config --strict-markers"
testpaths = ["tests"]
markers = [
  "contract: requires an external service such as PostgreSQL",
]

[tool.ruff]
target-version = "py313"
line-length = 88
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "RUF"]

[tool.mypy]
python_version = "3.13"
strict = true
packages = ["mltrade"]

[tool.coverage.run]
branch = true
source = ["mltrade"]

[tool.coverage.report]
show_missing = true
skip_covered = true
fail_under = 90
```

```gitignore
# .gitignore
.DS_Store
.env
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/
dist/
build/
*.egg-info/
data/raw/
data/normalized/
data/features/
artifacts/
mlruns/
```

```markdown
# MLTrade

MLTrade is a research, risk, and execution platform for systematic long/short
trading in liquid US stocks and ETFs. Live trading is intentionally disabled.
```

```python
# src/mltrade/__init__.py
__version__ = "0.1.0"
```

Create empty package markers:

```python
# src/mltrade/domain/__init__.py
```

```python
# src/mltrade/storage/__init__.py
```

```python
# src/mltrade/operations/__init__.py
```

- [ ] **Step 3: Install the locked development environment**

Run:

```bash
uv sync --extra dev
```

Expected: `uv.lock` is created and all dependencies install successfully under
Python 3.13.

- [ ] **Step 4: Run the smoke test**

Run:

```bash
uv run pytest tests/unit/test_package.py -v
```

Expected: `1 passed`.

- [ ] **Step 5: Run static checks**

Run:

```bash
uv run ruff check .
uv run mypy src
```

Expected: both commands exit successfully with no findings.

- [ ] **Step 6: Commit the bootstrap**

```bash
git add pyproject.toml uv.lock .gitignore README.md src tests/unit/test_package.py
git commit -m "build: bootstrap mltrade package"
```

## Task 2: Add Validated Configuration and Secret Boundaries

**Files:**
- Create: `.env.example`
- Create: `src/mltrade/config.py`
- Create: `tests/unit/test_config.py`

- [ ] **Step 1: Write failing configuration tests**

```python
# tests/unit/test_config.py
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from mltrade.config import Environment, Settings


def test_settings_use_safe_local_defaults(tmp_path: Path) -> None:
    settings = Settings(data_root=tmp_path)

    assert settings.environment is Environment.LOCAL
    assert settings.live_trading_enabled is False
    assert settings.data_root == tmp_path
    assert settings.database_url.startswith("sqlite+pysqlite://")


def test_secret_values_are_not_revealed(tmp_path: Path) -> None:
    settings = Settings(
        data_root=tmp_path,
        alpaca_api_key=SecretStr("paper-key"),
        alpaca_api_secret=SecretStr("paper-secret"),
    )

    rendered = repr(settings)
    assert "paper-key" not in rendered
    assert "paper-secret" not in rendered


def test_environment_variables_use_mltrade_prefix(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MLTRADE_ENVIRONMENT", "paper")
    monkeypatch.setenv("MLTRADE_DATA_ROOT", str(tmp_path))

    settings = Settings()

    assert settings.environment is Environment.PAPER
    assert settings.data_root == tmp_path


def test_live_trading_cannot_be_enabled_in_foundation(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="not available"):
        Settings(data_root=tmp_path, live_trading_enabled=True)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run pytest tests/unit/test_config.py -v
```

Expected: collection fails because `mltrade.config` does not exist.

- [ ] **Step 3: Implement the settings model**

```python
# src/mltrade/config.py
from enum import StrEnum
from pathlib import Path

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    LOCAL = "local"
    TEST = "test"
    PAPER = "paper"
    LIVE = "live"


class Settings(BaseSettings):
    environment: Environment = Environment.LOCAL
    data_root: Path = Path("data")
    database_url: str = "sqlite+pysqlite:///data/operations.db"
    alpaca_api_key: SecretStr | None = None
    alpaca_api_secret: SecretStr | None = None
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    live_trading_enabled: bool = False
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_prefix="MLTRADE_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="forbid",
    )

    @field_validator("data_root")
    @classmethod
    def make_data_root_absolute(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        normalized = value.upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if normalized not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return normalized

    @model_validator(mode="after")
    def reject_live_trading(self) -> "Settings":
        if self.live_trading_enabled:
            raise ValueError("live trading is not available in this release")
        return self
```

```dotenv
# .env.example
MLTRADE_ENVIRONMENT=local
MLTRADE_DATA_ROOT=./data
MLTRADE_DATABASE_URL=postgresql+psycopg://mltrade:mltrade@localhost:5432/mltrade
MLTRADE_ALPACA_API_KEY=
MLTRADE_ALPACA_API_SECRET=
MLTRADE_ALPACA_BASE_URL=https://paper-api.alpaca.markets
MLTRADE_LIVE_TRADING_ENABLED=false
MLTRADE_LOG_LEVEL=INFO
```

- [ ] **Step 4: Run configuration tests**

Run:

```bash
uv run pytest tests/unit/test_config.py -v
```

Expected: `4 passed`.

- [ ] **Step 5: Commit configuration**

```bash
git add .env.example src/mltrade/config.py tests/unit/test_config.py
git commit -m "feat: add validated application settings"
```

## Task 3: Define Instrument Identity and UTC Time Contracts

**Files:**
- Create: `src/mltrade/domain/instruments.py`
- Create: `src/mltrade/domain/time.py`
- Create: `tests/unit/test_instruments.py`
- Create: `tests/unit/test_time.py`

- [ ] **Step 1: Write failing domain tests**

```python
# tests/unit/test_instruments.py
import pytest
from pydantic import ValidationError

from mltrade.domain.instruments import AssetType, InstrumentId


def test_instrument_normalizes_symbol() -> None:
    instrument = InstrumentId(symbol=" spy ", asset_type=AssetType.ETF)

    assert instrument.symbol == "SPY"
    assert str(instrument) == "US:ETF:SPY"


def test_instrument_rejects_invalid_symbol() -> None:
    with pytest.raises(ValidationError):
        InstrumentId(symbol="BRK/B", asset_type=AssetType.STOCK)
```

```python
# tests/unit/test_time.py
from datetime import UTC, datetime, timedelta, timezone

import pytest

from mltrade.domain.time import require_utc


def test_require_utc_accepts_utc_datetime() -> None:
    value = datetime(2026, 6, 12, 20, 0, tzinfo=UTC)

    assert require_utc(value) is value


def test_require_utc_rejects_naive_datetime() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        require_utc(datetime(2026, 6, 12, 20, 0))


def test_require_utc_converts_offset_datetime() -> None:
    eastern = timezone(-timedelta(hours=4))
    value = datetime(2026, 6, 12, 16, 0, tzinfo=eastern)

    assert require_utc(value) == datetime(2026, 6, 12, 20, 0, tzinfo=UTC)
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run pytest tests/unit/test_instruments.py tests/unit/test_time.py -v
```

Expected: collection fails because the domain modules do not exist.

- [ ] **Step 3: Implement immutable instrument identity**

```python
# src/mltrade/domain/instruments.py
import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, field_validator

_SYMBOL_PATTERN = re.compile(r"^[A-Z][A-Z0-9.-]{0,14}$")


class AssetType(StrEnum):
    STOCK = "stock"
    ETF = "etf"


class InstrumentId(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    asset_type: AssetType
    country: str = "US"

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        symbol = value.strip().upper()
        if not _SYMBOL_PATTERN.fullmatch(symbol):
            raise ValueError("symbol must be a valid US stock or ETF symbol")
        return symbol

    @field_validator("country")
    @classmethod
    def require_us_country(cls, value: str) -> str:
        country = value.strip().upper()
        if country != "US":
            raise ValueError("the initial mandate supports US instruments only")
        return country

    def __str__(self) -> str:
        return f"{self.country}:{self.asset_type.value.upper()}:{self.symbol}"
```

- [ ] **Step 4: Implement UTC normalization**

```python
# src/mltrade/domain/time.py
from datetime import UTC, datetime


def require_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    if value.tzinfo is UTC:
        return value
    return value.astimezone(UTC)
```

- [ ] **Step 5: Run domain tests**

Run:

```bash
uv run pytest tests/unit/test_instruments.py tests/unit/test_time.py -v
```

Expected: `5 passed`.

- [ ] **Step 6: Commit domain contracts**

```bash
git add src/mltrade/domain tests/unit/test_instruments.py tests/unit/test_time.py
git commit -m "feat: define instrument and time contracts"
```

## Task 4: Add Exchange-Aware Session Logic

**Files:**
- Create: `src/mltrade/calendar.py`
- Create: `tests/unit/test_calendar.py`

- [ ] **Step 1: Write failing calendar tests**

```python
# tests/unit/test_calendar.py
from datetime import UTC, date, datetime

from mltrade.calendar import XNYSCalendar


def test_holiday_is_not_a_session() -> None:
    calendar = XNYSCalendar()

    assert calendar.is_session(date(2024, 7, 4)) is False
    assert calendar.is_session(date(2024, 7, 5)) is True


def test_last_completed_session_before_market_close() -> None:
    calendar = XNYSCalendar()
    now = datetime(2024, 7, 5, 17, 0, tzinfo=UTC)

    assert calendar.last_completed_session(now) == date(2024, 7, 3)


def test_last_completed_session_after_market_close() -> None:
    calendar = XNYSCalendar()
    now = datetime(2024, 7, 5, 21, 0, tzinfo=UTC)

    assert calendar.last_completed_session(now) == date(2024, 7, 5)
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run pytest tests/unit/test_calendar.py -v
```

Expected: collection fails because `mltrade.calendar` does not exist.

- [ ] **Step 3: Implement the XNYS calendar adapter**

```python
# src/mltrade/calendar.py
from datetime import UTC, date, datetime, timedelta
from functools import cached_property
from typing import Any

import exchange_calendars as xcals
import pandas as pd

from mltrade.domain.time import require_utc


class XNYSCalendar:
    @cached_property
    def _calendar(self) -> Any:
        return xcals.get_calendar("XNYS")

    def is_session(self, session_date: date) -> bool:
        return self._calendar.is_session(pd.Timestamp(session_date))

    def last_completed_session(self, now: datetime) -> date:
        utc_now = require_utc(now)
        candidate = utc_now.date()

        while not self.is_session(candidate):
            candidate -= timedelta(days=1)

        session = pd.Timestamp(candidate)
        close = self._calendar.session_close(session).to_pydatetime()
        if close.tzinfo is None:
            close = close.replace(tzinfo=UTC)

        if utc_now >= close.astimezone(UTC):
            return candidate

        previous = self._calendar.previous_session(session)
        return previous.date()
```

- [ ] **Step 4: Run calendar tests**

Run:

```bash
uv run pytest tests/unit/test_calendar.py -v
```

Expected: `3 passed`.

- [ ] **Step 5: Run the full unit suite**

Run:

```bash
uv run pytest tests/unit -v
```

Expected: all unit tests pass.

- [ ] **Step 6: Commit session logic**

```bash
git add src/mltrade/calendar.py tests/unit/test_calendar.py
git commit -m "feat: add exchange-aware session calendar"
```

## Task 5: Define Immutable Dataset Manifests and Snapshot Storage

**Files:**
- Create: `src/mltrade/storage/manifests.py`
- Create: `src/mltrade/storage/snapshots.py`
- Create: `tests/unit/test_snapshots.py`

- [ ] **Step 1: Write failing snapshot tests**

```python
# tests/unit/test_snapshots.py
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mltrade.storage.manifests import DatasetManifest
from mltrade.storage.snapshots import SnapshotStore


def make_manifest() -> DatasetManifest:
    return DatasetManifest(
        dataset="daily_prices",
        snapshot_id="2026-06-12T210000Z",
        created_at=datetime(2026, 6, 12, 21, 0, tzinfo=UTC),
        source="test-fixture",
        schema_version=1,
        row_count=2,
        content_sha256="a" * 64,
        data_files=("part-000.parquet",),
    )


def test_manifest_round_trips(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    manifest = make_manifest()

    saved_path = store.save_manifest(manifest)
    loaded = store.load_manifest(manifest.dataset, manifest.snapshot_id)

    assert saved_path.name == "manifest.json"
    assert loaded == manifest


def test_existing_snapshot_cannot_be_overwritten(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    manifest = make_manifest()
    store.save_manifest(manifest)

    with pytest.raises(FileExistsError):
        store.save_manifest(manifest)
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run pytest tests/unit/test_snapshots.py -v
```

Expected: collection fails because the storage modules do not exist.

- [ ] **Step 3: Implement the dataset manifest**

```python
# src/mltrade/storage/manifests.py
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mltrade.domain.time import require_utc


class DatasetManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    snapshot_id: str = Field(pattern=r"^[A-Za-z0-9_.:-]+$")
    created_at: datetime
    source: str = Field(min_length=1)
    schema_version: int = Field(ge=1)
    row_count: int = Field(ge=0)
    content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    data_files: tuple[str, ...] = ()

    @field_validator("created_at")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        return require_utc(value)

    @field_validator("data_files")
    @classmethod
    def require_relative_files(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            if value.startswith("/") or ".." in value.split("/"):
                raise ValueError("data_files must contain safe relative paths")
        return values
```

- [ ] **Step 4: Implement atomic, append-only manifest persistence**

```python
# src/mltrade/storage/snapshots.py
import os
from pathlib import Path
from uuid import uuid4

from mltrade.storage.manifests import DatasetManifest


class SnapshotStore:
    def __init__(self, root: Path) -> None:
        self._root = root

    def snapshot_dir(self, dataset: str, snapshot_id: str) -> Path:
        return self._root / dataset / snapshot_id

    def save_manifest(self, manifest: DatasetManifest) -> Path:
        directory = self.snapshot_dir(manifest.dataset, manifest.snapshot_id)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "manifest.json"
        if target.exists():
            raise FileExistsError(f"snapshot already exists: {target}")

        temporary = directory / f".manifest-{uuid4().hex}.tmp"
        payload = manifest.model_dump_json(indent=2)
        temporary.write_text(payload + "\n", encoding="utf-8")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return target

    def load_manifest(self, dataset: str, snapshot_id: str) -> DatasetManifest:
        target = self.snapshot_dir(dataset, snapshot_id) / "manifest.json"
        return DatasetManifest.model_validate_json(target.read_text(encoding="utf-8"))
```

- [ ] **Step 5: Run snapshot tests**

Run:

```bash
uv run pytest tests/unit/test_snapshots.py -v
```

Expected: `2 passed`.

- [ ] **Step 6: Commit snapshot contracts**

```bash
git add src/mltrade/storage tests/unit/test_snapshots.py
git commit -m "feat: add immutable dataset manifests"
```

## Task 6: Add Operational Database and Append-Only Audit Records

**Files:**
- Create: `src/mltrade/operations/database.py`
- Create: `src/mltrade/operations/models.py`
- Create: `src/mltrade/operations/audit.py`
- Create: `tests/unit/test_audit.py`

- [ ] **Step 1: Write failing audit tests**

```python
# tests/unit/test_audit.py
from datetime import UTC, datetime

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from mltrade.operations.audit import AuditService
from mltrade.operations.models import AuditEvent, Base


def test_audit_service_appends_structured_event() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        service = AuditService(session)
        event_id = service.record(
            event_type="risk.trading_blocked",
            occurred_at=datetime(2026, 6, 12, 21, 0, tzinfo=UTC),
            actor="system",
            correlation_id="run-123",
            payload={"reason": "stale_data"},
        )

        stored = session.scalar(
            select(AuditEvent).where(AuditEvent.id == event_id)
        )

    assert stored is not None
    assert stored.event_type == "risk.trading_blocked"
    assert stored.payload == {"reason": "stale_data"}
```

- [ ] **Step 2: Run the test to verify failure**

Run:

```bash
uv run pytest tests/unit/test_audit.py -v
```

Expected: collection fails because the operations modules do not exist.

- [ ] **Step 3: Implement database helpers**

```python
# src/mltrade/operations/database.py
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session


def build_engine(database_url: str) -> Engine:
    return create_engine(database_url, pool_pre_ping=True)


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    with Session(engine) as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
```

- [ ] **Step 4: Implement the audit database model**

```python
# src/mltrade/operations/models.py
from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import JSON, DateTime, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    event_type: Mapped[str] = mapped_column(String(120), index=True)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        index=True,
    )
    actor: Mapped[str] = mapped_column(String(120))
    correlation_id: Mapped[str] = mapped_column(String(120), index=True)
    payload: Mapped[dict[str, object]] = mapped_column(JSON)
```

- [ ] **Step 5: Implement append-only audit writes**

```python
# src/mltrade/operations/audit.py
from datetime import datetime
from uuid import UUID

from sqlalchemy.orm import Session

from mltrade.domain.time import require_utc
from mltrade.operations.models import AuditEvent


class AuditService:
    def __init__(self, session: Session) -> None:
        self._session = session

    def record(
        self,
        *,
        event_type: str,
        occurred_at: datetime,
        actor: str,
        correlation_id: str,
        payload: dict[str, object],
    ) -> UUID:
        event = AuditEvent(
            event_type=event_type,
            occurred_at=require_utc(occurred_at),
            actor=actor,
            correlation_id=correlation_id,
            payload=payload,
        )
        self._session.add(event)
        self._session.flush()
        return event.id
```

- [ ] **Step 6: Run audit tests**

Run:

```bash
uv run pytest tests/unit/test_audit.py -v
```

Expected: `1 passed`.

- [ ] **Step 7: Commit operational persistence**

```bash
git add src/mltrade/operations tests/unit/test_audit.py
git commit -m "feat: add append-only operational audit records"
```

## Task 7: Configure Structured JSON Logging

**Files:**
- Create: `src/mltrade/operations/logging.py`
- Create: `tests/unit/test_logging.py`

- [ ] **Step 1: Write the failing logging test**

```python
# tests/unit/test_logging.py
import json

import structlog

from mltrade.operations.logging import configure_logging


def test_logging_emits_json_without_secret_values(capsys) -> None:
    configure_logging("INFO")
    logger = structlog.get_logger("test")

    logger.info(
        "data_snapshot_validated",
        dataset="daily_prices",
        api_secret="paper-secret",
    )

    output = json.loads(capsys.readouterr().out)
    raw_output = json.dumps(output)
    assert output["event"] == "data_snapshot_validated"
    assert output["dataset"] == "daily_prices"
    assert output["api_secret"] == "[REDACTED]"
    assert output["level"] == "info"
    assert "paper-secret" not in raw_output
```

- [ ] **Step 2: Run the test to verify failure**

Run:

```bash
uv run pytest tests/unit/test_logging.py -v
```

Expected: collection fails because `mltrade.operations.logging` does not exist.

- [ ] **Step 3: Implement JSON logging**

```python
# src/mltrade/operations/logging.py
import logging
import sys

import structlog


def redact_secrets(
    _logger: object,
    _method_name: str,
    event_dict: dict[str, object],
) -> dict[str, object]:
    secret_markers = ("secret", "password", "token", "api_key")
    for key in tuple(event_dict):
        if any(marker in key.lower() for marker in secret_markers):
            event_dict[key] = "[REDACTED]"
    return event_dict


def configure_logging(level: str) -> None:
    numeric_level = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }[level]
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
        force=True,
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            redact_secrets,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            numeric_level
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )
```

- [ ] **Step 4: Run logging tests**

Run:

```bash
uv run pytest tests/unit/test_logging.py -v
```

Expected: `1 passed`.

- [ ] **Step 5: Commit structured logging**

```bash
git add src/mltrade/operations/logging.py tests/unit/test_logging.py
git commit -m "feat: add structured JSON logging"
```

## Task 8: Add a Foundation Health Command

**Files:**
- Create: `src/mltrade/cli.py`
- Create: `tests/integration/test_doctor.py`

- [ ] **Step 1: Write the failing CLI integration test**

```python
# tests/integration/test_doctor.py
from typer.testing import CliRunner

from mltrade.cli import app

runner = CliRunner()


def test_doctor_reports_foundation_health(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MLTRADE_DATA_ROOT", str(tmp_path))
    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "configuration: ok" in result.stdout
    assert "calendar: ok" in result.stdout
    assert "data root: ok" in result.stdout
    assert "live trading: disabled" in result.stdout
```

- [ ] **Step 2: Run the test to verify failure**

Run:

```bash
uv run pytest tests/integration/test_doctor.py -v
```

Expected: collection fails because `mltrade.cli` does not exist.

- [ ] **Step 3: Implement the health command**

```python
# src/mltrade/cli.py
from datetime import UTC, datetime

import typer

from mltrade.calendar import XNYSCalendar
from mltrade.config import Settings

app = typer.Typer(no_args_is_help=True)


@app.command()
def doctor() -> None:
    settings = Settings()
    settings.data_root.mkdir(parents=True, exist_ok=True)

    calendar = XNYSCalendar()
    calendar.last_completed_session(datetime.now(UTC))

    typer.echo("configuration: ok")
    typer.echo("calendar: ok")
    typer.echo("data root: ok")
    state = "enabled" if settings.live_trading_enabled else "disabled"
    typer.echo(f"live trading: {state}")
```

- [ ] **Step 4: Run the CLI integration test**

Run:

```bash
uv run pytest tests/integration/test_doctor.py -v
uv run mltrade doctor
```

Expected: the test passes and the command prints four healthy status lines with
live trading disabled.

- [ ] **Step 5: Commit the health command**

```bash
git add src/mltrade/cli.py tests/integration/test_doctor.py
git commit -m "feat: add foundation health command"
```

## Task 9: Add PostgreSQL and Containerized Local Runtime

**Files:**
- Create: `compose.yaml`
- Create: `Dockerfile`
- Create: `tests/contract/test_postgres_audit.py`

- [ ] **Step 1: Write the PostgreSQL contract test**

```python
# tests/contract/test_postgres_audit.py
import os
from datetime import UTC, datetime

import pytest
from sqlalchemy import select

from mltrade.operations.audit import AuditService
from mltrade.operations.database import build_engine, session_scope
from mltrade.operations.models import AuditEvent, Base

pytestmark = pytest.mark.contract


def test_postgres_persists_audit_event() -> None:
    database_url = os.environ["MLTRADE_TEST_DATABASE_URL"]
    engine = build_engine(database_url)
    Base.metadata.create_all(engine)

    with session_scope(engine) as session:
        event_id = AuditService(session).record(
            event_type="system.contract_test",
            occurred_at=datetime(2026, 6, 12, 21, 0, tzinfo=UTC),
            actor="pytest",
            correlation_id="contract-1",
            payload={"database": "postgresql"},
        )

    with session_scope(engine) as session:
        stored = session.scalar(
            select(AuditEvent).where(AuditEvent.id == event_id)
        )
        assert stored is not None
        assert stored.payload == {"database": "postgresql"}
```

- [ ] **Step 2: Create the local PostgreSQL service**

```yaml
# compose.yaml
services:
  postgres:
    image: postgres:17
    environment:
      POSTGRES_DB: mltrade
      POSTGRES_USER: mltrade
      POSTGRES_PASSWORD: mltrade
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U mltrade -d mltrade"]
      interval: 2s
      timeout: 3s
      retries: 15
    volumes:
      - mltrade-postgres:/var/lib/postgresql/data

volumes:
  mltrade-postgres:
```

- [ ] **Step 3: Create the application image**

```dockerfile
# Dockerfile
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN pip install --no-cache-dir uv \
    && uv sync --frozen --no-dev

ENTRYPOINT ["uv", "run", "mltrade"]
CMD ["doctor"]
```

- [ ] **Step 4: Start PostgreSQL**

Run:

```bash
docker compose up -d --wait postgres
```

Expected: service `postgres` reports healthy.

- [ ] **Step 5: Run the PostgreSQL contract**

Run:

```bash
MLTRADE_TEST_DATABASE_URL=postgresql+psycopg://mltrade:mltrade@localhost:5432/mltrade \
  uv run pytest tests/contract/test_postgres_audit.py -v
```

Expected: `1 passed`.

- [ ] **Step 6: Build and run the application container**

Run:

```bash
docker build -t mltrade:foundation .
docker run --rm mltrade:foundation doctor
```

Expected: image builds successfully and the command reports configuration,
calendar, and data-root health with live trading disabled.

- [ ] **Step 7: Commit local infrastructure**

```bash
git add compose.yaml Dockerfile tests/contract/test_postgres_audit.py
git commit -m "build: add PostgreSQL development runtime"
```

## Task 10: Add Deterministic Fixtures and Project Documentation

**Files:**
- Create: `tests/conftest.py`
- Modify: `README.md`

- [ ] **Step 1: Add deterministic test fixtures**

```python
# tests/conftest.py
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest


@pytest.fixture
def fixed_now() -> datetime:
    return datetime(2026, 6, 12, 21, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def isolate_mltrade_environment(monkeypatch) -> Iterator[None]:
    for name in (
        "MLTRADE_ENVIRONMENT",
        "MLTRADE_DATA_ROOT",
        "MLTRADE_DATABASE_URL",
        "MLTRADE_ALPACA_API_KEY",
        "MLTRADE_ALPACA_API_SECRET",
        "MLTRADE_ALPACA_BASE_URL",
        "MLTRADE_LIVE_TRADING_ENABLED",
        "MLTRADE_LOG_LEVEL",
    ):
        monkeypatch.delenv(name, raising=False)
    yield
```

- [ ] **Step 2: Write the project README**

````markdown
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
````

- [ ] **Step 3: Run all non-contract verification**

Run:

```bash
uv run ruff check .
uv run mypy src
uv run pytest tests/unit tests/integration --cov=mltrade --cov-report=term-missing
```

Expected: Ruff and mypy pass; all tests pass; branch coverage is at least 90%.

- [ ] **Step 4: Run the external-service contract**

Run:

```bash
docker compose up -d --wait postgres
MLTRADE_TEST_DATABASE_URL=postgresql+psycopg://mltrade:mltrade@localhost:5432/mltrade \
  uv run pytest tests/contract -v
```

Expected: all PostgreSQL contract tests pass.

- [ ] **Step 5: Verify safety defaults**

Run:

```bash
uv run mltrade doctor
```

Expected output includes:

```text
configuration: ok
calendar: ok
data root: ok
live trading: disabled
```

- [ ] **Step 6: Commit the completed foundation**

```bash
git add README.md tests/conftest.py
git commit -m "docs: document foundation development workflow"
```

## Task 11: Final Foundation Acceptance

**Files:**
- Review: all files created by Tasks 1 through 10
- Review: `docs/superpowers/specs/2026-06-12-us-equities-ml-trading-platform-design.md`

- [ ] **Step 1: Verify the complete repository**

Run:

```bash
uv sync --frozen --extra dev
uv run ruff check .
uv run mypy src
uv run pytest tests/unit tests/integration --cov=mltrade --cov-report=term-missing
docker compose up -d --wait postgres
MLTRADE_TEST_DATABASE_URL=postgresql+psycopg://mltrade:mltrade@localhost:5432/mltrade \
  uv run pytest tests/contract -v
docker build -t mltrade:foundation .
docker run --rm mltrade:foundation doctor
git status --short
```

Expected:

- dependency lock is reproducible;
- lint and type checks pass;
- all unit and integration tests pass with at least 90% branch coverage;
- all PostgreSQL contract tests pass;
- the container health command succeeds;
- live trading remains disabled; and
- `git status --short` prints no output.

- [ ] **Step 2: Report the acceptance evidence**

In the implementation handoff, report the exact test counts, branch coverage
percentage, PostgreSQL contract result, container image result, and final commit
hash. Do not claim Phase 1 complete if any acceptance command failed or was
skipped.
