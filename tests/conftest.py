from collections.abc import Iterator
from datetime import UTC, datetime

import pytest


@pytest.fixture
def fixed_now() -> datetime:
    return datetime(2026, 6, 12, 21, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def isolate_mltrade_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
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
