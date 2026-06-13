from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import pytest
from pydantic import PydanticDeprecatedSince20, ValidationError

from mltrade.data.bars import DailyBar
from mltrade.domain.instruments import AssetType, InstrumentId


def make_bar(**overrides: Any) -> DailyBar:
    values: dict[str, Any] = {
        "instrument": InstrumentId(symbol="SPY", asset_type=AssetType.ETF),
        "session": date(2026, 6, 12),
        "open": Decimal("100"),
        "high": Decimal("105"),
        "low": Decimal("99"),
        "close": Decimal("104"),
        "volume": 1_000_000,
        "vwap": Decimal("102.50"),
        "trade_count": 42_000,
        "source": "test",
        "ingested_at": datetime(2026, 6, 13, 1, 0, tzinfo=UTC),
    }
    values.update(overrides)
    return DailyBar(**values)


def test_daily_bar_rejects_high_below_an_ohlc_value() -> None:
    with pytest.raises(ValidationError, match="high"):
        make_bar(high=Decimal("103"))


def test_daily_bar_rejects_low_above_an_ohlc_value() -> None:
    with pytest.raises(ValidationError, match="low"):
        make_bar(low=Decimal("101"))


def test_daily_bar_normalizes_ingested_at_to_utc() -> None:
    eastern = timezone(-timedelta(hours=4))

    bar = make_bar(
        ingested_at=datetime(2026, 6, 12, 21, 0, tzinfo=eastern)
    )

    assert bar.ingested_at == datetime(2026, 6, 13, 1, 0, tzinfo=UTC)
    assert bar.ingested_at.tzinfo is UTC


def test_daily_bar_rejects_naive_ingested_at() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        make_bar(ingested_at=datetime(2026, 6, 13, 1, 0))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("open", Decimal("0")),
        ("high", Decimal("0")),
        ("low", Decimal("0")),
        ("close", Decimal("0")),
        ("vwap", Decimal("0")),
        ("volume", -1),
        ("trade_count", -1),
    ],
)
def test_daily_bar_rejects_invalid_numeric_constraints(
    field: str,
    value: Decimal | int,
) -> None:
    with pytest.raises(ValidationError, match=field):
        make_bar(**{field: value})


def test_daily_bar_rejects_empty_source() -> None:
    with pytest.raises(ValidationError, match="source"):
        make_bar(source="")


def test_daily_bar_is_frozen() -> None:
    bar = make_bar()

    with pytest.raises(ValidationError, match="frozen"):
        bar.close = Decimal("103")  # type: ignore[misc]


@pytest.mark.parametrize(
    "update",
    [
        {"high": Decimal("1")},
        {"volume": -1},
        {"ingested_at": datetime(2026, 6, 13, 1, 0)},
    ],
)
def test_daily_bar_rejects_updates_via_model_copy(
    update: dict[str, object],
) -> None:
    bar = make_bar()

    with pytest.raises(TypeError, match="DailyBar cannot be updated"):
        bar.model_copy(update=update)


@pytest.mark.parametrize(
    "update",
    [
        {"low": Decimal("1000")},
        {"trade_count": -1},
        {"ingested_at": datetime(2026, 6, 13, 1, 0)},
    ],
)
def test_daily_bar_rejects_updates_via_legacy_copy(
    update: dict[str, object],
) -> None:
    bar = make_bar()

    with pytest.warns(PydanticDeprecatedSince20):
        with pytest.raises(TypeError, match="DailyBar cannot be updated"):
            bar.copy(update=update)


def test_daily_bar_allows_update_free_deep_copies() -> None:
    bar = make_bar()

    copied = bar.model_copy(deep=True)
    with pytest.warns(PydanticDeprecatedSince20):
        legacy_copied = bar.copy(deep=True)

    assert copied == bar
    assert copied is not bar
    assert legacy_copied == bar
    assert legacy_copied is not bar


@pytest.mark.parametrize("field", ["volume", "trade_count"])
def test_daily_bar_rejects_boolean_counts(field: str) -> None:
    with pytest.raises(ValidationError, match=field):
        make_bar(**{field: True})
