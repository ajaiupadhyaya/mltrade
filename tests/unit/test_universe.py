import pytest
from pydantic import ValidationError

from mltrade.domain.instruments import AssetType, InstrumentId
from mltrade.universe import MVP_UNIVERSE, Universe


def test_mvp_universe_is_the_versioned_ordered_etf_universe() -> None:
    expected_symbols = (
        "SPY",
        "QQQ",
        "IWM",
        "EFA",
        "EEM",
        "TLT",
        "IEF",
        "GLD",
        "DBC",
        "VNQ",
    )

    assert MVP_UNIVERSE.version == "mvp-etf-v1"
    assert MVP_UNIVERSE.symbols == expected_symbols
    assert all(
        isinstance(instrument, InstrumentId)
        and instrument.asset_type is AssetType.ETF
        for instrument in MVP_UNIVERSE.instruments
    )


def test_universe_rejects_blank_version() -> None:
    instrument = InstrumentId(symbol="SPY", asset_type=AssetType.ETF)

    with pytest.raises(ValidationError, match="version"):
        Universe(version=" ", instruments=(instrument,))


def test_universe_rejects_duplicate_symbols() -> None:
    first = InstrumentId(symbol="SPY", asset_type=AssetType.ETF)
    duplicate = InstrumentId(symbol="SPY", asset_type=AssetType.STOCK)

    with pytest.raises(ValidationError, match="duplicate"):
        Universe(version="test-v1", instruments=(first, duplicate))


def test_universe_is_frozen() -> None:
    with pytest.raises(ValidationError, match="frozen"):
        MVP_UNIVERSE.version = "changed"  # type: ignore[misc]
