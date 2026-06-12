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


def test_instrument_rejects_country_update_via_model_copy() -> None:
    instrument = InstrumentId(symbol="SPY", asset_type=AssetType.ETF)

    with pytest.raises(TypeError, match="InstrumentId cannot be updated"):
        instrument.model_copy(update={"country": "CA"})


def test_instrument_rejects_symbol_update_via_model_copy() -> None:
    instrument = InstrumentId(symbol="SPY", asset_type=AssetType.ETF)

    with pytest.raises(TypeError, match="InstrumentId cannot be updated"):
        instrument.model_copy(update={"symbol": "BRK/B"})
