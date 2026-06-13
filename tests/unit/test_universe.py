import pytest
from pydantic import PydanticDeprecatedSince20, ValidationError

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


@pytest.mark.parametrize(
    "update",
    [
        {"version": ""},
        {
            "instruments": (
                MVP_UNIVERSE.instruments[0],
                MVP_UNIVERSE.instruments[0],
            )
        },
    ],
)
def test_universe_rejects_updates_via_model_copy(
    update: dict[str, object],
) -> None:
    with pytest.raises(TypeError, match="Universe cannot be updated"):
        MVP_UNIVERSE.model_copy(update=update)


@pytest.mark.parametrize(
    "update",
    [
        {"version": ""},
        {
            "instruments": (
                MVP_UNIVERSE.instruments[0],
                MVP_UNIVERSE.instruments[0],
            )
        },
    ],
)
def test_universe_rejects_updates_via_legacy_copy(
    update: dict[str, object],
) -> None:
    with pytest.warns(PydanticDeprecatedSince20):
        with pytest.raises(TypeError, match="Universe cannot be updated"):
            MVP_UNIVERSE.copy(update=update)


def test_universe_allows_update_free_deep_copies() -> None:
    copied = MVP_UNIVERSE.model_copy(deep=True)

    with pytest.warns(PydanticDeprecatedSince20):
        legacy_copied = MVP_UNIVERSE.copy(deep=True)

    assert copied == MVP_UNIVERSE
    assert copied is not MVP_UNIVERSE
    assert legacy_copied == MVP_UNIVERSE
    assert legacy_copied is not MVP_UNIVERSE
