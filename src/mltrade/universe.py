import warnings
from collections.abc import Mapping
from typing import Any, Self, override

from pydantic import (
    BaseModel,
    ConfigDict,
    PydanticDeprecatedSince20,
    field_validator,
    model_validator,
)

from mltrade.domain.instruments import AssetType, InstrumentId


class Universe(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: str
    instruments: tuple[InstrumentId, ...]

    @override
    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        if update:
            raise TypeError("Universe cannot be updated")
        return super().model_copy(update=update, deep=deep)

    @override
    def copy(
        self,
        *,
        include: Any = None,
        exclude: Any = None,
        update: dict[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        if update:
            warnings.warn(
                "The `copy` method is deprecated; use `model_copy` instead.",
                category=PydanticDeprecatedSince20,
                stacklevel=2,
            )
            raise TypeError("Universe cannot be updated")
        return super().copy(
            include=include,
            exclude=exclude,
            update=update,
            deep=deep,
        )

    @field_validator("version")
    @classmethod
    def require_version(cls, value: str) -> str:
        version = value.strip()
        if not version:
            raise ValueError("version must not be empty")
        return version

    @model_validator(mode="after")
    def reject_duplicate_symbols(self) -> Self:
        symbols = self.symbols
        if len(symbols) != len(set(symbols)):
            raise ValueError("universe contains duplicate symbols")
        return self

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(instrument.symbol for instrument in self.instruments)


MVP_UNIVERSE = Universe(
    version="mvp-etf-v1",
    instruments=tuple(
        InstrumentId(symbol=symbol, asset_type=AssetType.ETF)
        for symbol in (
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
    ),
)
