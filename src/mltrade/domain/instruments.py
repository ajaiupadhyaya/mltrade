import re
import warnings
from collections.abc import Mapping, Set
from enum import StrEnum
from typing import Any, Self, override

from pydantic import (
    BaseModel,
    ConfigDict,
    PydanticDeprecatedSince20,
    field_validator,
)

_SYMBOL_PATTERN = re.compile(r"^[A-Z][A-Z0-9.-]{0,14}$")
type _CopySelection = (
    Set[int] | Set[str] | Mapping[int, Any] | Mapping[str, Any]
)


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

    @override
    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        if update:
            raise TypeError("InstrumentId cannot be updated")
        return super().model_copy(update=update, deep=deep)

    @override
    def copy(
        self,
        *,
        include: _CopySelection | None = None,
        exclude: _CopySelection | None = None,
        update: dict[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        if update:
            warnings.warn(
                "The `copy` method is deprecated; use `model_copy` instead.",
                category=PydanticDeprecatedSince20,
                stacklevel=2,
            )
            raise TypeError("InstrumentId cannot be updated")
        return super().copy(
            include=include,
            exclude=exclude,
            update=update,
            deep=deep,
        )

    def __str__(self) -> str:
        return f"{self.country}:{self.asset_type.value.upper()}:{self.symbol}"
