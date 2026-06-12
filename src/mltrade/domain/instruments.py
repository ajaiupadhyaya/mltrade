import re
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Self, override

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

    def __str__(self) -> str:
        return f"{self.country}:{self.asset_type.value.upper()}:{self.symbol}"
