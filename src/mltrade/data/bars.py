import warnings
from collections.abc import Mapping
from datetime import date, datetime
from decimal import Decimal
from typing import Annotated, Any, Protocol, Self, override

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PydanticDeprecatedSince20,
    field_validator,
    model_validator,
)

from mltrade.domain.instruments import InstrumentId
from mltrade.domain.time import require_utc
from mltrade.universe import Universe

PositiveDecimal = Annotated[Decimal, Field(gt=0)]
NonnegativeInt = Annotated[int, Field(ge=0, strict=True)]


class DailyBar(BaseModel):
    model_config = ConfigDict(frozen=True)

    instrument: InstrumentId
    session: date
    open: PositiveDecimal
    high: PositiveDecimal
    low: PositiveDecimal
    close: PositiveDecimal
    volume: NonnegativeInt
    vwap: PositiveDecimal
    trade_count: NonnegativeInt
    source: Annotated[str, Field(min_length=1)]
    ingested_at: datetime

    @override
    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        if update:
            raise TypeError("DailyBar cannot be updated")
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
            raise TypeError("DailyBar cannot be updated")
        return super().copy(
            include=include,
            exclude=exclude,
            update=update,
            deep=deep,
        )

    @field_validator("ingested_at")
    @classmethod
    def normalize_ingested_at(cls, value: datetime) -> datetime:
        return require_utc(value)

    @model_validator(mode="after")
    def validate_ohlc(self) -> Self:
        if self.high < max(self.open, self.low, self.close):
            raise ValueError("high must be at least every other OHLC value")
        if self.low > min(self.open, self.high, self.close):
            raise ValueError("low must be at most every other OHLC value")
        return self


class DailyBarSource(Protocol):
    def fetch(
        self,
        universe: Universe,
        start: date,
        end: date,
        ingested_at: datetime,
    ) -> tuple[DailyBar, ...]: ...
