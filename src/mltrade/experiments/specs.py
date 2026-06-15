import warnings
from collections.abc import Mapping
from copy import deepcopy
from decimal import Decimal
from typing import Annotated, Any, Literal, Self, override

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PydanticDeprecatedSince20,
    StrictBool,
    field_validator,
    model_validator,
)

from mltrade.storage.manifests import require_safe_path_segment

type CostBps = Annotated[Decimal, Field(ge=0, le=100)]


def _to_validation_data(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _to_validation_data(
            value.model_dump(mode="python", round_trip=True)
        )
    if isinstance(value, Mapping):
        return {
            key: _to_validation_data(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_to_validation_data(item) for item in value)
    if isinstance(value, list):
        return [_to_validation_data(item) for item in value]
    return value


class StrictFrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    @override
    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        if update is None:
            return super().model_copy(deep=deep)

        values = self.model_dump(round_trip=True)
        values.update(_to_validation_data(update))
        values = _to_validation_data(values)
        if deep:
            values = deepcopy(values)
        return type(self).model_validate(values)

    @override
    def copy(
        self,
        *,
        include: Any = None,
        exclude: Any = None,
        update: dict[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        if include is not None or exclude is not None:
            warnings.warn(
                "The `copy` method is deprecated; use `model_copy` instead.",
                category=PydanticDeprecatedSince20,
                stacklevel=2,
            )
            raise TypeError(
                f"{type(self).__name__} cannot be partially copied"
            )
        if update is not None:
            warnings.warn(
                "The `copy` method is deprecated; use `model_copy` instead.",
                category=PydanticDeprecatedSince20,
                stacklevel=2,
            )
            return self.model_copy(update=update, deep=deep)
        return super().copy(deep=deep)


class DatasetSpec(StrictFrozenModel):
    name: Literal["daily_bars"] = "daily_bars"
    snapshot_id: str = Field(pattern=r"^[A-Za-z0-9_.:-]+$")
    universe_version: Literal["mvp-etf-v1"] = "mvp-etf-v1"
    feature_version: Literal["trend-momentum-v1"] = "trend-momentum-v1"

    @field_validator("snapshot_id")
    @classmethod
    def validate_snapshot_id(cls, value: str) -> str:
        if value == "latest":
            raise ValueError("snapshot_id must identify an immutable snapshot")
        return require_safe_path_segment(value)


class RidgeModelSpec(StrictFrozenModel):
    family: Literal["ridge"] = "ridge"
    version: Literal["ridge-trend-v1"] = "ridge-trend-v1"
    alpha: float = Field(
        default=1.0,
        strict=True,
        allow_inf_nan=False,
        gt=0.0,
        le=10_000.0,
    )
    fit_intercept: StrictBool = True


class ValidationSpec(StrictFrozenModel):
    minimum_training_sessions: int = Field(
        default=504,
        strict=True,
        ge=252,
        le=2520,
    )
    embargo_sessions: int = Field(default=21, strict=True, ge=1, le=126)
    retrain_every_sessions: int = Field(
        default=21,
        strict=True,
        ge=1,
        le=126,
    )


class CostSpec(StrictFrozenModel):
    headline_bps: CostBps = Decimal("5")
    sensitivity_bps: tuple[CostBps, ...] = Field(
        default=(
            Decimal("2"),
            Decimal("5"),
            Decimal("10"),
        ),
        min_length=1,
    )


class PortfolioSpec(StrictFrozenModel):
    reference_equity: Decimal = Field(default=Decimal("1000000"), gt=0)
    maximum_position_weight: Decimal = Field(
        default=Decimal("0.25"),
        gt=0,
        le=1,
    )
    minimum_cash_weight: Decimal = Field(default=Decimal("0.05"), gt=0, lt=1)
    target_annual_volatility: Decimal = Field(default=Decimal("0.15"), gt=0)

    @model_validator(mode="after")
    def validate_limits(self) -> "PortfolioSpec":
        if (
            self.maximum_position_weight
            > Decimal("1") - self.minimum_cash_weight
        ):
            raise ValueError(
                "maximum_position_weight cannot exceed "
                "1 - minimum_cash_weight"
            )
        return self


class ObjectiveSpec(StrictFrozenModel):
    name: Literal["robust_sharpe"] = "robust_sharpe"
    maximum_drawdown: float = Field(
        default=-0.35,
        strict=True,
        allow_inf_nan=False,
        ge=-1.0,
        le=0.0,
    )
    maximum_turnover: float = Field(
        default=1.0,
        strict=True,
        allow_inf_nan=False,
        ge=0.0,
    )


class ResourceBudget(StrictFrozenModel):
    max_trials: int = Field(default=40, strict=True, ge=1, le=500)
    timeout_minutes: int = Field(default=60, strict=True, ge=1, le=720)
    worker_count: int = Field(default=1, strict=True, ge=1, le=2)


class ExperimentSpec(StrictFrozenModel):
    schema_version: Literal[1] = 1
    name: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    description: str = ""
    dataset: DatasetSpec
    model: RidgeModelSpec = RidgeModelSpec()
    validation: ValidationSpec = ValidationSpec()
    costs: CostSpec = CostSpec()
    portfolio: PortfolioSpec = PortfolioSpec()
    objective: ObjectiveSpec = ObjectiveSpec()
    resources: ResourceBudget = ResourceBudget()
    seed: int = Field(default=42, strict=True, ge=0, le=2**32 - 1)

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version_type(cls, value: Any) -> Any:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1")
        return value
