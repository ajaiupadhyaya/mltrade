from decimal import Decimal
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from mltrade.storage.manifests import require_safe_path_segment


class StrictFrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


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
    alpha: float = Field(default=1.0, gt=0.0, le=10_000.0)
    fit_intercept: bool = True


class ValidationSpec(StrictFrozenModel):
    minimum_training_sessions: int = Field(default=504, ge=252, le=2520)
    embargo_sessions: int = Field(default=21, ge=1, le=126)
    retrain_every_sessions: int = Field(default=21, ge=1, le=126)


class CostSpec(StrictFrozenModel):
    headline_bps: Decimal = Field(default=Decimal("5"), ge=0, le=100)
    sensitivity_bps: tuple[Decimal, ...] = (
        Decimal("2"),
        Decimal("5"),
        Decimal("10"),
    )


class PortfolioSpec(StrictFrozenModel):
    reference_equity: Decimal = Field(default=Decimal("1000000"), gt=0)
    maximum_position_weight: Decimal = Field(
        default=Decimal("0.25"),
        gt=0,
        le=1,
    )
    minimum_cash_weight: Decimal = Field(default=Decimal("0.05"), ge=0, lt=1)
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
    maximum_drawdown: float = Field(default=-0.35, ge=-1.0, le=0.0)
    maximum_turnover: float = Field(default=1.0, ge=0.0)


class ResourceBudget(StrictFrozenModel):
    max_trials: int = Field(default=40, ge=1, le=500)
    timeout_minutes: int = Field(default=60, ge=1, le=720)
    worker_count: int = Field(default=1, ge=1, le=2)


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
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
