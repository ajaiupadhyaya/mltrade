"""Immutable forecast contracts for the walk-forward forecasting pipeline.

This module defines the value objects produced by the forecast runner and the
``ForecastBlocked`` exception used to signal fail-closed conditions.

Design notes
------------
- ``Forecast`` and ``ForecastBatch`` are frozen Pydantic models (consistent
  with the rest of the codebase).
- ``ForecastBlocked`` is a plain ``Exception`` subclass raised (never stored)
  when the pipeline cannot produce a valid forecast.
- No ``Any`` leaks; all fields are fully annotated.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from copy import deepcopy
from datetime import date
from typing import Any, Literal, Self, override

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PydanticDeprecatedSince20,
    StrictBool,
    StrictFloat,
    StrictInt,
)


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


class _StrictForecastConfig(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
    )

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


class RidgeForecastConfig(_StrictForecastConfig):
    """Validated Ridge and walk-forward forecast boundaries."""

    alpha: StrictFloat = Field(
        default=1.0,
        gt=0.0,
        allow_inf_nan=False,
    )
    fit_intercept: StrictBool = True
    minimum_training_sessions: StrictInt = Field(default=504, ge=1)
    embargo_sessions: StrictInt = Field(default=21, ge=1)


class ForecastBlocked(Exception):
    """Raised when the forecasting pipeline cannot produce a valid forecast.

    The message always identifies the root cause.  Two distinguished cases:

    - ``"non-finite"`` — a feature value or model prediction is NaN or ±inf.
    - ``"insufficient training history"`` — fewer than 504 distinct training
      sessions are available after embargo filtering.

    Callers should treat any ``ForecastBlocked`` as a fail-closed signal and
    skip the rebalance rather than proceeding with a stale or unreliable
    forecast.
    """


# ---------------------------------------------------------------------------
# Model version literal
# ---------------------------------------------------------------------------

MODEL_VERSION: Literal["ridge-trend-v1"] = "ridge-trend-v1"

# ---------------------------------------------------------------------------
# Individual symbol forecast
# ---------------------------------------------------------------------------


class Forecast(BaseModel):
    """Immutable point-in-time forecast for one symbol on one decision session.

    Attributes
    ----------
    symbol:
        Ticker symbol (e.g. ``"SPY"``).
    decision_session:
        The rebalance date for which this forecast was generated.
    predicted_forward_return:
        Ridge regression output: predicted 21-session simple forward return.
        Always finite (``ForecastBlocked`` is raised if the model outputs
        non-finite values).
    model_version:
        Identifies the model and feature set; always ``"ridge-trend-v1"``.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    decision_session: date
    predicted_forward_return: float
    model_version: Literal["ridge-trend-v1"] = MODEL_VERSION


# ---------------------------------------------------------------------------
# Batch of forecasts for one decision session
# ---------------------------------------------------------------------------


class ForecastBatch(BaseModel):
    """Immutable collection of forecasts generated for one decision session.

    Each ``Forecast`` in ``forecasts`` corresponds to exactly one symbol in
    the prediction cross-section (the set of symbols with a non-missing
    feature row at ``decision_session``).

    Attributes
    ----------
    decision_session:
        The rebalance date for all forecasts in this batch.
    forecasts:
        One ``Forecast`` per symbol in the current cross-section.
    training_start:
        Earliest ``decision_session`` among the usable training rows (after
        embargo filtering).
    training_end:
        Latest ``decision_session`` among the usable training rows (after
        embargo filtering).
    embargo_start:
        The first session date within the embargo zone.  Training labels must
        have ``label_end_session < embargo_start``.  Computed as the XNYS
        session that is ``embargo_sessions`` trading sessions before
        ``decision_session`` (inclusive of ``decision_session``).
    training_row_count:
        Number of usable training rows (after embargo filtering and
        missing/label availability checks).
    training_session_count:
        Number of distinct ``decision_session`` values in the usable training
        set.  Must be >= 504 for the forecast to proceed.
    model_version:
        Always ``"ridge-trend-v1"``.
    """

    model_config = ConfigDict(frozen=True)

    decision_session: date
    forecasts: tuple[Forecast, ...]
    training_start: date
    training_end: date
    embargo_start: date
    training_row_count: int
    training_session_count: int
    model_version: Literal["ridge-trend-v1"] = MODEL_VERSION
