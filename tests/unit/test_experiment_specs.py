from decimal import Decimal

import pytest
from pydantic import PydanticDeprecatedSince20, ValidationError

from mltrade.experiments.specs import (
    CostSpec,
    DatasetSpec,
    ExperimentSpec,
    PortfolioSpec,
    ResourceBudget,
    RidgeModelSpec,
)
from mltrade.portfolio.targets import PortfolioLimits


def make_spec() -> ExperimentSpec:
    return ExperimentSpec(
        name="ridge_baseline",
        dataset=DatasetSpec(snapshot_id="daily-bars:2026-06-12"),
    )


def test_baseline_spec_defaults_match_the_versioned_contract() -> None:
    spec = make_spec()

    assert spec.model_dump() == {
        "schema_version": 1,
        "name": "ridge_baseline",
        "description": "",
        "dataset": {
            "name": "daily_bars",
            "snapshot_id": "daily-bars:2026-06-12",
            "universe_version": "mvp-etf-v1",
            "feature_version": "trend-momentum-v1",
        },
        "model": {
            "family": "ridge",
            "version": "ridge-trend-v1",
            "alpha": 1.0,
            "fit_intercept": True,
        },
        "validation": {
            "minimum_training_sessions": 504,
            "embargo_sessions": 21,
            "retrain_every_sessions": 21,
        },
        "costs": {
            "headline_bps": Decimal("5"),
            "sensitivity_bps": (
                Decimal("2"),
                Decimal("5"),
                Decimal("10"),
            ),
        },
        "portfolio": {
            "reference_equity": Decimal("1000000"),
            "maximum_position_weight": Decimal("0.25"),
            "minimum_cash_weight": Decimal("0.05"),
            "target_annual_volatility": Decimal("0.15"),
        },
        "objective": {
            "name": "robust_sharpe",
            "maximum_drawdown": -0.35,
            "maximum_turnover": 1.0,
        },
        "resources": {
            "max_trials": 40,
            "timeout_minutes": 60,
            "worker_count": 1,
        },
        "seed": 42,
    }


def test_models_are_frozen() -> None:
    spec = make_spec()

    with pytest.raises(ValidationError, match="frozen"):
        spec.name = "changed"  # type: ignore[misc]

    with pytest.raises(ValidationError, match="frozen"):
        spec.dataset.snapshot_id = "changed"  # type: ignore[misc]


def test_dataset_spec_model_copy_revalidates_updates() -> None:
    dataset = make_spec().dataset

    with pytest.raises(ValidationError, match="immutable snapshot"):
        dataset.model_copy(update={"snapshot_id": "latest"})


def test_experiment_spec_model_copy_revalidates_updates() -> None:
    spec = make_spec()

    with pytest.raises(ValidationError, match="name"):
        spec.model_copy(update={"name": "Invalid Name"})


def test_portfolio_spec_model_copy_revalidates_cross_field_limits() -> None:
    portfolio = make_spec().portfolio

    with pytest.raises(
        ValidationError,
        match="maximum_position_weight cannot exceed",
    ):
        portfolio.model_copy(
            update={"maximum_position_weight": Decimal("0.96")}
        )


def test_model_copy_allows_valid_revalidated_updates() -> None:
    spec = make_spec()

    updated = spec.model_copy(
        update={"name": "ridge_candidate", "seed": 7},
    )

    assert updated.name == "ridge_candidate"
    assert updated.seed == 7
    assert spec.name == "ridge_baseline"
    assert spec.seed == 42


def test_model_copy_rejects_constructed_nested_model_injection() -> None:
    spec = make_spec()
    unsafe_dataset = DatasetSpec.model_construct(
        name="daily_bars",
        snapshot_id="latest",
        universe_version="mvp-etf-v1",
        feature_version="trend-momentum-v1",
    )

    with pytest.raises(ValidationError, match="immutable snapshot"):
        spec.model_copy(update={"dataset": unsafe_dataset})


def test_legacy_copy_rejects_constructed_nested_model_injection() -> None:
    spec = make_spec()
    unsafe_dataset = DatasetSpec.model_construct(
        name="daily_bars",
        snapshot_id="latest",
        universe_version="mvp-etf-v1",
        feature_version="trend-momentum-v1",
    )

    with pytest.warns(PydanticDeprecatedSince20):
        with pytest.raises(ValidationError, match="immutable snapshot"):
            spec.copy(update={"dataset": unsafe_dataset})


def test_model_copy_allows_valid_nested_model_updates() -> None:
    spec = make_spec()
    dataset = DatasetSpec(snapshot_id="daily-bars:2026-06-13")

    updated = spec.model_copy(update={"dataset": dataset})

    assert updated.dataset == dataset
    assert updated.dataset is not dataset


def test_model_copy_without_updates_preserves_deep_copy_semantics() -> None:
    spec = make_spec()

    shallow = spec.model_copy()
    deep = spec.model_copy(deep=True)

    assert shallow == spec
    assert shallow.dataset is spec.dataset
    assert deep == spec
    assert deep.dataset is not spec.dataset


def test_legacy_copy_updates_are_revalidated() -> None:
    spec = make_spec()

    with pytest.warns(PydanticDeprecatedSince20):
        updated = spec.copy(update={"name": "ridge_candidate"})

    assert updated.name == "ridge_candidate"

    with pytest.warns(PydanticDeprecatedSince20):
        with pytest.raises(ValidationError, match="name"):
            spec.copy(update={"name": "Invalid Name"})


@pytest.mark.parametrize(
    "selection",
    ({"include": {"name"}}, {"exclude": {"seed"}}),
)
def test_legacy_copy_rejects_partial_copies_with_warning(
    selection: dict[str, set[str]],
) -> None:
    spec = make_spec()

    with pytest.warns(PydanticDeprecatedSince20):
        with pytest.raises(TypeError, match="cannot be partially copied"):
            spec.copy(**selection)


def test_unknown_root_key_is_rejected() -> None:
    values = make_spec().model_dump()
    values["unexpected"] = True

    with pytest.raises(ValidationError, match="unexpected"):
        ExperimentSpec.model_validate(values)


def test_unknown_nested_key_is_rejected() -> None:
    values = make_spec().model_dump()
    values["dataset"]["unexpected"] = True

    with pytest.raises(ValidationError, match=r"dataset\.unexpected"):
        ExperimentSpec.model_validate(values)


@pytest.mark.parametrize(
    "snapshot_id",
    (
        "latest",
        "",
        ".",
        "..",
        "snap/child",
        r"snap\child",
        "../snapshot",
        "snapshot with spaces",
        "snapshot@version",
    ),
)
def test_snapshot_id_rejects_unsafe_values(snapshot_id: str) -> None:
    with pytest.raises(ValidationError):
        DatasetSpec(snapshot_id=snapshot_id)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("max_trials", 0),
        ("max_trials", 501),
        ("timeout_minutes", 0),
        ("timeout_minutes", 721),
        ("worker_count", 0),
        ("worker_count", 3),
    ),
)
def test_resource_budget_rejects_values_outside_boundaries(
    field: str,
    value: int,
) -> None:
    with pytest.raises(ValidationError, match=field):
        ResourceBudget.model_validate({field: value})


@pytest.mark.parametrize(
    ("factory", "field", "value"),
    (
        (ResourceBudget, "worker_count", True),
        (ExperimentSpec, "seed", True),
        (ExperimentSpec, "schema_version", True),
        (ExperimentSpec, "schema_version", 1.0),
        (RidgeModelSpec, "fit_intercept", 1),
        (RidgeModelSpec, "alpha", "1.5"),
    ),
)
def test_non_decimal_scalars_reject_coercive_types(
    factory: type[ResourceBudget] | type[ExperimentSpec] | type[RidgeModelSpec],
    field: str,
    value: object,
) -> None:
    if factory is ExperimentSpec:
        values: dict[str, object] = {
            "name": "ridge_baseline",
            "dataset": {"snapshot_id": "daily-bars:2026-06-12"},
            field: value,
        }
    else:
        values = {field: value}

    with pytest.raises(ValidationError, match=field):
        factory.model_validate(values)


@pytest.mark.parametrize(
    ("factory", "field", "value"),
    (
        (RidgeModelSpec, "alpha", float("inf")),
        (RidgeModelSpec, "alpha", float("nan")),
    ),
)
def test_ridge_model_rejects_non_finite_floats(
    factory: type[RidgeModelSpec],
    field: str,
    value: float,
) -> None:
    with pytest.raises(ValidationError, match=field):
        factory.model_validate({field: value})


@pytest.mark.parametrize(
    "sensitivity_bps",
    (
        (),
        (Decimal("-0.01"),),
        (Decimal("100.01"),),
    ),
)
def test_cost_sensitivity_rejects_invalid_ranges(
    sensitivity_bps: tuple[Decimal, ...],
) -> None:
    with pytest.raises(ValidationError, match="sensitivity_bps"):
        CostSpec(sensitivity_bps=sensitivity_bps)


def test_portfolio_rejects_conflicting_limits() -> None:
    with pytest.raises(
        ValidationError,
        match="maximum_position_weight cannot exceed",
    ):
        PortfolioSpec(
            maximum_position_weight=Decimal("0.96"),
            minimum_cash_weight=Decimal("0.05"),
        )


def test_portfolio_rejects_zero_minimum_cash_weight() -> None:
    with pytest.raises(ValidationError, match="minimum_cash_weight"):
        PortfolioSpec(minimum_cash_weight=Decimal("0"))


def test_portfolio_defaults_map_to_runtime_limits() -> None:
    portfolio = PortfolioSpec()

    limits = PortfolioLimits(
        maximum_position_weight=portfolio.maximum_position_weight,
        minimum_cash_weight=portfolio.minimum_cash_weight,
        target_annual_volatility=portfolio.target_annual_volatility,
    )

    assert limits.maximum_position_weight == Decimal("0.25")
    assert limits.minimum_cash_weight == Decimal("0.05")
    assert limits.target_annual_volatility == Decimal("0.15")


def test_decimal_inputs_preserve_exact_values() -> None:
    costs = CostSpec(
        headline_bps=Decimal("3.125"),
        sensitivity_bps=(Decimal("1.25"), Decimal("3.125")),
    )
    portfolio = PortfolioSpec(
        reference_equity=Decimal("1234567.89"),
        maximum_position_weight=Decimal("0.123456789"),
        minimum_cash_weight=Decimal("0.075"),
        target_annual_volatility=Decimal("0.101"),
    )

    assert costs.headline_bps == Decimal("3.125")
    assert costs.sensitivity_bps == (
        Decimal("1.25"),
        Decimal("3.125"),
    )
    assert portfolio.reference_equity == Decimal("1234567.89")
    assert portfolio.maximum_position_weight == Decimal("0.123456789")
    assert portfolio.minimum_cash_weight == Decimal("0.075")
    assert portfolio.target_annual_volatility == Decimal("0.101")
