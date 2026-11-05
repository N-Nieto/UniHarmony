"""Test ``fit_transform`` and the numerical options of every ComBat variant.

The numerical options (``var_epsilon``, ``max_iter``, ``df``, ...) are
constructor parameters, so ``fit`` and ``transform`` only take data: ``y``
(ignored) and the metadata keywords (sites, covariates). ``fit_transform``
passes the metadata to both ``fit`` and ``transform``.
"""

import inspect
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
from sklearn.base import clone

from uniharmony.combat import ComBatGAM, CovBat, NeuroComBat


@dataclass(frozen=True)
class RoutingCase:
    """How to call one ComBat variant.

    Attributes
    ----------
    estimator_cls : type
        The estimator.
    data_kwargs : dict of str to str
        Covariate keyword -> fixture key, passed to both ``fit`` and ``transform``.
    options : dict
        Every numerical option with a non-default value.
    params : dict
        Other constructor parameters (e.g., a fixed ``random_state``).

    """

    estimator_cls: type
    data_kwargs: dict[str, str]
    options: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)

    def make(self, **params: Any) -> Any:
        """Build a new estimator."""
        return self.estimator_cls(**{**self.params, **params})


_EPSILONS = {"var_epsilon": 1e-6, "delta_epsilon": 1e-6, "tau_2_epsilon": 1e-9, "max_iter": 500}

CASES = {
    "NeuroComBat": RoutingCase(
        NeuroComBat,
        {"categorical_covariates": "sex", "continuous_covariates": "age"},
        _EPSILONS,
    ),
    "CovBat": RoutingCase(
        CovBat,
        {"categorical_covariates": "sex", "continuous_covariates": "age"},
        _EPSILONS,
    ),
    "ComBatGAM": RoutingCase(
        ComBatGAM,
        {"smooth_covariates": "age", "continuous_covariates": "extra"},
        {**_EPSILONS, "df": 6, "degree": 2, "smooth_covariates_bounds": (10.0, 90.0)},
        # The GAM penalty is selected by shuffled k-fold CV: fix the seed so results are reproducible
        params={"random_state": 0},
    ),
}


@pytest.fixture(params=list(CASES), ids=list(CASES))
def case(request: pytest.FixtureRequest) -> RoutingCase:
    """Each ComBat variant."""
    return CASES[request.param]


@pytest.fixture
def data() -> dict[str, np.ndarray]:
    """Three sites with additive and multiplicative site effects and covariate effects."""
    rng = np.random.default_rng(0)
    n = 240
    sites = np.repeat(["a", "b", "c"], n // 3)
    age = rng.uniform(20, 80, n)
    sex = rng.integers(0, 2, n)
    extra = rng.standard_normal(n)
    site_shift = np.select([sites == "a", sites == "b"], [0.0, 2.0], -1.5)
    site_scale = np.select([sites == "a", sites == "b"], [1.0, 1.8], 0.6)
    X = 0.05 * age[:, None] + 0.5 * sex[:, None] + site_shift[:, None] + site_scale[:, None] * rng.standard_normal((n, 5))
    return {"X": X, "sites": sites, "age": age[:, None], "sex": sex[:, None], "extra": extra[:, None]}


def _data_kwargs(case: RoutingCase, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: data[key] for name, key in case.data_kwargs.items()}


def test_fit_transform_equals_fit_then_transform(case: RoutingCase, data) -> None:
    """With every option set, fit_transform gives the same result as fit().transform()."""
    kwargs = _data_kwargs(case, data)
    X_fit_transform = case.make(**case.options).fit_transform(data["X"], sites=data["sites"], **kwargs)
    estimator = case.make(**case.options).fit(data["X"], sites=data["sites"], **kwargs)
    X_fit_then_transform = estimator.transform(data["X"], sites=data["sites"], **kwargs)
    np.testing.assert_allclose(X_fit_transform, X_fit_then_transform, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("option", ["var_epsilon", "delta_epsilon", "tau_2_epsilon", "max_iter"])
def test_each_option_is_a_constructor_parameter(case: RoutingCase, data, option: str) -> None:
    """Every numerical option is a constructor parameter (get_params, set_params, clone)."""
    estimator = case.make(**{option: case.options[option]})
    assert estimator.get_params()[option] == case.options[option]
    assert clone(estimator).get_params()[option] == case.options[option]
    X_harmonized = estimator.fit_transform(data["X"], sites=data["sites"], **_data_kwargs(case, data))
    assert X_harmonized.shape == data["X"].shape
    assert np.all(np.isfinite(X_harmonized))


@pytest.mark.parametrize("option", ["df", "degree", "smooth_covariates_bounds"])
def test_each_gam_option_is_a_constructor_parameter(data, option: str) -> None:
    """ComBatGAM's spline options are constructor parameters, and they change the result."""
    case = CASES["ComBatGAM"]
    kwargs = _data_kwargs(case, data)
    default = case.make().fit_transform(data["X"], sites=data["sites"], **kwargs)
    changed = case.make(**{option: case.options[option]}).fit_transform(data["X"], sites=data["sites"], **kwargs)
    assert changed.shape == data["X"].shape
    assert not np.allclose(default, changed)


def test_max_iter_is_used(data) -> None:
    """max_iter in the constructor reaches the parametric empirical Bayes solver."""
    kwargs = {"continuous_covariates": data["age"]}
    one = NeuroComBat(max_iter=1).fit_transform(data["X"], sites=data["sites"], **kwargs)
    converged = NeuroComBat().fit_transform(data["X"], sites=data["sites"], **kwargs)
    assert not np.allclose(one, converged)


def test_covariates_reach_transform(case: RoutingCase, data) -> None:
    """Covariates given to fit_transform are also used by transform."""
    kwargs = _data_kwargs(case, data)
    X_fit_transform = case.make().fit_transform(data["X"], sites=data["sites"], **kwargs)
    X_fit_then_transform = (
        case.make().fit(data["X"], sites=data["sites"], **kwargs).transform(data["X"], sites=data["sites"], **kwargs)
    )
    np.testing.assert_allclose(X_fit_transform, X_fit_then_transform, rtol=1e-10, atol=1e-10)


def test_y_is_second_and_ignored(case: RoutingCase, data) -> None:
    """The target is the second positional argument of fit and fit_transform, and it is ignored."""
    kwargs = _data_kwargs(case, data)
    y = np.random.default_rng(1).integers(0, 2, len(data["X"]))
    with_y = case.make().fit_transform(data["X"], y, sites=data["sites"], **kwargs)
    without_y = case.make().fit_transform(data["X"], sites=data["sites"], **kwargs)
    np.testing.assert_array_equal(with_y, without_y)
    assert list(inspect.signature(case.estimator_cls.fit).parameters)[:3] == ["self", "X", "y"]


def test_metadata_are_keyword_only(case: RoutingCase) -> None:
    """The sites and covariates can only be passed by keyword (they cannot land in the slot of y)."""
    for method in ("fit", "transform"):
        parameters = inspect.signature(getattr(case.estimator_cls, method)).parameters
        for name in ("sites", *case.data_kwargs):
            assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY, f"{method}({name})"


def test_sites_are_required(case: RoutingCase, data) -> None:
    """Without site_column nor sites, fit explains both ways to give them."""
    with pytest.raises(ValueError, match=r"site_column=.*sites="):
        case.make().fit(data["X"], **_data_kwargs(case, data))


def test_unknown_argument_raises(case: RoutingCase, data) -> None:
    """Arguments that fit does not accept raise a TypeError naming them."""
    with pytest.raises(TypeError, match="not_an_option"):
        case.make().fit_transform(data["X"], sites=data["sites"], **_data_kwargs(case, data), not_an_option=1)


def test_missing_required_covariate_raises(data) -> None:
    """ComBatGAM needs smooth covariates, as a column or a keyword."""
    with pytest.raises(ValueError, match=r"smooth_covariates.*smooth_columns"):
        ComBatGAM().fit_transform(data["X"], sites=data["sites"])
