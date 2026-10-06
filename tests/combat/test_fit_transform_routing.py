"""Test ``fit_transform`` of every ComBat variant.

Each variant declares ``fit_transform`` with the same, typed signature as its
``fit`` (so IDEs and type checkers show every argument) and delegates to
``BaseComBat._fit_then_transform``: shared data (covariates) go to ``fit`` and
``transform``, fit-only options (``var_epsilon``, ``max_iter``, ``df``, ...)
only to ``fit``.
"""

import inspect
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
from statsmodels.gam.api import GLMGam
from statsmodels.gam.gam_cross_validation.cross_validators import KFold

from uniharmony.combat import ComBatGAM, CovBat, NeuroComBat


@dataclass(frozen=True)
class RoutingCase:
    """How to call one ComBat variant.

    Attributes
    ----------
    estimator_cls : type
        The estimator.
    data_kwargs : list of str
        Covariates (by fixture key) passed to both ``fit`` and ``transform``.
    fit_only_kwargs : dict
        Every fit-only option with a non-default value.

    """

    estimator_cls: type
    data_kwargs: dict[str, str]
    fit_only_kwargs: dict[str, Any] = field(default_factory=dict)


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
    ),
}


@pytest.fixture(autouse=True)
def deterministic_gam_penalty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ComBatGAM deterministic: its penalty search uses unseeded shuffled folds (audit #19b)."""
    select_penweight_kfold = GLMGam.select_penweight_kfold

    def unshuffled(self, *args, **kwargs):
        kwargs.setdefault("cv_iterator", KFold(k_folds=kwargs.get("k_folds", 5), shuffle=False))
        return select_penweight_kfold(self, *args, **kwargs)

    monkeypatch.setattr(GLMGam, "select_penweight_kfold", unshuffled)


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


def test_fit_only_options_are_not_passed_to_transform(case: RoutingCase, data) -> None:
    """Fit-only options are accepted and give the same result as fit().transform()."""
    kwargs = _data_kwargs(case, data)
    X_fit_transform = case.estimator_cls().fit_transform(data["X"], data["sites"], **kwargs, **case.fit_only_kwargs)
    estimator = case.estimator_cls().fit(data["X"], data["sites"], **kwargs, **case.fit_only_kwargs)
    X_fit_then_transform = estimator.transform(data["X"], data["sites"], **kwargs)
    np.testing.assert_allclose(X_fit_transform, X_fit_then_transform, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("option", ["var_epsilon", "delta_epsilon", "tau_2_epsilon", "max_iter"])
def test_each_fit_only_option(case: RoutingCase, data, option: str) -> None:
    """Every fit-only option can be passed on its own."""
    X_harmonized = case.estimator_cls().fit_transform(
        data["X"], data["sites"], **_data_kwargs(case, data), **{option: case.fit_only_kwargs[option]}
    )
    assert X_harmonized.shape == data["X"].shape
    assert np.all(np.isfinite(X_harmonized))


@pytest.mark.parametrize("option", ["df", "degree", "smooth_covariates_bounds"])
def test_each_gam_fit_only_option(data, option: str) -> None:
    """ComBatGAM's spline options are fit-only."""
    case = CASES["ComBatGAM"]
    X_harmonized = ComBatGAM().fit_transform(
        data["X"], data["sites"], **_data_kwargs(case, data), **{option: case.fit_only_kwargs[option]}
    )
    assert X_harmonized.shape == data["X"].shape


def test_covariates_reach_transform(case: RoutingCase, data) -> None:
    """Covariates given to fit_transform are also used by transform (keyword or positional)."""
    kwargs = _data_kwargs(case, data)
    X_keyword = case.estimator_cls().fit_transform(data["X"], data["sites"], **kwargs)
    # Positional arguments follow the order of fit: the first data argument after sites
    first, *rest = kwargs
    X_positional = case.estimator_cls().fit_transform(
        data["X"], data["sites"], kwargs[first], **{name: kwargs[name] for name in rest}
    )
    np.testing.assert_allclose(X_keyword, X_positional, rtol=1e-10, atol=1e-10)
    X_fit_then_transform = (
        case.estimator_cls().fit(data["X"], data["sites"], **kwargs).transform(data["X"], data["sites"], **kwargs)
    )
    np.testing.assert_allclose(X_keyword, X_fit_then_transform, rtol=1e-10, atol=1e-10)


def test_keyword_data_arguments(case: RoutingCase, data) -> None:
    """X and sites can be passed by keyword."""
    X_harmonized = case.estimator_cls().fit_transform(X=data["X"], sites=data["sites"], **_data_kwargs(case, data))
    assert X_harmonized.shape == data["X"].shape


def test_signature_mirrors_fit(case: RoutingCase) -> None:
    """fit_transform declares exactly the parameters of fit (names, order, defaults, annotations)."""
    fit = inspect.signature(case.estimator_cls.fit)
    fit_transform = inspect.signature(case.estimator_cls.fit_transform)
    assert list(fit_transform.parameters.values()) == list(fit.parameters.values())
    assert fit_transform.return_annotation == inspect.signature(case.estimator_cls.transform).return_annotation


def test_docstring_documents_every_parameter(case: RoutingCase) -> None:
    """Every parameter of fit_transform is documented in its docstring."""
    doc = inspect.getdoc(case.estimator_cls.fit_transform)
    for name in list(inspect.signature(case.estimator_cls.fit_transform).parameters)[1:]:
        assert re.search(rf"^{name} : ", doc, flags=re.MULTILINE), f"{name} is not documented"


def test_unknown_argument_raises(case: RoutingCase, data) -> None:
    """Arguments that fit does not accept raise a TypeError naming them."""
    with pytest.raises(TypeError, match="not_an_option"):
        case.estimator_cls().fit_transform(data["X"], data["sites"], **_data_kwargs(case, data), not_an_option=1)


def test_missing_required_argument_raises(data) -> None:
    """A missing required fit argument raises like a direct fit call."""
    with pytest.raises(TypeError, match="smooth_covariates"):
        ComBatGAM().fit_transform(data["X"], data["sites"])
