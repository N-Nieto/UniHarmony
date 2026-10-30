"""Shared tests for all ComBat-based methods.

Every test in this module runs for each ComBat variant listed in ``COMBAT_CASES``
(NeuroComBat, CovBat and ComBatGAM). Method-specific behaviour (hyperparameters,
reference implementations, smoothing, covariance correction, ...) is tested in
the per-method modules.

To add a new ComBat variant, add a ``ComBatCase`` to ``COMBAT_CASES`` and, if
needed, its expected sklearn check failures to ``_expected_failed_checks``.

Data comes from the shared fixtures in ``tests/conftest.py``.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from uniharmony.combat import ComBatGAM, CovBat, NeuroComBat


# ---------------------------------------------------------------------------
# ComBat variants under test
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComBatCase:
    """How to build and call one ComBat variant in the shared tests.

    Attributes
    ----------
    estimator_cls : type
        The ComBat estimator class.
    params : dict
        Parameters always passed to the constructor (e.g., a fixed ``random_state``).
    variants : list of dict
        Hyperparameter sets that must all run (e.g., empirical Bayes on/off).
    supports_categorical : bool
        Whether ``categorical_covariates`` is accepted.
    smooth_covariate : str or None
        Covariate passed as ``smooth_covariates`` (required by ComBatGAM).
        It is reserved for smoothing and not used as a continuous covariate.
    known_failures : dict of str to str
        Shared tests known to fail for this variant, mapped to the reason.
        They are marked as strict xfail, so they fail once the bug is fixed
        and the entry must then be removed.

    """

    estimator_cls: type
    params: dict[str, Any] = field(default_factory=dict)
    variants: list[dict[str, Any]] = field(default_factory=lambda: [{}])
    supports_categorical: bool = True
    smooth_covariate: str | None = None
    known_failures: dict[str, str] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """Name used in test ids."""
        return self.estimator_cls.__name__

    def make(self, **params: Any) -> Any:
        """Instantiate the estimator with the case parameters and ``params``."""
        return self.estimator_cls(**{**self.params, **params})

    def covariate_kwargs(self, data: Any, categorical: tuple = (), continuous: tuple = ()) -> dict[str, Any]:
        """Build the covariate keyword arguments for ``fit`` / ``transform``.

        Skips the test if the variant does not support the requested covariates.
        """
        kwargs: dict[str, Any] = {}
        if self.smooth_covariate is not None:
            kwargs["smooth_covariates"] = data.get(self.smooth_covariate)
        if categorical:
            if not self.supports_categorical:
                pytest.skip(f"{self.name} does not support categorical covariates")
            kwargs["categorical_covariates"] = data.get(*categorical)
        if continuous:
            kwargs["continuous_covariates"] = data.get(*continuous)
        return kwargs

    def age_preserving_kwargs(self, data: Any) -> dict[str, Any]:
        """Keyword arguments that ask the method to preserve the age effect."""
        if self.smooth_covariate == "age":
            return {"smooth_covariates": data.get("age")}
        return {"continuous_covariates": data.get("age")}


_EB_VARIANTS = [
    {"empirical_bayes": eb, "parametric_adjustments": parametric, "mean_only": mean_only}
    for eb in (True, False)
    for parametric in (True, False)
    for mean_only in (True, False)
]

COMBAT_CASES = [
    ComBatCase(NeuroComBat, variants=_EB_VARIANTS),
    ComBatCase(
        CovBat,
        variants=[
            {},
            {"first_combat_eb": False},
            {"first_combat_parametric": False},
            {"score_eb": True},
            {"std_var": False},
            {"n_pc": 3},
            {"residualize": True},
        ],
    ),
    ComBatCase(
        ComBatGAM,
        # Penalty weights are selected by shuffled k-fold CV: fix the seed so results are reproducible
        params={"random_state": 0},
        variants=_EB_VARIANTS,
        supports_categorical=False,
        smooth_covariate="age",
    ),
]


@pytest.fixture(params=COMBAT_CASES, ids=lambda case: case.name)
def combat_case(request: pytest.FixtureRequest) -> ComBatCase:
    """Run a test once per ComBat variant, marking its known failures as strict xfail."""
    case = request.param
    reason = case.known_failures.get(request.node.originalname)
    if reason is not None:
        request.applymarker(pytest.mark.xfail(reason=reason, strict=True))
    return case


# Covariate combinations used by the shared tests. "age" is not used as a
# continuous covariate here because it is ComBatGAM's smooth covariate.
CATEGORICAL_ONLY = [("sex",), ("sex", "education")]
CONTINUOUS_ONLY = [("extra",), ("extra", "education")]
MIXED = [(("sex",), ("extra",)), (("sex", "education"), ("extra",))]


# ---------------------------------------------------------------------------
# sklearn compatibility
# ---------------------------------------------------------------------------

_SITES_INSTEAD_OF_Y = "sites instead of y"
_MISSING_SMOOTH = "missing smooth covariates"

# Checks expected to fail for every ComBat variant
_COMMON_EXPECTED_FAILED_CHECKS = {
    "check_transformers_unfitted": "checked inside",
    "check_n_features_in_after_fitting": "not needed",
    "check_n_features_in": "not needed",
    "check_requires_y_none": "target cannot be None",
    "check_fit2d_1sample": "custom message",
    "check_fit_score_takes_y": _SITES_INSTEAD_OF_Y,
    "check_estimators_dtypes": _SITES_INSTEAD_OF_Y,
    "check_dtype_object": _SITES_INSTEAD_OF_Y,
    "check_estimators_pickle": _SITES_INSTEAD_OF_Y,
    "check_f_contiguous_array_estimator": _SITES_INSTEAD_OF_Y,
    "check_transformer_data_not_an_array": _SITES_INSTEAD_OF_Y,
    "check_transformer_preserve_dtypes": _SITES_INSTEAD_OF_Y,
    "check_transformer_general": _SITES_INSTEAD_OF_Y,
    "check_methods_sample_order_invariance": _SITES_INSTEAD_OF_Y,
    "check_methods_subset_invariance": _SITES_INSTEAD_OF_Y,
    "check_dict_unchanged": _SITES_INSTEAD_OF_Y,
    "check_fit_idempotent": _SITES_INSTEAD_OF_Y,
    "check_fit2d_predict1d": _SITES_INSTEAD_OF_Y,
}

# Additional checks expected to fail per variant
_EXTRA_EXPECTED_FAILED_CHECKS = {
    NeuroComBat: {
        "check_estimators_nan_inf": "checked inside",
    },
    CovBat: {
        "check_estimators_nan_inf": "checked inside",
        "check_fit2d_1feature": "Harmonization produced non-finite values",
    },
    ComBatGAM: dict.fromkeys(
        (
            "check_complex_data",
            "check_dont_overwrite_parameters",
            "check_estimator_sparse_array",
            "check_estimator_sparse_matrix",
            "check_estimator_sparse_tag",
            "check_estimators_empty_data_messages",
            "check_estimators_fit_returns_self",
            "check_estimators_overwrite_params",
            "check_fit1d",
            "check_fit2d_1feature",
            "check_fit_check_is_fitted",
            "check_pipeline_consistency",
            "check_positive_only_tag_during_fit",
            "check_readonly_memmap_input",
        ),
        _MISSING_SMOOTH,
    ),
}


def _expected_failed_checks(estimator: object) -> dict[str, str]:
    return {**_COMMON_EXPECTED_FAILED_CHECKS, **_EXTRA_EXPECTED_FAILED_CHECKS[type(estimator)]}


@parametrize_with_checks(
    [
        *(NeuroComBat(**params) for params in _EB_VARIANTS if params["parametric_adjustments"]),
        CovBat(),
        *(ComBatGAM(**params) for params in _EB_VARIANTS if params["parametric_adjustments"]),
    ],
    expected_failed_checks=_expected_failed_checks,
)
def test_combat_compat_sklearn(estimator: object, check: Callable) -> None:
    """Test sklearn compatibility of every ComBat variant."""
    check(estimator)


# ---------------------------------------------------------------------------
# API: fit / transform
# ---------------------------------------------------------------------------


def test_fit_transform_shape_and_finite(combat_case: ComBatCase, any_multisite_data) -> None:
    """Output has the input shape, is finite and differs from the input."""
    d = any_multisite_data
    X_harmonized = combat_case.make().fit_transform(d.X, d.sites, **combat_case.covariate_kwargs(d))
    assert X_harmonized.shape == d.X.shape
    assert np.all(np.isfinite(X_harmonized))
    assert not np.allclose(X_harmonized, d.X)


def test_all_variants_run(combat_case: ComBatCase, multisite_data) -> None:
    """Every hyperparameter variant fits and transforms."""
    d = multisite_data
    kwargs = combat_case.covariate_kwargs(d)
    for params in combat_case.variants:
        X_harmonized = combat_case.make(**params).fit_transform(d.X, d.sites, **kwargs)
        assert X_harmonized.shape == d.X.shape, params
        assert np.all(np.isfinite(X_harmonized)), params


def test_fit_then_transform_equals_fit_transform(combat_case: ComBatCase, multisite_data) -> None:
    """``fit(...).transform(...)`` equals ``fit_transform(...)``."""
    d = multisite_data
    kwargs = combat_case.covariate_kwargs(d, continuous=("extra",))
    X1 = combat_case.make().fit_transform(d.X, d.sites, **kwargs)
    X2 = combat_case.make().fit(d.X, d.sites, **kwargs).transform(d.X, d.sites, **kwargs)
    np.testing.assert_allclose(X1, X2)


def test_transform_held_out_data(combat_case: ComBatCase, multisite_data, rng) -> None:
    """A model fitted on training data harmonizes held-out data with the same sites."""
    d = multisite_data
    test_mask = rng.random(d.n_samples) < 0.2
    kwargs = combat_case.covariate_kwargs(d, continuous=("extra",))
    train_kwargs = {key: value[~test_mask] for key, value in kwargs.items()}
    test_kwargs = {key: value[test_mask] for key, value in kwargs.items()}
    model = combat_case.make().fit(d.X[~test_mask], d.sites[~test_mask], **train_kwargs)
    X_test = model.transform(d.X[test_mask], d.sites[test_mask], **test_kwargs)
    assert X_test.shape == d.X[test_mask].shape
    assert np.all(np.isfinite(X_test))


def test_transform_subset_of_sites(combat_case: ComBatCase, multisite_data) -> None:
    """Transforming data from only some of the fitted sites works."""
    d = multisite_data
    kwargs = combat_case.covariate_kwargs(d)
    model = combat_case.make().fit(d.X, d.sites, **kwargs)
    one_site = d.sites == d.sites[0]
    X_site = model.transform(d.X[one_site], d.sites[one_site], **{k: v[one_site] for k, v in kwargs.items()})
    assert X_site.shape == d.X[one_site].shape


def test_reproducible(combat_case: ComBatCase, multisite_data) -> None:
    """Two fits on the same data give identical results."""
    d = multisite_data
    kwargs = combat_case.covariate_kwargs(d)
    X1 = combat_case.make().fit_transform(d.X, d.sites, **kwargs)
    X2 = combat_case.make().fit_transform(d.X, d.sites, **kwargs)
    np.testing.assert_array_equal(X1, X2)


def test_repeated_transform_consistent(combat_case: ComBatCase, multisite_data) -> None:
    """Calling transform twice gives the same result."""
    d = multisite_data
    kwargs = combat_case.covariate_kwargs(d)
    model = combat_case.make().fit(d.X, d.sites, **kwargs)
    np.testing.assert_array_equal(model.transform(d.X, d.sites, **kwargs), model.transform(d.X, d.sites, **kwargs))


def test_max_iter(combat_case: ComBatCase, multisite_data) -> None:
    """``max_iter`` is accepted by fit (solver may stop before converging)."""
    d = multisite_data
    model = combat_case.make().fit(d.X, d.sites, max_iter=1, **combat_case.covariate_kwargs(d))
    assert np.all(np.isfinite(model.transform(d.X, d.sites, **combat_case.covariate_kwargs(d))))


# ---------------------------------------------------------------------------
# Site validation
# ---------------------------------------------------------------------------


def test_single_site_raises(combat_case: ComBatCase, multisite_data) -> None:
    """At least two sites are required."""
    d = multisite_data
    one_site = d.sites == d.sites[0]
    kwargs = {k: v[one_site] for k, v in combat_case.covariate_kwargs(d).items()}
    with pytest.raises(ValueError, match="At least 2 sites required"):
        combat_case.make().fit(d.X[one_site], d.sites[one_site], **kwargs)


def test_unseen_site_raises(combat_case: ComBatCase, multisite_data_string_sites) -> None:
    """Transforming data with a site not seen during fit raises."""
    d = multisite_data_string_sites
    kwargs = combat_case.covariate_kwargs(d)
    model = combat_case.make().fit(d.X, d.sites, **kwargs)
    unseen = d.sites.copy()
    unseen[0] = "Z"
    with pytest.raises(ValueError, match=r"(unseen|not seen) during"):
        model.transform(d.X, unseen, **kwargs)


def test_all_sites_renamed_raises(combat_case: ComBatCase, multisite_data_string_sites) -> None:
    """Same number of sites but different labels at transform raises."""
    d = multisite_data_string_sites
    kwargs = combat_case.covariate_kwargs(d)
    model = combat_case.make().fit(d.X, d.sites, **kwargs)
    renamed = np.char.add("new_", d.sites.astype(str))
    with pytest.raises(ValueError, match=r"(unseen|not seen) during"):
        model.transform(d.X, renamed, **kwargs)


def test_nan_in_sites_raises(combat_case: ComBatCase, multisite_data) -> None:
    """Missing site labels are rejected."""
    d = multisite_data
    sites = d.sites.astype(object)
    sites[0] = np.nan
    with pytest.raises(ValueError):
        combat_case.make().fit(d.X, sites, **combat_case.covariate_kwargs(d))


# ---------------------------------------------------------------------------
# Harmonization effect
# ---------------------------------------------------------------------------


def _mean_site_spread(X: np.ndarray, sites: np.ndarray) -> float:
    """Mean pairwise distance between per-site feature means."""
    means = [X[sites == s].mean(axis=0) for s in np.unique(sites)]
    return float(np.mean([np.linalg.norm(a - b) for i, a in enumerate(means) for b in means[i + 1 :]]))


def test_site_mean_differences_reduced(combat_case: ComBatCase, any_multisite_data) -> None:
    """Harmonization reduces the differences between site means."""
    d = any_multisite_data
    X_harmonized = combat_case.make().fit_transform(d.X, d.sites, **combat_case.covariate_kwargs(d))
    assert _mean_site_spread(X_harmonized, d.sites) < 0.5 * _mean_site_spread(d.X, d.sites)


def test_age_effect_preserved(combat_case: ComBatCase, three_site_data) -> None:
    """The age effect keeps its direction for most features when age is preserved."""
    d = three_site_data
    age = d.covariates["age"]
    X_harmonized = combat_case.make().fit_transform(d.X, d.sites, **combat_case.age_preserving_kwargs(d))
    raw_corr = np.array([np.corrcoef(d.X[:, j], age)[0, 1] for j in range(d.n_features)])
    harm_corr = np.array([np.corrcoef(X_harmonized[:, j], age)[0, 1] for j in range(d.n_features)])
    assert (np.sign(raw_corr) == np.sign(harm_corr)).mean() >= 0.8


# ---------------------------------------------------------------------------
# Covariates (any number of columns, e.g., age, sex and education)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("categorical", "continuous"),
    [*((c, ()) for c in CATEGORICAL_ONLY), *(((), c) for c in CONTINUOUS_ONLY), *MIXED],
)
def test_multiple_covariates(combat_case: ComBatCase, any_multisite_data, categorical: tuple, continuous: tuple) -> None:
    """Fit and transform work with any number of covariate columns."""
    d = any_multisite_data
    kwargs = combat_case.covariate_kwargs(d, categorical, continuous)
    model = combat_case.make().fit(d.X, d.sites, **kwargs)
    X_harmonized = model.transform(d.X, d.sites, **kwargs)
    assert X_harmonized.shape == d.X.shape
    assert np.all(np.isfinite(X_harmonized))


@pytest.mark.parametrize("kind", ["categorical", "continuous"])
def test_1d_and_single_column_covariates_are_equivalent(combat_case: ComBatCase, multisite_data, kind: str) -> None:
    """A 1D covariate and the same covariate as an (n, 1) column give the same result."""
    d = multisite_data
    name = "sex" if kind == "categorical" else "extra"
    kwargs_2d = combat_case.covariate_kwargs(d, **{kind: (name,)})
    kwargs_1d = {**kwargs_2d, f"{kind}_covariates": d.covariates[name]}
    out_1d = combat_case.make().fit(d.X, d.sites, **kwargs_1d).transform(d.X, d.sites, **kwargs_1d)
    out_2d = combat_case.make().fit(d.X, d.sites, **kwargs_2d).transform(d.X, d.sites, **kwargs_2d)
    np.testing.assert_allclose(out_1d, out_2d)


@pytest.mark.parametrize(("kind", "names"), [("categorical", ("sex", "education")), ("continuous", ("extra", "education"))])
def test_every_covariate_column_is_used(combat_case: ComBatCase, multisite_data, kind: str, names: tuple) -> None:
    """Adding a covariate column changes the result (it is not silently dropped)."""
    d = multisite_data
    out_one = combat_case.make().fit_transform(d.X, d.sites, **combat_case.covariate_kwargs(d, **{kind: names[:1]}))
    out_two = combat_case.make().fit_transform(d.X, d.sites, **combat_case.covariate_kwargs(d, **{kind: names}))
    assert not np.allclose(out_one, out_two)


def test_transform_rejects_different_number_of_categorical_columns(combat_case: ComBatCase, multisite_data) -> None:
    """Transform fails clearly if the number of categorical columns differs from fit."""
    d = multisite_data
    model = combat_case.make().fit(d.X, d.sites, **combat_case.covariate_kwargs(d, categorical=("sex", "education")))
    with pytest.raises(ValueError, match="categorical_covariates has 1 columns, but 2 were seen during fit"):
        model.transform(d.X, d.sites, **combat_case.covariate_kwargs(d, categorical=("sex",)))


@pytest.mark.parametrize("kind", ["categorical", "continuous"])
def test_3d_covariates_rejected(combat_case: ComBatCase, multisite_data, kind: str) -> None:
    """Covariates with more than 2 dimensions are rejected."""
    d = multisite_data
    name = "sex" if kind == "categorical" else "extra"
    kwargs = combat_case.covariate_kwargs(d, **{kind: (name,)})
    kwargs[f"{kind}_covariates"] = kwargs[f"{kind}_covariates"][:, :, np.newaxis]
    with pytest.raises(ValueError, match="dim 3"):
        combat_case.make().fit(d.X, d.sites, **kwargs)


def test_nan_in_categorical_covariates_raises(combat_case: ComBatCase, multisite_data) -> None:
    """Missing values in categorical covariates are rejected."""
    d = multisite_data
    kwargs = combat_case.covariate_kwargs(d, categorical=("sex",))
    sex = kwargs["categorical_covariates"].astype(object)
    sex[0, 0] = np.nan
    kwargs["categorical_covariates"] = sex
    with pytest.raises(ValueError):
        combat_case.make().fit(d.X, d.sites, **kwargs)
