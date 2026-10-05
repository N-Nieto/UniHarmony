"""Test BARTharm.

Data come from the BARTharm simulation in ``tests/iqm/conftest.py``. The
sampler runs with few iterations and trees to keep the tests fast.
"""

from collections.abc import Callable

import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from uniharmony.iqm import BARTharm
from uniharmony.iqm._bart import _QuantileNormalizer


# Fast sampler settings used by most tests
FAST = {"n_iter": 120, "burn_in": 40, "thinning_interval": 2, "n_trees_mu": 20, "n_trees_tau": 10}
# Kept draws with FAST: 120 // 2 - 40 // 2
N_KEPT = 40


def _between_site_variance(values: np.ndarray, sites: np.ndarray) -> float:
    """Variance of ``values`` explained by the site means."""
    labels, codes = np.unique(sites, return_inverse=True)
    means = np.array([values[codes == s].mean() for s in range(len(labels))])
    return float(np.var(means[codes]))


# ---------------------------------------------------------------------------
# sklearn compatibility
# ---------------------------------------------------------------------------

_IQMS_INSTEAD_OF_Y = "fit and transform require iqm_covariates instead of y"

_EXPECTED_FAILED_CHECKS = {
    "check_transformer_data_not_an_array": _IQMS_INSTEAD_OF_Y,
    "check_transformer_general": _IQMS_INSTEAD_OF_Y,
    "check_transformer_preserve_dtypes": _IQMS_INSTEAD_OF_Y,
    "check_methods_sample_order_invariance": _IQMS_INSTEAD_OF_Y,
    "check_methods_subset_invariance": _IQMS_INSTEAD_OF_Y,
    "check_dict_unchanged": _IQMS_INSTEAD_OF_Y,
    "check_fit_idempotent": _IQMS_INSTEAD_OF_Y,
    "check_fit2d_predict1d": _IQMS_INSTEAD_OF_Y,
    "check_n_features_in_after_fitting": _IQMS_INSTEAD_OF_Y,
    "check_estimators_dtypes": _IQMS_INSTEAD_OF_Y,
    "check_dtype_object": _IQMS_INSTEAD_OF_Y,
    "check_estimators_nan_inf": _IQMS_INSTEAD_OF_Y,
    "check_estimators_pickle": _IQMS_INSTEAD_OF_Y,
    "check_f_contiguous_array_estimator": _IQMS_INSTEAD_OF_Y,
    "check_transformers_unfitted": _IQMS_INSTEAD_OF_Y,
    "check_requires_y_none": "iqm_covariates cannot be None",
    "check_fit_score_takes_y": "fit takes iqm_covariates, not y",
}


@parametrize_with_checks(
    [BARTharm(n_iter=6, burn_in=2, n_trees_mu=3, n_trees_tau=2, n_stored_draws=2, random_state=0)],
    expected_failed_checks=lambda _: _EXPECTED_FAILED_CHECKS,
)
def test_bartharm_compat_sklearn(estimator: BARTharm, check: Callable) -> None:
    """Test sklearn compatibility."""
    check(estimator)


# ---------------------------------------------------------------------------
# Harmonization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("var_scaling", [False, True])
def test_removes_iqm_driven_scanner_effect(iqm_data, var_scaling: bool) -> None:
    """The harmonized outcome recovers the scanner-free outcome."""
    d = iqm_data
    harmonizer = BARTharm(n_iter=300, burn_in=100, n_trees_mu=30, n_trees_tau=20, var_scaling=var_scaling, random_state=0)
    X_harmonized = harmonizer.fit_transform(d.X, d.iqms, d.biological_covariates, sites=d.sites)

    raw_corr = np.corrcoef(d.X[:, 0], d.X_clean[:, 0])[0, 1]
    harmonized_corr = np.corrcoef(X_harmonized[:, 0], d.X_clean[:, 0])[0, 1]
    assert raw_corr < 0.7
    assert harmonized_corr > 0.99

    # The difference to the clean outcome no longer depends on the scanner.
    # (It has an arbitrary offset: the intercept is split between mu and tau.)
    raw_site_variance = _between_site_variance((d.X - d.X_clean)[:, 0], d.sites)
    assert _between_site_variance((X_harmonized - d.X_clean)[:, 0], d.sites) < raw_site_variance / 100
    assert harmonizer.rmse_[0] < 0.1


def test_nonlinear_effects(iqm_data_nonlinear) -> None:
    """Nonlinear scanner and biological effects are separated too."""
    d = iqm_data_nonlinear
    harmonizer = BARTharm(n_iter=300, burn_in=100, n_trees_mu=30, n_trees_tau=20, random_state=0)
    X_harmonized = harmonizer.fit_transform(d.X, d.iqms, d.biological_covariates)
    assert np.corrcoef(X_harmonized[:, 0], d.X_clean[:, 0])[0, 1] > 0.99
    raw_site_variance = _between_site_variance((d.X - d.X_clean)[:, 0], d.sites)
    assert _between_site_variance((X_harmonized - d.X_clean)[:, 0], d.sites) < raw_site_variance / 10


def test_multiple_features(iqm_data) -> None:
    """Every feature is harmonized independently, with its own scale."""
    d = iqm_data
    X = np.column_stack([d.X[:, 0], 0.01 * d.X[:, 0] + 3.0, d.X_clean[:, 0]])
    X_harmonized = BARTharm(**FAST, random_state=0).fit_transform(X, d.iqms, d.biological_covariates)
    assert X_harmonized.shape == X.shape
    assert np.all(np.isfinite(X_harmonized))
    # Feature 1 is a rescaled feature 0
    assert np.corrcoef(X_harmonized[:, 0], X_harmonized[:, 1])[0, 1] > 0.99
    assert np.std(X_harmonized[:, 1]) < 0.1 * np.std(X_harmonized[:, 0])


@pytest.mark.parametrize("posterior_summary", ["mean", "median"])
def test_transform_matches_fit_transform_with_all_draws(iqm_data, posterior_summary: str) -> None:
    """With all kept draws stored, transform reproduces fit_transform on the training data."""
    d = iqm_data
    params = {**FAST, "posterior_summary": posterior_summary, "n_stored_draws": N_KEPT, "random_state": 0}
    harmonizer = BARTharm(**params)
    X_fit_transform = harmonizer.fit_transform(d.X, d.iqms, d.biological_covariates)
    np.testing.assert_allclose(harmonizer.transform(d.X, d.iqms), X_fit_transform, rtol=1e-10, atol=1e-8)


def test_transform_var_scaling_matches_fit_transform(iqm_data) -> None:
    """With variance scaling, transform also needs sites and biological covariates."""
    d = iqm_data
    harmonizer = BARTharm(**FAST, var_scaling=True, n_stored_draws=N_KEPT, random_state=0)
    X_fit_transform = harmonizer.fit_transform(d.X, d.iqms, d.biological_covariates, sites=d.sites)
    X_transform = harmonizer.transform(d.X, d.iqms, d.biological_covariates, sites=d.sites)
    np.testing.assert_allclose(X_transform, X_fit_transform, rtol=1e-10, atol=1e-8)

    with pytest.raises(ValueError, match="sites are required"):
        harmonizer.transform(d.X, d.iqms, d.biological_covariates)
    with pytest.raises(ValueError, match="biological_covariates were used in fit"):
        harmonizer.transform(d.X, d.iqms, sites=d.sites)
    with pytest.raises(ValueError, match="not seen in fit"):
        harmonizer.transform(d.X, d.iqms, d.biological_covariates, sites=np.full(d.X.shape[0], 99))


def test_transform_new_data(iqm_data) -> None:
    """New samples, including IQMs outside the training range, are harmonized."""
    d = iqm_data
    train, test = slice(0, 200), slice(200, None)
    harmonizer = BARTharm(**FAST, n_stored_draws=10, random_state=0)
    harmonizer.fit(d.X[train], d.iqms[train], d.biological_covariates[train])
    assert len(harmonizer._mu_snapshots[0]) == 10

    X_test = harmonizer.transform(d.X[test], d.iqms[test])
    assert X_test.shape == d.X[test].shape
    assert np.all(np.isfinite(X_test))
    X_out_of_range = harmonizer.transform(d.X[test], d.iqms[test] * 10.0)
    assert np.all(np.isfinite(X_out_of_range))


def test_site_scales(iqm_data) -> None:
    """Site scales have one value per fitted site and geometric mean 1."""
    d = iqm_data
    harmonizer = BARTharm(**FAST, var_scaling=True, random_state=0)
    harmonizer.fit(d.X, d.iqms, d.biological_covariates, sites=d.sites)
    np.testing.assert_array_equal(harmonizer.sites_, np.unique(d.sites))
    assert harmonizer.site_scales_.shape == (1, len(np.unique(d.sites)))
    assert np.all(harmonizer.site_scales_ > 0)
    # Every draw has geometric mean 1; the posterior mean is close to it
    assert abs(np.mean(np.log(harmonizer.site_scales_))) < 0.05
    assert harmonizer.sigma_draws_.shape == (1, N_KEPT)


def test_without_biological_covariates(iqm_data) -> None:
    """The model runs with the scanner forest only."""
    d = iqm_data
    harmonizer = BARTharm(**FAST, random_state=0)
    X_harmonized = harmonizer.fit_transform(d.X, d.iqms)
    assert harmonizer.n_biological_covariates_ == 0
    assert np.all(np.isfinite(X_harmonized))
    np.testing.assert_allclose(harmonizer.transform(d.X, d.iqms).shape, d.X.shape)


def test_one_dimensional_covariates(iqm_data) -> None:
    """A single IQM and a single biological covariate can be passed as 1D arrays."""
    d = iqm_data
    harmonizer = BARTharm(**FAST, random_state=0)
    X_harmonized = harmonizer.fit_transform(d.X, d.iqms[:, 2], d.biological_covariates[:, 0])
    assert harmonizer.n_iqm_covariates_ == 1
    assert harmonizer.n_biological_covariates_ == 1
    assert X_harmonized.shape == d.X.shape


def test_constant_feature_unchanged(iqm_data) -> None:
    """A constant feature cannot be z-scored and is returned unchanged."""
    d = iqm_data
    X = np.column_stack([d.X[:, 0], np.full(d.X.shape[0], 5.0)])
    harmonizer = BARTharm(**FAST, random_state=0)
    X_harmonized = harmonizer.fit_transform(X, d.iqms, d.biological_covariates)
    np.testing.assert_array_equal(X_harmonized[:, 1], 5.0)
    np.testing.assert_array_equal(harmonizer.transform(X, d.iqms)[:, 1], 5.0)
    assert not np.allclose(X_harmonized[:, 0], X[:, 0])


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_random_state_reproducible(iqm_data) -> None:
    """The same random_state gives the same result, independently of n_jobs."""
    d = iqm_data
    X = np.column_stack([d.X[:, 0], d.X_clean[:, 0]])
    first = BARTharm(**FAST, random_state=0, n_jobs=1).fit_transform(X, d.iqms, d.biological_covariates)
    second = BARTharm(**FAST, random_state=0, n_jobs=2).fit_transform(X, d.iqms, d.biological_covariates)
    third = BARTharm(**FAST, random_state=1).fit_transform(X, d.iqms, d.biological_covariates)
    np.testing.assert_array_equal(first, second)
    assert not np.allclose(first, third)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"n_iter": 1, "thinning_interval": 2}, "must be at least thinning_interval"),
        ({"n_iter": 100, "burn_in": 100}, "No posterior draws left after burn-in"),
        ({"posterior_summary": "mode"}, "posterior_summary"),
        ({"n_trees_mu": 0}, "n_trees_mu"),
    ],
)
def test_invalid_params(iqm_data, params: dict, match: str) -> None:
    """Invalid sampler settings raise errors."""
    d = iqm_data
    with pytest.raises(ValueError, match=match):
        BARTharm(**{**FAST, **params}).fit(d.X, d.iqms, d.biological_covariates)


def test_invalid_inputs(iqm_data) -> None:
    """Invalid data raise errors."""
    d = iqm_data
    with pytest.raises(ValueError, match="sites are required"):
        BARTharm(**FAST, var_scaling=True).fit(d.X, d.iqms, d.biological_covariates)
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        BARTharm(**FAST).fit(d.X, d.iqms[:-1])
    X_nan = d.X.copy()
    X_nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        BARTharm(**FAST).fit(X_nan, d.iqms)

    harmonizer = BARTharm(**FAST, random_state=0).fit(d.X, d.iqms, d.biological_covariates)
    with pytest.raises(ValueError, match="iqm_covariates has 3 columns"):
        harmonizer.transform(d.X, d.iqms[:, :3])
    with pytest.raises(ValueError, match="features"):
        harmonizer.transform(np.hstack([d.X, d.X]), d.iqms)


# ---------------------------------------------------------------------------
# Covariate normalization
# ---------------------------------------------------------------------------


def test_quantile_normalizer_matches_trank_and_extrapolates() -> None:
    """The normalizer reproduces quantile_normalize_bart on training data and clips new values."""
    X = np.array([[3.0, 1.0], [1.0, 1.0], [2.0, 1.0], [2.0, 1.0], [5.0, 1.0]])
    normalizer = _QuantileNormalizer().fit(X)
    np.testing.assert_allclose(normalizer.transform(X)[:, 0], [2 / 3, 0, 1 / 3, 1 / 3, 1])
    np.testing.assert_array_equal(normalizer.transform(X)[:, 1], 0.0)  # constant column
    np.testing.assert_allclose(normalizer.transform(np.array([[0.0, 7.0], [4.0, 1.0], [9.0, 1.0]]))[:, 0], [0, 5 / 6, 1])
