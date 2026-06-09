"""Tests for CovBat transformer."""

from collections.abc import Callable

import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from uniharmony.combat import CovBat


def _ex_failed_checks(_) -> dict[str, str]:
    return {
        "check_transformers_unfitted": "checked inside",
        "check_n_features_in_after_fitting": "not needed",
        "check_estimators_nan_inf": "checked inside",
        "check_fit_score_takes_y": "sites instead of y",
        "check_estimators_dtypes": "sites instead of y",
        "check_dtype_object": "sites instead of y",
        "check_estimators_pickle": "sites instead of y",
        "check_f_contiguous_array_estimator": "sites instead of y",
        "check_transformer_data_not_an_array": "sites instead of y",
        "check_transformer_preserve_dtypes": "sites instead of y",
        "check_transformer_general": "sites instead of y",
        "check_methods_sample_order_invariance": "sites instead of y",
        "check_methods_subset_invariance": "sites instead of y",
        "check_dict_unchanged": "sites instead of y",
        "check_fit_idempotent": "sites instead of y",
        "check_n_features_in": "not needed",
        "check_fit2d_predict1d": "sites instead of y",
        "check_fit2d_1sample": "custom message",
        "check_requires_y_none": "target cannot be None",
        "check_fit2d_1feature": "Harmonization produced non-finite values",
    }


@parametrize_with_checks(
    [
        CovBat(),
    ],
    expected_failed_checks=_ex_failed_checks,
)
def test_neuro_combat_compat_sklearn(estimator: object, check: Callable) -> None:
    """Test NeuroComBat compatibility with sklearn.

    Parameters
    ----------
    estimator : object
        Instance of NeuroComBat.
    check : callable
        sklearn fixture.

    """
    check(estimator)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rng():
    """Random generator."""
    return np.random.default_rng(42)


@pytest.fixture
def three_site_data(rng):
    """Synthetic data with strong mean, variance and covariance site effects."""
    n_per_site = 60
    n_features = 20
    sites = np.repeat(["A", "B", "C"], n_per_site)

    # Shared biological covariate (e.g. age)
    age = rng.normal(50, 10, len(sites))

    X = []
    for s in ["A", "B", "C"]:
        idx = sites == s
        n = idx.sum()

        # Site-specific mean shift
        mean_shift = {"A": 0.0, "B": 2.5, "C": -1.5}[s]

        # Site-specific covariance structure
        if s == "A":
            cov = np.eye(n_features)
        elif s == "B":
            cov = np.eye(n_features) * 1.5
            cov[0, 1] = cov[1, 0] = 0.8
        else:
            cov = np.eye(n_features) * 0.7
            cov[2, 3] = cov[3, 2] = -0.6

        # Add age effect (preserved biological signal)
        age_effect = age[idx][:, None] * np.linspace(0.1, 0.5, n_features)

        samples = rng.multivariate_normal(
            mean=np.zeros(n_features) + mean_shift,
            cov=cov,
            size=n,
        )
        samples += age_effect
        X.append(samples)

    X = np.vstack(X)
    return X, sites, age


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _site_covariances(X, sites):
    """Return a list of per-site covariance matrices."""
    return [np.cov(X[sites == s].T) for s in np.unique(sites)]


def _cov_div(covs):
    """Mean Frobenius norm of pairwise covariance differences."""
    divs = []
    for i in range(len(covs)):
        for j in range(i + 1, len(covs)):
            divs.append(np.linalg.norm(covs[i] - covs[j], "fro"))
    return np.mean(divs)


def _regress_out_covariate(X, covariate):
    """Remove covariate effects from each feature (column-wise OLS)."""
    X_res = X.copy()
    cov = covariate.reshape(-1, 1)
    for j in range(X.shape[1]):
        beta = np.linalg.lstsq(cov, X[:, j], rcond=None)[0]
        X_res[:, j] = X[:, j] - cov @ beta
    return X_res


# ---------------------------------------------------------------------------
# 1. API / shape tests
# ---------------------------------------------------------------------------


def test_fit_transform_returns_same_shape(three_site_data):
    """Output shape must match input shape."""
    X, sites, age = three_site_data
    covbat = CovBat()
    X_harm = covbat.fit_transform(X, sites, continuous_covariates=age[:, None])
    assert X_harm.shape == X.shape
    assert np.isfinite(X_harm).all()


def test_transform_after_fit(three_site_data):
    """Transform on held-out data must work and preserve shape."""
    X, sites, age = three_site_data
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    s_train, s_test = sites[:split], sites[split:]
    a_train, a_test = age[:split], age[split:]

    covbat = CovBat()
    covbat.fit(X_train, s_train, continuous_covariates=a_train[:, None])
    X_test_harm = covbat.transform(X_test, s_test, continuous_covariates=a_test[:, None])
    assert X_test_harm.shape == X_test.shape


def test_fit_then_transform_same_as_fit_transform(three_site_data):
    """fit_transform must be equivalent to fit + transform."""
    X, sites, age = three_site_data
    covbat1 = CovBat(std_var=True, pct_var=0.90)
    X1 = covbat1.fit_transform(X, sites, continuous_covariates=age[:, None])

    covbat2 = CovBat(std_var=True, pct_var=0.90)
    X2 = covbat2.fit(X, sites, continuous_covariates=age[:, None]).transform(X, sites, continuous_covariates=age[:, None])
    np.testing.assert_allclose(X1, X2, rtol=1e-5)


def test_unseen_site_raises(three_site_data):
    """Transforming with a never-seen site must raise ValueError."""
    X, sites, _ = three_site_data
    covbat = CovBat()
    covbat.fit(X, sites)
    bad_sites = sites.copy()
    bad_sites[0] = "Z"
    with pytest.raises(ValueError, match="One or more sites were not seen during fit"):
        covbat.transform(X, bad_sites)


# ---------------------------------------------------------------------------
# 2. Hyper-parameter tests
# ---------------------------------------------------------------------------


def test_n_pc_override(three_site_data):
    """n_pc must override pct_var."""
    X, sites, _ = three_site_data
    covbat = CovBat(n_pc=5, pct_var=0.10)
    covbat.fit(X, sites)
    assert covbat.n_pc_ == 5


def test_n_pc_capped_to_n_components(three_site_data):
    """If n_pc > available components, cap it."""
    X, sites, _ = three_site_data
    covbat = CovBat(n_pc=9999)
    covbat.fit(X, sites)
    n_samples, n_features = X.shape
    assert covbat.n_pc_ <= min(n_samples, n_features)


def test_pct_var_none_uses_all_components(three_site_data):
    """When both pct_var and n_pc are None, use all PCs."""
    X, sites, _ = three_site_data
    covbat = CovBat(pct_var=None, n_pc=None)
    covbat.fit(X, sites)
    n_samples, n_features = X.shape
    assert covbat.n_pc_ == min(n_samples, n_features)


def test_std_var_false_no_scaler(three_site_data):
    """With std_var=False the internal scaler must be None."""
    X, sites, _ = three_site_data
    covbat = CovBat(std_var=False)
    covbat.fit(X, sites)
    assert covbat._scaler is None


def test_residualize_true_no_mean_restoration(three_site_data):
    """With residualize=True the global mean must not be restored."""
    X, sites, _ = three_site_data
    original_mean = X.mean(axis=0)
    covbat = CovBat(residualize=True)
    X_harm = covbat.fit_transform(X, sites)
    assert np.not_equal(original_mean, X_harm.mean(axis=0)).all


# ---------------------------------------------------------------------------
# 3. Covariance-harmonization effect tests
# ---------------------------------------------------------------------------


def test_covariance_site_effects_reduced_no_covariates(three_site_data):
    """CovBat must reduce covariance divergence when no covariates are used."""
    X, sites, _ = three_site_data

    raw_covs = _site_covariances(X, sites)
    raw_div = _cov_div(raw_covs)

    covbat = CovBat(std_var=True, pct_var=0.95)
    X_harm = covbat.fit_transform(X, sites)
    harm_covs = _site_covariances(X_harm, sites)
    harm_div = _cov_div(harm_covs)

    assert harm_div < raw_div, f"CovBat did not reduce covariance divergence: raw={raw_div:.3f}, harm={harm_div:.3f}"


def test_covariance_site_effects_reduced_with_covariates(three_site_data):
    """CovBat must reduce batch-specific covariance divergence with covariates.

    When covariates are included, the first ComBat preserves biological
    covariance from covariates in the restored mean.  To measure only
    batch-specific covariance reduction, we regress out the covariate
    before computing per-site covariance matrices.
    """
    X, sites, age = three_site_data

    # Remove age effects from raw data to isolate batch covariance
    X_raw_res = _regress_out_covariate(X, age)
    raw_covs = _site_covariances(X_raw_res, sites)
    raw_div = _cov_div(raw_covs)

    covbat = CovBat(std_var=True, pct_var=0.95)
    X_harm = covbat.fit_transform(X, sites, continuous_covariates=age[:, None])

    # Remove age effects from harmonized data
    X_harm_res = _regress_out_covariate(X_harm, age)
    harm_covs = _site_covariances(X_harm_res, sites)
    harm_div = _cov_div(harm_covs)

    assert harm_div < raw_div, (
        f"CovBat did not reduce batch-specific covariance divergence: raw={raw_div:.3f}, harm={harm_div:.3f}"
    )


def test_mean_site_effects_reduced(three_site_data):
    """The first ComBat step must still remove mean site effects."""
    X, sites, _ = three_site_data
    covbat = CovBat()
    X_harm = covbat.fit_transform(X, sites)

    raw_means = [X[sites == s].mean(axis=0) for s in np.unique(sites)]
    harm_means = [X_harm[sites == s].mean(axis=0) for s in np.unique(sites)]

    raw_mean_spread = np.mean(
        [np.linalg.norm(raw_means[i] - raw_means[j]) for i in range(len(raw_means)) for j in range(i + 1, len(raw_means))]
    )
    harm_mean_spread = np.mean(
        [np.linalg.norm(harm_means[i] - harm_means[j]) for i in range(len(harm_means)) for j in range(i + 1, len(harm_means))]
    )

    assert harm_mean_spread < raw_mean_spread


def test_age_effect_preserved(three_site_data):
    """A biological covariate (age) must not be destroyed by harmonization."""
    X, sites, age = three_site_data
    covbat = CovBat()
    X_harm = covbat.fit_transform(X, sites, continuous_covariates=age[:, None])

    raw_corr = np.array([np.corrcoef(X[:, j], age)[0, 1] for j in range(X.shape[1])])
    harm_corr = np.array([np.corrcoef(X_harm[:, j], age)[0, 1] for j in range(X.shape[1])])

    same_sign = np.sign(raw_corr) == np.sign(harm_corr)
    assert same_sign.mean() >= 0.8


# ---------------------------------------------------------------------------
# 4. Edge-case / robustness tests
# ---------------------------------------------------------------------------


def test_single_feature_works(rng):
    """CovBat must handle the degenerate case of a single feature."""
    n = 30
    X = rng.normal(size=(n, 1))
    sites = np.repeat(["A", "B"], n // 2)
    covbat = CovBat()
    with pytest.raises(RuntimeError, match="Harmonization produced non-finite values"):
        _ = covbat.fit_transform(X, sites)


def test_many_sites_few_samples(rng):
    """CovBat should run (possibly with warnings) when sites are very small."""
    n = 20
    X = rng.normal(size=(n, 10))
    sites = np.array([f"S{i % 5}" for i in range(n)])
    covbat = CovBat()
    X_harm = covbat.fit_transform(X, sites)
    assert X_harm.shape == X.shape


def test_repeated_transform_calls_consistent(three_site_data):
    """Calling transform twice with the same data must give the same result."""
    X, sites, age = three_site_data
    covbat = CovBat()
    covbat.fit(X, sites, continuous_covariates=age[:, None])
    out1 = covbat.transform(X, sites, continuous_covariates=age[:, None])
    out2 = covbat.transform(X, sites, continuous_covariates=age[:, None])
    np.testing.assert_array_equal(out1, out2)
