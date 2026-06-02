"""Battery of tests for the CovBat harmonizer.

Run with::

    pytest test_covbat.py -v

These tests assume the UniHarmony package is importable (i.e. the
``_covbat.py`` file lives inside ``uniharmony/combat/``).
"""

import numpy as np
import pytest

# Adjust the import path if your package layout differs
from uniharmony.combat._covbat import CovBat
from uniharmony.datasets import make_multisite_classification


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rng():
    """Rng."""
    return np.random.default_rng(42)


@pytest.fixture
def three_site_data(rng):
    """Synthetic data with strong mean, variance and covariance site effects."""
    X, _, sites, covars = make_multisite_classification(
        n_samples=300, n_sites=3, n_features=20, covariates=["age"], site_effect_type="location+scale", site_effect_strength=20
    )

    return X, sites, covars["age"]


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
    # Split 80 / 20
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    s_train, s_test = sites[:split], sites[split:]
    a_train, a_test = age[:split], age[split:]

    covbat = CovBat()
    covbat.fit(X_train, s_train, continuous_covariates=a_train[:, None])
    X_test_harm = covbat.transform(X_test, s_test, continuous_covariates=a_test[:, None])
    assert X_test_harm.shape == X_test.shape


def test_fit_then_transform_same_as_fit_transform(three_site_data):
    """``fit_transform`` must be equivalent to ``fit`` + ``transform``."""
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
    bad_sites[0] = 10
    with pytest.raises(ValueError, match="One or more sites were not seen during fit"):
        covbat.transform(X, bad_sites)


# ---------------------------------------------------------------------------
# 2. Hyper-parameter tests
# ---------------------------------------------------------------------------


def test_n_pc_override(three_site_data):
    """``n_pc`` must override ``pct_var``."""
    X, sites, _ = three_site_data
    covbat = CovBat(n_pc=5, pct_var=0.10)
    covbat.fit(X, sites)
    assert covbat.n_pc_ == 5


def test_n_pc_capped_to_n_components(three_site_data):
    """If ``n_pc`` > available components, cap it."""
    X, sites, _ = three_site_data
    covbat = CovBat(n_pc=9999)
    covbat.fit(X, sites)
    n_samples, n_features = X.shape
    assert covbat.n_pc_ <= min(n_samples, n_features)


def test_pct_var_none_uses_all_components(three_site_data):
    """When both ``pct_var`` and ``n_pc`` are None, use all PCs."""
    X, sites, _ = three_site_data
    covbat = CovBat(pct_var=None, n_pc=None)
    covbat.fit(X, sites)
    n_samples, n_features = X.shape
    assert covbat.n_pc_ == min(n_samples, n_features)


def test_std_var_false_no_scaler(three_site_data):
    """With ``std_var=False`` the internal scaler must be None."""
    X, sites, _ = three_site_data
    covbat = CovBat(std_var=False)
    covbat.fit(X, sites)
    assert covbat._scaler is None


def test_residualize_true_no_mean_restoration(three_site_data):
    """With ``residualize=True`` the global mean must not be restored."""
    X, sites, _ = three_site_data
    covbat = CovBat(residualize=True)
    X_harm = covbat.fit_transform(X, sites)
    # Mean should be close to zero because the first ComBat step
    # intentionally leaves it at zero.
    np.testing.assert_allclose(X_harm.mean(axis=0), 0, atol=1e-3)


# ---------------------------------------------------------------------------
# 3. Covariance-harmonization effect tests
# ---------------------------------------------------------------------------


def _site_covariances(X, sites):
    """Return a list of per-site covariance matrices."""
    return [np.cov(X[sites == s].T) for s in np.unique(sites)]


def test_covariance_site_effects_reduced(three_site_data):
    """CovBat must reduce differences in covariance structure across sites."""
    X, sites, age = three_site_data

    # Raw covariance divergence
    raw_covs = _site_covariances(X, sites)
    raw_div = np.mean(
        [np.linalg.norm(raw_covs[i] - raw_covs[j], "fro") for i in range(len(raw_covs)) for j in range(i + 1, len(raw_covs))]
    )

    # Harmonized divergence
    covbat = CovBat(std_var=True, pct_var=0.95)
    X_harm = covbat.fit_transform(X, sites, continuous_covariates=age[:, None])
    harm_covs = _site_covariances(X_harm, sites)
    harm_div = np.mean(
        [np.linalg.norm(harm_covs[i] - harm_covs[j], "fro") for i in range(len(harm_covs)) for j in range(i + 1, len(harm_covs))]
    )

    assert harm_div < raw_div, f"CovBat did not reduce covariance divergence: raw={raw_div:.3f}, harm={harm_div:.3f}"


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
    X_harm = covbat.fit_transform(X, sites, continuous_covariates=age.reshape(-1, 1))

    # Simple correlation between each feature and age should remain similar
    raw_corr = np.array([np.corrcoef(X[:, j], age)[0, 1] for j in range(X.shape[1])])
    harm_corr = np.array([np.corrcoef(X_harm[:, j], age)[0, 1] for j in range(X.shape[1])])

    np.testing.assert_array_equal(raw_corr, harm_corr)


# ---------------------------------------------------------------------------
# 4. Edge-case / robustness tests
# ---------------------------------------------------------------------------


def test_many_sites_few_samples(rng):
    """CovBat should run (possibly with warnings) when sites are very small."""
    n = 20
    X = rng.normal(size=(n, 10))
    sites = np.array([f"S{i % 5}" for i in range(n)])
    covbat = CovBat()
    X_harm = covbat.fit_transform(X, sites)
    assert X_harm.shape == X.shape


def test_repeated_transform_calls_consistent(three_site_data):
    """Calling ``transform`` twice with the same data must give the same result."""
    X, sites, age = three_site_data
    covbat = CovBat()
    covbat.fit(X, sites, continuous_covariates=age[:, None])
    out1 = covbat.transform(X, sites, continuous_covariates=age[:, None])
    out2 = covbat.transform(X, sites, continuous_covariates=age[:, None])
    np.testing.assert_array_equal(out1, out2)
