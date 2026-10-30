"""CovBat-specific tests.

Behaviour shared by all ComBat variants (API, site validation, covariates,
harmonization effect, sklearn compatibility) is tested in ``test_combat_common.py``.
Data comes from the shared fixtures in ``tests/conftest.py`` (``three_site_data``
has strong mean, variance and covariance site effects).
"""

import numpy as np
import pytest

from uniharmony.combat import CovBat


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
# Hyper-parameters
# ---------------------------------------------------------------------------


def test_n_pc_override(three_site_data):
    """n_pc must override pct_var."""
    X, sites = three_site_data.X, three_site_data.sites
    covbat = CovBat(n_pc=5, pct_var=0.10)
    covbat.fit(X, sites)
    assert covbat.n_pc_ == 5


def test_n_pc_capped_to_n_components(three_site_data):
    """If n_pc > available components, cap it."""
    X, sites = three_site_data.X, three_site_data.sites
    covbat = CovBat(n_pc=9999)
    covbat.fit(X, sites)
    n_samples, n_features = X.shape
    assert covbat.n_pc_ <= min(n_samples, n_features)


def test_pct_var_none_uses_all_components(three_site_data):
    """When both pct_var and n_pc are None, use all PCs."""
    X, sites = three_site_data.X, three_site_data.sites
    covbat = CovBat(pct_var=None, n_pc=None)
    covbat.fit(X, sites)
    n_samples, n_features = X.shape
    assert covbat.n_pc_ == min(n_samples, n_features)


def test_std_var_false_no_scaler(three_site_data):
    """With std_var=False the internal scaler must be None."""
    X, sites = three_site_data.X, three_site_data.sites
    covbat = CovBat(std_var=False)
    covbat.fit(X, sites)
    assert covbat._scaler is None


def test_residualize_true_no_mean_restoration(three_site_data):
    """With residualize=True the global mean must not be restored."""
    X, sites = three_site_data.X, three_site_data.sites
    original_mean = X.mean(axis=0)
    covbat = CovBat(residualize=True)
    X_harm = covbat.fit_transform(X, sites)
    assert np.not_equal(original_mean, X_harm.mean(axis=0)).all()


# ---------------------------------------------------------------------------
# Covariance harmonization effect
# ---------------------------------------------------------------------------


def test_covariance_site_effects_reduced_no_covariates(three_site_data):
    """CovBat must reduce covariance divergence when no covariates are used."""
    X, sites = three_site_data.X, three_site_data.sites

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
    X, sites, age = three_site_data.X, three_site_data.sites, three_site_data.covariates["age"]

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


# ---------------------------------------------------------------------------
# Edge cases
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
