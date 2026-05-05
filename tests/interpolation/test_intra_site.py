"""Test IntraSiteInterpolation transformer."""

import numbers

import numpy as np
import pytest

from uniharmony.datasets import make_multisite_classification
from uniharmony.interpolation import IntraSiteInterpolation


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def binary_data():
    """Generate binary classification dataset with site imbalance."""
    return make_multisite_classification(
        n_samples=2000,
        n_features=4,
        n_sites=2,
        n_classes=2,
        random_state=42,
        balance_per_site=[0.1, 0.9],
    )


@pytest.fixture
def regression_data():
    """Generate regression dataset with continuous targets."""
    rng = np.random.default_rng(53)
    X = rng.standard_normal((200, 4))
    sites = np.array([0] * 100 + [1] * 100)
    y = rng.standard_normal(200) * 10 + 50
    return X, y, sites


@pytest.fixture
def covariate_data():
    """Generate dataset with categorical and continuous covariates."""
    rng = np.random.default_rng(54)
    n_samples = 2000

    X, y, sites = make_multisite_classification(
        n_samples=n_samples,
        n_features=4,
        n_sites=2,
        n_classes=2,
        random_state=42,
        balance_per_site=[0.1, 0.9],
    )

    sex = rng.integers(0, 2, (n_samples, 1))
    age = rng.standard_normal((n_samples, 1)) * 10 + 50

    return X, y, sites, sex, age


# ==============================================================================
# Basic functionality
# ==============================================================================


def test_basic_run(binary_data):
    """Model should run and return valid shapes."""
    X, y, sites = binary_data
    isi = IntraSiteInterpolation("random")

    Xr, yr = isi.fit_resample(X, y, sites=sites)

    assert len(Xr) == len(yr)
    assert Xr.ndim == 2
    assert yr.ndim == 1


# ==============================================================================
# Balance correctness
# ==============================================================================


@pytest.mark.parametrize("strategy", ["per_site", "global_max"])
def test_balance(strategy, binary_data):
    """Each site must be class-balanced after resampling."""
    X, y, sites = binary_data
    isi = IntraSiteInterpolation("random", balance_strategy=strategy)

    _, yr = isi.fit_resample(X, y, sites=sites)
    sr = isi.sites_resampled_

    for site in np.unique(sr):
        counts = np.unique(yr[sr == site], return_counts=True)[1]
        assert len(set(counts)) == 1


# ==============================================================================
# samples_created_
# ==============================================================================


def test_samples_created(binary_data):
    """samples_created_ should be a dict with non-negative integers."""
    X, y, sites = binary_data
    isi = IntraSiteInterpolation("random")

    isi.fit_resample(X, y, sites=sites)

    assert isinstance(isi.samples_created_, dict)

    for d in isi.samples_created_.values():
        for v in d.values():
            assert isinstance(v, numbers.Integral)
            assert v >= 0


# ==============================================================================
# Validation
# ==============================================================================


def test_invalid_balance_strategy(binary_data):
    """Invalid balance_strategy should raise ValueError."""
    X, y, sites = binary_data
    isi = IntraSiteInterpolation(balance_strategy="invalid")

    with pytest.raises(ValueError):
        isi.fit_resample(X, y, sites=sites)


def test_invalid_interpolator(binary_data):
    """Invalid interpolator name should raise ValueError."""
    X, y, sites = binary_data
    isi = IntraSiteInterpolation("invalid")

    with pytest.raises(ValueError):
        isi.fit_resample(X, y, sites=sites)


# ==============================================================================
# Covariates
# ==============================================================================


@pytest.mark.parametrize("strategy", ["per_site", "global_max"])
def test_covariates(strategy, covariate_data):
    """Covariate stratification should work with binning-based grouping."""
    X, y, sites, sex, age = covariate_data

    isi = IntraSiteInterpolation(
        interpolator="random",
        balance_strategy=strategy,
    )

    Xr, yr = isi.fit_resample(
        X,
        y,
        sites=sites,
        categorical_covariate=sex,
        continuous_covariate=age,
        n_bins_cont_cov=5,
        binning_strategy_cont_cov="quantile",
    )

    assert len(Xr) == len(yr)


def test_covariates_requires_bins(covariate_data):
    """Continuous covariates without n_bins_cont_cov must raise ValueError."""
    X, y, sites, _, age = covariate_data

    isi = IntraSiteInterpolation(
        interpolator="random",
    )

    with pytest.raises(ValueError):
        isi.fit_resample(
            X,
            y,
            sites=sites,
            continuous_covariate=age,
        )


# ==============================================================================
# Regression
# ==============================================================================


def test_regression_requires_bins(regression_data):
    """Regression without n_bins must raise ValueError."""
    X, y, sites = regression_data

    isi = IntraSiteInterpolation(task="regression")

    with pytest.raises(ValueError):
        isi.fit_resample(X, y, sites=sites)


@pytest.mark.parametrize("strategy", ["per_site", "global_max"])
def test_regression_runs(strategy, regression_data):
    """Regression should work when n_bins is provided."""
    X, y, sites = regression_data

    isi = IntraSiteInterpolation(
        interpolator="random",
        task="regression",
        n_bins=5,
        balance_strategy=strategy,
    )

    Xr, yr = isi.fit_resample(X, y, sites=sites)

    assert len(Xr) == len(yr)
    assert yr.dtype.kind == "f"


# ==============================================================================
# Reproducibility
# ==============================================================================


def test_reproducibility(binary_data):
    """Same random_state must produce identical results."""
    X, y, sites = binary_data

    isi1 = IntraSiteInterpolation("random", random_state=42)
    isi2 = IntraSiteInterpolation("random", random_state=42)

    X1, y1 = isi1.fit_resample(X, y, sites=sites)
    X2, y2 = isi2.fit_resample(X, y, sites=sites)

    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


# ==============================================================================
# Monkeypatch robustness
# ==============================================================================


def test_assertion_failure(monkeypatch):
    """If resampling fails to balance, an assertion should be triggered."""
    rng = np.random.default_rng(99)
    X = rng.standard_normal((200, 4))
    sites = np.array([0] * 100 + [1] * 100)
    y = np.array([0] * 80 + [1] * 20 + [0] * 30 + [1] * 70)

    isi = IntraSiteInterpolation("random")

    def bad_resample(*args, **kwargs):
        return args[1], args[2]

    monkeypatch.setattr(IntraSiteInterpolation, "_resample_site", bad_resample)

    with pytest.raises(AssertionError):
        isi.fit_resample(X, y, sites=sites)
