"""Test IntraSiteInterpolation transformer."""

import numbers

import numpy as np
import pytest
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

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


def test_basic_run_invalid_instance(binary_data):
    """Model should run and return valid shapes."""
    X, y, sites = binary_data
    interpolator = LogisticRegression()
    isi = IntraSiteInterpolation(interpolator=interpolator)
    with pytest.raises(ValueError):
        _, _ = isi.fit_resample(X, y, sites=sites)


def test_basic_run_no_balance():
    """Model should run but no resampling."""
    X, y, sites = make_multisite_classification()
    interpolator = SMOTE()

    isi = IntraSiteInterpolation(interpolator=interpolator)
    _, _ = isi.fit_resample(X, y, sites=sites)


def test_basic_run_no_balance_small():
    """Model should run but no resampling with small data."""
    X, y, sites = make_multisite_classification(n_samples=[2, 19])
    interpolator = SMOTE()

    isi = IntraSiteInterpolation(interpolator=interpolator)
    _, _ = isi.fit_resample(X, y, sites=sites)


# ==============================================================================
# Balance correctness
# ==============================================================================


@pytest.mark.parametrize("strategy", ["per_site", "global_max"])
def test_balance_strategy(strategy, binary_data):
    """Each site must be class-balanced after resampling."""
    X, y, sites = binary_data
    isi = IntraSiteInterpolation("random", balance_strategy=strategy)

    _, yr = isi.fit_resample(X, y, sites=sites)
    sr = isi.sites_resampled_

    for site in np.unique(sr):
        counts = np.unique(yr[sr == site], return_counts=True)[1]
        assert len(set(counts)) == 1


## Binning strategies
@pytest.mark.parametrize("strategy", ["uniform", "quantile"])
def test_binning_strategy(strategy, regression_data):
    """Each strategies for binning_strategy."""
    X, y, sites = regression_data
    isi = IntraSiteInterpolation("random", binning_strategy=strategy)

    _, _ = isi.fit_resample(X, y, sites=sites)


def test_binning_strategy_invalid(regression_data):
    """Invalid strategies for binning."""
    X, y, sites = regression_data
    isi = IntraSiteInterpolation("random", binning_strategy="invalid")
    with pytest.raises(ValueError):
        _, _ = isi.fit_resample(X, y, sites=sites)


## continuos Binning strategies
@pytest.mark.parametrize("strategy", ["uniform", "quantile"])
def test_binning_strategy_cont_cov(strategy, covariate_data):
    """Each strategies for binning_strategy_cont_cov(."""
    X, y, sites, sex, age = covariate_data
    isi = IntraSiteInterpolation("random")

    _, _ = isi.fit_resample(
        X,
        y,
        sites=sites,
        categorical_covariate=sex,
        continuous_covariate=age,
        n_bins_cont_cov=2,
        binning_strategy_cont_cov=strategy,
    )


def test_binning_strategy_invalid_cont_cov(covariate_data):
    """Invalid strategies for binning_strategy_cont_cov."""
    X, y, sites, sex, age = covariate_data
    isi = IntraSiteInterpolation("random")
    with pytest.raises(ValueError):
        _, _ = isi.fit_resample(
            X,
            y,
            sites=sites,
            categorical_covariate=sex,
            continuous_covariate=age,
            n_bins_cont_cov=2,
            binning_strategy_cont_cov="invalid",
        )


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


def test___sklearn_tags__():
    """Test __sklearn_tags__."""
    isi = IntraSiteInterpolation(
        interpolator="random",
    )
    isi.__sklearn_tags__()


def test_compatibility(binary_data):
    """Test compatibility."""
    X, y, _ = binary_data
    isi = IntraSiteInterpolation()
    isi._fit_resample(X, y)


def test_covariates_categorical(covariate_data):
    """Covariate stratification should work with binning-based grouping."""
    X, y, sites, sex, _ = covariate_data

    isi = IntraSiteInterpolation(
        interpolator="random",
    )

    Xr, yr = isi.fit_resample(
        X,
        y,
        sites=sites,
        categorical_covariate=sex,
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
