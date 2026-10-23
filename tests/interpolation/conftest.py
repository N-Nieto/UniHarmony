"""Shared fixtures for the interpolation tests."""

import numpy as np
import pytest

from uniharmony.datasets import make_multisite_classification


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
def opposite_sites_data():
    """Two sites with opposite class imbalance (site predicts the target)."""
    rng = np.random.default_rng(3)
    n_features = 5
    X0 = rng.standard_normal((400, n_features))
    X1 = rng.standard_normal((400, n_features)) + 2.0
    y0 = np.r_[np.zeros(320), np.ones(80)].astype(int)
    y1 = np.r_[np.zeros(80), np.ones(320)].astype(int)
    X = np.vstack([X0, X1]) + np.outer(np.r_[y0, y1], np.ones(n_features))
    return X, np.r_[y0, y1], np.repeat([0, 1], 400)


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


@pytest.fixture
def linear_regression_data():
    """Regression data whose target is an exact linear function of X, with imbalanced sites."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((200, 3))
    weights = np.array([3.0, -2.0, 0.5])
    y = X @ weights + 50.0
    # Site 0 over-represents high targets, so its low bins are oversampled
    sites = np.where(rng.random(200) < np.where(y > 50.0, 0.8, 0.3), 0, 1)
    return X, y, sites, weights
