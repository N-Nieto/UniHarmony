"""Fixtures for the PrettYharmonize tests."""

import numpy as np
import numpy.typing as npt
import pytest

from uniharmony.datasets import make_multisite_classification


@pytest.fixture
def clf_data() -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Binary data, 3 sites with different class balance and location+scale effects."""
    return make_multisite_classification(
        n_sites=3,
        n_samples=300,
        n_features=6,
        balance_per_site=[[0.7, 0.3], [0.3, 0.7], [0.5, 0.5]],
        site_effect_type="location+scale",
        random_state=0,
    )


@pytest.fixture
def reg_data() -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Regression data following the ComBat model, 3 sites with different target ranges."""
    rng = np.random.default_rng(1)
    n_samples, n_features = 240, 6
    sites = rng.permutation(np.arange(n_samples) % 3)
    y = 50 + 10 * rng.normal(size=n_samples) + 5 * sites
    weights = rng.choice([-1.0, 1.0], n_features) * 0.1
    noise = rng.normal(size=(n_samples, n_features)) * np.exp(rng.normal(scale=0.4, size=(3, n_features)))[sites]
    X = np.outer(y, weights) + rng.normal(scale=2, size=(3, n_features))[sites] + noise
    return X, y, sites


@pytest.fixture
def rng() -> np.random.Generator:
    """Random generator."""
    return np.random.default_rng(42)
