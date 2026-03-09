"""Test ISI harmonization resampling."""

import numpy as np
import pytest
from imblearn.over_sampling import SMOTE

from uniharmony import make_multisite_classification
from uniharmony.interpolation import IntraSiteInterpolation


def generate_data(multiclass=False):
    """Generate synthetic data for testing.

    :param multiclass: Description
    """
    x = np.random.randn(200, 4)
    sites = np.array([0] * 100 + [1] * 100)

    if multiclass:
        y = np.array([0] * 120 + [1] * 50 + [2] * 30)
    else:
        y = np.array([0] * 150 + [1] * 50)

    return x, y, sites


def test_binary_runs():
    """Test matching sample length."""
    x, y, sites = generate_data()
    y = np.random.permutation(y)
    ici = IntraSiteInterpolation("smote")
    xr, yr = ici.fit_resample(x, y, sites=sites)
    sr = ici.sites_resampled_
    assert len(xr) == len(yr) == len(sr)


def test_multiclass_balance():
    """Test multiclass balance."""
    x, y, sites = generate_data(multiclass=True)
    y = np.random.permutation(y)
    ici = IntraSiteInterpolation("random")
    _, yr = ici.fit_resample(x, y, sites=sites)
    sr = ici.sites_resampled_

    for site in np.unique(sr):
        counts = np.unique(yr[sr == site], return_counts=True)[1]
        assert len(set(counts)) == 1


def test_invalid_site():
    """Test site generation."""
    x = np.random.randn(10, 2)
    y = np.zeros(10)
    y = np.random.permutation(y)
    sites = np.zeros(10)

    ici = IntraSiteInterpolation()
    with pytest.raises(ValueError):
        ici.fit_resample(x, y, sites=sites)


def test_invalid_model_name():
    """Test wrong model warning."""
    x = np.random.randn(10, 2)
    y = np.zeros(10)
    y = np.random.permutation(y)
    sites = np.zeros(10)
    with pytest.raises(ValueError):
        ici = IntraSiteInterpolation(interpolator="wrong_name")
        ici.fit_resample(x, y, sites=sites)


def test_shape_missmatch():
    """Test data missmatch."""
    _, y, sites = make_multisite_classification(2, 100)
    X, y, _ = make_multisite_classification(2, 400)

    ici = IntraSiteInterpolation("smote")
    with pytest.raises(ValueError):
        _, _ = ici.fit_resample(X, y, sites=sites)


def test_interpolator_as_instance():
    """Test passing intepolator instance."""
    interpolator = SMOTE()
    _ = IntraSiteInterpolation(interpolator=interpolator)


def test_verbosity():
    """Test verbosity."""
    x, y, sites = generate_data()
    y = np.random.permutation(y)
    ici = IntraSiteInterpolation("smote", verbose=True)
    _, _ = ici.fit_resample(x, y, sites=sites)


def test_single_class_in_a_site():
    """Test site generation."""
    x = np.random.randn(300, 10)
    y = np.array([0] * 180 + [1] * 80 + [2] * 40)
    sites = np.array([0] * 150 + [1] * 150)
    ici = IntraSiteInterpolation()
    with pytest.raises(ValueError):
        ici.fit_resample(x, y, sites=sites)
