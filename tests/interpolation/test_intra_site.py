"""Test IntraSiteInterpolation transformer."""

import numpy as np
import numpy.typing as npt
import pytest
from imblearn.over_sampling import SMOTE

from uniharmony import make_multisite_classification
from uniharmony.interpolation import IntraSiteInterpolation


def generate_data(
    multiclass: bool = False,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Generate synthetic data for testing.

    Parameters
    ----------
    multiclass : bool, optional (default False)
        Whether to generate multi-class data or not.

    Returns
    -------
    array
        Features.
    array
        Targets.
    array
        Sites.

    """
    x = np.random.randn(200, 4)
    sites = np.array([0] * 100 + [1] * 100)

    if multiclass:
        y = np.array([0] * 120 + [1] * 50 + [2] * 30)
    else:
        y = np.array([0] * 150 + [1] * 50)

    return x, y, sites


def test_binary_runs() -> None:
    """Test matching sample length."""
    x, y, sites = generate_data()
    y = np.random.permutation(y)
    isi = IntraSiteInterpolation("smote")
    xr, yr = isi.fit_resample(x, y, sites=sites)
    sr = isi.sites_resampled_
    assert len(xr) == len(yr) == len(sr)


def test_multiclass_balance() -> None:
    """Test multiclass balance."""
    x, y, sites = generate_data(multiclass=True)
    y = np.random.permutation(y)
    isi = IntraSiteInterpolation("random")
    _, yr = isi.fit_resample(x, y, sites=sites)
    sr = isi.sites_resampled_

    for site in np.unique(sr):
        counts = np.unique(yr[sr == site], return_counts=True)[1]
        assert len(set(counts)) == 1


def test_invalid_site() -> None:
    """Test site generation."""
    x = np.random.randn(10, 2)
    y = np.zeros(10)
    y = np.random.permutation(y)
    sites = np.zeros(10)

    isi = IntraSiteInterpolation()
    with pytest.raises(ValueError):
        isi.fit_resample(x, y, sites=sites)


def test_shape_missmatch() -> None:
    """Test data missmatch."""
    _, y, sites = make_multisite_classification(2, 100)
    X, y, _ = make_multisite_classification(2, 400)

    isi = IntraSiteInterpolation("smote")
    with pytest.raises(ValueError):
        _, _ = isi.fit_resample(X, y, sites=sites)


def test_verbosity() -> None:
    """Test verbosity."""
    x, y, sites = generate_data()
    y = np.random.permutation(y)
    isi = IntraSiteInterpolation("smote", verbose=True)
    _, _ = isi.fit_resample(x, y, sites=sites)


def test_single_class_in_a_site() -> None:
    """Test site generation."""
    x = np.random.randn(300, 10)
    y = np.array([0] * 180 + [1] * 80 + [2] * 40)
    sites = np.array([0] * 150 + [1] * 150)
    isi = IntraSiteInterpolation()
    with pytest.raises(ValueError):
        isi.fit_resample(x, y, sites=sites)
