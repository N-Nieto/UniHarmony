"""Test suite for plotting functionalities."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from sklearn.decomposition import FastICA
from sklearn.linear_model import LogisticRegression

from uniharmony import make_multisite_classification
from uniharmony.plot import plot_2d_components_by_value, plot_2d_projection, plot_decision_boundary_2d, plot_features_by_site


# Fixtures
@pytest.fixture
def multisite_data() -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Provide simple 2-site binary classification data."""
    X, y, sites = make_multisite_classification(n_samples=10)

    return X, y, sites


@pytest.mark.docs
def test_plot_2d_projection_default(multisite_data) -> None:
    """Plot 2D projection with default parameters."""
    X, y, sites = multisite_data
    # Using the function as default. The function will use t-SNE as default dimensionality reductor.
    plot_2d_projection(X, sites, y)


@pytest.mark.docs
def test_plot_2d_projection_default_without_y(multisite_data) -> None:
    """Plot 2D projection with default parameters without y."""
    X, _, sites = multisite_data
    # Using the function as default. The function will use t-SNE as default dimensionality reductor.
    plot_2d_projection(X, sites)


# TODO: Parametrize this test
@pytest.mark.docs
def test_plot_2d_projection_methods_as_str(multisite_data) -> None:
    """Plot 2D projection passing method as str."""
    X, y, sites = multisite_data
    # Using the function as default. The function will use t-SNE as default dimensionality reductor.
    plot_2d_projection(X, sites, y, method="pca")
    plot_2d_projection(X, sites, y, method="tsne")
    plot_2d_projection(X, sites, y, method="kernelpca")
    plot_2d_projection(X, sites, y, method="MDS")
    plot_2d_projection(X, sites, y, method="isomap")


@pytest.mark.docs
def test_plot_2d_projection_methods_as_str_invalid(multisite_data) -> None:
    """Plot 2D projection passing method as str invalid."""
    X, y, sites = multisite_data
    # Using the function as default. The function will use t-SNE as default dimensionality reductor.
    with pytest.raises(ValueError):
        plot_2d_projection(X, sites, y, method="invalid")


@pytest.mark.docs
def test_plot_2d_projection_with_instance(multisite_data) -> None:
    """Plot 2D projection passing instance."""
    X, y, sites = multisite_data
    dim_reductor = FastICA(n_components=2)
    # We can also pass directly an instance of the dimensionality reductor that we want to use.
    plot_2d_projection(X, sites, y, dim_reductor=dim_reductor)


@pytest.mark.docs
def test_plot_2d_projection_with_invalida_params(multisite_data) -> None:
    """Plot 2D projection passing instance."""
    X, y, sites = multisite_data
    dim_reductor = FastICA(n_components=10)
    # We can also pass directly an instance of the dimensionality reductor that we want to use.
    with pytest.raises(ValueError):
        plot_2d_projection(X, sites, y, dim_reductor=dim_reductor)
    dim_reductor = LogisticRegression()
    # We can also pass directly an instance of the dimensionality reductor that we want to use.
    with pytest.raises(TypeError):
        plot_2d_projection(X, sites, y, dim_reductor=dim_reductor)


@pytest.mark.docs
def test_plot_features_by_site_with_default(multisite_data) -> None:
    """Test plot features by site with defaults."""
    X, _, sites = multisite_data
    # basic functionality
    plot_features_by_site(X, sites)


@pytest.mark.docs
def test_plot_features_by_site(multisite_data) -> None:
    """Plot features by site with with more parameters."""
    X, _, sites = multisite_data
    # basic functionality
    _, _ = plot_features_by_site(
        X,
        sites,
        figsize=(14, 7),
        rotation=45,
        show_points=True,
        title="All Features by Site (with individual points)",
    )


@pytest.mark.docs
def test_plot_decision_boundary_2d() -> None:
    """Plot decision boundary 2D."""
    X, y, _ = make_multisite_classification(n_features=2, n_samples=100)
    _, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Fit the model and plot the decision boundary,
    # this is just for visualization purposes, the real evaluation was be done with cross-validation
    clf = LogisticRegression()
    clf.fit(X, y)
    plot_decision_boundary_2d(ax, clf)


@pytest.mark.docs
def test_plot_2d_components_by_value() -> None:
    """Test plot 2D components by value."""
    X, y, sites = make_multisite_classification(n_features=2, n_samples=100)

    df = pd.DataFrame({"comp1": X[:, 0], "comp2": X[:, 1], "site": sites, "target": y})

    # Initialize figure
    _, axes = plt.subplots(1, 1, figsize=(16, 14))

    plot_2d_components_by_value(df, "site", "tSNE", ax=axes)
