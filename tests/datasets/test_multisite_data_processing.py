"""Test suite for multisite data processing functions."""

import numpy as np
import pandas as pd
import pytest

from uniharmony.datasets import filter_data_by_site


def test_filter_data_by_site_includes_requested_sites() -> None:
    """Filter numpy data to the requested sites."""
    X = np.arange(12).reshape(6, 2)
    y = np.array([0, 1, 0, 1, 0, 1])
    sites = np.array(["site_a", "site_b", "site_a", "site_c", "site_b", "site_a"])

    X_filtered, y_filtered, sites_filtered = filter_data_by_site(X, y, sites, include_sites=["site_a", "site_c"])

    np.testing.assert_array_equal(X_filtered, X[[0, 2, 3, 5]])
    np.testing.assert_array_equal(y_filtered, y[[0, 2, 3, 5]])
    np.testing.assert_array_equal(sites_filtered, sites[[0, 2, 3, 5]])


def test_filter_data_by_site_excludes_requested_sites() -> None:
    """Exclude numpy rows that belong to a site."""
    X = np.arange(12).reshape(6, 2)
    y = np.array([0, 1, 0, 1, 0, 1])
    sites = np.array(["site_a", "site_b", "site_a", "site_c", "site_b", "site_a"])

    X_filtered, y_filtered, sites_filtered = filter_data_by_site(X, y, sites, exclude_sites="site_b")

    np.testing.assert_array_equal(X_filtered, X[[0, 2, 3, 5]])
    np.testing.assert_array_equal(y_filtered, y[[0, 2, 3, 5]])
    np.testing.assert_array_equal(sites_filtered, sites[[0, 2, 3, 5]])


def test_filter_data_by_site_applies_exclusions_after_inclusions() -> None:
    """Apply exclusions after the include list."""
    X = np.arange(12).reshape(6, 2)
    y = np.array([0, 1, 0, 1, 0, 1])
    sites = np.array(["site_a", "site_b", "site_a", "site_c", "site_b", "site_a"])

    X_filtered, y_filtered, sites_filtered = filter_data_by_site(
        X,
        y,
        sites,
        include_sites=["site_a", "site_b"],
        exclude_sites=["site_b"],
    )

    np.testing.assert_array_equal(X_filtered, X[[0, 2, 5]])
    np.testing.assert_array_equal(y_filtered, y[[0, 2, 5]])
    np.testing.assert_array_equal(sites_filtered, sites[[0, 2, 5]])


def test_filter_data_by_site_preserves_pandas_containers() -> None:
    """Preserve pandas data types and indexes when filtering."""
    index = ["sample_0", "sample_1", "sample_2", "sample_3"]
    X = pd.DataFrame({"feature_0": [0.1, 0.2, 0.3, 0.4], "feature_1": [1.1, 1.2, 1.3, 1.4]}, index=index)
    y = pd.Series([0, 1, 0, 1], index=index, name="target")
    sites = pd.Series(["site_a", "site_b", "site_a", "site_c"], index=index, name="site")

    X_filtered, y_filtered, sites_filtered = filter_data_by_site(X, y, sites, include_sites="site_a")

    pd.testing.assert_frame_equal(X_filtered, X.iloc[[0, 2]])
    pd.testing.assert_series_equal(y_filtered, y.iloc[[0, 2]])
    pd.testing.assert_series_equal(sites_filtered, sites.iloc[[0, 2]])


def test_filter_data_by_site_rejects_inconsistent_lengths() -> None:
    """Reject inputs that do not describe the same number of samples."""
    X = np.arange(8).reshape(4, 2)
    y = np.array([0, 1, 0])
    sites = np.array(["site_a", "site_b", "site_a", "site_c"])

    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        filter_data_by_site(X, y, sites, include_sites="site_a")


def test_filter_data_by_site_rejects_two_dimensional_sites() -> None:
    """Reject site labels that are not one-dimensional."""
    X = np.arange(8).reshape(4, 2)
    y = np.array([0, 1, 0, 1])
    sites = np.array([["site_a"], ["site_b"], ["site_a"], ["site_c"]])

    with pytest.raises(ValueError, match="sites must be one-dimensional"):
        filter_data_by_site(X, y, sites, include_sites="site_a")
