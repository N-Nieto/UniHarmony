"""Tests for report_metric_by_site module."""

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, mean_squared_error

from uniharmony.metrics import report_metric_by_site


def test_with_callable_metric():
    """Test computation with callable metric."""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1])
    sites = np.array([1, 1, 2, 2, 3, 3])

    results = report_metric_by_site(y_true, y_pred, sites, accuracy_score)

    assert isinstance(results, dict)
    assert len(results) == 3


def test_mismatched_array_lengths():
    """Test ValueError for mismatched array lengths."""
    y_true = np.array([0, 1, 0])
    y_pred = np.array([0, 1])
    sites = np.array([1, 1])

    with pytest.raises(ValueError, match="same length"):
        report_metric_by_site(y_true, y_pred, sites, accuracy_score)


def test_non_array_y_pred():
    """Test TypeError for non-array y_pred."""
    with pytest.raises(TypeError):
        report_metric_by_site(
            np.array([0, 1]),
            [0, 1],  # type: ignore
            np.array([1, 1]),
            accuracy_score,
        )


def test_non_array_y_true():
    """Test TypeError for non-array y_true."""
    with pytest.raises(TypeError):
        report_metric_by_site(
            [0, 1],  # type: ignore
            np.array([0, 1]),
            np.array([1, 1]),
            accuracy_score,
        )


def test_non_array_sites():
    """Test TypeError for non-array sites."""
    with pytest.raises(TypeError):
        report_metric_by_site(
            np.array([0, 1]),
            np.array([0, 1]),
            [1, 1],  # type: ignore
            accuracy_score,
        )


def test_invalid_metric_type():
    """Test TypeError for invalid metric type."""
    with pytest.raises(TypeError, match="metric must be a callable"):
        report_metric_by_site(
            np.array([0, 1]),
            np.array([0, 1]),
            np.array([1, 1]),
            123,  # type: ignore
        )


def test_regression_metric():
    """Test with regression metric."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.0, 5.2, 5.8])
    sites = np.array([1, 1, 2, 2, 3, 3])

    results = report_metric_by_site(y_true, y_pred, sites, mean_squared_error)

    assert len(results) == 3
    assert all(v >= 0 for v in results.values())


def test_string_site_identifiers():
    """Test with string site identifiers."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    sites = np.array(["A", "A", "B", "B"])

    results = report_metric_by_site(
        y_true, y_pred, sites, accuracy_score, overall_performance=False
    )

    assert set(results.keys()) == {"A", "B"}


def test_overall_performance_functionality():
    """Test with string site identifiers."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    sites = np.array(["A", "A", "B", "B"])

    results = report_metric_by_site(
        y_true, y_pred, sites, accuracy_score, overall_performance=False
    )
    assert set(results.keys()) == {"A", "B"}
    results = report_metric_by_site(
        y_true, y_pred, sites, accuracy_score, overall_performance=True
    )
    assert set(results.keys()) == {"overall", "A", "B"}
    with pytest.raises(TypeError):
        results = report_metric_by_site(
            y_true, y_pred, sites, accuracy_score, overall_performance=1
        )
