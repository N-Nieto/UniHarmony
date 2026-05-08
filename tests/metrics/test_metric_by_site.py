"""Tests for report_metrics_by_site module."""

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

from uniharmony.metrics import report_metrics_by_site
from uniharmony.metrics._report_metric_by_site import (
    _binarize,
    _input_checks,
    _input_checks_multi,
    _is_binary_or_multiclass,
    _is_probability_like,
    _metric_needs_y_pred,
    _validate_metric_kwargs,
)


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def binary_classification_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sample binary classification data with 3 sites."""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1])
    sites = np.array([1, 1, 2, 2, 3, 3])
    return y_true, y_pred, sites


@pytest.fixture
def binary_scores_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sample binary classification data with probability scores."""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_scores = np.array([0.1, 0.9, 0.2, 0.4, 0.3, 0.8])
    sites = np.array([1, 1, 2, 2, 3, 3])
    return y_true, y_scores, sites


@pytest.fixture
def string_sites_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sample data with string site identifiers."""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1])
    sites = np.array(["A", "A", "B", "B", "C", "C"])
    return y_true, y_pred, sites


@pytest.fixture
def regression_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sample regression data."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.0, 5.2, 5.8])
    sites = np.array([1, 1, 2, 2, 3, 3])
    return y_true, y_pred, sites


@pytest.fixture
def multiclass_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sample multiclass classification data."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 1])
    sites = np.array([1, 1, 2, 2, 3, 3])
    return y_true, y_pred, sites


# =========================================================================
# Single metric (callable)
# =========================================================================


@pytest.mark.parametrize(
    "metric,expected_keys",
    [
        (accuracy_score, {1, 2, 3}),
        (f1_score, {1, 2, 3}),
        (mean_squared_error, {1, 2, 3}),
    ],
)
def test_single_callable_returns_nested_dict(binary_classification_data, metric, expected_keys) -> None:
    """Test that single callable returns {metric_name: {site: value}}."""
    y_true, y_pred, sites = binary_classification_data
    results = report_metrics_by_site(y_true, y_pred, sites, metric, overall_performance=False)

    metric_name = metric.__name__
    assert isinstance(results, dict)
    assert metric_name in results
    assert isinstance(results[metric_name], dict)
    assert set(results[metric_name].keys()) == expected_keys


@pytest.mark.parametrize(
    "site,expected_value",
    [
        (1, 1.0),  # y_true=[0,1], y_pred=[0,1] -> accuracy=1.0
        (2, 0.5),  # y_true=[0,1], y_pred=[0,0] -> accuracy=0.5
        (3, 1.0),  # y_true=[0,1], y_pred=[0,1] -> accuracy=1.0
    ],
)
def test_single_callable_correct_values(binary_classification_data, site, expected_value) -> None:
    """Test that single callable computes correct metric values."""
    y_true, y_pred, sites = binary_classification_data
    results = report_metrics_by_site(y_true, y_pred, sites, accuracy_score)

    assert results["accuracy_score"][site] == pytest.approx(expected_value)


def test_single_callable_with_overall(binary_classification_data) -> None:
    """Test overall_performance with single callable."""
    y_true, y_pred, sites = binary_classification_data
    results = report_metrics_by_site(y_true, y_pred, sites, accuracy_score, overall_performance=True)

    assert "overall" in results["accuracy_score"]
    assert results["accuracy_score"]["overall"] == pytest.approx(accuracy_score(y_true, y_pred))
    assert set(results["accuracy_score"].keys()) == {"overall", 1, 2, 3}


def test_single_callable_with_kwargs(binary_classification_data) -> None:
    """Test single callable with metric_kwargs dict."""
    y_true, y_pred, sites = binary_classification_data
    results = report_metrics_by_site(y_true, y_pred, sites, f1_score, metric_kwargs={"zero_division": 0.0})

    assert "f1_score" in results
    assert len(results["f1_score"]) == 4


def test_single_callable_string_sites(string_sites_data) -> None:
    """Test single callable with string site identifiers."""
    y_true, y_pred, sites = string_sites_data
    results = report_metrics_by_site(y_true, y_pred, sites, accuracy_score)

    assert set(results["accuracy_score"].keys()) == {"overall", "A", "B", "C"}


def test_single_callable_regression(regression_data) -> None:
    """Test single callable with regression metric."""
    y_true, y_pred, sites = regression_data
    results = report_metrics_by_site(y_true, y_pred, sites, mean_squared_error)

    assert "mean_squared_error" in results
    assert len(results["mean_squared_error"]) == 4
    assert all(v >= 0 for v in results["mean_squared_error"].values())


# =========================================================================
# Multiple metrics (sequence)
# =========================================================================


@pytest.mark.parametrize(
    "metrics,expected_names",
    [
        ([accuracy_score], {"accuracy_score"}),
        ([accuracy_score, precision_score, recall_score], {"accuracy_score", "precision_score", "recall_score"}),
        ([accuracy_score, f1_score], {"accuracy_score", "f1_score"}),
    ],
)
def test_multiple_metrics_basic(binary_classification_data, metrics, expected_names) -> None:
    """Test computing multiple metrics simultaneously."""
    y_true, y_pred, sites = binary_classification_data
    results = report_metrics_by_site(y_true, y_pred, sites, metrics)

    assert isinstance(results, dict)
    assert len(results) == len(expected_names)
    assert set(results.keys()) == expected_names

    for _, site_results in results.items():
        assert isinstance(site_results, dict)
        assert len(site_results) == 4
        assert set(site_results.keys()) == {"overall", 1, 2, 3}


def test_multiple_metrics_with_overall(binary_classification_data) -> None:
    """Test multi-metric with overall performance."""
    y_true, y_pred, sites = binary_classification_data
    metrics = [accuracy_score, f1_score]

    results = report_metrics_by_site(y_true, y_pred, sites, metrics, overall_performance=True)

    assert "overall" in results["accuracy_score"]
    assert "overall" in results["f1_score"]


def test_multiple_metrics_with_none_kwargs(binary_classification_data) -> None:
    """Test multi-metric with metric_kwargs=None."""
    y_true, y_pred, sites = binary_classification_data
    metrics = [accuracy_score, balanced_accuracy_score]

    results = report_metrics_by_site(y_true, y_pred, sites, metrics, metric_kwargs=None)
    assert len(results) == 2


def test_multiple_metrics_with_shared_kwargs(binary_classification_data) -> None:
    """Test multi-metric with single dict shared across all metrics."""
    y_true, y_pred, sites = binary_classification_data
    metrics = [f1_score, precision_score]

    results = report_metrics_by_site(
        y_true, y_pred, sites, metrics, metric_kwargs={"zero_division": 0.0}, overall_performance=False
    )
    assert len(results) == 2
    assert all(len(v) == 3 for v in results.values())


def test_multiple_metrics_with_individual_kwargs(binary_classification_data) -> None:
    """Test multi-metric with individual kwargs per metric."""
    y_true, y_pred, sites = binary_classification_data
    metrics = [f1_score, precision_score]
    kwargs = [{"average": "binary"}, {"zero_division": 0.0}]

    results = report_metrics_by_site(y_true, y_pred, sites, metrics, metric_kwargs=kwargs)
    assert len(results) == 2


def test_multiple_metrics_kwargs_length_mismatch(binary_classification_data) -> None:
    """Test ValueError when metric_kwargs length doesn't match metrics."""
    y_true, y_pred, sites = binary_classification_data
    metrics = [accuracy_score, f1_score]

    with pytest.raises(ValueError, match="same length as metrics"):
        report_metrics_by_site(y_true, y_pred, sites, metrics, metric_kwargs=[{}, {}, {}])


def test_multiple_metrics_empty_list() -> None:
    """Test ValueError for empty metrics list."""
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    sites = np.array([1, 1])

    with pytest.raises(ValueError, match="at least one callable"):
        report_metrics_by_site(y_true, y_pred, sites, [])


def test_multiple_metrics_non_callable_in_list() -> None:
    """Test TypeError when metrics contains non-callable."""
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    sites = np.array([1, 1])

    with pytest.raises(TypeError, match=r"metrics\[1\] must be callable"):
        report_metrics_by_site(y_true, y_pred, sites, [accuracy_score, "bad"])


# =========================================================================
# Auto-binarization
# =========================================================================


def test_roc_auc_with_scores_no_binarization(binary_scores_data) -> None:
    """Test roc_auc_score uses scores directly without binarization."""
    y_true, y_scores, sites = binary_scores_data
    results = report_metrics_by_site(y_true, y_scores, sites, roc_auc_score)

    assert "roc_auc_score" in results
    assert len(results["roc_auc_score"]) == 4


def test_mixed_metrics_some_binarize_some_dont(binary_scores_data) -> None:
    """Test mix of score-based and pred-based metrics with same y_scores."""
    y_true, y_scores, sites = binary_scores_data
    metrics = [roc_auc_score, accuracy_score, f1_score]
    kwargs = [
        {},
        {"threshold": 0.5},
        {"threshold": 0.5, "average": "macro"},
    ]

    results = report_metrics_by_site(y_true, y_scores, sites, metrics, metric_kwargs=kwargs)

    assert set(results.keys()) == {"roc_auc_score", "accuracy_score", "f1_score"}
    for metric_name in results:
        assert len(results[metric_name]) == 4


def test_binarization_threshold_not_passed_to_metric(binary_scores_data) -> None:
    """Test that threshold is consumed and not passed to the metric."""
    y_true, y_scores, sites = binary_scores_data
    # f1_score does not accept 'threshold' as a parameter
    # If threshold were passed through, this would raise TypeError
    results = report_metrics_by_site(
        y_true,
        y_scores,
        sites,
        f1_score,
        metric_kwargs={"threshold": 0.5, "average": "binary", "zero_division": 0.0},
    )
    assert "f1_score" in results


def test_labels_passed_no_binarization(binary_classification_data) -> None:
    """Test that discrete labels are not binarized."""
    y_true, y_pred, sites = binary_classification_data
    results = report_metrics_by_site(y_true, y_pred, sites, accuracy_score)

    assert results["accuracy_score"][1] == pytest.approx(1.0)


# =========================================================================
# Input validation
# =========================================================================


@pytest.mark.parametrize(
    "y_true,y_pred,sites,metric,error_match",
    [
        (
            np.array([0, 1, 0]),
            np.array([0, 1]),
            np.array([1, 1]),
            accuracy_score,
            "same length",
        ),
        (
            np.array([0, 1]),
            np.array([0, 1, 0]),
            np.array([1]),
            accuracy_score,
            "same length",
        ),
    ],
)
def test_mismatched_array_lengths(y_true, y_pred, sites, metric, error_match) -> None:
    """Test ValueError for mismatched array lengths."""
    with pytest.raises(ValueError, match=error_match):
        report_metrics_by_site(y_true, y_pred, sites, metric)


@pytest.mark.parametrize(
    "y_true,y_pred,sites,metric,error_match",
    [
        (
            [0, 1],
            np.array([0, 1]),
            np.array([1, 1]),
            accuracy_score,
            "y_true must be a numpy.ndarray",
        ),
        (
            np.array([0, 1]),
            [0, 1],
            np.array([1, 1]),
            accuracy_score,
            "y_pred must be a numpy.ndarray",
        ),
        (
            np.array([0, 1]),
            np.array([0, 1]),
            [1, 1],
            accuracy_score,
            "sites must be a numpy.ndarray",
        ),
        (
            np.array([0, 1]),
            np.array([0, 1]),
            np.array([1, 1]),
            123,
            r"must be callable",
        ),
        (
            np.array([0, 1]),
            np.array([0, 1]),
            np.array([1, 1]),
            "accuracy_score",
            r"must be callable",
        ),
    ],
)
def test_invalid_inputs_raise_typeerror(y_true, y_pred, sites, metric, error_match) -> None:
    """Test TypeError for invalid input types."""
    with pytest.raises(TypeError, match=error_match):
        report_metrics_by_site(y_true, y_pred, sites, metric)


def test_overall_performance_invalid_type(binary_classification_data) -> None:
    """Test TypeError for non-bool overall_performance."""
    y_true, y_pred, sites = binary_classification_data
    with pytest.raises(TypeError, match="overall_performance must be a bool"):
        report_metrics_by_site(y_true, y_pred, sites, accuracy_score, overall_performance="yes")


# =========================================================================
# Site handling
# =========================================================================


@pytest.mark.parametrize(
    "sites,expected_keys,site_type",
    [
        (np.array(["A", "A", "B", "B", "C", "C"]), {"A", "B", "C"}, str),
        (np.array([1, 1, 2, 2, 3, 3]), {1, 2, 3}, int),
        (np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0]), {1, 2}, int),
        (np.array([True, True, True, False, False, False]), {0, 1}, int),
    ],
)
def test_site_identifier_types(binary_classification_data, sites, expected_keys, site_type) -> None:
    """Test various site identifier types are handled correctly."""
    y_true, y_pred, _ = binary_classification_data
    results = report_metrics_by_site(y_true, y_pred, sites, accuracy_score, overall_performance=False)

    assert set(results["accuracy_score"].keys()) == expected_keys
    if site_type is int:
        assert all(isinstance(k, int) and not isinstance(k, bool) for k in results["accuracy_score"].keys())


def test_single_site() -> None:
    """Test with only one unique site."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    sites = np.array([1, 1, 1, 1])

    results = report_metrics_by_site(y_true, y_pred, sites, accuracy_score, overall_performance=False)
    assert set(results["accuracy_score"].keys()) == {1}
    assert results["accuracy_score"][1] == pytest.approx(0.75)


def test_empty_site_skipped_by_default() -> None:
    """Test that empty sites are skipped when skip_empty_sites=True."""
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    sites = np.array([1, 1])

    results = report_metrics_by_site(y_true, y_pred, sites, accuracy_score)
    assert set(results["accuracy_score"].keys()) == {"overall", 1}


def test_empty_site_not_skipped() -> None:
    """Test that empty sites don't break when skip_empty_sites=False."""
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    sites = np.array([1, 1])

    results = report_metrics_by_site(y_true, y_pred, sites, accuracy_score, skip_empty_sites=False)
    assert set(results["accuracy_score"].keys()) == {"overall", 1}


# =========================================================================
# _binarize
# =========================================================================


@pytest.mark.parametrize(
    "scores,threshold,expected",
    [
        (np.array([0.1, 0.5, 0.9, 0.3]), 0.5, np.array([0, 1, 1, 0])),
        (np.array([0.1, 0.5, 0.9, 0.3]), 0.3, np.array([0, 1, 1, 1])),
        (np.array([0.6, 0.7, 0.8]), 0.5, np.array([1, 1, 1])),
        (np.array([0.1, 0.2, 0.3]), 0.5, np.array([0, 0, 0])),
        (np.array([0.5, 0.5, 0.5]), 0.5, np.array([1, 1, 1])),
    ],
)
def test_binarize(scores, threshold, expected) -> None:
    """Test binarization at various thresholds."""
    result = _binarize(scores, threshold)
    np.testing.assert_array_equal(result, expected)


def test_binarize_empty_array() -> None:
    """Test empty array binarization."""
    scores = np.array([])
    result = _binarize(scores)
    expected = np.array([], dtype=int)
    np.testing.assert_array_equal(result, expected)


# =========================================================================
# _is_binary_or_multiclass
# =========================================================================


@pytest.mark.parametrize(
    "arr,expected",
    [
        (np.array([0, 1, 0, 1]), True),
        (np.array([0, 1, 2, 3]), True),
        (np.array([0.1, 0.9, 0.3, 0.8]), False),
        (np.array([0.0, 1.0, 0.0, 1.0]), True),
        (np.array([0.000000001, 0.999999999]), True),
        (np.array([]), True),
    ],
)
def test_is_binary_or_multiclass(arr, expected) -> None:
    """Test _is_binary_or_multiclass with various inputs."""
    assert _is_binary_or_multiclass(arr) == expected


# =========================================================================
# _is_probability_like
# =========================================================================


@pytest.mark.parametrize(
    "arr,expected",
    [
        (np.array([0.1, 0.5, 0.9, 0.3]), True),
        (np.array([0, 1, 0, 1]), False),
        (np.array([0.1, 1.5, 0.9]), False),
        (np.array([-0.1, 0.5, 0.9]), False),
        (np.array([]), False),
        (np.array([0.0, 1.0]), False),
    ],
)
def test_is_probability_like(arr, expected) -> None:
    """Test _is_probability_like with various inputs."""
    assert _is_probability_like(arr) == expected


# =========================================================================
# _metric_needs_y_score / _metric_needs_y_pred
# =========================================================================


@pytest.mark.parametrize(
    "metric,expected_score,expected_pred",
    [
        (roc_auc_score, True, False),
        (average_precision_score, True, False),
        (accuracy_score, False, True),
        (f1_score, False, True),
        (precision_score, False, True),
    ],
)
def test_metric_signature_detection(metric, expected_score, expected_pred) -> None:
    """Test metric signature detection for known metrics."""
    assert _metric_needs_y_pred(metric) == expected_pred


# =========================================================================
# _validate_metric_kwargs
# =========================================================================


@pytest.mark.parametrize(
    "metric_kwargs,n_metrics,expected",
    [
        (None, 3, [{}, {}, {}]),
        ({"average": "macro"}, 2, [{"average": "macro"}, {"average": "macro"}]),
        ([{}, {"average": "weighted"}], 2, [{}, {"average": "weighted"}]),
    ],
)
def test_validate_metric_kwargs_valid(metric_kwargs, n_metrics, expected) -> None:
    """Test valid metric_kwargs normalization."""
    result = _validate_metric_kwargs(metric_kwargs, n_metrics)
    assert result == expected


@pytest.mark.parametrize(
    "metric_kwargs,n_metrics,error_type,error_match",
    [
        ([{}, {}], 3, ValueError, "same length as metrics"),
        (123, 2, TypeError, "must be a dict, sequence of dicts, or None"),
        ("not_valid", 2, TypeError, "must be a dict, sequence of dicts, or None"),
        (None, 0, ValueError, "must be positive"),
        (None, -1, ValueError, "must be positive"),
        (None, "3", TypeError, "n_metrics must be an int"),
        ([{}, "bad"], 2, TypeError, r"metric_kwargs\[1\] must be a dict"),
    ],
)
def test_validate_metric_kwargs_invalid(metric_kwargs, n_metrics, error_type, error_match) -> None:
    """Test invalid metric_kwargs raise appropriate errors."""
    with pytest.raises(error_type, match=error_match):
        _validate_metric_kwargs(metric_kwargs, n_metrics)


# =========================================================================
# _input_checks and _input_checks_multi
# =========================================================================


@pytest.mark.parametrize(
    "y_true,y_pred,sites,metric,overall_performance,should_raise",
    [
        (np.array([0, 1]), np.array([0, 1]), np.array([1, 1]), accuracy_score, False, False),
        (np.array([0, 1, 0]), np.array([0, 1]), np.array([1, 1]), accuracy_score, False, True),
        (np.array([0, 1]), np.array([0, 1]), np.array([1, 1]), "not_callable", False, True),
    ],
)
def test_input_checks(y_true, y_pred, sites, metric, overall_performance, should_raise) -> None:
    """Test input validation."""
    if should_raise:
        with pytest.raises((TypeError, ValueError)):
            _input_checks(y_true, y_pred, sites, metric, overall_performance)
    else:
        _input_checks(y_true, y_pred, sites, metric, overall_performance)


def test_input_checks_multi_empty_metrics() -> None:
    """Test empty metrics list raises ValueError."""
    with pytest.raises(ValueError, match="at least one callable"):
        _input_checks_multi(
            np.array([0, 1]),
            np.array([0, 1]),
            np.array([1, 1]),
            [],
            False,
        )


def test_input_checks_multi_non_list_raises() -> None:
    """Test non-list metrics raises TypeError."""
    with pytest.raises(TypeError, match="must be a list"):
        _input_checks_multi(
            np.array([0, 1]),
            np.array([0, 1]),
            np.array([1, 1]),
            accuracy_score,
            False,
        )


# =========================================================================
# Edge cases
# =========================================================================


@pytest.mark.parametrize(
    "y_true,y_pred,expected_site1,expected_site2",
    [
        (np.array([0, 1, 0, 1]), np.array([0, 1, 0, 1]), 1.0, 1.0),  # perfect
        (np.array([0, 1, 0, 1]), np.array([1, 0, 1, 0]), 0.0, 0.0),  # all wrong
    ],
)
def test_perfect_and_wrong_predictions(y_true, y_pred, expected_site1, expected_site2) -> None:
    """Test with perfect and completely wrong predictions."""
    sites = np.array([1, 1, 2, 2])
    results = report_metrics_by_site(y_true, y_pred, sites, accuracy_score)

    assert results["accuracy_score"][1] == pytest.approx(expected_site1)
    assert results["accuracy_score"][2] == pytest.approx(expected_site2)


def test_large_number_of_sites() -> None:
    """Test with many unique sites."""
    n = 100
    y_true = np.random.randint(0, 2, size=n)
    y_pred = np.random.randint(0, 2, size=n)
    sites = np.arange(n)

    results = report_metrics_by_site(y_true, y_pred, sites, accuracy_score, overall_performance=False)
    assert len(results["accuracy_score"]) == n
    assert all(v in {0.0, 1.0} for v in results["accuracy_score"].values())


def test_multiclass_predictions_accepted(multiclass_data) -> None:
    """Test that multiclass labels are accepted as discrete predictions."""
    y_true, y_pred, sites = multiclass_data
    results = report_metrics_by_site(y_true, y_pred, sites, accuracy_score)
    assert isinstance(results, dict)
    assert len(results["accuracy_score"]) == 4
