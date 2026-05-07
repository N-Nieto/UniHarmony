"""Tests for report_metrics_by_site module."""

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

from uniharmony.metrics import (
    report_metrics_by_site,
)
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
# Single metric (callable) — NEW behavior
# =========================================================================


def test_single_callable_returns_nested_dict(binary_classification_data) -> None:
    """Test that single callable returns {metric_name: {site: value}}."""
    y_true, y_pred, sites = binary_classification_data
    results = report_metrics_by_site(y_true, y_pred, sites, accuracy_score)

    assert isinstance(results, dict)
    assert "accuracy_score" in results
    assert isinstance(results["accuracy_score"], dict)
    assert set(results["accuracy_score"].keys()) == {1, 2, 3}


def test_single_callable_correct_values(binary_classification_data) -> None:
    """Test that single callable computes correct metric values."""
    y_true, y_pred, sites = binary_classification_data
    results = report_metrics_by_site(y_true, y_pred, sites, accuracy_score)

    # Site 1: y_true=[0,1], y_pred=[0,1] -> accuracy=1.0
    assert results["accuracy_score"][1] == pytest.approx(1.0)
    # Site 2: y_true=[0,1], y_pred=[0,0] -> accuracy=0.5
    assert results["accuracy_score"][2] == pytest.approx(0.5)
    # Site 3: y_true=[0,1], y_pred=[0,1] -> accuracy=1.0
    assert results["accuracy_score"][3] == pytest.approx(1.0)


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
    assert len(results["f1_score"]) == 3


def test_single_callable_string_sites(string_sites_data) -> None:
    """Test single callable with string site identifiers."""
    y_true, y_pred, sites = string_sites_data
    results = report_metrics_by_site(y_true, y_pred, sites, accuracy_score)

    assert set(results["accuracy_score"].keys()) == {"A", "B", "C"}


def test_single_callable_regression(regression_data) -> None:
    """Test single callable with regression metric."""
    y_true, y_pred, sites = regression_data
    results = report_metrics_by_site(y_true, y_pred, sites, mean_squared_error)

    assert "mean_squared_error" in results
    assert len(results["mean_squared_error"]) == 3
    assert all(v >= 0 for v in results["mean_squared_error"].values())


# =========================================================================
# Multiple metrics (sequence) — existing behavior
# =========================================================================


def test_multiple_metrics_basic(binary_classification_data) -> None:
    """Test computing multiple metrics simultaneously."""
    y_true, y_pred, sites = binary_classification_data
    metrics = [accuracy_score, precision_score, recall_score]

    results = report_metrics_by_site(y_true, y_pred, sites, metrics)

    assert isinstance(results, dict)
    assert len(results) == 3
    assert set(results.keys()) == {
        "accuracy_score",
        "precision_score",
        "recall_score",
    }

    for _, site_results in results.items():
        assert isinstance(site_results, dict)
        assert len(site_results) == 3
        assert set(site_results.keys()) == {1, 2, 3}


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

    results = report_metrics_by_site(y_true, y_pred, sites, metrics, metric_kwargs={"zero_division": 0.0})
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


def test_accuracy_with_scores_auto_binarizes(binary_scores_data) -> None:
    """Test accuracy_score auto-binarizes continuous scores at threshold=0.5."""
    y_true, y_scores, sites = binary_scores_data
    results = report_metrics_by_site(y_true, y_scores, sites, accuracy_score)

    # y_scores=[0.1, 0.9, 0.2, 0.4, 0.3, 0.8]
    # binarized at 0.5: [0, 1, 0, 0, 0, 1]
    # Site 1: y_true=[0,1], binarized=[0,1] -> accuracy=1.0
    assert results["accuracy_score"][1] == pytest.approx(1.0)
    # Site 2: y_true=[0,1], binarized=[0,0] -> accuracy=0.5
    assert results["accuracy_score"][2] == pytest.approx(0.5)


def test_accuracy_with_scores_custom_threshold(binary_scores_data) -> None:
    """Test accuracy_score with custom threshold."""
    y_true, y_scores, sites = binary_scores_data
    results = report_metrics_by_site(y_true, y_scores, sites, accuracy_score, metric_kwargs={"threshold": 0.3})

    # y_scores=[0.1, 0.9, 0.2, 0.4, 0.3, 0.8]
    # binarized at 0.3: [0, 1, 0, 1, 1, 1]
    # Site 1: y_true=[0,1], binarized=[0,1] -> accuracy=1.0
    assert results["accuracy_score"][1] == pytest.approx(1.0)
    # Site 2: y_true=[0,1], binarized=[0,1] -> accuracy=1.0 (changed from 0.5!)
    assert results["accuracy_score"][2] == pytest.approx(1.0)


def test_roc_auc_with_scores_no_binarization(binary_scores_data) -> None:
    """Test roc_auc_score uses scores directly without binarization."""
    y_true, y_scores, sites = binary_scores_data
    results = report_metrics_by_site(y_true, y_scores, sites, roc_auc_score)

    assert "roc_auc_score" in results
    assert len(results["roc_auc_score"]) == 3


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
    # roc_auc should use scores, accuracy and f1 should use binarized
    for metric_name in results:
        assert len(results[metric_name]) == 3


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

    # Should work normally without any binarization
    assert results["accuracy_score"][1] == pytest.approx(1.0)


# =========================================================================
# Input validation
# =========================================================================


def test_mismatched_array_lengths() -> None:
    """Test ValueError for mismatched array lengths."""
    y_true = np.array([0, 1, 0])
    y_pred = np.array([0, 1])
    sites = np.array([1, 1])

    with pytest.raises(ValueError, match="same length"):
        report_metrics_by_site(y_true, y_pred, sites, accuracy_score)


def test_mismatched_all_three_lengths() -> None:
    """Test ValueError when all three arrays have different lengths."""
    with pytest.raises(ValueError, match="same length"):
        report_metrics_by_site(
            np.array([0, 1]),
            np.array([0, 1, 0]),
            np.array([1]),
            accuracy_score,
        )


def test_non_array_y_true() -> None:
    """Test TypeError for non-array y_true."""
    with pytest.raises(TypeError, match=r"y_true must be a numpy.ndarray"):
        report_metrics_by_site(
            [0, 1],
            np.array([0, 1]),
            np.array([1, 1]),
            accuracy_score,
        )


def test_non_array_y_pred() -> None:
    """Test TypeError for non-array y_pred."""
    with pytest.raises(TypeError, match=r"y_pred must be a numpy.ndarray"):
        report_metrics_by_site(
            np.array([0, 1]),
            [0, 1],
            np.array([1, 1]),
            accuracy_score,
        )


def test_non_array_sites() -> None:
    """Test TypeError for non-array sites."""
    with pytest.raises(TypeError, match=r"sites must be a numpy.ndarray"):
        report_metrics_by_site(
            np.array([0, 1]),
            np.array([0, 1]),
            [1, 1],
            accuracy_score,
        )


def test_invalid_metric_type_single() -> None:
    """Test TypeError for invalid single metric type."""
    with pytest.raises(TypeError, match="metrics must be a sequence of callables,"):
        report_metrics_by_site(
            np.array([0, 1]),
            np.array([0, 1]),
            np.array([1, 1]),
            123,
        )


def test_invalid_metric_type_sequence() -> None:
    """Test TypeError for invalid metric in sequence."""
    with pytest.raises(TypeError, match=r"metrics\[0\] must be callable"):
        report_metrics_by_site(
            np.array([0, 1]),
            np.array([0, 1]),
            np.array([1, 1]),
            [123],
        )


def test_metric_not_callable_string() -> None:
    """Test TypeError when metric is a string instead of callable."""
    with pytest.raises(TypeError, match="metrics must be a sequence of callables,"):
        report_metrics_by_site(
            np.array([0, 1]),
            np.array([0, 1]),
            np.array([1, 1]),
            "accuracy_score",
        )


def test_overall_performance_invalid_type(binary_classification_data) -> None:
    """Test TypeError for non-bool overall_performance."""
    y_true, y_pred, sites = binary_classification_data
    with pytest.raises(TypeError, match="overall_performance must be a bool"):
        report_metrics_by_site(y_true, y_pred, sites, accuracy_score, overall_performance=1)


# =========================================================================
# Site handling
# =========================================================================


def test_string_site_identifiers(string_sites_data) -> None:
    """Test with string site identifiers."""
    y_true, y_pred, sites = string_sites_data
    results = report_metrics_by_site(y_true, y_pred, sites, accuracy_score)

    assert set(results["accuracy_score"].keys()) == {"A", "B", "C"}


def test_integer_site_conversion(binary_classification_data) -> None:
    """Test that integer site identifiers are kept as ints."""
    y_true, y_pred, sites = binary_classification_data
    results = report_metrics_by_site(y_true, y_pred, sites, accuracy_score)

    for key in results["accuracy_score"]:
        assert isinstance(key, int)
        assert not isinstance(key, bool)


def test_float_site_identifiers_converted_to_int() -> None:
    """Test that float site IDs are converted to int keys."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    sites = np.array([1.0, 1.0, 2.0, 2.0])

    results = report_metrics_by_site(y_true, y_pred, sites, accuracy_score)
    assert set(results["accuracy_score"].keys()) == {1, 2}
    assert all(isinstance(k, int) for k in results["accuracy_score"].keys())


def test_boolean_site_identifiers() -> None:
    """Test with boolean site identifiers."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    sites = np.array([True, True, False, False])

    results = report_metrics_by_site(y_true, y_pred, sites, accuracy_score)
    # bool is subclass of int, so True->1, False->0
    assert set(results["accuracy_score"].keys()) == {0, 1}
    assert all(isinstance(k, int) for k in results["accuracy_score"].keys())


def test_single_site() -> None:
    """Test with only one unique site."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    sites = np.array([1, 1, 1, 1])

    results = report_metrics_by_site(y_true, y_pred, sites, accuracy_score)
    assert set(results["accuracy_score"].keys()) == {1}
    assert results["accuracy_score"][1] == pytest.approx(0.75)


def test_empty_site_skipped_by_default() -> None:
    """Test that empty sites are skipped when skip_empty_sites=True."""
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    sites = np.array([1, 1])

    results = report_metrics_by_site(y_true, y_pred, sites, accuracy_score)
    assert set(results["accuracy_score"].keys()) == {1}


def test_empty_site_not_skipped() -> None:
    """Test that empty sites don't break when skip_empty_sites=False."""
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    sites = np.array([1, 1])

    results = report_metrics_by_site(y_true, y_pred, sites, accuracy_score, skip_empty_sites=False)
    assert set(results["accuracy_score"].keys()) == {1}


# =========================================================================
# _binarize — Unit tests
# =========================================================================


class TestBinarize:
    """Unit tests for _binarize helper."""

    def test_default_threshold(self) -> None:
        """Test binarization at default threshold 0.5."""
        scores = np.array([0.1, 0.5, 0.9, 0.3])
        result = _binarize(scores)
        expected = np.array([0, 1, 1, 0])
        np.testing.assert_array_equal(result, expected)

    def test_custom_threshold(self) -> None:
        """Test binarization at custom threshold."""
        scores = np.array([0.1, 0.5, 0.9, 0.3])
        result = _binarize(scores, threshold=0.3)
        expected = np.array([0, 1, 1, 1])
        np.testing.assert_array_equal(result, expected)

    def test_all_above_threshold(self) -> None:
        """Test when all values are above threshold."""
        scores = np.array([0.6, 0.7, 0.8])
        result = _binarize(scores)
        expected = np.array([1, 1, 1])
        np.testing.assert_array_equal(result, expected)

    def test_all_below_threshold(self) -> None:
        """Test when all values are below threshold."""
        scores = np.array([0.1, 0.2, 0.3])
        result = _binarize(scores)
        expected = np.array([0, 0, 0])
        np.testing.assert_array_equal(result, expected)

    def test_empty_array(self) -> None:
        """Test empty array."""
        scores = np.array([])
        result = _binarize(scores)
        expected = np.array([], dtype=int)
        np.testing.assert_array_equal(result, expected)

    def test_exactly_at_threshold(self) -> None:
        """Test values exactly at threshold are mapped to 1."""
        scores = np.array([0.5, 0.5, 0.5])
        result = _binarize(scores, threshold=0.5)
        expected = np.array([1, 1, 1])
        np.testing.assert_array_equal(result, expected)


# =========================================================================
# _is_binary_or_multiclass — Unit tests
# =========================================================================


def test_binary_array() -> None:
    """Test binary labels are recognized."""
    arr = np.array([0, 1, 0, 1])
    assert _is_binary_or_multiclass(arr)


def test_multiclass_array() -> None:
    """Test multiclass labels are recognized."""
    arr = np.array([0, 1, 2, 3])
    assert _is_binary_or_multiclass(arr)


def test_continuous_scores() -> None:
    """Test continuous values are not recognized as discrete."""
    arr = np.array([0.1, 0.9, 0.3, 0.8])
    assert not _is_binary_or_multiclass(arr)


def test_mixed_int_float() -> None:
    """Test float representations of integers are recognized."""
    arr = np.array([0.0, 1.0, 0.0, 1.0])
    assert _is_binary_or_multiclass(arr)


def test_near_integer_with_tolerance() -> None:
    """Test values near integers within tolerance are recognized."""
    arr = np.array([0.000000001, 0.999999999])
    assert _is_binary_or_multiclass(arr)


def test_empty_array_binary() -> None:
    """Test empty array returns True."""
    arr = np.array([])
    assert _is_binary_or_multiclass(arr)


# =========================================================================
# _is_probability_like — Unit tests
# =========================================================================


def test_probabilities() -> None:
    """Test probability scores are recognized."""
    arr = np.array([0.1, 0.5, 0.9, 0.3])
    assert _is_probability_like(arr)


def test_binary_not_probability() -> None:
    """Test binary labels are not probability-like."""
    arr = np.array([0, 1, 0, 1])
    assert not _is_probability_like(arr)


def test_values_out_of_range() -> None:
    """Test values outside [0, 1] are not probability-like."""
    arr = np.array([0.1, 1.5, 0.9])
    assert not _is_probability_like(arr)


def test_negative_values() -> None:
    """Test negative values are not probability-like."""
    arr = np.array([-0.1, 0.5, 0.9])
    assert not _is_probability_like(arr)


def test_empty_array() -> None:
    """Test empty array returns False."""
    arr = np.array([])
    assert not _is_probability_like(arr)


def test_exactly_two_values_in_range() -> None:
    """Test two distinct values in [0,1] are not probability-like."""
    arr = np.array([0.0, 1.0])
    assert not _is_probability_like(arr)


# =========================================================================
# MetricSignatureDetection _metric_needs_y_score / _metric_needs_y_pred — Unit tests
# =========================================================================


def test_roc_auc_needs_score() -> None:
    """Test roc_auc_score is detected as score-based."""
    assert not _metric_needs_y_pred(roc_auc_score)


def test_accuracy_needs_pred() -> None:
    """Test accuracy_score is detected as pred-based."""
    assert _metric_needs_y_pred(accuracy_score)


def test_unknown_metric() -> None:
    """Test unknown metric returns False for both."""

    def custom_metric(y_true, y_pred):
        return 0.0

    assert not _metric_needs_y_pred(custom_metric)


# =========================================================================
# ValidateMetricKwargs _validate_metric_kwargs — Unit tests
# =========================================================================


def test_none_input() -> None:
    """Test None returns list of empty dicts."""
    result = _validate_metric_kwargs(None, 3)
    assert result == [{}, {}, {}]
    assert isinstance(result, list)


def test_single_dict_broadcast() -> None:
    """Test single dict is broadcast to all metrics."""
    kwargs = {"average": "macro"}
    result = _validate_metric_kwargs(kwargs, 2)
    assert result == [{"average": "macro"}, {"average": "macro"}]


def test_sequence_of_dicts() -> None:
    """Test sequence of dicts preserved one-to-one."""
    kwargs = [{}, {"average": "weighted"}]
    result = _validate_metric_kwargs(kwargs, 2)
    assert result == [{}, {"average": "weighted"}]


def test_tuple_input() -> None:
    """Test tuple is converted to list."""
    kwargs = ({}, {"average": "binary"})
    result = _validate_metric_kwargs(kwargs, 2)
    assert isinstance(result, list)
    assert len(result) == 2


def test_length_mismatch_raises() -> None:
    """Test ValueError on length mismatch."""
    with pytest.raises(ValueError, match="same length as metrics"):
        _validate_metric_kwargs([{}, {}], 3)


def test_invalid_type_raises() -> None:
    """Test TypeError for invalid metric_kwargs type."""
    with pytest.raises(TypeError, match="must be a dict, sequence of dicts, or None"):
        _validate_metric_kwargs(123, 2)


def test_string_raises() -> None:
    """Test TypeError when metric_kwargs is a string."""
    with pytest.raises(TypeError, match="must be a dict, sequence of dicts, or None"):
        _validate_metric_kwargs("not_valid", 2)


def test_zero_metrics_raises() -> None:
    """Test ValueError for n_metrics <= 0."""
    with pytest.raises(ValueError, match="must be positive"):
        _validate_metric_kwargs(None, 0)


def test_negative_metrics_raises() -> None:
    """Test ValueError for negative n_metrics."""
    with pytest.raises(ValueError, match="must be positive"):
        _validate_metric_kwargs(None, -1)


def test_non_int_n_metrics_raises() -> None:
    """Test TypeError for non-int n_metrics."""
    with pytest.raises(TypeError, match="n_metrics must be an int"):
        _validate_metric_kwargs(None, "3")


def test_sequence_with_non_dict_element_raises() -> None:
    """Test TypeError when sequence contains non-dict."""
    with pytest.raises(TypeError, match=r"metric_kwargs\[1\] must be a dict"):
        _validate_metric_kwargs([{}, "bad"], 2)


# =========================================================================
# _input_checks and _input_checks_multi — Unit tests
# =========================================================================


def test_valid_inputs_pass() -> None:
    """Test valid inputs do not raise."""
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    sites = np.array([1, 1])
    _input_checks(y_true, y_pred, sites, accuracy_score, False)


def test_mismatched_lengths_raises() -> None:
    """Test mismatched lengths raise ValueError."""
    with pytest.raises(ValueError, match="same length"):
        _input_checks(
            np.array([0, 1, 0]),
            np.array([0, 1]),
            np.array([1, 1]),
            accuracy_score,
            False,
        )


def test_non_callable_metric_raises() -> None:
    """Test non-callable metric raises TypeError."""
    with pytest.raises(TypeError, match="metric must be a callable"):
        _input_checks(
            np.array([0, 1]),
            np.array([0, 1]),
            np.array([1, 1]),
            "not_callable",
            False,
        )


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


def test_input_checks_multi_non_sequence_raises() -> None:
    """Test non-sequence metrics raises TypeError."""
    with pytest.raises(TypeError, match="must be a sequence"):
        _input_checks_multi(
            np.array([0, 1]),
            np.array([0, 1]),
            np.array([1, 1]),
            accuracy_score,  # single callable, not sequence
            False,
        )


# =========================================================================
# Edge cases
# =========================================================================


def test_all_correct_predictions() -> None:
    """Test with perfect predictions."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    sites = np.array([1, 1, 2, 2])

    results = report_metrics_by_site(y_true, y_pred, sites, accuracy_score)
    assert results["accuracy_score"][1] == pytest.approx(1.0)
    assert results["accuracy_score"][2] == pytest.approx(1.0)


def test_all_incorrect_predictions() -> None:
    """Test with completely wrong predictions."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0])
    sites = np.array([1, 1, 2, 2])

    results = report_metrics_by_site(y_true, y_pred, sites, accuracy_score)
    assert results["accuracy_score"][1] == pytest.approx(0.0)
    assert results["accuracy_score"][2] == pytest.approx(0.0)


def test_large_number_of_sites() -> None:
    """Test with many unique sites."""
    n = 100
    y_true = np.random.randint(0, 2, size=n)
    y_pred = np.random.randint(0, 2, size=n)
    sites = np.arange(n)

    results = report_metrics_by_site(y_true, y_pred, sites, accuracy_score)
    assert len(results["accuracy_score"]) == n
    # Each site has exactly one sample, so accuracy is either 0 or 1
    assert all(v in {0.0, 1.0} for v in results["accuracy_score"].values())


def test_multiclass_predictions_accepted(multiclass_data) -> None:
    """Test that multiclass labels are accepted as discrete predictions."""
    y_true, y_pred, sites = multiclass_data
    results = report_metrics_by_site(y_true, y_pred, sites, accuracy_score)
    assert isinstance(results, dict)
    assert len(results["accuracy_score"]) == 3
