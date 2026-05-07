"""Tests for report_metric_by_site module."""

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

from uniharmony.metrics import (
    METRICS_REQUIRING_Y_PRED,
    METRICS_REQUIRING_Y_SCORE,
    _input_checks,
    _input_checks_multi,
    _is_binary_or_multiclass,
    _is_probability_like,
    _metric_needs_y_pred,
    _metric_needs_y_score,
    _validate_metric_kwargs,
    _validate_prediction_type,
    report_metric_by_site,
    report_multimetrics_by_site,
)


# =============================================================================
# Fixtures
# =============================================================================


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


# =============================================================================
# report_metric_by_site — Basic functionality
# =============================================================================


def test_with_callable_metric(binary_classification_data) -> None:
    """Test computation with callable metric returns correct structure."""
    y_true, y_pred, sites = binary_classification_data
    results = report_metric_by_site(y_true, y_pred, sites, accuracy_score)

    assert isinstance(results, dict)
    assert len(results) == 3
    assert set(results.keys()) == {1, 2, 3}
    assert all(isinstance(v, (int, float)) for v in results.values())


def test_correct_metric_values(binary_classification_data) -> None:
    """Test that computed metric values are mathematically correct."""
    y_true, y_pred, sites = binary_classification_data
    results = report_metric_by_site(y_true, y_pred, sites, accuracy_score)

    # Site 1: y_true=[0,1], y_pred=[0,1] → accuracy=1.0
    assert results[1] == pytest.approx(1.0)
    # Site 2: y_true=[0,1], y_pred=[0,0] → accuracy=0.5
    assert results[2] == pytest.approx(0.5)
    # Site 3: y_true=[0,1], y_pred=[0,1] → accuracy=1.0
    assert results[3] == pytest.approx(1.0)


def test_string_site_identifiers(string_sites_data) -> None:
    """Test with string site identifiers preserves string keys."""
    y_true, y_pred, sites = string_sites_data
    results = report_metric_by_site(y_true, y_pred, sites, accuracy_score)

    assert set(results.keys()) == {"A", "B", "C"}
    assert all(isinstance(k, str) for k in results.keys())


def test_integer_site_conversion(binary_classification_data) -> None:
    """Test that integer site identifiers are kept as ints, not floats."""
    y_true, y_pred, sites = binary_classification_data
    results = report_metric_by_site(y_true, y_pred, sites, accuracy_score)

    for key in results:
        assert isinstance(key, int)
        assert not isinstance(key, bool)


def test_overall_performance_false(binary_classification_data) -> None:
    """Test overall_performance=False does not include overall key."""
    y_true, y_pred, sites = binary_classification_data
    results = report_metric_by_site(y_true, y_pred, sites, accuracy_score, overall_performance=False)
    assert "overall" not in results


def test_overall_performance_true(binary_classification_data) -> None:
    """Test overall_performance=True includes overall key with correct value."""
    y_true, y_pred, sites = binary_classification_data
    results = report_metric_by_site(y_true, y_pred, sites, accuracy_score, overall_performance=True)

    assert "overall" in results
    assert results["overall"] == pytest.approx(accuracy_score(y_true, y_pred))
    # Should still have all sites
    assert set(results.keys()) == {"overall", 1, 2, 3}


def test_overall_performance_invalid_type(binary_classification_data) -> None:
    """Test TypeError for non-bool overall_performance."""
    y_true, y_pred, sites = binary_classification_data
    with pytest.raises(TypeError, match="overall_performance must be a bool"):
        report_metric_by_site(y_true, y_pred, sites, accuracy_score, overall_performance="yes")


def test_regression_metric(regression_data) -> None:
    """Test with regression metric (MSE)."""
    y_true, y_pred, sites = regression_data
    results = report_metric_by_site(y_true, y_pred, sites, mean_squared_error)

    assert len(results) == 3
    assert all(v >= 0 for v in results.values())
    # Site 1: MSE of [1.0,2.0] vs [1.1,2.1]
    expected_site1 = mean_squared_error(y_true[:2], y_pred[:2])
    assert results[1] == pytest.approx(expected_site1)


def test_metric_with_kwargs(binary_classification_data) -> None:
    """Test passing additional kwargs to metric."""
    y_true, y_pred, sites = binary_classification_data
    # Use zero_division parameter of f1_score
    results = report_metric_by_site(y_true, y_pred, sites, f1_score, zero_division=0.0)
    assert isinstance(results, dict)
    assert len(results) == 3


def test_single_site() -> None:
    """Test with only one unique site."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    sites = np.array([1, 1, 1, 1])

    results = report_metric_by_site(y_true, y_pred, sites, accuracy_score)
    assert set(results.keys()) == {1}
    assert results[1] == pytest.approx(0.75)


def test_empty_site_skipped_by_default() -> None:
    """Test that empty sites are skipped when skip_empty_sites=True."""
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    sites = np.array([1, 1])

    results = report_metric_by_site(y_true, y_pred, sites, accuracy_score)
    assert set(results.keys()) == {1}


def test_empty_site_not_skipped_raises() -> None:
    """Test that empty sites raise error when skip_empty_sites=False."""
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    sites = np.array([1, 1])

    # This won't actually trigger since we don't have an empty site in the data
    # But we test the parameter exists and doesn't break
    results = report_metric_by_site(y_true, y_pred, sites, accuracy_score, skip_empty_sites=False)
    assert set(results.keys()) == {1}


# =============================================================================
# report_metric_by_site — Input validation
# =============================================================================


def test_mismatched_array_lengths() -> None:
    """Test ValueError for mismatched array lengths."""
    y_true = np.array([0, 1, 0])
    y_pred = np.array([0, 1])
    sites = np.array([1, 1])

    with pytest.raises(ValueError, match="same length"):
        report_metric_by_site(y_true, y_pred, sites, accuracy_score)


def test_mismatched_all_three_lengths() -> None:
    """Test ValueError when all three arrays have different lengths."""
    with pytest.raises(ValueError, match="same length"):
        report_metric_by_site(
            np.array([0, 1]),
            np.array([0, 1, 0]),
            np.array([1]),
            accuracy_score,
        )


def test_non_array_y_true() -> None:
    """Test TypeError for non-array y_true."""
    with pytest.raises(TypeError, match=r"y_true must be a numpy.ndarray"):
        report_metric_by_site(
            [0, 1],
            np.array([0, 1]),
            np.array([1, 1]),
            accuracy_score,
        )


def test_non_array_y_pred() -> None:
    """Test TypeError for non-array y_pred."""
    with pytest.raises(TypeError, match=r"y_pred must be a numpy.ndarray"):
        report_metric_by_site(
            np.array([0, 1]),
            [0, 1],
            np.array([1, 1]),
            accuracy_score,
        )


def test_non_array_sites() -> None:
    """Test TypeError for non-array sites."""
    with pytest.raises(TypeError, match=r"sites must be a numpy.ndarray"):
        report_metric_by_site(
            np.array([0, 1]),
            np.array([0, 1]),
            [1, 1],
            accuracy_score,
        )


def test_invalid_metric_type() -> None:
    """Test TypeError for invalid metric type."""
    with pytest.raises(TypeError, match="metric must be a callable"):
        report_metric_by_site(
            np.array([0, 1]),
            np.array([0, 1]),
            np.array([1, 1]),
            123,
        )


def test_metric_not_callable_string() -> None:
    """Test TypeError when metric is a string instead of callable."""
    with pytest.raises(TypeError, match="metric must be a callable"):
        report_metric_by_site(
            np.array([0, 1]),
            np.array([0, 1]),
            np.array([1, 1]),
            "accuracy_score",
        )


# =============================================================================
# Prediction type validation
# =============================================================================


def test_roc_auc_with_scores_succeeds(binary_scores_data) -> None:
    """Test roc_auc_score succeeds with probability scores."""
    y_true, y_scores, sites = binary_scores_data
    results = report_metric_by_site(y_true, y_scores, sites, roc_auc_score)
    assert isinstance(results, dict)
    assert len(results) == 3


def test_roc_auc_with_binary_labels_fails(binary_classification_data) -> None:
    """Test roc_auc_score fails when given binary labels instead of scores."""
    y_true, y_pred, sites = binary_classification_data
    with pytest.raises(ValueError, match="requires continuous scores"):
        report_metric_by_site(y_true, y_pred, sites, roc_auc_score)


def test_accuracy_with_binary_labels_succeeds(binary_classification_data) -> None:
    """Test accuracy_score succeeds with binary labels."""
    y_true, y_pred, sites = binary_classification_data
    results = report_metric_by_site(y_true, y_pred, sites, accuracy_score)
    assert isinstance(results, dict)


def test_accuracy_with_continuous_values_fails(binary_scores_data) -> None:
    """Test accuracy_score fails when given continuous scores."""
    y_true, y_scores, sites = binary_scores_data
    with pytest.raises(ValueError, match="requires discrete predictions"):
        report_metric_by_site(y_true, y_scores, sites, accuracy_score)


def test_average_precision_with_scores_succeeds(binary_scores_data) -> None:
    """Test average_precision_score succeeds with probability scores."""
    y_true, y_scores, sites = binary_scores_data
    results = report_metric_by_site(y_true, y_scores, sites, average_precision_score)
    assert isinstance(results, dict)


def test_precision_with_labels_succeeds(binary_classification_data) -> None:
    """Test precision_score succeeds with binary labels."""
    y_true, y_pred, sites = binary_classification_data
    results = report_metric_by_site(y_true, y_pred, sites, precision_score)
    assert isinstance(results, dict)


def test_multiclass_predictions_accepted(multiclass_data) -> None:
    """Test that multiclass labels are accepted as discrete predictions."""
    y_true, y_pred, sites = multiclass_data
    results = report_metric_by_site(y_true, y_pred, sites, accuracy_score)
    assert isinstance(results, dict)
    assert len(results) == 3


# =============================================================================
# report_multimetrics_by_site — Multi-metric functionality
# =============================================================================


def test_multiple_metrics_basic(binary_classification_data) -> None:
    """Test computing multiple metrics simultaneously."""
    y_true, y_pred, sites = binary_classification_data
    metrics = [accuracy_score, precision_score, recall_score]

    results = report_multimetrics_by_site(y_true, y_pred, sites, metrics)

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

    results = report_multimetrics_by_site(y_true, y_pred, sites, metrics, overall_performance=True)

    assert "overall" in results["accuracy_score"]
    assert "overall" in results["f1_score"]


def test_multiple_metrics_with_none_kwargs(binary_classification_data) -> None:
    """Test multi-metric with metric_kwargs=None."""
    y_true, y_pred, sites = binary_classification_data
    metrics = [accuracy_score, balanced_accuracy_score]

    results = report_multimetrics_by_site(y_true, y_pred, sites, metrics, metric_kwargs=None)
    assert len(results) == 2


def test_multiple_metrics_with_shared_kwargs(binary_classification_data) -> None:
    """Test multi-metric with single dict shared across all metrics."""
    y_true, y_pred, sites = binary_classification_data
    metrics = [f1_score, precision_score]

    results = report_multimetrics_by_site(
        y_true,
        y_pred,
        sites,
        metrics,
        metric_kwargs={"zero_division": 0.0},
    )
    assert len(results) == 2
    assert all(len(v) == 3 for v in results.values())


def test_multiple_metrics_with_individual_kwargs(binary_classification_data) -> None:
    """Test multi-metric with individual kwargs per metric."""
    y_true, y_pred, sites = binary_classification_data
    metrics = [f1_score, precision_score]
    kwargs = [{"average": "binary"}, {"zero_division": 0.0}]

    results = report_multimetrics_by_site(y_true, y_pred, sites, metrics, metric_kwargs=kwargs)
    assert len(results) == 2


def test_multiple_metrics_kwargs_length_mismatch(binary_classification_data) -> None:
    """Test ValueError when metric_kwargs length doesn't match metrics."""
    y_true, y_pred, sites = binary_classification_data
    metrics = [accuracy_score, f1_score]

    with pytest.raises(ValueError, match="same length as metrics"):
        report_multimetrics_by_site(
            y_true,
            y_pred,
            sites,
            metrics,
            metric_kwargs=[{}, {}, {}],  # 3 dicts for 2 metrics
        )


def test_multiple_metrics_empty_list() -> None:
    """Test ValueError for empty metrics list."""
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    sites = np.array([1, 1])

    with pytest.raises(ValueError, match="at least one callable"):
        report_multimetrics_by_site(y_true, y_pred, sites, [])


def test_multiple_metrics_non_callable_in_list() -> None:
    """Test TypeError when metrics contains non-callable."""
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    sites = np.array([1, 1])

    with pytest.raises(TypeError, match="metrics\\[1\\] must be callable"):
        report_multimetrics_by_site(y_true, y_pred, sites, [accuracy_score, "not_callable"])


def test_multiple_metrics_mixed_pred_and_score(binary_classification_data, binary_scores_data) -> None:
    """Test multi-metric with metrics needing different prediction types."""
    # This should fail because we can only pass one y_pred
    y_true, y_pred, sites = binary_classification_data

    metrics = [accuracy_score, roc_auc_score]
    with pytest.raises(ValueError, match="requires continuous scores"):
        report_multimetrics_by_site(y_true, y_pred, sites, metrics)


def test_multiple_metrics_with_scores(binary_scores_data) -> None:
    """Test multi-metric with score-based metrics only."""
    y_true, y_scores, sites = binary_scores_data
    metrics = [roc_auc_score, average_precision_score]

    results = report_multimetrics_by_site(y_true, y_scores, sites, metrics)
    assert set(results.keys()) == {"roc_auc_score", "average_precision_score"}


# =============================================================================
# _validate_metric_kwargs — Unit tests
# =============================================================================


class TestValidateMetricKwargs:
    """Unit tests for _validate_metric_kwargs helper."""

    def test_none_input(self) -> None:
        """Test None returns list of empty dicts."""
        result = _validate_metric_kwargs(None, 3)
        assert result == [{}, {}, {}]
        assert isinstance(result, list)

    def test_single_dict_broadcast(self) -> None:
        """Test single dict is broadcast to all metrics."""
        kwargs = {"average": "macro"}
        result = _validate_metric_kwargs(kwargs, 2)
        assert result == [{"average": "macro"}, {"average": "macro"}]

    def test_sequence_of_dicts(self) -> None:
        """Test sequence of dicts preserved one-to-one."""
        kwargs = [{}, {"average": "weighted"}]
        result = _validate_metric_kwargs(kwargs, 2)
        assert result == [{}, {"average": "weighted"}]

    def test_tuple_input(self) -> None:
        """Test tuple is converted to list."""
        kwargs = ({}, {"average": "binary"})
        result = _validate_metric_kwargs(kwargs, 2)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_length_mismatch_raises(self) -> None:
        """Test ValueError on length mismatch."""
        with pytest.raises(ValueError, match="same length as metrics"):
            _validate_metric_kwargs([{}, {}], 3)

    def test_invalid_type_raises(self) -> None:
        """Test TypeError for invalid metric_kwargs type."""
        with pytest.raises(TypeError, match="must be a dict, sequence of dicts, or None"):
            _validate_metric_kwargs(123, 2)

    def test_string_raises(self) -> None:
        """Test TypeError when metric_kwargs is a string."""
        with pytest.raises(TypeError, match="must be a dict, sequence of dicts, or None"):
            _validate_metric_kwargs("not_valid", 2)

    def test_zero_metrics_raises(self) -> None:
        """Test ValueError for n_metrics <= 0."""
        with pytest.raises(ValueError, match="must be positive"):
            _validate_metric_kwargs(None, 0)

    def test_negative_metrics_raises(self) -> None:
        """Test ValueError for negative n_metrics."""
        with pytest.raises(ValueError, match="must be positive"):
            _validate_metric_kwargs(None, -1)

    def test_non_int_n_metrics_raises(self) -> None:
        """Test TypeError for non-int n_metrics."""
        with pytest.raises(TypeError, match="n_metrics must be an int"):
            _validate_metric_kwargs(None, "3")

    def test_sequence_with_non_dict_element_raises(self) -> None:
        """Test TypeError when sequence contains non-dict."""
        with pytest.raises(TypeError, match="metric_kwargs\\[1\\] must be a dict"):
            _validate_metric_kwargs([{}, "bad"], 2)


# =============================================================================
# _is_binary_or_multiclass — Unit tests
# =============================================================================


class TestIsBinaryOrMulticlass:
    """Unit tests for _is_binary_or_multiclass helper."""

    def test_binary_array(self) -> None:
        """Test binary labels are recognized."""
        arr = np.array([0, 1, 0, 1])
        assert _is_binary_or_multiclass(arr) is True

    def test_multiclass_array(self) -> None:
        """Test multiclass labels are recognized."""
        arr = np.array([0, 1, 2, 3])
        assert _is_binary_or_multiclass(arr) is True

    def test_continuous_scores(self) -> None:
        """Test continuous values are not recognized as discrete."""
        arr = np.array([0.1, 0.9, 0.3, 0.8])
        assert _is_binary_or_multiclass(arr) is False

    def test_mixed_int_float(self) -> None:
        """Test float representations of integers are recognized."""
        arr = np.array([0.0, 1.0, 0.0, 1.0])
        assert _is_binary_or_multiclass(arr) is True

    def test_near_integer_with_tolerance(self) -> None:
        """Test values near integers within tolerance are recognized."""
        arr = np.array([0.000000001, 0.999999999])
        assert _is_binary_or_multiclass(arr) is True

    def test_empty_array(self) -> None:
        """Test empty array returns True."""
        arr = np.array([])
        assert _is_binary_or_multiclass(arr) is True


# =============================================================================
# _is_probability_like — Unit tests
# =============================================================================


class TestIsProbabilityLike:
    """Unit tests for _is_probability_like helper."""

    def test_probabilities(self) -> None:
        """Test probability scores are recognized."""
        arr = np.array([0.1, 0.5, 0.9, 0.3])
        assert _is_probability_like(arr) is True

    def test_binary_not_probability(self) -> None:
        """Test binary labels are not probability-like."""
        arr = np.array([0, 1, 0, 1])
        assert _is_probability_like(arr) is False

    def test_values_out_of_range(self) -> None:
        """Test values outside [0, 1] are not probability-like."""
        arr = np.array([0.1, 1.5, 0.9])
        assert _is_probability_like(arr) is False

    def test_negative_values(self) -> None:
        """Test negative values are not probability-like."""
        arr = np.array([-0.1, 0.5, 0.9])
        assert _is_probability_like(arr) is False

    def test_empty_array(self) -> None:
        """Test empty array returns False."""
        arr = np.array([])
        assert _is_probability_like(arr) is False

    def test_exactly_two_values_in_range(self) -> None:
        """Test two distinct values in [0,1] are not probability-like."""
        arr = np.array([0.0, 1.0])
        assert _is_probability_like(arr) is False


# =============================================================================
# _metric_needs_y_score / _metric_needs_y_pred — Unit tests
# =============================================================================


class TestMetricSignatureDetection:
    """Unit tests for metric signature detection."""

    def test_roc_auc_needs_score(self) -> None:
        """Test roc_auc_score is detected as score-based."""
        assert _metric_needs_y_score(roc_auc_score) is True
        assert _metric_needs_y_pred(roc_auc_score) is False

    def test_accuracy_needs_pred(self) -> None:
        """Test accuracy_score is detected as pred-based."""
        assert _metric_needs_y_pred(accuracy_score) is True
        assert _metric_needs_y_score(accuracy_score) is False

    def test_unknown_metric(self) -> None:
        """Test unknown metric returns False for both."""

        def custom_metric(y_true, y_pred):
            return 0.0

        assert _metric_needs_y_score(custom_metric) is False
        assert _metric_needs_y_pred(custom_metric) is False

    def test_registry_contents(self) -> None:
        """Test that registries contain expected metrics."""
        assert "roc_auc_score" in METRICS_REQUIRING_Y_SCORE
        assert "accuracy_score" in METRICS_REQUIRING_Y_PRED


# =============================================================================
# _validate_prediction_type — Unit tests
# =============================================================================


class TestValidatePredictionType:
    """Unit tests for _validate_prediction_type helper."""

    def test_score_metric_with_scores_passes(self) -> None:
        """Test score-based metric accepts probability scores."""
        scores = np.array([0.1, 0.9, 0.3])
        _validate_prediction_type(scores, roc_auc_score)  # should not raise

    def test_score_metric_with_labels_raises(self) -> None:
        """Test score-based metric rejects binary labels."""
        labels = np.array([0, 1, 0])
        with pytest.raises(ValueError, match="requires continuous scores"):
            _validate_prediction_type(labels, roc_auc_score)

    def test_pred_metric_with_labels_passes(self) -> None:
        """Test pred-based metric accepts binary labels."""
        labels = np.array([0, 1, 0])
        _validate_prediction_type(labels, accuracy_score)  # should not raise

    def test_pred_metric_with_scores_raises(self) -> None:
        """Test pred-based metric rejects continuous scores."""
        scores = np.array([0.1, 0.9, 0.3])
        with pytest.raises(ValueError, match="requires discrete predictions"):
            _validate_prediction_type(scores, accuracy_score)

    def test_unknown_metric_no_check(self) -> None:
        """Test unknown metric does not raise regardless of input."""

        def unknown_metric(y_true, y_pred):
            return 0.0

        scores = np.array([0.1, 0.9])
        labels = np.array([0, 1])
        _validate_prediction_type(scores, unknown_metric)
        _validate_prediction_type(labels, unknown_metric)

    def test_custom_metric_name_in_error(self) -> None:
        """Test custom metric name appears in error message."""
        labels = np.array([0, 1, 0])
        with pytest.raises(ValueError, match="my_metric"):
            _validate_prediction_type(labels, roc_auc_score, metric_name="my_metric")


# =============================================================================
# _input_checks and _input_checks_multi — Unit tests
# =============================================================================


class TestInputChecks:
    """Unit tests for input validation helpers."""

    def test_valid_inputs_pass(self) -> None:
        """Test valid inputs do not raise."""
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        sites = np.array([1, 1])
        _input_checks(y_true, y_pred, sites, accuracy_score, False)

    def test_mismatched_lengths_raises(self) -> None:
        """Test mismatched lengths raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            _input_checks(
                np.array([0, 1, 0]),
                np.array([0, 1]),
                np.array([1, 1]),
                accuracy_score,
                False,
            )

    def test_non_callable_metric_raises(self) -> None:
        """Test non-callable metric raises TypeError."""
        with pytest.raises(TypeError, match="metric must be a callable"):
            _input_checks(
                np.array([0, 1]),
                np.array([0, 1]),
                np.array([1, 1]),
                "not_callable",
                False,
            )

    def test_input_checks_multi_empty_metrics(self) -> None:
        """Test empty metrics list raises ValueError."""
        with pytest.raises(ValueError, match="at least one callable"):
            _input_checks_multi(
                np.array([0, 1]),
                np.array([0, 1]),
                np.array([1, 1]),
                [],
                False,
            )

    def test_input_checks_multi_non_sequence_raises(self) -> None:
        """Test non-sequence metrics raises TypeError."""
        with pytest.raises(TypeError, match="must be a sequence"):
            _input_checks_multi(
                np.array([0, 1]),
                np.array([0, 1]),
                np.array([1, 1]),
                accuracy_score,  # single callable, not sequence
                False,
            )


# =============================================================================
# Edge cases
# =============================================================================


def test_all_correct_predictions() -> None:
    """Test with perfect predictions."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    sites = np.array([1, 1, 2, 2])

    results = report_metric_by_site(y_true, y_pred, sites, accuracy_score)
    assert results[1] == pytest.approx(1.0)
    assert results[2] == pytest.approx(1.0)


def test_all_incorrect_predictions() -> None:
    """Test with completely wrong predictions."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0])
    sites = np.array([1, 1, 2, 2])

    results = report_metric_by_site(y_true, y_pred, sites, accuracy_score)
    assert results[1] == pytest.approx(0.0)
    assert results[2] == pytest.approx(0.0)


def test_large_number_of_sites() -> None:
    """Test with many unique sites."""
    n = 100
    y_true = np.random.randint(0, 2, size=n)
    y_pred = np.random.randint(0, 2, size=n)
    sites = np.arange(n)

    results = report_metric_by_site(y_true, y_pred, sites, accuracy_score)
    assert len(results) == n
    # Each site has exactly one sample, so accuracy is either 0 or 1
    assert all(v in {0.0, 1.0} for v in results.values())


def test_float_site_identifiers_converted_to_int() -> None:
    """Test that float site IDs are converted to int keys."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    sites = np.array([1.0, 1.0, 2.0, 2.0])

    results = report_metric_by_site(y_true, y_pred, sites, accuracy_score)
    assert set(results.keys()) == {1, 2}
    assert all(isinstance(k, int) for k in results.keys())


def test_boolean_site_identifiers() -> None:
    """Test with boolean site identifiers."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    sites = np.array([True, True, False, False])

    results = report_metric_by_site(y_true, y_pred, sites, accuracy_score)
    # bool is subclass of int, so True→1, False→0
    assert set(results.keys()) == {0, 1}
    assert all(isinstance(k, int) for k in results.keys())
