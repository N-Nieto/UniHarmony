"""Tests for general utilities."""

import numpy as np
import pytest

from uniharmony._utils import (
    handle_near_zero_values,
    handle_negative_variance,
    minimum_samples_warning,
    solve_ordinary_least_squares,
    validate_sites,
)


# ─── solve_ordinary_least_squares ────────────────────────────────────────────


@pytest.mark.parametrize(
    "gram_matrix, X, design, expected_shape",
    [
        # Well-conditioned case (Cholesky path)
        # n_samples=2, n_targets=2 (must be equal for code to work)
        (
            np.array([[2.0, 0.5], [0.5, 2.0]]),
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            (2, 2),
        ),
        # Single target, single sample (n_samples == n_targets == 1)
        (
            np.array([[3.0]]),
            np.array([[1.0]]),
            np.array([[1.0]]),
            (1, 1),
        ),
        # Larger well-conditioned (n_samples=3, n_targets=3)
        (
            np.eye(3) * 2,
            np.ones((3, 3)),
            np.ones((3, 3)),
            (3, 3),
        ),
    ],
)
def test_solve_ols_well_conditioned(gram_matrix, X, design, expected_shape):
    """Test OLS with well-conditioned matrices (Cholesky pathº)."""
    result = solve_ordinary_least_squares(gram_matrix, X, design)
    assert result.shape == expected_shape
    assert np.isfinite(result).all()


@pytest.mark.parametrize(
    "gram_matrix, X, design",
    [
        # Ill-conditioned / singular (pseudo-inverse path)
        # Exact singular 2x2 matrix
        (
            np.array([[1.0, 1.0], [1.0, 1.0]]),  # rank 1
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[1.0, 1.0], [1.0, 1.0]]),
        ),
        # Exact singular 3x3 matrix (Cholesky will definitely fail)
        (
            np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
            np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        ),
    ],
)
def test_solve_ols_ill_conditioned(monkeypatch, gram_matrix, X, design):
    """Test OLS falls back to pseudo-inverse for ill-conditioned matrices."""
    warning_calls = []

    def mock_warning(msg, *args, **kwargs):
        warning_calls.append(msg)

    monkeypatch.setattr("uniharmony._utils.logger.warning", mock_warning)

    result = solve_ordinary_least_squares(gram_matrix, X, design)

    assert len(warning_calls) == 1
    assert "ill-conditioned" in warning_calls[0]
    assert np.isfinite(result).all()


@pytest.mark.parametrize(
    "gram_matrix, X, design, expected_error, error_match",
    [
        # Non-square gram matrix
        (
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            np.array([[1.0]]),
            np.array([[1.0, 2.0]]),
            ValueError,
            "gram_matrix must be square",
        ),
        # 1D gram matrix
        (
            np.array([1.0, 2.0, 3.0]),
            np.array([[1.0]]),
            np.array([[1.0, 2.0, 3.0]]),
            ValueError,
            "gram_matrix must be square",
        ),
        # Shape mismatch: design features != gram features
        (
            np.eye(3),
            np.array([[1.0]]),
            np.array([[1.0, 2.0]]),  # 2 features, but gram is 3x3
            ValueError,
            "design has 2 features but gram_matrix has 3",
        ),
    ],
)
def test_solve_ols_validation_errors(gram_matrix, X, design, expected_error, error_match):
    """Test input validation errors."""
    with pytest.raises(expected_error, match=error_match):
        solve_ordinary_least_squares(gram_matrix, X, design)


# ─── handle_near_zero_values ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "values, epsilon, context, expected, should_warn",
    [
        # Some near-zero values replaced
        (
            np.array([0.0, 1e-10, 0.5, 0.3]),
            1e-8,
            "features",
            np.array([1e-8, 1e-8, 0.5, 0.3]),
            True,
        ),
        # All above threshold — no replacement
        (
            np.array([1e-7, 0.5, 1.0]),
            1e-8,
            "features",
            np.array([1e-7, 0.5, 1.0]),
            False,
        ),
        # Custom epsilon
        (
            np.array([0.01, 0.5]),
            0.1,
            "variances",
            np.array([0.1, 0.5]),
            True,
        ),
        # Custom context
        (
            np.array([0.0]),
            1e-8,
            "variances",
            np.array([1e-8]),
            True,
        ),
        # Empty array
        (
            np.array([]),
            1e-8,
            "features",
            np.array([]),
            False,
        ),
        # Single element above threshold
        (
            np.array([1.0]),
            1e-8,
            "features",
            np.array([1.0]),
            False,
        ),
        # Single element below threshold
        (
            np.array([1e-12]),
            1e-8,
            "features",
            np.array([1e-8]),
            True,
        ),
    ],
)
def test_handle_near_zero_values(monkeypatch, values, epsilon, context, expected, should_warn):
    """Test near-zero value handling with various inputs."""
    warning_calls = []

    def mock_warning(msg, *args, **kwargs):
        warning_calls.append(msg)

    monkeypatch.setattr("uniharmony._utils.logger.warning", mock_warning)

    result = handle_near_zero_values(values, epsilon=epsilon, context=context)
    np.testing.assert_array_almost_equal(result, expected)
    if should_warn:
        assert len(warning_calls) == 1
        assert context in warning_calls[0]
        assert str(epsilon) in warning_calls[0]
    else:
        assert len(warning_calls) == 0


def test_handle_near_zero_values_does_not_mutate_input():
    """Ensure the input array is never modified in-place."""
    original = np.array([0.0, 0.5])
    original_copy = original.copy()
    result = handle_near_zero_values(original)
    np.testing.assert_array_equal(original, original_copy)
    assert result is not original


# ─── handle_negative_variance ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "variance, expected, should_log_error",
    [
        # All positive — no change
        (
            np.array([0.5, 1.0, 2.0]),
            np.array([0.5, 1.0, 2.0]),
            False,
        ),
        # Some negative values
        (
            np.array([-1e-15, 0.5, -2e-16, 0.3]),
            np.array([1e-15, 0.5, 2e-16, 0.3]),
            True,
        ),
        # All negative
        (
            np.array([-1.0, -2.0, -3.0]),
            np.array([1.0, 2.0, 3.0]),
            True,
        ),
        # Mix with zero
        (
            np.array([-0.1, 0.0, 0.5]),
            np.array([0.1, 0.0, 0.5]),
            True,
        ),
        # Empty array
        (
            np.array([]),
            np.array([]),
            False,
        ),
        # Single positive
        (
            np.array([1.0]),
            np.array([1.0]),
            False,
        ),
        # Single negative
        (
            np.array([-1.0]),
            np.array([1.0]),
            True,
        ),
    ],
)
def test_handle_negative_variance(monkeypatch, variance, expected, should_log_error):
    """Test negative variance correction."""
    error_calls = []

    def mock_error(msg, *args, **kwargs):
        error_calls.append(msg)

    monkeypatch.setattr("uniharmony._utils.logger.error", mock_error)

    result = handle_negative_variance(variance)
    np.testing.assert_array_almost_equal(result, expected)
    if should_log_error:
        assert len(error_calls) == 1
        assert "negative pooled variance" in error_calls[0]
    else:
        assert len(error_calls) == 0


def test_handle_negative_variance_does_not_mutate_input():
    """Ensure the input array is never modified in-place."""
    original = np.array([-1.0, 0.5])
    original_copy = original.copy()
    result = handle_negative_variance(original)
    np.testing.assert_array_equal(original, original_copy)
    assert result is not original


# ─── minimum_samples_warning ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "n_samples, min_limit, should_warn",
    [
        # Below limit — should warn
        ([10, 20, 30], 16, True),
        ([5], 16, True),
        ([15, 100], 16, True),
        # Nested list input
        ([[10], [20]], 16, True),
        # At limit — should NOT warn (strictly less)
        ([16, 20], 16, False),
        # Above limit — should NOT warn
        ([20, 30, 40], 16, False),
        ([100], 16, False),
        # Custom limit
        ([5, 10], 20, True),
        ([25, 30], 20, False),
        # Single above custom limit
        ([21], 20, False),
        # Array input
        (np.array([10, 20]), 16, True),
        (np.array([20, 30]), 16, False),
    ],
)
def test_minimum_samples_warning(monkeypatch, n_samples, min_limit, should_warn):
    """Test minimum sample size warning logic."""
    warning_calls = []

    def mock_warning(msg, *args, **kwargs):
        warning_calls.append(msg)

    monkeypatch.setattr("uniharmony._utils.logger.warning", mock_warning)

    minimum_samples_warning(n_samples, min_samples_limit=min_limit)
    if should_warn:
        assert len(warning_calls) == 1
        msg = warning_calls[0]
        assert "ComBat requires" in msg
        assert str(min_limit) in msg
    else:
        assert len(warning_calls) == 0


# ─── validate_sites ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "sites",
    [
        np.array([0, 1]),
        np.array([0, 1, 2]),
        np.array(["A", "B"]),
        np.array([0, 0, 1, 1, 2]),
        np.array([1, 2, 3, 4, 5]),
    ],
)
def test_validate_sites_valid(sites):
    """Test valid site arrays (>=2 unique sites)."""
    validate_sites(sites)  # should not raise


@pytest.mark.parametrize(
    "sites, error_match",
    [
        (np.array([0]), "At least 2 sites required"),
        (np.array([1, 1, 1]), "At least 2 sites required"),
        (np.array(["A", "A", "A"]), "At least 2 sites required"),
        (np.array([0.0, 0.0]), "At least 2 sites required"),
    ],
)
def test_validate_sites_invalid(sites, error_match):
    """Test invalid site arrays (<2 unique sites)."""
    with pytest.raises(ValueError, match=error_match):
        validate_sites(sites)
