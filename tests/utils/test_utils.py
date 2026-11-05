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


@pytest.mark.parametrize(("n_targets", "n_samples", "n_regressors"), [(1, 1, 1), (2, 5, 2), (7, 50, 4)])
def test_solve_ols_shape(n_targets, n_samples, n_regressors):
    """X is (n_targets, n_samples); the solution is (n_regressors, n_targets)."""
    rng = np.random.default_rng(0)
    result = solve_ordinary_least_squares(rng.normal(size=(n_targets, n_samples)), rng.normal(size=(n_samples, n_regressors)))
    assert result.shape == (n_regressors, n_targets)
    assert np.isfinite(result).all()


def test_solve_ols_matches_normal_equations():
    """Full-rank design: the solution equals (D'D)^-1 D'X, without a warning."""
    rng = np.random.default_rng(0)
    design = rng.normal(size=(50, 4))
    X = rng.normal(size=(7, 50))
    expected = np.linalg.solve(design.T @ design, design.T @ X.T)
    np.testing.assert_allclose(solve_ordinary_least_squares(X, design), expected, atol=1e-12)


@pytest.mark.parametrize(
    "design",
    [
        np.ones((4, 2)),  # two identical columns
        np.column_stack([np.arange(6.0), np.ones(6), np.arange(6.0) + 1]),  # third = first + second
    ],
)
def test_solve_ols_rank_deficient(monkeypatch, design):
    """Rank-deficient design: minimum-norm solution (as the pseudo-inverse), with a warning."""
    warning_calls = []
    monkeypatch.setattr("uniharmony._utils.logger.warning", lambda msg, *a, **k: warning_calls.append(msg))
    X = np.random.default_rng(1).normal(size=(3, design.shape[0]))
    result = solve_ordinary_least_squares(X, design)
    np.testing.assert_allclose(result, np.linalg.pinv(design) @ X.T, atol=1e-10)
    assert len(warning_calls) == 1
    assert "rank deficient" in warning_calls[0]


def test_solve_ols_nearly_collinear_design():
    """A nearly collinear design (Cholesky of D'D still succeeds) is solved stably."""
    rng = np.random.default_rng(2)
    base = rng.normal(size=(30, 2))
    design = np.column_stack([base, base[:, 0] + base[:, 1]])
    X = rng.normal(size=(3, 30))
    np.testing.assert_allclose(solve_ordinary_least_squares(X, design), np.linalg.pinv(design) @ X.T, atol=1e-10)


@pytest.mark.parametrize(
    ("X", "design", "error_match"),
    [
        (np.ones((2, 3)), np.ones(3), "design must be a 2D array"),
        (np.ones((2, 4)), np.ones((3, 2)), r"X must have shape \(n_targets, 3\)"),
        (np.ones(3), np.ones((3, 2)), r"X must have shape \(n_targets, 3\)"),
    ],
)
def test_solve_ols_validation_errors(X, design, error_match):
    """Incompatible shapes raise a clear error."""
    with pytest.raises(ValueError, match=error_match):
        solve_ordinary_least_squares(X, design)


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
