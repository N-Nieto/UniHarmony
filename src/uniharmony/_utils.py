"""General utilities."""

import numpy as np
import numpy.typing as npt

from uniharmony._verbose import get_logger


__all__ = [
    "handle_near_zero_values",
    "handle_negative_variance",
    "minimum_samples_warning",
    "solve_ordinary_least_squares",
    "validate_sites",
]


logger = get_logger(src="utils")


def solve_ordinary_least_squares(X: npt.NDArray, design: npt.NDArray) -> npt.NDArray:
    """Fit ordinary least squares for many targets at once.

    Solves ``beta = argmin ||X.T - design @ beta||^2`` with an SVD-based
    least-squares solver, which is numerically stable for any design. For a
    full-rank design this is the usual estimate
    ``(design.T @ design)^-1 @ design.T @ X.T``; for a rank-deficient design
    (e.g., a covariate collinear with the sites) it is the minimum-norm
    solution, and a warning reports the condition number of the design.

    Parameters
    ----------
    X : array, shape (n_targets, n_samples)
        Target values, one row per target.
    design : array, shape (n_samples, n_regressors)
        Design matrix.

    Returns
    -------
    solution : ndarray, shape (n_regressors, n_targets)
        OLS coefficients.

    Raises
    ------
    ValueError
        If the shapes of ``X`` and ``design`` are incompatible.

    """
    X = np.asarray(X)
    design = np.asarray(design)
    if design.ndim != 2:
        raise ValueError(f"design must be a 2D array, got shape {design.shape}")
    if X.ndim != 2 or X.shape[1] != design.shape[0]:
        raise ValueError(f"X must have shape (n_targets, {design.shape[0]}), got {X.shape}")

    solution, _, rank, singular_values = np.linalg.lstsq(design, X.T, rcond=None)
    if rank < design.shape[1]:
        condition_number = singular_values[0] / singular_values[-1] if singular_values[-1] > 0 else np.inf
        logger.warning(
            f"Design matrix is rank deficient (rank {rank} < {design.shape[1]} columns, "
            f"condition number: {condition_number:.2e}). Using the minimum-norm least-squares solution. "
            "Consider removing redundant covariates or checking for perfect "
            "correlation between covariates and sites."
        )
    return solution


def handle_near_zero_values(
    values: npt.ArrayLike,
    epsilon: float = 1e-8,
    context: str = "features",
) -> npt.NDArray:
    """Replace near-zero values with epsilon to prevent division by zero.

    Creates a copy of the input array and replaces values below epsilon
    with epsilon. This is useful for variance estimates that might be
    zero or near-zero due to constant features or numerical errors.

    Parameters
    ----------
    values : array-like
        Input array that may contain near-zero values.
    epsilon : float, optional (default 1e-8)
        Threshold below which values are replaced. Also the replacement value.
    context : str, optional (default "values")
        Description of what the values represent (for logging).

    Returns
    -------
    safe_values : ndarray
        Copy of input with near-zero values replaced by epsilon.
        Always returns a new array, never modifies input in place.

    Examples
    --------
    >>> var = np.array([0.0, 1e-10, 0.5, 0.3])
    >>> safe_var = handle_near_zero_values(var, epsilon=1e-8)
    >>> safe_var
    array([1.e-08, 1.e-08, 5.e-01, 3.e-01])

    """
    # Ensure we're working with a copy, not modifying input
    values_arr = np.asarray(values, dtype=np.float64).copy()
    # Find near-zero values and replace with epsilon
    zero_var_mask = values_arr < epsilon
    if np.any(zero_var_mask):
        n_zero_var = np.sum(zero_var_mask)
        logger.warning(
            f"{n_zero_var} {context} have near-zero values. "
            f"These steps added a small epsilon: {epsilon} to avoid division by zero. "
            "Consider removing constant features."
        )
        # Replace near-zero variances with epsilon to prevent numerical issues during standardization
        values_arr[zero_var_mask] = epsilon  # Small epsilon
    return values_arr


def handle_negative_variance(variance: npt.ArrayLike) -> npt.NDArray:
    """Handle negative variance values caused by numerical errors.

    Variance should always be non-negative, but numerical errors in
    computations can produce small negative values. This function
    corrects them by taking absolute value or clipping to epsilon.

    Parameters
    ----------
    variance : array-like
        Variance estimates that may contain negative values due to
        numerical errors.

    Returns
    -------
    safe_variance : ndarray
        Corrected variance with negative values replaced by absolute
        values (and optionally clipped to epsilon).

    Examples
    --------
    >>> var = np.array([-1e-15, 0.5, -2e-16, 0.3])
    >>> safe_var = handle_negative_variance(var)
    >>> safe_var
    array([1.e-15, 5.e-01, 2.e-16, 3.e-01])

    """
    variance_arr = np.asarray(variance, dtype=np.float64).copy()

    # Find negative values
    negative_mask = variance_arr < 0
    if np.any(negative_mask):
        n_neg = np.sum(negative_mask)
        logger.error(
            f"{n_neg} features have negative pooled variance due to numerical errors. "
            "Setting to absolute value, but check your data for issues."
        )
        variance_arr = np.abs(variance_arr)
    return variance_arr


def minimum_samples_warning(n_samples_per_site: list[list[int]] | npt.NDArray, min_samples_limit: int = 16) -> None:
    """Check if sites meet minimum sample size requirements.

    Parameters
    ----------
    n_samples_per_site : list or array of int
        Number of samples per site.
    min_samples_limit : int, optional (default 16)
        Minimum number of samples required. Below this is considered critical.

    Returns
    -------
    None

    """
    n_samples_per_site = np.asarray(n_samples_per_site)

    min_samples = int(np.min(n_samples_per_site))
    if min_samples < min_samples_limit:
        logger.warning(
            f"Site with only {min_samples} samples detected. "
            "ComBat requires 16-32+ subjects per site for reliable harmonization. "
            f"Specified minimum is {min_samples_limit}. "
            "Results may be unstable or overfit."
        )


def validate_sites(sites: npt.NDArray) -> None:
    """Validate sites.

    Parameters
    ----------
    sites : array
        The sites.

    Raises
    ------
    ValueError
        If fewer than 2 samples or a single site is provided.

    """
    if len(sites) < 2:
        raise ValueError(f"At least 2 samples required, got n_samples = {len(sites)}.")
    if len(np.unique(sites)) < 2:
        raise ValueError("At least 2 sites required")
