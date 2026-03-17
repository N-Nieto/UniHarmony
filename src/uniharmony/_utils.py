"""General utilities."""

import numpy as np
import numpy.typing as npt
import structlog


__all__ = [
    "handle_near_zero_values",
    "handle_negative_variance",
    "minimum_samples_warning",
    "solve_ordinary_least_squares",
    "validate_covariates",
]


logger = structlog.get_logger()


def solve_ordinary_least_squares(gram_matrix, X: npt.ArrayLike, design: npt.NDArray) -> npt.NDArray:
    """Solve Ordinary Least Squares using normal equations with stability checks.

    Solves: beta = (X_design^T @ X_design)^(-1) @ X_design^T @ X_data

    Uses Cholesky decomposition for well-conditioned matrices and falls back
    to pseudo-inverse for ill-conditioned cases.

    Parameters
    ----------
    gram_matrix : array, shape (n_features, n_features)
        The Gram matrix X_design^T @ X_design.
    X : array-like, shape (n_samples, n_targets)
        Target values (transposed to (n_targets, n_samples) internally).
    design : array, shape (n_samples, n_features)
        Design matrix X_design.
    check_condition : bool, optional (default True)
        Whether to compute and log the condition number.

    Returns
    -------
    solution : ndarray, shape (n_features, n_targets)
        OLS coefficients.
    condition_number : float
        Condition number of the Gram matrix (useful for diagnostics).

    Raises
    ------
    ValueError
        If gram_matrix is not square or shapes are incompatible.
    np.linalg.LinAlgError
        If both Cholesky and pseudo-inverse fail.

    Notes
    -----
    The condition number is computed as the ratio of largest to smallest
    singular value. Large condition numbers (>1e10) indicate potential
    numerical instability.

    """
    # Validate inputs
    gram_matrix = np.asarray(gram_matrix)
    design = np.asarray(design)
    X = np.asarray(X)

    if gram_matrix.ndim != 2 or gram_matrix.shape[0] != gram_matrix.shape[1]:
        raise ValueError(f"gram_matrix must be square 2D array, got shape {gram_matrix.shape}")

    n_features = gram_matrix.shape[0]
    if design.shape[1] != n_features:
        raise ValueError(f"design has {design.shape[1]} features but gram_matrix has {n_features}")

    try:
        # Quick check: try Cholesky decomposition (only works for well-conditioned SPD matrices)
        # This is faster than computing full condition number
        np.linalg.cholesky(gram_matrix)
        # If Cholesky succeeds, matrix is well-conditioned, use fast inv
        solution = np.linalg.inv(gram_matrix) @ design.T @ X.T
    except np.linalg.LinAlgError:
        # Cholesky failed - matrix is singular or ill-conditioned
        # Fall to pseudo-inverse for numerical stability and reise a warning
        cond_num = np.linalg.cond(gram_matrix)
        logger.warning(
            f"Design matrix is ill-conditioned (condition number: {cond_num:.2e}). "
            "Using pseudo-inverse for numerical stability. "
            "Consider removing redundant covariates or checking for perfect "
            "correlation between covariates and sites."
        )
        solution = np.linalg.pinv(design) @ X.T
    return solution


def handle_near_zero_values(
    values: npt.NDArray,
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


def handle_negative_variance(variance: npt.NDArray) -> npt.NDArray:
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


def minimum_samples_warning(n_samples_per_site: list[int] | npt.NDArray, min_samples_limit: int = 16) -> None:
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
    min_samples = np.min(n_samples_per_site)
    if min_samples < min_samples_limit:
        logger.warning(
            f"Site with only {min_samples} samples detected. "
            "ComBat requires 16-32+ subjects per site for reliable harmonization. "
            f"Specified minimum is {min_samples_limit}. "
            "Results may be unstable or overfit."
        )


def validate_covariates(covariates: npt.ArrayLike | None, n_samples: int, name: str) -> npt.NDArray | None:
    if covariates is not None:
        covariates = np.asarray(covariates)
        if covariates.ndim == 1:
            covariates = covariates.reshape(-1, 1)
        if covariates.shape[0] != n_samples:
            raise ValueError(f"{name} has {covariates.shape[0]} samples but sites has {n_samples}")
        return covariates
    return None


def validate_sites(sites):
    # Ensure sites is 2D for sklearn (n_samples, 1)
    if sites.ndim == 1:
        sites = sites.reshape(-1, 1)
    elif sites.ndim != 2 or sites.shape[1] != 1:
        raise ValueError(f"sites must be shape (n_samples,) or (n_samples, 1), got {sites.shape}")
    return sites
