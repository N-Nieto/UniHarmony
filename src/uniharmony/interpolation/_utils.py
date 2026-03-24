"""Utility functions for interpolation-based harmonization methods."""

from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import structlog
from imblearn.over_sampling import (
    ADASYN,
    SMOTE,
    SVMSMOTE,
    BorderlineSMOTE,
    KMeansSMOTE,
    RandomOverSampler,
)
from numpy.typing import ArrayLike, NDArray
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array


__all__ = [
    "class_representation_checks",
    "create_interpolator",
    "interpolate_sample",
    "interpolate_to_average",
    "sites_sanity_checks",
    "validate_alpha",
    "validate_covariates",
    "validate_k_parameter",
]

logger = structlog.get_logger()


def create_interpolator(
    name: str, random_state: int | np.random.RandomState = 23, **kwargs
) -> type[SMOTE] | type[BorderlineSMOTE] | type[SVMSMOTE] | type[ADASYN] | type[KMeansSMOTE] | type[RandomOverSampler]:
    """Create an imblearn interpolator based on a string name.

    Parameters
    ----------
    name : str
        Name of interpolator.
    random_state : int or RandomState instance, optional (default 23)
        The seed of the pseudo random number generator or RandomState for
        reproducibility.
    **kwargs : dict
        Extra keyword arguments for the interpolator.

    Returns
    -------
    object
        Initialised interpolator instance.

    Raises
    ------
    ValueError
        If ``name`` is invalid.

    """
    random_state = check_random_state(random_state)
    mapping = {
        "smote": SMOTE,
        "borderline-smote": BorderlineSMOTE,
        "svm-smote": SVMSMOTE,
        "adasyn": ADASYN,
        "kmeans-smote": KMeansSMOTE,
        "random": RandomOverSampler,
    }

    name = name.lower()
    if name not in mapping:
        raise ValueError(f"Unsupported interpolator: {name}")

    return mapping[name](random_state=random_state, sampling_strategy="not majority", **kwargs)


def sites_sanity_checks(x: npt.NDArray, sites: npt.NDArray) -> None:
    """Sanity checks for site array.

    Parameters
    ----------
    x : array
        Features.
    sites : array
        Sites.

    Raises
    ------
    ValueError
        If ``x`` and ``sites`` have different length or
        if single site is provided.

    """
    if x.shape[0] != sites.shape[0]:
        raise ValueError("X and sites must have same length")

    if len(np.unique(sites)) < 2:
        raise ValueError("At least two sites required")


def class_representation_checks(y: npt.NDArray, sites: npt.NDArray) -> None:
    """Check that each site has at least two classes.

    Parameters
    ----------
    y : array
        Targets.
    sites : array
        Sites.

    Raises
    ------
    ValueError
        If ``sites`` have single class.

    """
    for site in np.unique(sites):
        if len(np.unique(y[sites == site])) < 2:
            raise ValueError(f"Site {site} has only one class; cannot resample.")


def validate_covariates(
    n_samples: int,
    categorical_covariate: ArrayLike | None,
    continuous_covariate: ArrayLike | None,
    covariate_tolerance: ArrayLike | None,
    *,
    allow_nan: bool = False,
) -> tuple[NDArray[Any] | None, NDArray[np.float64] | None, NDArray[np.float64] | None]:
    """Validate covariate arrays and tolerance.

    Validates shapes and ensures all values are finite (unless ``allow_nan``
    is True). Processes tolerance into a properly shaped array.

    Parameters
    ----------
    n_samples : int
        Expected number of samples (from X).
    categorical_covariate : array-like or None
        Categorical covariates with shape (n_samples, n_categorical).
    continuous_covariate : array-like or None
        Continuous covariates with shape (n_samples, n_continuous).
    covariate_tolerance : array-like or None
        Tolerance values for continuous covariates. If None and
        continuous_covariate is provided, defaults to zeros (exact matching).
    allow_nan : bool, default=False
        If False, raises ValueError if NaN or infinite values are found.

    Returns
    -------
    cat_cov : ndarray or None
        Validated categorical covariates with shape (n_samples, n_categorical).
    cont_cov : ndarray or None
        Validated continuous covariates with shape (n_samples, n_continuous).
    cov_tolerance_arr : ndarray or None
        Validated tolerance array with shape (n_continuous,).

    Raises
    ------
    ValueError
        If shapes are incompatible, if tolerance shape doesn't match
        continuous covariates, or if NaN/Inf found when not allowed.

    """
    cat_cov: NDArray[Any] | None = None
    cont_cov: NDArray[np.float64] | None = None
    tol_arr: NDArray[np.float64] | None = None

    if categorical_covariate is not None:
        cat_cov = check_array(categorical_covariate, dtype=None, ensure_all_finite=not allow_nan)
        if cat_cov.shape[0] != n_samples:
            raise ValueError(f"categorical_covariate has {cat_cov.shape[0]} samples, but X has {n_samples} samples")
        logger.debug(f"[ISMI] Using {cat_cov.shape[1]} categorical covariates")

    if continuous_covariate is not None:
        cont_cov = check_array(continuous_covariate, ensure_all_finite=not allow_nan)
        if cont_cov.shape[0] != n_samples:
            raise ValueError(
                f"continuous_covariate has {cont_cov.shape[0]} samples, but X has {n_samples} samples."
                "Both must have same number of samples."
            )

        if covariate_tolerance is None:
            tol_arr = np.zeros(cont_cov.shape[1])
            logger.debug("[ISMI] No tolerance specified, using exact matching")
        else:
            tol_arr = np.asarray(covariate_tolerance, dtype=np.float64).flatten()
            if tol_arr.shape[0] != cont_cov.shape[1]:
                raise ValueError(
                    f"covariate_tolerance has {tol_arr.shape[0]} values,"
                    f"but continuous_covariate has {cont_cov.shape[1]} columns."
                    "One tolerance value per continuous covariate (column) is required."
                )

        logger.debug(f"[ISMI] Using {cont_cov.shape[1]} continuous covariates with tolerance: {tol_arr}")

    elif covariate_tolerance is not None:
        raise ValueError(
            "covariate_tolerance provided but continuous_covariate is None. Cannot use tolerance without continuous covariates."
        )

    return cat_cov, cont_cov, tol_arr


def validate_alpha(
    alpha: float | tuple[float, float],
) -> tuple[float, float]:
    """Validate and parse the alpha parameter.

    Parameters
    ----------
    alpha : float or tuple of float
        Interpolation weight specification. If float, used as constant.
        If tuple (min, max), defines uniform sampling range.

    Returns
    -------
    alpha_min : float
        Minimum alpha value.
    alpha_max : float
        Maximum alpha value.

    Raises
    ------
    ValueError
        If alpha values are outside [0, 1] or if tuple doesn't satisfy
        0 <= min <= max <= 1.

    """
    if isinstance(alpha, (int, float)):
        alpha_val = float(alpha)
        if not (0 <= alpha_val <= 1):
            logger.warning(f"[ISMI] alpha={alpha_val} is outside [0, 1]")
        alpha_min = alpha_val
        alpha_max = alpha_val
    elif isinstance(alpha, (tuple, list)) and len(alpha) == 2:
        alpha_min, alpha_max = float(alpha[0]), float(alpha[1])
        if not (0 <= alpha_min <= alpha_max <= 1):
            raise ValueError(f"alpha range must satisfy 0 <= min <= max <= 1, got ({alpha_min}, {alpha_max})")
    else:
        raise ValueError("alpha must be a float or a tuple of two floats (min, max)")

    return alpha_min, alpha_max


def validate_k_parameter(
    k: int | Literal["max", "average"] = 1,
) -> tuple[int | Literal["max", "average"], bool]:
    """Validate the k parameter.

    Parameters
    ----------
    k : int or str
        Number of matches specification. Can be an integer >= 1,
        "max" for all matches, or "average" for averaging.

    Returns
    -------
    k_val : int or str
        Validated k value.
    use_average : bool
        True if k is "average".

    Raises
    ------
    ValueError
        If k is not a valid integer >= 1 or not one of the allowed strings.

    """
    if isinstance(k, str):
        k_lower = k.lower()
        if k_lower == "max":
            k_val = "max"
            use_average = False
        elif k_lower == "average":
            k_val = "average"
            use_average = True
        else:
            raise ValueError(f"k must be an int >= 1, 'max', or 'average', got '{k}'")

    elif isinstance(k, int):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        k_val = k
        use_average = False
    elif isinstance(k, float) and k.is_integer() and k >= 1:
        logger.warning(f"k={k} is a float but represents an integer; converting to int")
        k_val = int(k)
        use_average = False
    else:
        raise ValueError(f"k must be int or str, got {type(k).__name__}")

    return k_val, use_average


def interpolate_sample(
    x_src: NDArray[np.float64],
    y_src: Any,
    x_dst: NDArray[np.float64],
    y_dst: Any,
    alpha: float,
    y_kind: str,
) -> tuple[NDArray[np.float64], Any]:
    """Interpolate between a single source and destination sample.

    Parameters
    ----------
    x_src : ndarray of shape (n_features,)
        Source features.
    y_src : scalar
        Source target.
    x_dst : ndarray of shape (n_features,)
        Destination features.
    y_dst : scalar
        Destination target.
    alpha : float
        Interpolation weight in [0, 1].
    y_kind : str
        Data type kind of y ('i', 'u', 'b' for classification, 'f' for regression).

    Returns
    -------
    x_new : ndarray
        Interpolated features.
    y_new : scalar
        Interpolated or copied target.

    """
    x_new = x_src + alpha * (x_dst - x_src)

    if y_kind in "iub":  # Classification: keep source label
        y_new = y_src
    elif y_kind == "f":  # Regression: interpolate target
        y_new = y_src + alpha * (y_dst - y_src)
    else:
        raise ValueError(f"Unsupported y_kind: {y_kind}")

    return x_new, y_new


def interpolate_to_average(
    x_src: NDArray[np.float64],
    y_src: Any,
    X_dst: NDArray[np.float64],
    y_dst: NDArray[Any],
    alpha: float,
    y_kind: str,
) -> tuple[NDArray[np.float64], Any]:
    """Interpolate source sample with average of multiple destination samples.

    Parameters
    ----------
    x_src : ndarray of shape (n_features,)
        Source features.
    y_src : scalar
        Source target.
    X_dst : ndarray of shape (n_matches, n_features)
        Destination features.
    y_dst : ndarray of shape (n_matches,)
        Destination targets.
    alpha : float
        Interpolation weight.
    y_kind : str
        Data type kind of y.

    Returns
    -------
    x_new : ndarray
        Interpolated features.
    y_new : scalar
        Interpolated or copied target.

    """
    x_avg = np.mean(X_dst, axis=0)
    x_new = x_src + alpha * (x_avg - x_src)

    if y_kind in "iub":  # Classification
        y_new = y_src
    elif y_kind == "f":  # Regression
        y_avg = np.mean(y_dst)
        y_new = y_src + alpha * (y_avg - y_src)
    else:
        raise ValueError(f"Unsupported y_kind: {y_kind}")

    return x_new, y_new
