"""Utility functions for interpolation-based harmonization methods."""

from typing import Any

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
    "validate_covariates",
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
