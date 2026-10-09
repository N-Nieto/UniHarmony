"""Utility functions for interpolation-based harmonization methods."""

import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
import structlog
from imblearn.base import SamplerMixin
from imblearn.over_sampling import (
    ADASYN,
    SMOTE,
    SVMSMOTE,
    BorderlineSMOTE,
    KMeansSMOTE,
    RandomOverSampler,
)
from imblearn.under_sampling import (
    ClusterCentroids,
    InstanceHardnessThreshold,
    NearMiss,
    RandomUnderSampler,
)
from numpy.typing import ArrayLike, NDArray
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array


__all__ = [
    "INTERPOLATORS",
    "UNDERSAMPLERS",
    "allocate_proportionally",
    "create_interpolator",
    "create_undersampler",
    "effective_dimension",
    "validate_all_classes_per_site",
    "validate_class_representation",
    "validate_covariates",
    "variance_ratio",
]

logger = structlog.get_logger()


INTERPOLATORS: dict[str, type] = {
    "smote": SMOTE,
    "borderline-smote": BorderlineSMOTE,
    "svm-smote": SVMSMOTE,
    "adasyn": ADASYN,
    "kmeans-smote": KMeansSMOTE,
    "random": RandomOverSampler,
}

UNDERSAMPLERS: dict[str, tuple[type, dict]] = {
    "random": (RandomUnderSampler, {}),
    "nearmiss": (NearMiss, {"version": 1}),
    "nearmiss-1": (NearMiss, {"version": 1}),
    "nearmiss-2": (NearMiss, {"version": 2}),
    "nearmiss-3": (NearMiss, {"version": 3}),
    "cluster-centroids": (ClusterCentroids, {"voting": "hard"}),
    "instance-hardness": (InstanceHardnessThreshold, {}),
}


def create_interpolator(name: str, random_state: int | np.random.RandomState = 23, **kwargs) -> SamplerMixin:
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
        Initialized interpolator instance.

    Raises
    ------
    ValueError
        If ``name`` is invalid.

    """
    random_state = check_random_state(random_state)
    name_lower = name.lower()
    if name_lower not in INTERPOLATORS:
        raise ValueError(f"Unsupported interpolator: {name}. Choose from {sorted(INTERPOLATORS)}.")

    return INTERPOLATORS[name_lower](random_state=random_state, **kwargs)


def create_undersampler(name: str, random_state: int | np.random.RandomState | None = None, **kwargs) -> SamplerMixin:
    """Create an imblearn under-sampler based on a string name.

    Only under-samplers that accept a target count per class (a ``dict``
    ``sampling_strategy``) are listed. Cleaning methods (Tomek links, ENN, ...)
    cannot reach an exact count; apply them before ISI if needed.

    Parameters
    ----------
    name : str
        Name of the under-sampler: ``"cluster-centroids"`` (k-means on the
        class, keeping the real sample nearest to each centroid:
        ``ClusterCentroids(voting="hard")``), ``"nearmiss"`` (alias of
        ``"nearmiss-1"``), ``"nearmiss-2"``, ``"nearmiss-3"``,
        ``"instance-hardness"`` or ``"random"``.
    random_state : int, RandomState instance or None, optional (default None)
        Seed, passed to the under-sampler when it accepts one.
    **kwargs : dict
        Extra keyword arguments for the under-sampler.

    Returns
    -------
    object
        Initialized under-sampler instance.

    Raises
    ------
    ValueError
        If ``name`` is invalid.

    """
    name_lower = name.lower()
    if name_lower not in UNDERSAMPLERS:
        raise ValueError(f"Unsupported undersampler: {name}. Choose from {sorted(UNDERSAMPLERS)}.")
    cls, defaults = UNDERSAMPLERS[name_lower]
    params = {**defaults, **kwargs}
    if "random_state" in cls().get_params():
        params.setdefault("random_state", random_state)
    return cls(**params)


def allocate_proportionally(total: int, weights: npt.ArrayLike, capacity: npt.ArrayLike | None = None) -> npt.NDArray[np.int64]:
    """Split an integer ``total`` into integer parts proportional to ``weights``.

    Uses the largest-remainder method, so the parts sum exactly to ``total``.
    When ``capacity`` is given, no part exceeds its capacity (the excess is
    redistributed proportionally among the parts that still have room).

    Parameters
    ----------
    total : int
        Number of units to allocate.
    weights : array-like of shape (n_parts,)
        Non-negative weights.
    capacity : array-like of shape (n_parts,) or None, optional (default None)
        Maximum number of units per part.

    Returns
    -------
    numpy.ndarray of shape (n_parts,)
        Integer allocation.

    Raises
    ------
    ValueError
        If the weights are all zero while ``total > 0``, or if ``total``
        exceeds the total capacity.

    """
    weights = np.asarray(weights, dtype=float)
    cap = np.full(len(weights), np.inf) if capacity is None else np.asarray(capacity, dtype=float)
    if total > 0 and (weights.sum() <= 0 or cap[weights > 0].sum() < total):
        raise ValueError(f"Cannot allocate {total} units with weights {weights} and capacity {cap}.")
    alloc = np.zeros(len(weights), dtype=np.int64)
    remaining = int(total)
    while remaining > 0:
        open_ = (weights > 0) & (alloc < cap)
        share = np.where(open_, weights, 0.0)
        exact = remaining * share / share.sum()
        add = np.minimum(np.floor(exact), cap - alloc).astype(np.int64)
        left = remaining - int(add.sum())
        if left > 0:
            # largest remainders first, among parts that still have room after `add`
            order = np.argsort(-(exact - np.floor(exact)), kind="stable")
            for i in order:
                if left == 0:
                    break
                if open_[i] and alloc[i] + add[i] < cap[i]:
                    add[i] += 1
                    left -= 1
        alloc += add
        remaining = int(total - alloc.sum())
    return alloc


def variance_ratio(X_synthetic: npt.ArrayLike, X_real: npt.ArrayLike) -> float:
    """Average per-feature ratio between synthetic and real variance.

    Parameters
    ----------
    X_synthetic : array-like of shape (n_synthetic, n_features)
        Synthetic samples.
    X_real : array-like of shape (n_real, n_features)
        Real samples they were generated from.

    Returns
    -------
    float
        ``mean_j var(X_synthetic[:, j]) / var(X_real[:, j])`` over the
        features with non-zero real variance (unbiased variances). ``1``
        means the synthetic samples have the same spread as the real ones;
        SMOTE-like interpolation gives values below one. ``nan`` if it
        cannot be computed (fewer than two samples or constant features).

    """
    X_synthetic = np.asarray(X_synthetic, dtype=float)
    X_real = np.asarray(X_real, dtype=float)
    if len(X_synthetic) < 2 or len(X_real) < 2:
        return float("nan")
    var_real = X_real.var(axis=0, ddof=1)
    keep = var_real > np.finfo(float).eps * np.maximum(1.0, np.abs(X_real).max(axis=0)) ** 2
    if not np.any(keep):
        return float("nan")
    return float(np.mean(X_synthetic[:, keep].var(axis=0, ddof=1) / var_real[keep]))


def effective_dimension(
    X: npt.ArrayLike,
    groups: npt.ArrayLike | None = None,
    max_samples: int | None = 2000,
    random_state: int | np.random.RandomState | None = 0,
) -> float:
    """Effective number of dimensions (participation ratio) of the data.

    The features are centered within each group (e.g. each site-class cell),
    scaled to unit pooled variance, and the participation ratio of the
    eigenvalues ``l`` of their correlation matrix is returned:
    ``(sum l) ** 2 / sum(l ** 2)``. It equals the number of features for
    uncorrelated features and approaches 1 when all features are collinear.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data.
    groups : array-like of shape (n_samples,) or None, optional (default None)
        Group labels; each group is centered separately.
    max_samples : int or None, optional (default 2000)
        After centering, the ratio is computed from at most this many randomly
        chosen samples, which keeps the cost at ``O(max_samples ** 2 *
        n_features)`` for large data sets (e.g. voxel-wise images). ``None``
        uses all samples.
    random_state : int, RandomState instance or None, optional (default 0)
        Random state for the choice of samples.

    Returns
    -------
    float
        Participation ratio, between 1 and ``min(n_samples, n_features)``.

    """
    X = np.asarray(X, dtype=float)
    Xc = X.copy()
    if groups is None:
        Xc -= Xc.mean(axis=0)
    else:
        groups = np.asarray(groups)
        for g in np.unique(groups):
            mask = groups == g
            Xc[mask] -= Xc[mask].mean(axis=0)
    sd = Xc.std(axis=0)
    Xc = Xc[:, sd > 0] / sd[sd > 0]
    if Xc.shape[1] == 0:
        return float("nan")
    if max_samples is not None and Xc.shape[0] > max_samples:
        rows = check_random_state(random_state).choice(Xc.shape[0], max_samples, replace=False)
        Xc = Xc[rows]
    # the non-zero eigenvalues l of X'X and XX' coincide: sum(l) = trace, sum(l**2) = squared Frobenius norm
    gram = Xc.T @ Xc if Xc.shape[1] <= Xc.shape[0] else Xc @ Xc.T
    return float(np.trace(gram) ** 2 / np.sum(gram**2))


def validate_all_classes_per_site(y: npt.NDArray, sites: npt.NDArray, kind: str = "class", raise_error: bool = True) -> None:
    """Check that every class is present in every site.

    A class that is missing from a site keeps the site predictive of the
    target, which no within-site resampling can fix.

    Parameters
    ----------
    y : array
        Classes (or target bins).
    sites : array
        Sites.
    kind : str, optional (default "class")
        Word used in the error message ("class" or "target bin").
    raise_error : bool, optional (default True)
        Raise a ``ValueError`` if a class is missing; otherwise warn.

    Raises
    ------
    ValueError
        If a site lacks one or more classes and ``raise_error`` is True.

    """
    classes = np.unique(y)
    missing = {}
    for site in np.unique(sites):
        absent = np.setdiff1d(classes, np.unique(y[sites == site]))
        if len(absent):
            missing[site] = absent.tolist()
    if missing:
        detail = "; ".join(f"site {s}: {m}" for s, m in missing.items())
        msg = (
            f"Every {kind} should be present in every site, otherwise the site still predicts the target. Missing: {detail}."
            + (
                " Use fewer bins (n_bins), or restrict the training data to the target range that all sites share; "
                "interpolation cannot create targets that a site does not have."
                if kind != "class"
                else ""
            )
        )
        if raise_error:
            raise ValueError(msg)
        warnings.warn(msg, UserWarning, stacklevel=3)


def validate_class_representation(y: npt.NDArray, sites: npt.NDArray) -> None:
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
        logger.debug(f"Using {cat_cov.shape[1]} categorical covariates")

    if continuous_covariate is not None:
        cont_cov = check_array(continuous_covariate, ensure_all_finite=not allow_nan)
        if cont_cov.shape[0] != n_samples:
            raise ValueError(
                f"continuous_covariate has {cont_cov.shape[0]} samples, but X has {n_samples} samples."
                "Both must have same number of samples."
            )

        if covariate_tolerance is None:
            tol_arr = np.zeros(cont_cov.shape[1])
            logger.debug("No tolerance specified, using exact matching")
        else:
            tol_arr = np.asarray(covariate_tolerance, dtype=np.float64).flatten()
            if tol_arr.shape[0] != cont_cov.shape[1]:
                raise ValueError(
                    f"covariate_tolerance has {tol_arr.shape[0]} values,"
                    f"but continuous_covariate has {cont_cov.shape[1]} columns."
                    "One tolerance value per continuous covariate (column) is required."
                )

        logger.debug(f"Using {cont_cov.shape[1]} continuous covariates with tolerance: {tol_arr}")

    elif covariate_tolerance is not None:
        raise ValueError(
            "covariate_tolerance provided but continuous_covariate is None. Cannot use tolerance without continuous covariates."
        )

    return cat_cov, cont_cov, tol_arr
