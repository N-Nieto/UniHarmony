"""Module for computing metrics stratified by site.

This module provides functionality to compute metrics for different sites,
allowing stratified evaluation across multiple locations or groups.
It supports both single metrics and multiple metrics, with automatic
validation of prediction types (binary vs. continuous scores) based on
metric requirements.
"""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


__all__ = [
    "METRICS_REQUIRING_Y_PRED",
    "METRICS_REQUIRING_Y_SCORE",
    "report_metric_by_site",
    "report_metrics_by_site",
]


# -------------------------------------------------------------------------
# Registry of metric signatures
# -------------------------------------------------------------------------

METRICS_REQUIRING_Y_SCORE: set[str] = {
    "roc_auc_score",
    "average_precision_score",
    "roc_curve",
    "precision_recall_curve",
    "det_curve",
    "brier_score_loss",
    "log_loss",
}

METRICS_REQUIRING_Y_PRED: set[str] = {
    "accuracy_score",
    "balanced_accuracy_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "fbeta_score",
    "jaccard_score",
    "cohen_kappa_score",
    "matthews_corrcoef",
    "hamming_loss",
    "zero_one_loss",
    "confusion_matrix",
    "classification_report",
}


def report_metric_by_site(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sites: np.ndarray,
    metric: Callable,
    overall_performance: bool = False,
    skip_empty_sites: bool = True,
    **kwargs: Any,
) -> dict[str | int, float]:
    """Compute a metric stratified by site.

    Automatically validates that ``y_pred`` matches the metric's expected
    input type (discrete predictions vs. continuous scores). For example,
    ``roc_auc_score`` requires probability-like scores, while
    ``accuracy_score`` requires class labels.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth (correct) target values.
    y_pred : np.ndarray
        Estimated targets as returned by a classifier, or probability
        estimates / decision function outputs for score-based metrics.
    sites : np.ndarray
        Site identifiers for stratification. Can be strings or integers.
    metric : Callable
        Metric function to compute (e.g., from ``sklearn.metrics``).
    overall_performance : bool, default=False
        If True, include an ``"overall"`` key with the metric computed
        across all sites.
    skip_empty_sites : bool, default=True
        If True, skip sites with no samples. If False, raises an error
        when a site has no samples.
    **kwargs : Any
        Additional keyword arguments forwarded to ``metric``.

    Returns
    -------
    dict[str | int, float]
        Dictionary mapping site identifiers to computed metric values.
        If ``overall_performance`` is True, includes an ``"overall"`` key.

    Raises
    ------
    TypeError
        If inputs have incorrect types.
    ValueError
        If input arrays have mismatched lengths, or if ``y_pred`` type
        does not match the metric's requirements.

    Examples
    --------
    >>> from sklearn.metrics import accuracy_score, roc_auc_score
    >>> y_true = np.array([0, 1, 0, 1, 0, 1])
    >>> y_pred = np.array([0, 1, 0, 0, 0, 1])
    >>> sites = np.array(["A", "A", "B", "B", "A", "B"])
    >>> report_metric_by_site(y_true, y_pred, sites, accuracy_score)
    {'A': 1.0, 'B': 0.5}

    >>> # Using probability scores for ROC-AUC
    >>> y_scores = np.array([0.1, 0.9, 0.2, 0.4, 0.3, 0.8])
    >>> report_metric_by_site(y_true, y_scores, sites, roc_auc_score)
    {'A': 1.0, 'B': 0.5}

    """
    # Validate inputs
    _input_checks(y_true, y_pred, sites, metric, overall_performance)

    # Validate prediction type against metric requirements
    _validate_prediction_type(y_pred, metric)

    # Compute metric per site
    results: dict[str | int, float] = {}

    if overall_performance:
        results["overall"] = metric(y_true, y_pred, **kwargs)

    for site in np.unique(sites):
        mask = sites == site

        if not np.any(mask):
            if skip_empty_sites:
                continue
            raise ValueError(f"Site {site!r} has no samples.")

        site_key = site if isinstance(site, str) else int(site)
        results[site_key] = metric(y_true[mask], y_pred[mask], **kwargs)

    return results


def report_metrics_by_site(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sites: np.ndarray,
    metrics: Sequence[Callable],
    metric_kwargs: dict[str, Any] | Sequence[dict[str, Any]] | None = None,
    overall_performance: bool = False,
    skip_empty_sites: bool = True,
) -> dict[str, dict[str | int, float]]:
    """Compute multiple metrics stratified by site.

    This is the multi-metric variant of :func:`report_metric_by_site`.
    Each metric can receive its own set of keyword arguments via
    ``metric_kwargs``.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth (correct) target values.
    y_pred : np.ndarray
        Estimated targets as returned by a classifier, or probability
        estimates / decision function outputs for score-based metrics.
    sites : np.ndarray
        Site identifiers for stratification. Can be strings or integers.
    metrics : Sequence[Callable]
        List of metric functions to compute (e.g., from ``sklearn.metrics``).
    metric_kwargs : dict[str, Any] | Sequence[dict[str, Any]] | None, default=None
        Keyword arguments for each metric. If a single dict, it is passed
        to all metrics. If a sequence, ``metric_kwargs[i]`` is passed to
        ``metrics[i]``. Must have the same length as ``metrics``.
    overall_performance : bool, default=False
        If True, include an ``"overall"`` key for each metric computed
        across all sites.
    skip_empty_sites : bool, default=True
        If True, skip sites with no samples.

    Returns
    -------
    dict[str, dict[str | int, float]]
        Dictionary mapping metric names to site-wise results.
        Each inner dictionary maps site identifiers to metric values.

    Raises
    ------
    TypeError
        If inputs have incorrect types.
    ValueError
        If ``metric_kwargs`` length does not match ``metrics`` length,
        or if ``y_pred`` type does not match a metric's requirements.

    Examples
    --------
    >>> from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
    >>> y_true = np.array([0, 1, 0, 1, 0, 1])
    >>> y_pred_labels = np.array([0, 1, 0, 0, 0, 1])
    >>> y_pred_scores = np.array([0.1, 0.9, 0.2, 0.4, 0.3, 0.8])
    >>> sites = np.array(["A", "A", "B", "B", "A", "B"])
    >>>
    >>> # Note: in practice you would use either labels or scores,
    >>> # not both, depending on your metrics. This example shows
    >>> # the API structure.
    >>> metrics = [accuracy_score, f1_score]
    >>> kwargs = [{}, {"average": "macro"}]
    >>> report_metrics_by_site(
    ...     y_true, y_pred_labels, sites, metrics, metric_kwargs=kwargs
    ... )
    {'accuracy_score': {'A': 1.0, 'B': 0.5}, 'f1_score': {'A': 1.0, 'B': 0.5}}

    """
    # Validate inputs
    _input_checks_multi(y_true, y_pred, sites, metrics, overall_performance)

    # Normalize metric_kwargs
    n_metrics = len(metrics)

    metric_kwargs_seq = _validate_metric_kwargs(metric_kwargs, n_metrics)

    # Validate each metric's prediction type
    for metric in metrics:
        _validate_prediction_type(y_pred, metric)

    # Compute all metrics per site
    results: dict[str, dict[str | int, float]] = {}

    for metric, kwargs in zip(metrics, metric_kwargs_seq, strict=True):
        metric_name = getattr(metric, "__name__", repr(metric))
        results[metric_name] = {}

        if overall_performance:
            results[metric_name]["overall"] = metric(y_true, y_pred, **kwargs)

        for site in np.unique(sites):
            mask = sites == site

            if not np.any(mask):
                if skip_empty_sites:
                    continue
                raise ValueError(f"Site {site!r} has no samples.")

            site_key = site if isinstance(site, str) else int(site)
            results[metric_name][site_key] = metric(y_true[mask], y_pred[mask], **kwargs)

    return results


def _metric_needs_y_score(metric: Callable) -> bool:
    """Determine if a metric requires continuous scores (y_score).

    Parameters
    ----------
    metric : Callable
        The metric function to inspect.

    Returns
    -------
    bool
        True if the metric is known to require continuous scores,
        False otherwise.

    """
    return getattr(metric, "__name__", "") in METRICS_REQUIRING_Y_SCORE


def _metric_needs_y_pred(metric: Callable) -> bool:
    """Determine if a metric requires discrete predictions (y_pred).

    Parameters
    ----------
    metric : Callable
        The metric function to inspect.

    Returns
    -------
    bool
        True if the metric is known to require discrete predictions,
        False otherwise.

    """
    return getattr(metric, "__name__", "") in METRICS_REQUIRING_Y_PRED


def _is_binary_or_multiclass(y: np.ndarray, tol: float = 1e-9) -> bool:
    """Check if array contains only discrete class labels.

    Parameters
    ----------
    y : np.ndarray
        Array to check.
    tol : float, default=1e-9
        Tolerance for checking if values are close to integers.

    Returns
    -------
    bool
        True if all values are effectively integers (discrete labels),
        False if values appear to be continuous scores.

    """
    # Handle empty arrays
    if y.size == 0:
        return True

    # Check if values are close to integers
    rounded = np.rint(y)
    return bool(np.allclose(y, rounded, atol=tol))


def _is_probability_like(y: np.ndarray) -> bool:
    """Check if array values look like continuous scores/probabilities.

    Parameters
    ----------
    y : np.ndarray
        Array to check.

    Returns
    -------
    bool
        True if values are in [0, 1] and not all binary,
        suggesting probability scores.

    """
    if y.size == 0:
        return False

    unique_vals = np.unique(y)
    # If more than 2 unique values and all in [0, 1], likely scores
    return len(unique_vals) > 2 and np.min(y) >= 0.0 and np.max(y) <= 1.0


def _validate_prediction_type(
    y_pred: np.ndarray,
    metric: Callable,
    metric_name: str | None = None,
) -> None:
    """Validate that y_pred matches the metric's expected input type.

    Parameters
    ----------
    y_pred : np.ndarray
        Predictions or scores to validate.
    metric : Callable
        The metric that will be applied.
    metric_name : str | None, optional
        Optional name for the metric in error messages.

    Raises
    ------
    ValueError
        If y_pred type does not match metric requirements.

    """
    name = metric_name or getattr(metric, "__name__", repr(metric))
    needs_score = _metric_needs_y_score(metric)
    needs_pred = _metric_needs_y_pred(metric)

    # If metric is unknown, try to infer from data
    if not needs_score and not needs_pred:
        # Heuristic: if y_pred looks like scores and metric isn't
        # explicitly in y_pred registry, assume it accepts scores
        return

    if needs_score:
        if _is_binary_or_multiclass(y_pred) and not _is_probability_like(y_pred):
            raise ValueError(
                f"Metric '{name}' requires continuous scores (y_score), "
                f"but y_pred appears to contain discrete class labels. "
                f"Got unique values: {np.unique(y_pred)}. "
                f"Please provide probability scores or decision function outputs."
            )
    elif needs_pred:
        if not _is_binary_or_multiclass(y_pred):
            raise ValueError(
                f"Metric '{name}' requires discrete predictions (y_pred), "
                f"but y_pred appears to contain continuous values. "
                f"Got unique values: {np.unique(y_pred)}. "
                f"Please provide class labels or thresholded predictions."
            )


def _input_checks(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sites: np.ndarray,
    metric: Callable,
    overall_performance: bool,
) -> None:
    """Validate input types and shapes for site-wise performance evaluation.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted values or scores.
    sites : np.ndarray
        Site identifiers for stratification.
    metric : Callable
        Metric to compute.
    overall_performance : bool
        Whether overall performance will also be computed.

    Raises
    ------
    TypeError
        If input arrays are not numpy arrays or metric is not callable.
    ValueError
        If arrays have mismatched lengths.

    """
    if not isinstance(y_true, np.ndarray):
        raise TypeError(f"y_true must be a numpy.ndarray, got {type(y_true).__name__!r} instead.")

    if not isinstance(y_pred, np.ndarray):
        raise TypeError(f"y_pred must be a numpy.ndarray, got {type(y_pred).__name__!r} instead.")

    if not isinstance(sites, np.ndarray):
        raise TypeError(f"sites must be a numpy.ndarray, got {type(sites).__name__!r} instead.")

    if not callable(metric):
        raise TypeError(f"metric must be a callable, got {type(metric).__name__!r} instead.")

    if not isinstance(overall_performance, bool):
        raise TypeError(f"overall_performance must be a bool, got {type(overall_performance).__name__!r} instead.")

    if not (len(y_true) == len(y_pred) == len(sites)):
        raise ValueError(
            "y_true, y_pred, and sites must have the same length, "
            f"got lengths y_true={len(y_true)}, "
            f"y_pred={len(y_pred)}, sites={len(sites)}."
        )


def _input_checks_multi(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sites: np.ndarray,
    metrics: Sequence[Callable],
    overall_performance: bool,
) -> None:
    """Validate inputs for multi-metric site-wise evaluation.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted values or scores.
    sites : np.ndarray
        Site identifiers for stratification.
    metrics : Sequence[Callable]
        Metrics to compute.
    overall_performance : bool
        Whether overall performance will also be computed.

    Raises
    ------
    TypeError
        If inputs have invalid types.
    ValueError
        If arrays have mismatched lengths or metrics is empty.

    """
    if not isinstance(metrics, Sequence) or isinstance(metrics, (str, bytes)):
        raise TypeError(f"metrics must be a sequence of callables, got {type(metrics).__name__!r}.")

    if len(metrics) == 0:
        raise ValueError("metrics must contain at least one callable.")

    for i, m in enumerate(metrics):
        if not callable(m):
            raise TypeError(f"metrics[{i}] must be callable, got {type(m).__name__!r} instead.")

    # Reuse single-metric checks for the rest
    _input_checks(y_true, y_pred, sites, metrics[0], overall_performance)


def _validate_metric_kwargs(
    metric_kwargs: dict[str, Any] | Sequence[dict[str, Any]] | None,
    n_metrics: int,
) -> list[dict[str, Any]]:
    """Normalize and validate keyword arguments for multiple metrics.

    Converts flexible input formats into a uniform list of dicts, one per
    metric, while validating lengths and types.

    Parameters
    ----------
    metric_kwargs : dict[str, Any] | Sequence[dict[str, Any]] | None
        Keyword arguments for metric functions. Supported formats:

        - ``None``: No keyword arguments for any metric (empty dicts).
        - ``dict``: A single dictionary applied to **all** metrics.
        - ``Sequence[dict]``: A sequence (list/tuple) of dictionaries,
          where ``metric_kwargs[i]`` is passed to ``metrics[i]``.

    n_metrics : int
        Expected number of metrics. Must be a positive integer.

    Returns
    -------
    list[dict[str, Any]]
        A list of length ``n_metrics`` where each element is a dict of
        keyword arguments for the corresponding metric.

    Raises
    ------
    TypeError
        If ``metric_kwargs`` is not a dict, sequence of dicts, or None.
    ValueError
        If ``n_metrics`` is not positive, or if ``metric_kwargs`` is a
        sequence whose length does not match ``n_metrics``.

    Examples
    --------
    >>> # None → list of empty dicts
    >>> _validate_metric_kwargs(None, 3)
    [{}, {}, {}]

    >>> # Single dict → broadcasted to all metrics
    >>> _validate_metric_kwargs({"average": "macro"}, 2)
    [{"average": "macro"}, {"average": "macro"}]

    >>> # Sequence of dicts → one per metric
    >>> _validate_metric_kwargs([{}, {"average": "weighted"}], 2)
    [{}, {"average": "weighted"}]

    """
    if not isinstance(n_metrics, int):
        raise TypeError(f"n_metrics must be an int, got {type(n_metrics).__name__!r}.")

    if n_metrics <= 0:
        raise ValueError(f"n_metrics must be positive, got {n_metrics}.")

    if metric_kwargs is None:
        return [{} for _ in range(n_metrics)]

    if isinstance(metric_kwargs, dict):
        return [metric_kwargs for _ in range(n_metrics)]

    if isinstance(metric_kwargs, Sequence) and not isinstance(metric_kwargs, (str, bytes)):
        if len(metric_kwargs) != n_metrics:
            raise ValueError(f"metric_kwargs must have the same length as metrics ({n_metrics}), got {len(metric_kwargs)}.")
        # Validate each element is a dict
        for i, item in enumerate(metric_kwargs):
            if not isinstance(item, dict):
                raise TypeError(f"metric_kwargs[{i}] must be a dict, got {type(item).__name__!r} instead.")
        return list(metric_kwargs)

    raise TypeError(f"metric_kwargs must be a dict, sequence of dicts, or None, got {type(metric_kwargs).__name__!r}.")
