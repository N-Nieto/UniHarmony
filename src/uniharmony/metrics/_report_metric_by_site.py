"""Module for computing metrics stratified by site.

This module provides functionality to compute metrics for different sites,
allowing stratified evaluation across multiple locations or groups.
It supports computing one or many metrics simultaneously, with automatic
binarization of continuous scores when discrete predictions are required.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt


__all__ = [
    "report_metrics_by_site",
]


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


def report_metrics_by_site(
    y_true: npt.NDarray,
    y_pred: npt.NDarray,
    sites: npt.NDarray,
    metrics: Callable | list[Callable],
    metric_kwargs: dict[str, Any] | list[dict[str, Any]] | None = None,
    overall_performance: bool = True,
    skip_empty_sites: bool = True,
) -> dict[str, dict[str | int, float]]:
    """Compute one or more metrics stratified by site.

    Accepts either a single metric function or a sequence of metrics.
    Each metric can receive its own set of keyword arguments via
    ``metric_kwargs``. If ``y_pred`` contains continuous scores but a
    metric requires discrete predictions, the scores are automatically
    binarized using the ``threshold`` keyword argument for that metric
    (default: 0.5).

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth (correct) target values.
    y_pred : np.ndarray
        Estimated targets as returned by a classifier, or probability
        estimates / decision function outputs.
    sites : np.ndarray
        Site identifiers for stratification. Can be strings or integers.
    metrics : callable or list of callable
        Metric function or list of metric functions to compute (e.g., from
        ``sklearn.metrics``). Pass a single callable for one metric, or
        a sequence for multiple metrics.
    metric_kwargs : dict or list of dict or None, optional (default None)
        Keyword arguments for each metric. If a single dict, it is passed
        to all metrics. If a list, ``metric_kwargs[i]`` is passed to
        ``metrics[i]``. Must have the same length as ``metrics``. Include
        ``threshold`` (default: 0.5) for metrics that require discrete
        predictions when ``y_pred`` contains continuous scores.
    overall_performance : bool, optional (default True)
        If True, include an ``"overall"`` key for each metric computed
        across all sites.
    skip_empty_sites : bool, optional (default True)
        If True, skip sites with no samples.

    Returns
    -------
    dict
        Dictionary mapping metric names to site-wise results.
        Each inner dictionary maps site identifiers to metric values.
        When a single metric is passed, the result contains one top-level
        key (the metric's ``__name__``).

    Raises
    ------
    TypeError
        If inputs have incorrect types.
    ValueError
        If ``metric_kwargs`` length does not match ``metrics`` length or
        if input arrays have mismatched lengths.

    Examples
    --------
    Single metric:

    >>> from sklearn.metrics import accuracy_score
    >>> y_true = np.array([0, 1, 0, 1, 0, 1])
    >>> y_scores = np.array([0.1, 0.9, 0.2, 0.4, 0.3, 0.8])
    >>> sites = np.array(["A", "A", "B", "B", "A", "B"])
    >>> report_metrics_by_site(y_true, y_scores, sites, accuracy_score)
    {'accuracy_score': {'A': 1.0, 'B': 0.5}}

    Single metric with custom threshold:

    >>> report_metrics_by_site(
    ...     y_true, y_scores, sites, accuracy_score, metric_kwargs={"threshold": 0.3}
    ... )
    {'accuracy_score': {'A': 1.0, 'B': 0.5}}

    Multiple metrics:

    >>> from sklearn.metrics import roc_auc_score, f1_score
    >>> report_metrics_by_site(
    ...     y_true,
    ...     y_scores,
    ...     sites,
    ...     metrics=[roc_auc_score, accuracy_score, f1_score],
    ...     metric_kwargs=[
    ...         {},
    ...         {"threshold": 0.5},
    ...         {"threshold": 0.5, "average": "macro"},
    ...     ],
    ...     overall_performance=True,
    ... )
    {'roc_auc_score': {'overall': 0.833, 'A': 1.0, 'B': 0.5},
     'accuracy_score': {'overall': 0.833, 'A': 1.0, 'B': 0.5},
     'f1_score': {'overall': 0.833, 'A': 1.0, 'B': 0.5}}

    """
    # Normalize metrics to list
    if not isinstance(list, metrics):
        metrics_seq = [metrics]
    else:
        metrics_seq = metrics

    # Validate inputs
    _input_checks_multi(y_true, y_pred, sites, metrics_seq, overall_performance)

    # Normalize metric_kwargs
    n_metrics = len(metrics_seq)
    metric_kwargs_seq = _validate_metric_kwargs(metric_kwargs, n_metrics)

    # Determine if y_pred is score-like or label-like once
    y_is_scores = _is_probability_like(y_pred) or not _is_binary_or_multiclass(y_pred)

    # Compute all metrics per site
    results: dict[str, dict[str | int, float]] = {}

    for metric, kwargs in zip(metrics_seq, metric_kwargs_seq, strict=True):
        metric_name = getattr(metric, "__name__", repr(metric))
        results[metric_name] = {}

        # Auto-binarize if metric needs predictions but we have scores
        y_pred_metric = y_pred
        if _metric_needs_y_pred(metric) and y_is_scores:
            threshold = kwargs.pop("threshold", 0.5)
            y_pred_metric = _binarize(y_pred, threshold)

        if overall_performance:
            if _metric_needs_y_pred(metric) and y_is_scores:
                threshold = kwargs.pop("threshold", 0.5)
                y_pred_metric = _binarize(y_pred, threshold)
                results[metric_name]["overall"] = metric(y_true, y_pred_metric, **kwargs)
            else:
                results[metric_name]["overall"] = metric(y_true, y_pred_metric, **kwargs)

        for site in np.unique(sites):
            mask = sites == site

            if not np.any(mask):
                if skip_empty_sites:
                    continue
                raise ValueError(f"Site {site!r} has no samples.")

            site_key = site if isinstance(site, str) else int(site)
            results[metric_name][site_key] = metric(y_true[mask], y_pred_metric[mask], **kwargs)

    return results


def _binarize(y: npt.NDarray, threshold: float = 0.5) -> npt.NDarray:
    """Binarize continuous scores using a threshold.

    Parameters
    ----------
    y : np.ndarray
        Continuous scores or probability estimates.
    threshold : float, optional (default 0.5)
        Values >= threshold are mapped to 1, values < threshold to 0.

    Returns
    -------
    np.ndarray
        Binary array of 0s and 1s with same shape as ``y``.

    """
    return (y >= threshold).astype(int)


def _metric_needs_y_pred(metric: Callable) -> bool:
    """Determine if a metric requires discrete predictions (y_pred)."""
    return getattr(metric, "__name__", "") in METRICS_REQUIRING_Y_PRED


def _is_binary_or_multiclass(y: npt.NDarray, tol: float = 1e-9) -> bool:
    """Check if array contains only discrete class labels.

    Parameters
    ----------
    y : np.ndarray
        Array to check.
    tol : float, optional (default 1e-9)
        Tolerance for checking if values are close to integers.

    Returns
    -------
    bool
        True if all values are effectively integers (discrete labels),
        False if values appear to be continuous scores.

    """
    if y.size == 0:
        return True
    rounded = np.rint(y)
    return bool(np.allclose(y, rounded, atol=tol))


def _is_probability_like(y: npt.NDarray) -> bool:
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
    return bool(len(unique_vals) > 2 and np.min(y) >= 0.0 and np.max(y) <= 1.0)


def _input_checks(
    y_true: npt.NDarray,
    y_pred: npt.NDarray,
    sites: npt.NDarray,
    metric: Callable,
    overall_performance: bool,
) -> None:
    """Validate input types and shapes for site-wise performance evaluation.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted values.
    sites : np.ndarray
        Site identifiers for stratification.
    metric : Callable
        Metric to compute from sklearn.metrics.
    overall_performance: bool
        Add an additional dictionary entry with the overall performance.

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
    y_true: npt.NDarray,
    y_pred: npt.NDarray,
    sites: npt.NDarray,
    metrics: list[Callable],
    overall_performance: bool,
) -> None:
    """Validate inputs for multi-metric site-wise evaluation.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted values.
    sites : np.ndarray
        Site identifiers for stratification.
    metrics : list of callable
        Metrics to compute from sklearn.metrics.
    overall_performance: bool
        Add an additional dictionary entry with the overall performance.

    Raises
    ------
    TypeError
        If inputs have invalid types.
    ValueError
        If arrays have mismatched lengths or metrics is empty.

    """
    if not isinstance(metrics, list) or isinstance(metrics, str):
        raise TypeError(f"metrics must be a list of callables, got {type(metrics).__name__!r}.")

    if len(metrics) == 0:
        raise ValueError("metrics must contain at least one callable.")

    for i, m in enumerate(metrics):
        if not callable(m):
            raise TypeError(f"metrics[{i}] must be callable, got {type(m).__name__!r} instead.")

    # Reuse single-metric checks for the rest
    _input_checks(y_true, y_pred, sites, metrics[0], overall_performance)


def _validate_metric_kwargs(
    metric_kwargs: dict[str, Any] | list[dict[str, Any]] | None,
    n_metrics: int,
) -> list[dict[str, Any]]:
    """Normalize and validate keyword arguments for multiple metrics.

    Converts flexible input formats into a uniform list of dicts, one per
    metric, while validating lengths and types.

    Parameters
    ----------
    metric_kwargs : dict or list of dict or None
        Keyword arguments for metric functions. Supported formats:

        - ``None``: No keyword arguments for any metric (empty dicts).
        - ``dict``: A single dictionary applied to **all** metrics.
        - ``list of dict``: A list of dictionaries,
          where ``metric_kwargs[i]`` is passed to ``metrics[i]``.

    n_metrics : int
        Expected number of metrics. Must be a positive integer.

    Returns
    -------
    list of dict
        A list of length ``n_metrics`` where each element is a dict of
        keyword arguments for the corresponding metric.

    Raises
    ------
    TypeError
        If ``metric_kwargs`` is not a dict, list of dict or None.
    ValueError
        If ``n_metrics`` is not positive or
        if ``metric_kwargs`` is a list whose length does not match ``n_metrics``.

    Examples
    --------
    >>> # None -> list of empty dicts
    >>> _validate_metric_kwargs(None, 3)
    [{}, {}, {}]

    >>> # Single dict -> broadcasted to all metrics
    >>> _validate_metric_kwargs({"average": "macro"}, 2)
    [{"average": "macro"}, {"average": "macro"}]

    >>> # Sequence of dicts -> one per metric
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

    if isinstance(metric_kwargs, list) and not isinstance(metric_kwargs, str):
        if len(metric_kwargs) != n_metrics:
            raise ValueError(f"metric_kwargs must have the same length as metrics ({n_metrics}), got {len(metric_kwargs)}.")
        for i, item in enumerate(metric_kwargs):
            if not isinstance(item, dict):
                raise TypeError(f"metric_kwargs[{i}] must be a dict, got {type(item).__name__!r} instead.")
        return list(metric_kwargs)

    raise TypeError(f"metric_kwargs must be a dict, sequence of dicts, or None, got {type(metric_kwargs).__name__!r}.")
