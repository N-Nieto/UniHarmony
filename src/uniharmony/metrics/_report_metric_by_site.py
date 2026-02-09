"""Module for computing metrics stratified by site.

This module provides functionality to compute metrics for different sites,
allowing stratified evaluation across multiple locations or groups.
"""

from collections.abc import Callable

import numpy as np


def report_metric_by_site(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sites: np.ndarray,
    metric: Callable,
    overall_performance: bool = False,
) -> dict[str | int, float]:
    """Compute a metric stratified by site.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted values.
    sites : np.ndarray
        Site identifiers for stratification.
    metric : Callable
        Metric to compute from sklean.metrics.
    overall_performance: bool
        Add an aditional dictionary entry with the overall performance.

    Returns
    -------
    Dict[Union[str, int], float]
        Dictionary with sites as keys and computed metrics as values.

    Raises
    ------
    TypeError
        If input arrays are not numpy arrays or metric is not str/callable.
    ValueError
        If arrays have mismatched lengths.

    """
    # Validate inputs
    _input_checks(y_true, y_pred, sites, metric, overall_performance)

    # Compute metric per site
    results = {}
    if overall_performance:
        results["overall"] = metric(y_true, y_pred)

    for site in np.unique(sites):
        # Create a mask for the values corresponding to the site.
        mask = sites == site
        # Put the results in the dict with keys named as the sites.
        # Keep site as-is if it's a string, convert to int otherwise
        site_key = site if isinstance(site, str) else int(site)
        results[site_key] = metric(y_true[mask], y_pred[mask])

    return results


def _input_checks(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sites: np.ndarray,
    metric: Callable,
    overall_performance: bool,
) -> None:
    """Validate input types and shapes for site-wise performance evaluation."""
    if not isinstance(y_true, np.ndarray):
        raise TypeError(
            "y_true must be a numpy.ndarray, "
            f"got {type(y_true).__name__!r} instead."
        )

    if not isinstance(y_pred, np.ndarray):
        raise TypeError(
            "y_pred must be a numpy.ndarray, "
            f"got {type(y_pred).__name__!r} instead."
        )

    if not isinstance(sites, np.ndarray):
        raise TypeError(
            "sites must be a numpy.ndarray, "
            f"got {type(sites).__name__!r} instead."
        )

    if not callable(metric):
        raise TypeError(
            "metric must be a callable, "
            f"got {type(metric).__name__!r} instead."
        )

    if not isinstance(overall_performance, bool):
        raise TypeError(
            "overall_performance must be a bool, "
            f"got {type(overall_performance).__name__!r} instead."
        )

    if not (len(y_true) == len(y_pred) == len(sites)):
        raise ValueError(
            "y_true, y_pred, and sites must have the same length, "
            f"got lengths y_true={len(y_true)}, "
            f"y_pred={len(y_pred)}, sites={len(sites)}."
        )
