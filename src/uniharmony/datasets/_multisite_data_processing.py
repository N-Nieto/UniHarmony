"""Process multisite datasets."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_array, check_consistent_length


__all__ = [
    "filter_data_by_site",
]

SiteLabel = str | int | float
SiteSelection = SiteLabel | Iterable[SiteLabel] | np.ndarray | pd.Series | None
DataLike = pd.DataFrame | pd.Series | np.ndarray | list[Any]


def filter_data_by_site(
    X: DataLike,
    y: DataLike,
    sites: DataLike,
    include_sites: SiteSelection = None,
    exclude_sites: SiteSelection = None,
) -> tuple[DataLike, DataLike, DataLike]:
    """Filter multisite data by site labels.

    Parameters
    ----------
    X : pandas.DataFrame, pandas.Series, numpy.ndarray or list
        Feature data with one row per sample.
    y : pandas.DataFrame, pandas.Series, numpy.ndarray or list
        Target labels with one row per sample.
    sites : pandas.DataFrame, pandas.Series, numpy.ndarray or list
        Site labels with one row per sample.
    include_sites : scalar, iterable or None, default=None
        Site labels to keep. When ``None``, all sites are initially kept.
    exclude_sites : scalar, iterable or None, default=None
        Site labels to remove. Exclusions are applied after inclusions.

    Returns
    -------
    X_filtered : same type as X for pandas inputs, otherwise numpy.ndarray
        Filtered feature data.
    y_filtered : same type as y for pandas inputs, otherwise numpy.ndarray
        Filtered target labels.
    sites_filtered : same type as sites for pandas inputs, otherwise numpy.ndarray
        Filtered site labels.

    Raises
    ------
    ValueError
        If ``X``, ``y`` and ``sites`` do not have consistent lengths, or if
        ``sites`` is not one-dimensional.

    Examples
    --------
    >>> X_filtered, y_filtered, sites_filtered = filter_data_by_site(
    ...     X,
    ...     y,
    ...     sites,
    ...     include_sites=["site_a", "site_b"],
    ...     exclude_sites="site_b",
    ... )

    """
    check_consistent_length(X, y, sites)
    sites_array = check_array(sites, dtype=None, ensure_2d=False)

    if sites_array.ndim != 1:
        msg = "sites must be one-dimensional."
        raise ValueError(msg)

    mask = np.ones(sites_array.shape[0], dtype=bool)
    include_values = _normalize_site_selection(include_sites)
    exclude_values = _normalize_site_selection(exclude_sites)

    if include_values is not None:
        mask &= np.isin(sites_array, include_values)

    if exclude_values is not None:
        mask &= ~np.isin(sites_array, exclude_values)

    return _filter_rows(X, mask), _filter_rows(y, mask), _filter_rows(sites, mask)


def _normalize_site_selection(selection: SiteSelection) -> np.ndarray | None:
    """Return a one-dimensional array of selected site labels."""
    if selection is None:
        return None

    if isinstance(selection, str) or np.isscalar(selection):
        values = np.asarray([selection], dtype=object)
    else:
        values = np.asarray(list(selection), dtype=object)

    if values.ndim != 1:
        msg = "Site selections must be scalar labels or one-dimensional collections of labels."
        raise ValueError(msg)

    return values


def _filter_rows(data: DataLike, mask: np.ndarray) -> DataLike:
    """Filter rows while preserving pandas containers."""
    if isinstance(data, pd.DataFrame | pd.Series):
        return data.iloc[mask]

    return np.asarray(data)[mask]
