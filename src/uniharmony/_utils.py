"""General utilities."""

import numpy as np
import numpy.typing as npt


__all__ = [
    "filter_site_by_size",
]


def filter_site_by_size(
    sites: npt.ArrayLike,
    min_size: float,
    max_size: float = np.inf,
    sites_ignore: npt.ArrayLike | None = None,
) -> npt.NDArray:
    """Filter sites by size.

    Parameters
    ----------
    sites : array-like
        Sites to filter.
    min_size : float
        Minimum site size.
    max_size : float, optional (default np.inf)
        Maximum size size.
    sites_ignore : array-like or None, optional (default None)
        Sites to ignore.

    Returns
    -------
    array
        Indices to filter using.

    """
    idx = np.zeros(len(sites))
    if sites_ignore is None:
        sites_ignore = np.array([])
    for su in sites_ignore:
        idx_su = sites == su
        idx = np.logical_or(idx, idx_su)
    sites_check = np.setdiff1d(np.unique(sites), sites_ignore)
    for su in sites_check:
        idx_su = sites == su
        n = np.sum(idx_su)
        if n >= min_size and n < max_size:
            idx = np.logical_or(idx, idx_su)
        else:
            print(f"excluding site: {su} #sub {n}")
    return idx
