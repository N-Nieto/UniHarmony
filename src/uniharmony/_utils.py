"""General utilities."""

import numpy as np


__all__ = [
    "filter_site_by_size",
]


def filter_site_by_size(sites, min_size, max_size=np.inf, sites_ignore=None):
    """Filter sites by size."""
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
