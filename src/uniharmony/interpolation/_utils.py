"""Utility functions for interpolation-based harmonization methods."""

import numpy as np
from imblearn.over_sampling import (
    ADASYN,
    SMOTE,
    SVMSMOTE,
    BorderlineSMOTE,
    KMeansSMOTE,
    RandomOverSampler,
)
from sklearn.utils import check_random_state


__all__ = [
    "class_representation_checks",
    "create_interpolator",
    "sites_sanity_checks",
]


def create_interpolator(
    name: str, random_state: int | np.random.RandomState = 23, **kwargs
):
    """Create an imblearn interpolator based on a string name."""
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

    return mapping[name](
        random_state=random_state, sampling_strategy="not majority", **kwargs
    )


def sites_sanity_checks(x, sites):
    """Sanity checks for site array."""
    if x.shape[0] != sites.shape[0]:
        raise ValueError("X and sites must have same length")

    if len(np.unique(sites)) < 2:
        raise ValueError("At least two sites required")


def class_representation_checks(y, sites):
    """Check that each site has at least two classes."""
    for site in np.unique(sites):
        if len(np.unique(y[sites == site])) < 2:
            raise ValueError(
                f"Site {site} has only one class; cannot resample."
            )
