"""Utility functions for interpolation-based harmonization methods."""

import numpy as np


def _create_interpolator(name: str, random_state: int = 23, **kwargs):
    from imblearn.over_sampling import (
        ADASYN,
        SMOTE,
        SVMSMOTE,
        BorderlineSMOTE,
        KMeansSMOTE,
        RandomOverSampler,
    )

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


def _sites_sanity_checks(x, sites):
    if x.shape[0] != sites.shape[0]:
        raise ValueError("X and sites must have same length")

    if len(np.unique(sites)) < 2:
        raise ValueError("At least two sites required")


def _class_representation_checks(y, sites):
    for site in np.unique(sites):
        if len(np.unique(y[sites == site])) < 2:
            raise ValueError(
                f"Site {site} has only one class; cannot resample."
            )
