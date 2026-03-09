"""Utility functions for interpolation-based harmonization methods."""

import numpy as np
import numpy.typing as npt
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
) -> (
    type[SMOTE]
    | type[BorderlineSMOTE]
    | type[SVMSMOTE]
    | type[ADASYN]
    | type[KMeansSMOTE]
    | type[RandomOverSampler]
):
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

    return mapping[name](
        random_state=random_state, sampling_strategy="not majority", **kwargs
    )


def sites_sanity_checks(x: npt.NDArray, sites: npt.NDArray) -> None:
    """Sanity checks for site array.

    Parameters
    ----------
    x : array
        Features.
    sites : array
        Sites.

    Raises
    ------
    ValueError
        If ``x`` and ``sites`` have different length or
        if single site is provided.

    """
    if x.shape[0] != sites.shape[0]:
        raise ValueError("X and sites must have same length")

    if len(np.unique(sites)) < 2:
        raise ValueError("At least two sites required")


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
            raise ValueError(
                f"Site {site} has only one class; cannot resample."
            )
