from __future__ import annotations

from collections import Counter

import numpy as np
from imblearn.base import SamplerMixin
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_X_y

from uniharmony.interpolation._utils import (
    _class_representation_checks,
    _clone_sampler,
    _create_interpolator,
    _sites_sanity_checks,
)


class ICIHarmonization(BaseEstimator, SamplerMixin):
    """Inter-Class Interpolation (ICI) Harmonization.

    Performs site-wise oversampling to remove site class correlation.
    Works for binary and multi-class classification.
    :param interpolator: The interpolator to use. Can be a string
        specifying a built-in method or an instance of SamplerMixin.
        Supported string methods are:
            - "smote": Synthetic Minority Over-sampling Technique
            - "borderline-smote": Borderline-SMOTE
            - "svm-smote": SVM-SMOTE
            - "adasyn": Adaptive Synthetic Sampling
            - "kmeans-smote": KMeans-SMOTE
            - "random": Random Over-Sampling
    :param random_state: Random state for reproducibility.
    :param verbose: If True, prints progress information.
    :param kwargs: Additional keyword arguments passed to the interpolator.
    """

    def __init__(
        self,
        interpolator: str | SamplerMixin = "smote",
        *,
        random_state: int = 42,
        verbose: bool = False,
        **kwargs,
    ):
        self.interpolator = interpolator
        self.random_state = random_state
        self.verbose = verbose
        self.kwargs = kwargs

        if isinstance(interpolator, str):
            self._base_sampler = _create_interpolator(
                interpolator, random_state=random_state, **kwargs
            )
        else:
            self._base_sampler = interpolator

    def fit_resample(  # type: ignore
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        sites: np.ndarray,
    ):
        """Fit and resample the dataset using site-wise harmonization."""
        x, y = check_X_y(x, y)
        sites = check_array(sites, ensure_2d=False)

        # Sanity checks for site length and number of sites
        _sites_sanity_checks(x, sites)

        # This methods needs at least two classes per site
        _class_representation_checks(y, sites)

        x_out, y_out, sites_out = [], [], []

        for site in np.unique(sites):
            mask = sites == site
            x_site, y_site = x[mask], y[mask]

            if self.verbose:
                print(f"[ICI] Site {site}: {Counter(y_site)}")

            strategy = self._build_sampling_strategy(y_site)

            if strategy:
                sampler = _clone_sampler(self._base_sampler, strategy)
                x_rs, y_rs = sampler.fit_resample(x_site, y_site)  # type: ignore
            else:
                x_rs, y_rs = x_site, y_site

            x_out.append(x_rs)
            y_out.append(y_rs)
            sites_out.append(np.full(len(x_rs), site))

        self.sites_resampled_ = np.concatenate(sites_out)

        return np.vstack(x_out), np.concatenate(y_out)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _build_sampling_strategy(self, y: np.ndarray) -> dict[int, int] | None:
        counts = Counter(y)
        if len(counts) < 2:
            return None

        max_count = max(counts.values())
        return {
            cls: max_count for cls, c in counts.items() if c < max_count
        } or None

    def _fit_resample(self, X, y, **params):
        raise NotImplementedError(
            "_fit_resample is not used. Use fit_resample(x, y, sites=...) instead."  # noqa: E501
        )
