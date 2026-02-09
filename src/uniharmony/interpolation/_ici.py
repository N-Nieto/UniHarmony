from __future__ import annotations

from collections import Counter

import numpy as np
from imblearn.base import SamplerMixin
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_X_y

from uniharmony.interpolation._utils import (
    _class_representation_checks,
    _create_interpolator,
    _sites_sanity_checks,
)


class ICIHarmonization(BaseEstimator, SamplerMixin):
    """Intra-Class Interpolation (ICI) Harmonization.

    This sampler performs **site-wise class balancing** to reduce spurious
    correlations between site membership and class labels.

    For each site independently:
    - The majority class is identified.
    - All minority classes are oversampled to match the majority count.
    - Any imblearn-compatible oversampling strategy may be used.

    The method supports both binary and multi-class classification and
    returns a globally concatenated, site-harmonized dataset.

    Parameters
    ----------
    interpolator : str or SamplerMixin, default="smote"
        Oversampling strategy to apply within each site.

        If a string is provided, one of the following built-in methods
        is used:

        - ``"smote"`` : Synthetic Minority Over-sampling Technique
        - ``"borderline-smote"`` : Borderline-SMOTE
        - ``"svm-smote"`` : SVM-SMOTE
        - ``"adasyn"`` : Adaptive Synthetic Sampling
        - ``"kmeans-smote"`` : KMeans-SMOTE
        - ``"random"`` : Random Over-Sampling

        If an instance is provided, it must implement
        :class:`imblearn.base.SamplerMixin`.

    random_state : int, default=42
        Random seed used to initialize stochastic interpolators.

    verbose : bool, default=False
        If True, prints per-site class distributions before resampling.

    **kwargs
        Additional keyword arguments passed to the interpolator constructor
        when ``interpolator`` is provided as a string.

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
            # Make sure the provided interpolator
            # has "not majority" as sampling_strategy
            assert interpolator.sampling_strategy in ["auto", "not majority"]  # type: ignore
            self._base_sampler = interpolator

    def fit_resample(  # type: ignore
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sites: np.ndarray,
    ):
        """Fit and resample the dataset using site-wise harmonization.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Feature matrix containing the input samples.

        y : numpy.ndarray of shape (n_samples,)
            Target class labels associated with each sample in ``X``.

        sites : numpy.ndarray of shape (n_samples,)
            Site or domain identifiers indicating the origin of each sample.
            Resampling is performed independently within each site.

        Returns
        -------
        X_resampled : numpy.ndarray of shape (n_samples_new, n_features)
            The feature matrix after site-wise oversampling.

        y_resampled : numpy.ndarray of shape (n_samples_new,)
            The corresponding class labels after resampling.

        Raises
        ------
        ValueError
            If ``X``, ``y``, and ``sites`` have incompatible shapes, if fewer
            than two unique sites are present, or if any site contains samples
            from only a single class.

        Notes
        -----
        For each site, the majority class count is used as the target.
        All minority classes within that site are oversampled to match
        this count using the configured interpolator.

        """
        X, y = check_X_y(X, y)
        sites = check_array(sites, ensure_2d=False)

        # Sanity checks for site length and number of sites
        _sites_sanity_checks(X, sites)

        # This methods needs at least two classes per site
        _class_representation_checks(y, sites)

        X_out, y_out, sites_out = [], [], []

        for site in np.unique(sites):
            mask = sites == site
            X_site, y_site = X[mask], y[mask]

            if self.verbose:
                print(f"[ICI] Site {site}: {Counter(y_site)}")

            X_rs, y_rs = self._base_sampler.fit_resample(X_site, y_site)  # type: ignore

            X_out.append(X_rs)
            y_out.append(y_rs)
            sites_out.append(np.full(len(X_rs), site))

        self.sites_resampled_ = np.concatenate(sites_out)

        return np.vstack(X_out), np.concatenate(y_out)

    # ------------------------------------------------------------------ #
    # Compatibility
    # ------------------------------------------------------------------ #
    def _fit_resample(self, X, y, **params):
        """No-use implementation required by SamplerMixin.

        This sampler overrides ``fit_resample`` directly because it
        requires the additional ``sites`` argument.
        """
        raise NotImplementedError(
            "ICIHarmonization requires the `sites` argument. "
            "Call fit_resample(X, y, sites=...) instead."
        )
