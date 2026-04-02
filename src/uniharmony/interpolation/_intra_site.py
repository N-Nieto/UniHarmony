"""Provide IntraSiteInterpolation transformer."""

from collections import Counter

import numpy as np
import structlog
from imblearn.base import SamplerMixin
from sklearn.base import BaseEstimator
from sklearn.utils import Tags, check_random_state
from sklearn.utils.validation import check_array, check_X_y

from uniharmony.interpolation._utils import (
    class_representation_checks,
    create_interpolator,
    sites_sanity_checks,
)


logger = structlog.get_logger()


class IntraSiteInterpolation(SamplerMixin, BaseEstimator):
    """Intra-Site Interpolation (ISI) Harmonization.

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
    interpolator : str or SamplerMixin instance, optional (default "smote")
        The interpolator to use. Can be a str specifying a built-in method or
        an instance of SamplerMixin.
        Supported str methods are:

          - "smote": Synthetic Minority Over-sampling Technique
          - "borderline-smote": Borderline-SMOTE
          - "svm-smote": SVM-SMOTE
          - "adasyn": Adaptive Synthetic Sampling
          - "kmeans-smote": KMeans-SMOTE
          - "random": Random Over-Sampling

    interpolator_kwargs : dict or None, optional (default None)
        Additional keyword arguments passed to ``interpolator``.
    random_state : int or RandomState instance or None, optional (default None)
        The seed of the pseudo random number generator or RandomState for
        reproducibility.

    """

    def __init__(
        self,
        interpolator: str | SamplerMixin = "smote",
        interpolator_kwargs: dict | None = None,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.interpolator = interpolator
        self.interpolator_kwargs = interpolator_kwargs
        self.random_state = random_state

    def fit_resample(
        self,
        X: np.ndarray,
        y: np.ndarray,
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
        sites_sanity_checks(X, sites)

        # This methods needs at least two classes per site
        class_representation_checks(y, sites)

        random_state = check_random_state(self.random_state)
        if isinstance(self.interpolator, str):
            self.interpolator = create_interpolator(
                self.interpolator,
                random_state=random_state,
                **self.interpolator_kwargs if self.interpolator_kwargs is not None else {},
            )
        elif isinstance(self.interpolator, SamplerMixin):
            # Make sure the provided interpolator
            # has "not majority" as sampling_strategy
            if self.interpolator.sampling_strategy not in ["auto", "not majority"]:
                raise ValueError("IntraSiteInterpolation requires the interpolator to have `sampling_strategy='not majority'`.")
        else:
            raise ValueError("interpolator must be either a string or an instance of SamplerMixin.")

        X_out, y_out, sites_out = [], [], []

        for site in np.unique(sites):
            mask = sites == site
            X_site, y_site = X[mask], y[mask]

            logger.info(f"[ISI] Site {site}: {Counter(y_site)}")

            X_rs, y_rs = self.interpolator.fit_resample(X=X_site, y=y_site)

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
        pass

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.estimator_type = "sampler"
        tags.input_tags.two_d_array = True
        tags.input_tags.sparse = False
        tags.input_tags.allow_nan = True
        tags.requires_fit = False
        return tags
