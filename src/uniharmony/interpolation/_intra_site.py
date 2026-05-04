"""Provide IntraSiteInterpolation transformer."""

from collections import Counter
from typing import Literal

import numpy as np
import structlog
from imblearn.base import SamplerMixin
from sklearn.base import BaseEstimator, clone
from sklearn.utils import Tags, check_random_state
from sklearn.utils.validation import (
    check_array,
    check_consistent_length,
    check_X_y,
)

from uniharmony._utils import validate_sites
from uniharmony.interpolation._utils import (
    create_interpolator,
    validate_class_representation,
)


__all__ = ["IntraSiteInterpolation"]

logger = structlog.get_logger()


class IntraSiteInterpolation(SamplerMixin, BaseEstimator):
    """Intra-Site Interpolation (ISI) Harmonization.

    This sampler performs **site-wise class balancing** to reduce spurious
    correlations between site membership and class labels.

    For each site independently:
    - The target class count is determined by ``balance_strategy``.
    - All classes below the target are oversampled to match the target.
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
    balance_strategy : {"per_site", "global_max"}, optional (default "per_site")
        Strategy to determine the target count for oversampling:

        - "per_site": Each site is balanced independently to its own majority
          class count (original behavior).
        - "global_max": All sites are balanced to the global maximum class
          count across all sites. The target is the largest class count found
          in any single site.

    Attributes
    ----------
    samples_created_ : dict
        A nested dictionary mapping ``{site: {class_label: n_created}}``,
        where ``n_created`` is the number of synthetic samples generated
        for that class in that site. Classes that were not oversampled
        have a value of ``0``.
    sites_resampled_ : ndarray of shape (n_samples_new,)
        Site identifiers for the resampled dataset.
    target_count_ : int or None
        The target sample count per class used for balancing. Set to the
        global maximum when ``balance_strategy="global_max"``, otherwise
        ``None`` (targets are per-site).

    """

    def __init__(
        self,
        interpolator: str | SamplerMixin = "smote",
        interpolator_kwargs: dict | None = None,
        random_state: int | np.random.RandomState | None = None,
        balance_strategy: str | Literal["per_site", "global_max"] = "per_site",
    ) -> None:
        self.interpolator = interpolator
        self.interpolator_kwargs = interpolator_kwargs
        self.random_state = random_state
        self.balance_strategy = balance_strategy

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
            than two unique sites are present, if any site contains samples
            from only a single class, if ``balance_strategy`` is invalid,
            or if a site is missing a class required for ``global_max``
            balancing.

        Notes
        -----
        For each site, the target class count is determined by
        ``balance_strategy``:

        - ``per_site``: the majority class count within that site.
        - ``global_max``: the maximum class count found across all sites.

        All classes within each site that are below the target count are
        oversampled to match the target using the configured interpolator.

        """
        X, y = check_X_y(X, y, estimator=self)
        sites = check_array(sites, dtype=None, ensure_2d=False, estimator=self)
        check_consistent_length(X, y, sites)
        validate_sites(sites)
        validate_class_representation(y, sites)

        if self.balance_strategy not in ("per_site", "global_max"):
            raise ValueError(f"balance_strategy must be 'per_site' or 'global_max', got {self.balance_strategy!r}.")

        random_state = check_random_state(self.random_state)

        # Resolve interpolator to an instance we can clone later
        if isinstance(self.interpolator, str):
            interpolator_template = create_interpolator(
                self.interpolator,
                random_state=random_state,
                **self.interpolator_kwargs if self.interpolator_kwargs is not None else {},
            )
        elif isinstance(self.interpolator, SamplerMixin):
            if self.interpolator.sampling_strategy not in ["auto", "not majority"]:
                raise ValueError("IntraSiteInterpolation requires the interpolator to have `sampling_strategy='not majority'`.")
            interpolator_template = self.interpolator
        else:
            raise ValueError("interpolator must be either a string or an instance of SamplerMixin.")

        # ------------------------------------------------------------------ #
        # Compute target count based on balance_strategy
        # ------------------------------------------------------------------ #
        unique_sites = np.unique(sites)
        unique_classes = np.unique(y)

        if self.balance_strategy == "global_max":
            self.target_count_ = max(np.sum((sites == site) & (y == cls)) for site in unique_sites for cls in unique_classes)
            logger.info(f"[ISI] Global max target count: {self.target_count_}")
        else:
            self.target_count_ = None

        # ------------------------------------------------------------------ #
        # Resample each site
        # ------------------------------------------------------------------ #
        X_out, y_out, sites_out = [], [], []
        self.samples_created_ = {}

        for site in unique_sites:
            mask = sites == site
            X_site, y_site = X[mask], y[mask]
            site_counts = Counter(y_site)
            logger.info(f"[ISI] Site {site}: {site_counts}")

            # Determine target count for this site
            if self.balance_strategy == "per_site":
                target = max(site_counts.values())
            else:  # global_max
                target = self.target_count_

            # Track how many samples will be created per class
            self.samples_created_[site] = {}
            for cls in unique_classes:
                n_cls = site_counts.get(cls, 0)
                self.samples_created_[site][cls] = max(0, target - n_cls)

            # Resample this site so every class has exactly `target` samples
            X_rs, y_rs = self._resample_site(X_site, y_site, target, unique_classes, interpolator_template)

            X_out.append(X_rs)
            y_out.append(y_rs)
            sites_out.append(np.full(len(X_rs), site))

        self.sites_resampled_ = np.concatenate(sites_out)
        X_resampled = np.vstack(X_out)
        y_resampled = np.concatenate(y_out)

        # ------------------------------------------------------------------ #
        # Post-hoc assertion: verify balancing
        # ------------------------------------------------------------------ #
        self._assert_balanced(y_resampled, self.sites_resampled_, unique_classes)

        return X_resampled, y_resampled

    def _resample_site(self, X_site, y_site, target, unique_classes, interpolator_template):
        """Resample a single site so that every class has exactly ``target`` samples.

        Parameters
        ----------
        X_site : ndarray of shape (n_site_samples, n_features)
            Feature matrix for the samples in this site.
        y_site : ndarray of shape (n_site_samples,)
            Class labels for the samples in this site.
        target : int
            Desired number of samples per class.
        unique_classes: ndarray
            All class labels present in the full dataset.
        interpolator_template : SamplerMixin
            The base interpolator instance to clone and configure.

        Returns
        -------
        X_rs : ndarray of shape (n_classes * target, n_features)
        y_rs : ndarray of shape (n_classes * target,)

        Raises
        ------
        ValueError
            If a required class is missing from the site and cannot be
            oversampled (only possible with ``global_max`` strategy).

        """
        site_counts = Counter(y_site)
        X_parts, y_parts = [], []

        # Check for missing classes (only an issue with global_max)
        for cls in unique_classes:
            if site_counts.get(cls, 0) == 0:
                raise ValueError(
                    f"Site has 0 samples for class {cls}, cannot oversample to "
                    f"target {target}. Ensure all classes are present in every "
                    f"site when using balance_strategy='global_max'."
                )

        # Build sampling_strategy dict: only list classes that need oversampling
        sampling_strategy = {}
        for cls in unique_classes:
            n_cls = site_counts[cls]
            if n_cls < target:
                sampling_strategy[cls] = target

        if not sampling_strategy:
            # No oversampling needed - trim to exact target if needed
            for cls in unique_classes:
                mask = y_site == cls
                X_cls = X_site[mask]
                X_parts.append(X_cls[:target])
                y_parts.append(np.full(target, cls))
            return np.vstack(X_parts), np.concatenate(y_parts)

        # Clone the interpolator and set the custom sampling strategy
        interpolator = clone(interpolator_template)
        interpolator.set_params(sampling_strategy=sampling_strategy)

        # Run the interpolator on the site's data
        X_temp, y_temp = interpolator.fit_resample(X_site, y_site)

        # Extract exactly `target` samples per class
        for cls in unique_classes:
            mask = y_temp == cls
            X_cls = X_temp[mask]
            n_have = len(X_cls)
            if n_have < target:
                raise RuntimeError(
                    f"Interpolator failed to produce enough samples for class {cls} "
                    f"in site: needed {target}, got {n_have}. "
                    f"Original count was {site_counts[cls]}. "
                    f"This can happen with ADASYN when a class has too few "
                    f"samples to generate meaningful synthetic neighbors."
                )
            X_parts.append(X_cls[:target])
            y_parts.append(np.full(target, cls))

        return np.vstack(X_parts), np.concatenate(y_parts)

    def _assert_balanced(self, y_resampled, sites_resampled, unique_classes):
        """Assert that the resampled dataset is correctly balanced.

        For ``balance_strategy="per_site"``, each site must have the same
        count for every class (within-site balance).
        For ``balance_strategy="global_max"``, every site must have the
        same count for every class, and that count must equal the global
        maximum (cross-site balance).

        """
        unique_sites = np.unique(sites_resampled)

        if self.balance_strategy == "per_site":
            for site in unique_sites:
                mask = sites_resampled == site
                counts = Counter(y_resampled[mask])
                counts_values = list(counts.values())
                assert all(c == counts_values[0] for c in counts_values), f"Site {site} is not balanced: {dict(counts)}"
        else:  # global_max
            site_counts = {}
            for site in unique_sites:
                mask = sites_resampled == site
                counts = Counter(y_resampled[mask])
                site_counts[site] = counts

            # Check within-site balance
            for site, counts in site_counts.items():
                counts_values = list(counts.values())
                assert all(c == counts_values[0] for c in counts_values), f"Site {site} is not balanced: {dict(counts)}"

            # Check cross-site balance (all sites have same counts)
            first_counts = next(iter(site_counts.values()))
            for site, counts in site_counts.items():
                assert counts == first_counts, (
                    f"Site {site} counts {dict(counts)} do not match first site counts {dict(first_counts)}"
                )

            # Check that the count equals the global target
            for site, counts in site_counts.items():
                actual = next(iter(counts.values()))
                assert actual == self.target_count_, (
                    f"Site {site} has {actual} samples per class, but global target is {self.target_count_}"
                )

        logger.info("[ISI] Balance assertion passed.")

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
