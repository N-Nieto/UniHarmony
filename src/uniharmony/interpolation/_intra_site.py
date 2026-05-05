"""Intra-site interpolation-based harmonization.

This module provides the ``IntraSiteInterpolation`` transformer, a sampler
designed to mitigate site-induced bias by enforcing class balance within each
site independently.

Key features
------------
- Site-wise class balancing using interpolation-based oversampling.
- Optional stratification via categorical and/or continuous covariates.
- Support for both classification and regression problems.
- Regression targets are discretized into bins for balancing purposes.
- Compatible with imbalanced-learn samplers.

Design principles
-----------------
- Preserve covariate distributions when requested.
- Guarantee exact class balance per site (or globally).
- Provide robust fallbacks when interpolation is insufficient.
"""

from collections import Counter
from typing import Literal

import numpy as np
import numpy.typing as npt
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
    validate_covariates,
)


__all__ = ["IntraSiteInterpolation"]

logger = structlog.get_logger()


class IntraSiteInterpolation(SamplerMixin, BaseEstimator):
    """Intra-Site Interpolation (ISI) Harmonization.

    This sampler performs **site-wise class balancing** to reduce spurious
    correlations between site membership and class labels.

    For each site independently:
    - The target class count is determined by ``balance_strategy``.
    - All minority classes are oversampled to match the target count.
    - Any imblearn-compatible oversampling strategy may be used.
    - Alternatively, all classes in the smaller sites are oversampled to matched the biggest site.

    When covariates are provided, balancing is performed within each
    covariate stratum (unique combination of covariate values) within
    each site, preserving the joint distribution of covariates and
    target labels.

    For regression tasks, the continuous target is binned into discrete
    intervals and each bin is treated as a class for balancing purposes.

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
          class count.
        - "global_max": All sites are balanced to the global maximum class
          count across all sites.

    n_bins : int or None, optional (default None)
        Number of bins for regression target binning. If ``None`` and the task
        is detected as regression, a default of 10 bins is used.

    binning_strategy : {"uniform", "quantile"}, optional (default "uniform")
        Strategy for creating bins when the task is regression:

        - "uniform": Bins of equal width covering the target range.
        - "quantile": Bins with approximately equal number of samples.

    task : {"auto", "classification", "regression"}, optional (default "auto")
        Task type. If ``"auto"``, inferred from ``y`` dtype (integer types
        imply classification, floating types imply regression).

    Attributes
    ----------
    samples_created_ : dict
        A nested dictionary mapping ``{site: {class_label: n_created}}``,
        where ``n_created`` is the number of synthetic samples generated
        for that class in that site. For regression, ``class_label`` is
        the bin index.
    sites_resampled_ : ndarray of shape (n_samples_new,)
        Site identifiers for the resampled dataset.
    target_count_ : int or None
        The target sample count per class used for balancing. Set to the
        global maximum when ``balance_strategy="global_max"``, otherwise
        ``None`` (targets are per-site).
    bins_ : ndarray or None
        Bin edges used for regression target binning. ``None`` for
        classification tasks.
    task_ : str
        Inferred or specified task type ("classification" or "regression").

    """

    def __init__(
        self,
        interpolator: str
        | Literal["smote", "borderline-smote", "svm-smote", "adasyn", "kmeans-smote", "random"]
        | SamplerMixin = "smote",
        interpolator_kwargs: dict | None = None,
        random_state: int | np.random.RandomState | None = None,
        balance_strategy: str | Literal["per_site", "global_max"] = "per_site",
        n_bins: int | None = None,
        binning_strategy: str | Literal["uniform", "quantile"] = "uniform",
        task: str | Literal["auto", "classification", "regression"] = "auto",
    ) -> None:
        self.interpolator = interpolator
        self.interpolator_kwargs = interpolator_kwargs
        self.random_state = random_state
        self.balance_strategy = balance_strategy
        self.n_bins = n_bins
        self.binning_strategy = binning_strategy
        self.task = task

    def fit_resample(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        sites: npt.ArrayLike,
        *,
        categorical_covariate: npt.ArrayLike | None = None,
        continuous_covariate: npt.ArrayLike | None = None,
        covariate_tolerance: npt.ArrayLike | None = None,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Fit and resample the dataset using site-wise harmonization.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix containing the input samples.

        y : array-like of shape (n_samples,)
            Target values. Integer labels for classification, continuous
            values for regression.

        sites : array-like of shape (n_samples,)
            Site or domain identifiers indicating the origin of each sample.
            Resampling is performed independently within each site.

        categorical_covariate : array-like of shape (n_samples, n_categorical), default=None
            Categorical covariates used for stratified balancing. When
            provided, classes are balanced within each unique covariate
            combination within each site.

        continuous_covariate : array-like of shape (n_samples, n_continuous), default=None
            Continuous covariates used for stratified balancing. Samples
            are grouped by approximate matching within ``covariate_tolerance``.

        covariate_tolerance : array-like of shape (n_continuous,), default=None
            Maximum allowed absolute difference for continuous covariate
            grouping. Must have one value per continuous covariate column.
            If ``None``, exact matching is required.

        Returns
        -------
        X_resampled : numpy.ndarray of shape (n_samples_new, n_features)
            The feature matrix after site-wise oversampling.

        y_resampled : numpy.ndarray of shape (n_samples_new,)
            The corresponding targets after resampling.

        Raises
        ------
        ValueError
            If ``X``, ``y``, and ``sites`` have incompatible shapes, if fewer
            than two unique sites are present, if any site is missing any
            class, or if ``balance_strategy`` is invalid.


        Notes
        -----
        Sites can be retrieved from IntraSiteInterpolation.sites_resampled_

        """
        logger.info("[ISI] Starting fit_resample")

        (
            X,
            y,
            sites,
            y_work,
            cat_cov,
            cont_cov,
            cov_tol,
        ) = self._validate_input(
            X,
            y,
            sites,
            categorical_covariate,
            continuous_covariate,
            covariate_tolerance,
        )

        interpolator_template = self._resolve_interpolator()

        unique_sites = np.unique(sites)
        unique_classes = np.unique(y_work)

        # Global target
        if self.balance_strategy == "global_max":
            self.target_count_ = max(np.sum((sites == site) & (y_work == cls)) for site in unique_sites for cls in unique_classes)
            logger.debug(f"N target for global_max strategy = {self.target_count_}")
        else:
            self.target_count_ = None

        X_out, y_out, sites_out = [], [], []
        self.samples_created_ = {}

        for site in unique_sites:
            logger.info(f"[ISI] Processing site {site}")

            mask = sites == site
            Xs, ys, yw = X[mask], y[mask], y_work[mask]

            cat_s = cat_cov[mask] if cat_cov is not None else None
            cont_s = cont_cov[mask] if cont_cov is not None else None

            target_N = max(Counter(yw).values()) if self.balance_strategy == "per_site" else self.target_count_
            logger.debug(f"for site {site}, N target for per_site strategy = {target_N}")

            Xr, yr = self._resample_site(
                Xs,
                ys,
                yw,
                target_N,
                unique_classes,
                interpolator_template,
                cat_s,
                cont_s,
                cov_tol,
            )

            self.samples_created_[site] = {
                c: max(
                    0,
                    np.sum((self._bin_target(yr)[0] if self.task_ == "regression" else yr) == c) - np.sum(yw == c),
                )
                for c in unique_classes
            }

            X_out.append(Xr)
            y_out.append(yr)
            sites_out.append(np.full(len(Xr), site))

        self.sites_resampled_ = np.concatenate(sites_out)

        return np.vstack(X_out), np.concatenate(y_out)

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def _validate_input(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        sites: npt.ArrayLike,
        categorical_covariate: npt.ArrayLike | None,
        continuous_covariate: npt.ArrayLike | None,
        covariate_tolerance: npt.ArrayLike | None,
    ) -> tuple[
        npt.NDArray,
        npt.NDArray,
        npt.NDArray,
        npt.NDArray,
        npt.NDArray | None,
        npt.NDArray | None,
        npt.NDArray | None,
    ]:
        """Validate and preprocess all inputs."""
        X, y = check_X_y(X, y, estimator=self)
        sites = check_array(sites, ensure_2d=False)
        check_consistent_length(X, y, sites)

        validate_sites(sites)

        cat_cov, cont_cov, cov_tol = validate_covariates(
            X.shape[0],
            categorical_covariate,
            continuous_covariate,
            covariate_tolerance,
            allow_nan=True,
        )

        self.task_ = self._infer_task(y)

        if self.task_ == "regression":
            if self.n_bins is None:
                raise ValueError("n_bins must be provided for regression.")
            y_work, self.bins_ = self._bin_target(y)
        else:
            y_work = y
            self.bins_ = None

        validate_class_representation(y_work, sites)

        return X, y, sites, y_work, cat_cov, cont_cov, cov_tol

    def _resolve_interpolator(self) -> SamplerMixin:
        """Create or validate interpolator instance."""
        random_state = check_random_state(self.random_state)

        if isinstance(self.interpolator, str):
            return create_interpolator(
                self.interpolator,
                random_state=random_state,
                **(self.interpolator_kwargs or {}),
            )

        if isinstance(self.interpolator, SamplerMixin):
            return self.interpolator

        raise ValueError("Invalid interpolator")

    # ------------------------------------------------------------------ #
    # Core function
    # ------------------------------------------------------------------ #
    def _resample_site(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_work: np.ndarray,
        target: int,
        classes: np.ndarray,
        interpolator_template: SamplerMixin,
        cat: np.ndarray | None,
        cont: np.ndarray | None,
        cov_tol: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resample a single site."""
        if cat is None and cont is None:
            group_labels = np.zeros(len(X), dtype=int)
        else:
            group_labels = self._create_group_labels(cat, cont, cov_tol)

        X_parts, y_parts = [], []

        for g in np.unique(group_labels):
            m = group_labels == g
            Xg, yg, yw = X[m], y[m], y_work[m]

            if len(Xg) == 0:
                continue

            counts = Counter(yw)
            group_target = max(counts.values()) if self.balance_strategy == "per_site" else target

            sampling_strategy = {c: group_target for c in classes if counts.get(c, 0) < group_target}

            if not sampling_strategy:
                for c in classes:
                    mc = yw == c
                    if np.any(mc):
                        X_parts.append(Xg[mc][:group_target])
                        y_parts.append(yg[mc][:group_target])
                continue

            interp = clone(interpolator_template)
            interp.set_params(sampling_strategy=sampling_strategy)

            X_tmp, y_tmp = interp.fit_resample(Xg, yg)

            y_tmp_work = self._bin_target(y_tmp)[0] if self.task_ == "regression" else y_tmp

            for c in classes:
                mc = y_tmp_work == c
                Xc, yc = X_tmp[mc], y_tmp[mc]

                if len(Xc) < group_target:
                    idx = np.random.choice(len(Xc), group_target - len(Xc), True)
                    Xc = np.vstack([Xc, Xc[idx]])
                    yc = np.concatenate([yc, yc[idx]])

                X_parts.append(Xc[:group_target])
                y_parts.append(yc[:group_target])

        return np.vstack(X_parts), np.concatenate(y_parts)

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #
    def _infer_task(self, y: np.ndarray) -> str:
        """Infer task type."""
        if self.task != "auto":
            return self.task
        return "classification" if y.dtype.kind in "biu" else "regression"

    def _bin_target(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Bin continuous targets."""
        bins = (
            np.linspace(y.min(), y.max(), self.n_bins + 1)
            if self.binning_strategy == "uniform"
            else np.quantile(y, np.linspace(0, 1, self.n_bins + 1))
        )

        yb = np.digitize(y, bins[1:-1])
        return np.clip(yb, 0, len(bins) - 2), bins

    def _create_group_labels(
        self,
        cat: np.ndarray | None,
        cont: np.ndarray | None,
        cov_tol: np.ndarray | None,
    ) -> np.ndarray:
        """Create group labels using covariates and tolerance."""
        n = len(cat) if cat is not None else len(cont)
        labels = np.zeros(n, dtype=int)

        if cat is not None:
            _, labels = np.unique(cat, axis=0, return_inverse=True)

        if cont is not None:
            cont_labels = np.zeros(n, dtype=int)

            for i in range(cont.shape[1]):
                col = cont[:, i]
                tol = cov_tol[i] if cov_tol is not None else 0.0

                if tol > 0:
                    bins = np.floor((col - col.min()) / tol)
                else:
                    _, bins = np.unique(col, return_inverse=True)

                cont_labels = cont_labels * (bins.max() + 1) + bins

            labels = labels * (cont_labels.max() + 1) + cont_labels

        return labels

    def _fit_resample(self, X, y, **params) -> None:
        """Unused method required by sklearn."""
        pass

    def __sklearn_tags__(self) -> Tags:
        """Return sklearn compatibility tags."""
        tags = super().__sklearn_tags__()
        tags.estimator_type = "sampler"
        return tags
