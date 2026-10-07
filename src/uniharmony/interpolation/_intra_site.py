"""Intra-site interpolation-based harmonization."""

import warnings
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import structlog
from imblearn.base import SamplerMixin
from imblearn.over_sampling import RandomOverSampler
from sklearn.base import BaseEstimator, clone
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import Tags, check_random_state
from sklearn.utils.validation import check_array, check_consistent_length, check_is_fitted, check_X_y

from uniharmony._utils import validate_sites
from uniharmony.interpolation._utils import (
    allocate_proportionally,
    create_interpolator,
    create_undersampler,
    effective_dimension,
    validate_all_classes_per_site,
    validate_covariates,
    variance_ratio,
)


__all__ = ["IntraSiteInterpolation"]

logger = structlog.get_logger()

# Number of synthetic samples drawn to estimate how well an interpolator preserves the variance of a cell.
_PILOT_MIN, _PILOT_MAX = 200, 2000
# Neighbour parameters of imblearn samplers that must not exceed the number of anchors (or of samples).
_ANCHOR_NEIGHBOR_PARAMS = ("k_neighbors", "n_neighbors")
_ALL_NEIGHBOR_PARAMS = ("m_neighbors",)
_SEED_MAX = np.iinfo(np.int32).max


class IntraSiteInterpolation(SamplerMixin, BaseEstimator):
    """Intra-Site Interpolation (ISI).

    ISI removes the association between site and target in a training set by
    **balancing the classes within every site**. Minority classes are
    over-sampled by interpolating between real samples of the same class (and,
    optionally, the same covariate stratum) *of the same site*, so the
    synthetic samples carry the biological variability of that class and the
    effect of site of that site. When interpolation cannot safely create
    enough samples, the remaining imbalance is closed by under-sampling the
    majority classes, so both meet in the middle.

    For each site ``s`` and class ``c`` with ``n_sc`` real samples:

    1. The amplification of the cell, ``r = n_synthetic / n_sc``, is capped
       at ``r*_sc`` (see ``max_amplification``).
    2. The site target is the count every class of the site is brought to::

           T_s = min(n_max, min_c floor(n_sc * (1 + r*_sc)))

       with ``n_max`` the largest class of the site (``"per_site"``) or the
       largest class of any site (``"global_max"``).
    3. Classes below ``T_s`` are over-sampled with ``interpolator``; classes
       above ``T_s`` are under-sampled with ``undersampler``.

    After resampling every class has ``T_s`` samples in site ``s``, so
    ``P(y | site)`` is uniform and the site no longer predicts the target.

    Parameters
    ----------
    interpolator : str or imblearn over-sampler, optional (default "smote")
        Over-sampler used to create synthetic samples. Strings:
        ``"smote"``, ``"borderline-smote"``, ``"svm-smote"``, ``"adasyn"``,
        ``"kmeans-smote"`` or ``"random"`` (random over-sampling, i.e.
        duplication, kept as a baseline). Any imblearn-compatible over-sampler
        instance that returns the original samples first can be passed.
        Neighbour parameters (``k_neighbors``, ``n_neighbors``,
        ``m_neighbors``) are reduced automatically for small cells.

    interpolator_kwargs : dict or None, optional (default None)
        Keyword arguments for ``interpolator`` when it is given as a string.

    undersampler : str, imblearn under-sampler or None, optional (default "random")
        Under-sampler used for the classes above the site target. Strings:
        ``"random"``, ``"nearmiss"`` (``"nearmiss-1"``), ``"nearmiss-2"``,
        ``"nearmiss-3"``, ``"cluster-centroids"`` or ``"instance-hardness"``.
        Any imblearn under-sampler that accepts a target count per class
        (``dict`` ``sampling_strategy``) can be passed; cleaning methods
        (Tomek links, ENN, ...) cannot reach an exact count and are rejected.
        Prototype generators such as ``ClusterCentroids`` return new samples,
        which are flagged in ``is_synthetic_``. If ``None``, no sample is
        removed and an error is raised when the amplification cap would
        require it.

    undersampler_kwargs : dict or None, optional (default None)
        Keyword arguments for ``undersampler`` when it is given as a string.

    balance_strategy : {"per_site", "global_max"}, optional (default "per_site")
        - ``"per_site"``: every site is balanced to its own largest class.
        - ``"global_max"``: every site is balanced towards the largest class
          of any site, so sites also get the same size. The amplification
          cap still applies, so sites whose classes cannot be amplified that
          much end up smaller (but still balanced).

    max_amplification : float, "auto", callable or None, optional (default "auto")
        Maximum number of synthetic samples per real sample in a site-class
        cell (``r*``).

        - ``"auto"``: variance-preservation rule. A pilot batch of synthetic
          samples is drawn for the cell and the ratio between their variance
          and the variance of the real samples is measured,
          ``rho = mean_j var_synthetic_j / var_real_j``. Mixing ``r``
          synthetic samples per real sample changes the variance of the
          class by ``r / (1 + r) * |1 - rho|``; keeping this below
          ``variance_tolerance`` (``eps``) gives::

              r* = eps / (|1 - rho| - eps)   if |1 - rho| > eps, else no cap

          ``rho`` shrinks when few samples have to span many effective
          dimensions, so the cap depends on the number of samples, the
          number of features and their collinearity through the data
          themselves. Duplication (``"random"``) keeps ``rho`` close to one
          and is therefore not capped by this rule.
        - float ``>= 0``: the same cap for every cell (``0`` means pure
          under-sampling).
        - callable: ``f(X_cell) -> float`` returning the cap of a cell from
          its real samples.
        - ``None``: no cap (pure over-sampling, as in earlier versions).

        A cell with fewer real samples than the interpolator needs (two, one
        for ``"random"``) cannot be over-sampled: its cap is ``0``.

    variance_tolerance : float, optional (default 0.1)
        Maximum relative change of the within-class variance accepted by
        ``max_amplification="auto"`` (``eps`` above).

    n_bins : int, optional (default 10)
        Number of bins of the target for regression.

    binning_strategy : {"uniform", "quantile"}, optional (default "quantile")
        How the regression target is binned: equal-width (``"uniform"``) or
        equal-frequency (``"quantile"``) bins.

    task : {"auto", "classification", "regression"}, optional (default "auto")
        Task type. ``"auto"`` infers it from the dtype of ``y`` (boolean,
        integer, string and object mean classification; floating point means
        regression). For regression, each target bin is treated as a class.

    n_bins_cont_cov : int or None, optional (default None)
        Number of bins used to stratify continuous covariates (required when
        ``continuous_covariate`` is given).

    binning_strategy_cont_cov : {"uniform", "quantile"}, optional (default "quantile")
        How continuous covariates are binned within each site.

    random_state : int, RandomState instance or None, optional (default None)
        Seed of the pseudo random number generator.

    Attributes
    ----------
    sites_resampled_ : ndarray of shape (n_samples_new,)
        Site of each resampled sample.

    sample_indices_ : ndarray of shape (n_samples_new,)
        Index in the input of each resampled sample, ``-1`` for samples
        created by ISI. Use it to carry along covariates of the real samples.

    is_synthetic_ : ndarray of shape (n_samples_new,)
        Whether each resampled sample was created by ISI.

    target_counts_ : dict
        ``{site: T_s}``, the number of samples per class in each site after
        resampling.

    target_count_ : int or None
        Largest class count of any site for ``balance_strategy="global_max"``,
        ``None`` otherwise.

    samples_created_ : dict
        ``{site: {class: n_created}}``. For regression, classes are bins.

    samples_removed_ : dict
        ``{site: {class: n_removed}}``.

    amplification_ : dict
        ``{site: {class: n_created / n_real}}``.

    amplification_cap_ : dict
        ``{site: {class: r*}}`` (``inf`` when uncapped).

    variance_ratio_ : dict
        ``{site: {class: rho}}`` measured by ``max_amplification="auto"``
        (``nan`` for cells that did not need over-sampling).

    effective_dim_ : float
        Effective number of dimensions of the data (participation ratio of
        the correlation matrix of the features, centered within each
        site-class cell). Compare it with the size of the cells: few samples
        per effective dimension means interpolation fills the space poorly.

    n_features_in_ : int
        Number of features.

    bins_ : ndarray or None
        Bin edges of the regression target, ``None`` for classification.

    task_ : str
        ``"classification"`` or ``"regression"``.

    interpolator_ : imblearn over-sampler
        Template of the over-sampler used.

    undersampler_ : imblearn under-sampler or None
        Template of the under-sampler used.

    See Also
    --------
    InterSiteMatchedInterpolation : Interpolation between matched samples of different sites.

    Notes
    -----
    Use ISI on training data only (e.g. inside an ``imblearn.pipeline.Pipeline``
    evaluated with cross-validation). As ``sites`` is not part of ``X``, enable
    scikit-learn metadata routing to pass it::

        import sklearn
        from imblearn.pipeline import Pipeline

        sklearn.set_config(enable_metadata_routing=True)
        isi = IntraSiteInterpolation().set_fit_resample_request(sites=True)
        pipe = Pipeline([("isi", isi), ("clf", LogisticRegression())])
        cross_validate(pipe, X, y, params={"sites": sites})

    Covariates (``categorical_covariate``, ``continuous_covariate``) restrict
    who is interpolated with whom: synthetic samples are interpolated between
    samples of the same class, site and covariate stratum, and they are spread
    over the strata in proportion to the real samples of that class, so
    ``P(covariates | class, site)`` is preserved. Under-sampling is stratified
    the same way.

    A warning is raised when more than half of the real samples of a site
    have to be removed, which happens when a site is extremely imbalanced and
    its minority class is too small to be interpolated (for instance 10000
    vs 2 samples): no method can balance such a site without either inventing
    or discarding most of the data.

    Original samples are returned unchanged, with their original targets.
    For regression, the target of a synthetic sample is interpolated between
    its two parent samples, ``y = y_a + lam * (y_b - y_a)``, which recovers
    SMOTE's interpolation exactly.

    """

    def __init__(
        self,
        interpolator: str
        | Literal["smote", "borderline-smote", "svm-smote", "adasyn", "kmeans-smote", "random"]
        | SamplerMixin = "smote",
        interpolator_kwargs: dict | None = None,
        undersampler: str
        | Literal["random", "nearmiss", "nearmiss-1", "nearmiss-2", "nearmiss-3", "cluster-centroids", "instance-hardness"]
        | SamplerMixin
        | None = "random",
        undersampler_kwargs: dict | None = None,
        balance_strategy: str | Literal["per_site", "global_max"] = "per_site",
        max_amplification: float | str | Literal["auto"] | Callable[[np.ndarray], float] | None = "auto",
        variance_tolerance: float = 0.1,
        n_bins: int = 10,
        binning_strategy: str | Literal["uniform", "quantile"] = "quantile",
        task: str | Literal["auto", "classification", "regression"] = "auto",
        n_bins_cont_cov: int | None = None,
        binning_strategy_cont_cov: str | Literal["uniform", "quantile"] = "quantile",
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.interpolator = interpolator
        self.interpolator_kwargs = interpolator_kwargs
        self.undersampler = undersampler
        self.undersampler_kwargs = undersampler_kwargs
        self.balance_strategy = balance_strategy
        self.max_amplification = max_amplification
        self.variance_tolerance = variance_tolerance
        self.n_bins = n_bins
        self.binning_strategy = binning_strategy
        self.task = task
        self.n_bins_cont_cov = n_bins_cont_cov
        self.binning_strategy_cont_cov = binning_strategy_cont_cov
        self.random_state = random_state

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fit_resample(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        sites: npt.ArrayLike,
        *,
        categorical_covariate: npt.ArrayLike | None = None,
        continuous_covariate: npt.ArrayLike | None = None,
        n_bins_cont_cov: int | None = None,
        binning_strategy_cont_cov: str | None = None,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Balance the classes within every site.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features.

        y : array-like of shape (n_samples,)
            Target: class labels for classification, continuous values for
            regression.

        sites : array-like of shape (n_samples,)
            Site of each sample.

        categorical_covariate : array-like of shape (n_samples,) or (n_samples, n_categorical), default=None
            Categorical covariates. Synthetic samples are only interpolated
            between samples with the same values.

        continuous_covariate : array-like of shape (n_samples,) or (n_samples, n_continuous), default=None
            Continuous covariates, binned within each site into
            ``n_bins_cont_cov`` bins that act as strata.

        n_bins_cont_cov : int or None, default=None
            Deprecated, set it in the constructor instead.

        binning_strategy_cont_cov : {"uniform", "quantile"} or None, default=None
            Deprecated, set it in the constructor instead.

        Returns
        -------
        X_resampled : numpy.ndarray of shape (n_samples_new, n_features)
            Resampled features, grouped by site.

        y_resampled : numpy.ndarray of shape (n_samples_new,)
            Resampled targets. Original samples keep their target and dtype.

        Raises
        ------
        ValueError
            If inputs are inconsistent, fewer than two sites are given, a
            class (or target bin) is missing from a site, a parameter is
            invalid, or the amplification cap requires under-sampling while
            ``undersampler=None``.

        Notes
        -----
        The site of each resampled sample is stored in ``sites_resampled_``
        and its origin in ``sample_indices_`` / ``is_synthetic_``.

        """
        n_bins_cc, strategy_cc = self._resolve_deprecated_covariate_params(n_bins_cont_cov, binning_strategy_cont_cov)
        X, y, sites, y_cls, cat_cov, cont_cov = self._validate_input(
            X, y, sites, categorical_covariate, continuous_covariate, n_bins_cc, strategy_cc
        )
        self._validate_params_values()
        rng = check_random_state(self.random_state)
        self.interpolator_ = self._resolve_interpolator()
        self.undersampler_ = self._resolve_undersampler()
        self.n_features_in_ = X.shape[1]

        unique_sites = np.unique(sites)
        classes = np.unique(y_cls)  # bins for regression
        counts = {site: {c: int(np.sum((sites == site) & (y_cls == c))) for c in classes} for site in unique_sites}
        # classes (bins) present in each site; only regression can miss some (see _validate_input)
        counts = {site: {c: n for c, n in cs.items() if n > 0} for site, cs in counts.items()}
        self.target_count_ = max(max(cs.values()) for cs in counts.values()) if self.balance_strategy == "global_max" else None

        groups = self._group_labels(sites, cat_cov, cont_cov, n_bins_cc, strategy_cc)
        self.effective_dim_ = effective_dimension(X, groups=self._cell_labels(sites, y_cls))

        X_out, y_out, site_out, idx_out = [], [], [], []
        self.target_counts_, self.samples_created_, self.samples_removed_ = {}, {}, {}
        self.amplification_, self.amplification_cap_, self.variance_ratio_ = {}, {}, {}

        for site in unique_sites:
            in_site = np.flatnonzero(sites == site)
            n_max = self.target_count_ if self.target_count_ is not None else max(counts[site].values())
            X_site, y_site, kept = self._resample_site(
                site, X[in_site], y[in_site], y_cls[in_site], groups[in_site], counts[site], n_max, rng
            )
            X_out.append(X_site)
            y_out.append(y_site)
            site_out.append(np.full(len(X_site), site, dtype=sites.dtype))
            idx_out.append(np.concatenate([in_site[kept], np.full(len(X_site) - len(kept), -1)]))

        self.sites_resampled_ = np.concatenate(site_out)
        self.sample_indices_ = np.concatenate(idx_out)
        self.is_synthetic_ = self.sample_indices_ < 0
        X_res, y_res = np.vstack(X_out), np.concatenate(y_out)
        if self.task_ == "classification":
            y_res = y_res.astype(y.dtype, copy=False)
        logger.debug(f"[ISI] {len(X)} samples -> {len(X_res)} ({int(self.is_synthetic_.sum())} synthetic)")
        return X_res, y_res

    def summary(self) -> pd.DataFrame:
        """Return a per site-class report of the resampling.

        Returns
        -------
        pandas.DataFrame
            One row per site and class with the number of real, removed,
            created and final samples, the amplification, its cap, the
            variance ratio of the interpolator and the number of real samples
            per effective dimension.

        """
        check_is_fitted(self, "target_counts_")
        rows = []
        for site, target in self.target_counts_.items():
            for c, n_created in self.samples_created_[site].items():
                n_removed = self.samples_removed_[site][c]
                n_real = target - n_created + n_removed
                rows.append(
                    {
                        "site": site,
                        "class": c,
                        "n_real": n_real,
                        "n_removed": n_removed,
                        "n_created": n_created,
                        "n_final": target,
                        "amplification": self.amplification_[site][c],
                        "amplification_cap": self.amplification_cap_[site][c],
                        "variance_ratio": self.variance_ratio_[site][c],
                        "samples_per_effective_dim": n_real / self.effective_dim_,
                    }
                )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def _resolve_deprecated_covariate_params(
        self, n_bins_cont_cov: int | None, binning_strategy_cont_cov: str | None
    ) -> tuple[int | None, str]:
        """Merge the deprecated ``fit_resample`` covariate options with the constructor ones."""
        n_bins, strategy = self.n_bins_cont_cov, self.binning_strategy_cont_cov
        if n_bins_cont_cov is not None or binning_strategy_cont_cov is not None:
            warnings.warn(
                "Passing `n_bins_cont_cov` / `binning_strategy_cont_cov` to `fit_resample` is deprecated and will be "
                "removed in uniharmony 0.1; set them in the IntraSiteInterpolation constructor.",
                FutureWarning,
                stacklevel=3,
            )
            n_bins = n_bins_cont_cov if n_bins_cont_cov is not None else n_bins
            strategy = binning_strategy_cont_cov if binning_strategy_cont_cov is not None else strategy
        return n_bins, strategy

    def _validate_params_values(self) -> None:
        """Validate the constructor parameters."""
        if self.balance_strategy not in {"per_site", "global_max"}:
            raise ValueError(f"balance_strategy must be 'per_site' or 'global_max', got {self.balance_strategy!r}")
        cap = self.max_amplification
        if isinstance(cap, str):
            if cap != "auto":
                raise ValueError(f"max_amplification must be 'auto', a number >= 0, a callable or None, got {cap!r}")
        elif cap is not None and not callable(cap) and (not np.isscalar(cap) or not cap >= 0):
            raise ValueError(f"max_amplification must be 'auto', a number >= 0, a callable or None, got {cap!r}")
        if not 0 < self.variance_tolerance < 1:
            raise ValueError(f"variance_tolerance must be in (0, 1), got {self.variance_tolerance}")

    def _validate_input(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        sites: npt.ArrayLike,
        categorical_covariate: npt.ArrayLike | None,
        continuous_covariate: npt.ArrayLike | None,
        n_bins_cont_cov: int | None,
        binning_strategy_cont_cov: str,
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray | None, npt.NDArray | None]:
        """Validate the data and derive the classes used for balancing.

        Returns
        -------
        tuple
            ``X``, ``y``, ``sites``, the classes (bins for regression) and the
            categorical and continuous covariates as 2D arrays (or ``None``).

        Raises
        ------
        ValueError
            If the inputs are inconsistent or a class is missing from a site.

        """
        X, y = check_X_y(X, y, estimator=self, dtype="numeric")
        sites = check_array(sites, dtype=None, ensure_2d=False, estimator=self)
        check_consistent_length(X, y, sites)
        validate_sites(sites)

        def _as_2d(cov: npt.ArrayLike | None) -> npt.ArrayLike | None:
            if cov is None:
                return None
            cov = np.asarray(cov)
            return cov.reshape(-1, 1) if cov.ndim == 1 else cov

        cat_cov, cont_cov, _ = validate_covariates(
            X.shape[0], _as_2d(categorical_covariate), _as_2d(continuous_covariate), None, allow_nan=False
        )
        if cont_cov is not None:
            if n_bins_cont_cov is None or n_bins_cont_cov < 2:
                raise ValueError("n_bins_cont_cov (>= 2) must be set when continuous_covariate is given.")
            if binning_strategy_cont_cov not in {"uniform", "quantile"}:
                raise ValueError(f"binning_strategy_cont_cov must be 'uniform' or 'quantile', got {binning_strategy_cont_cov!r}")

        self.task_ = self._infer_task(y)
        if self.task_ == "regression":
            y = y.astype(float)
            y_cls, self.bins_ = self._bin_target(y)
        else:
            y_cls, self.bins_ = y, None
        # A missing class keeps the site predictive of the target: an error for classification. For regression, sparse
        # tail bins are common, so each site is balanced over the bins it has, with a warning.
        is_clf = self.task_ == "classification"
        validate_all_classes_per_site(y_cls, sites, kind="class" if is_clf else "target bin", raise_error=is_clf)
        return X, y, sites, y_cls, cat_cov, cont_cov

    def _resolve_interpolator(self) -> SamplerMixin:
        """Return the over-sampler template (never mutates ``interpolator``)."""
        if isinstance(self.interpolator, str):
            return create_interpolator(self.interpolator, random_state=None, **(self.interpolator_kwargs or {}))
        if isinstance(self.interpolator, SamplerMixin):
            if getattr(self.interpolator, "_sampling_type", "over-sampling") != "over-sampling":
                raise ValueError(f"interpolator must be an over-sampler, got {self.interpolator!r}")
            return clone(self.interpolator)
        raise ValueError(
            f"Invalid interpolator: {self.interpolator!r}. Must be a string (e.g. 'smote') or an imblearn over-sampler."
        )

    def _resolve_undersampler(self) -> SamplerMixin | None:
        """Return the under-sampler template, or ``None``."""
        if self.undersampler is None:
            return None
        if isinstance(self.undersampler, str):
            return create_undersampler(self.undersampler, random_state=None, **(self.undersampler_kwargs or {}))
        if isinstance(self.undersampler, SamplerMixin):
            kind = getattr(self.undersampler, "_sampling_type", "under-sampling")
            if kind == "clean-sampling":
                raise ValueError(
                    f"{type(self.undersampler).__name__} is a cleaning method and cannot reach a target count per class. "
                    "Use RandomUnderSampler, NearMiss, ClusterCentroids or InstanceHardnessThreshold (apply cleaning "
                    "methods before ISI if needed)."
                )
            if kind != "under-sampling":
                raise ValueError(f"undersampler must be an under-sampler, got {self.undersampler!r}")
            return clone(self.undersampler)
        raise ValueError(f"Invalid undersampler: {self.undersampler!r}. Must be a string, an imblearn under-sampler or None.")

    # ------------------------------------------------------------------ #
    # Per-site resampling
    # ------------------------------------------------------------------ #
    def _resample_site(
        self,
        site: Any,
        Xs: np.ndarray,
        ys: np.ndarray,
        ys_cls: np.ndarray,
        groups: np.ndarray,
        counts: dict,
        n_max: int,
        rng: np.random.RandomState,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Balance the classes of one site and record its statistics.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Resampled features and targets of the site (kept real samples
            first, in their original order, then the new samples) and the
            local indices of the kept real samples.

        Raises
        ------
        ValueError
            If the cap requires under-sampling but ``undersampler`` is None.

        """
        classes = list(counts)
        caps, rhos = {}, {}
        for c in classes:
            caps[c], rhos[c] = self._amplification_cap(Xs, ys_cls, groups, c, n_needed=n_max - counts[c], rng=rng)
        target = int(min([n_max] + [np.floor(counts[c] * (1 + caps[c])) for c in classes if np.isfinite(caps[c])]))
        if target < max(counts.values()) and self.undersampler_ is None:
            raise ValueError(
                f"Site {site}: the amplification cap allows at most {target} samples per class, but the largest "
                f"class has {max(counts.values())}. Set an `undersampler`, or relax `max_amplification`."
            )
        logger.debug(f"[ISI] Site {site}: counts={counts}, caps={caps}, target={target}")
        n_removed = sum(max(0, n - target) for n in counts.values())
        if n_removed > 0.5 * sum(counts.values()):
            warnings.warn(
                f"Site {site}: interpolation can only bring its classes to {target} samples each, so {n_removed} of its "
                f"{sum(counts.values())} real samples are removed by under-sampling (class counts: "
                f"{', '.join(f'{c}={n}' for c, n in counts.items())}). Consider "
                "excluding this site, relaxing `max_amplification` / `variance_tolerance`, or checking the site's data.",
                UserWarning,
                stacklevel=3,
            )

        kept, X_new, y_new = [], [], []
        for c in classes:
            if counts[c] > target:
                kept_c, X_c, y_c = self._undersample(Xs, ys, ys_cls, groups, c, target, rng)
            else:
                kept_c = np.flatnonzero(ys_cls == c)
                X_c, y_c = (
                    self._oversample(Xs, ys, ys_cls, groups, c, target - counts[c], rng) if counts[c] < target else (None, None)
                )
            kept.append(kept_c)
            if X_c is not None and len(X_c):
                X_new.append(X_c)
                y_new.append(y_c)
        kept_idx = np.sort(np.concatenate(kept))

        kept_cls = ys_cls[kept_idx]
        self.target_counts_[site] = target
        self.samples_removed_[site] = {c: int(counts[c] - np.sum(kept_cls == c)) for c in classes}
        self.samples_created_[site] = {c: int(target - np.sum(kept_cls == c)) for c in classes}
        self.amplification_[site] = {c: self.samples_created_[site][c] / counts[c] for c in classes}
        self.amplification_cap_[site] = caps
        self.variance_ratio_[site] = rhos
        return np.vstack([Xs[kept_idx], *X_new]), np.concatenate([ys[kept_idx], *y_new]), kept_idx

    # ------------------------------------------------------------------ #
    # Amplification cap
    # ------------------------------------------------------------------ #
    def _min_anchors(self) -> int:
        """Smallest number of real samples the interpolator needs in a cell."""
        return 1 if isinstance(self.interpolator_, RandomOverSampler) else 2

    def _amplification_cap(
        self,
        Xs: np.ndarray,
        ys_cls: np.ndarray,
        groups: np.ndarray,
        c: Any,
        n_needed: int,
        rng: np.random.RandomState,
    ) -> tuple[float, float]:
        """Return the amplification cap ``r*`` of a cell and the variance ratio used for it.

        Returns
        -------
        tuple[float, float]
            ``(r*, rho)``; ``rho`` is ``nan`` unless measured.

        """
        in_cell = ys_cls == c
        n_c = int(in_cell.sum())
        if n_needed <= 0:
            return np.inf, np.nan
        if n_c < self._min_anchors():
            logger.warning(f"[ISI] Class {c!r} has {n_c} sample(s) in a site; it cannot be interpolated.")
            return 0.0, np.nan
        cap = self.max_amplification
        if cap is None:
            return np.inf, np.nan
        if callable(cap) and not isinstance(cap, str):
            return float(cap(Xs[in_cell])), np.nan
        if isinstance(cap, str):  # "auto"
            n_pilot = int(np.clip(2 * n_c, _PILOT_MIN, _PILOT_MAX))
            X_pilot, _ = self._oversample(Xs, None, ys_cls, groups, c, n_pilot, rng)
            rho = variance_ratio(X_pilot, Xs[in_cell])
            if not np.isfinite(rho):
                return np.inf, rho
            distortion = abs(1.0 - rho)
            eps = self.variance_tolerance
            return (np.inf if distortion <= eps else eps / (distortion - eps)), rho
        return float(cap), np.nan

    # ------------------------------------------------------------------ #
    # Over-sampling
    # ------------------------------------------------------------------ #
    def _oversample(
        self,
        Xs: np.ndarray,
        ys: np.ndarray | None,
        ys_cls: np.ndarray,
        groups: np.ndarray,
        c: Any,
        n_new: int,
        rng: np.random.RandomState,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create ``n_new`` synthetic samples of class ``c`` in one site.

        The synthetic samples are spread over the covariate strata in
        proportion to the real samples of the class, and interpolated within
        each stratum. The other classes of the site are passed to the
        interpolator as context (used by borderline / SVM / ADASYN variants).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Synthetic samples and their targets (``ys=None`` skips targets).

        """
        in_cell = ys_cls == c
        context = ~in_cell
        g_cell = groups[in_cell]
        strata, n_per_stratum = np.unique(g_cell, return_counts=True)
        usable = n_per_stratum >= self._min_anchors()
        if not np.any(usable):
            logger.warning(f"[ISI] Class {c!r}: no covariate stratum has enough samples to interpolate; ignoring strata.")
            strata, n_per_stratum, usable = np.array([0]), np.array([int(in_cell.sum())]), np.array([True])
            g_cell = np.zeros(int(in_cell.sum()), dtype=int)
        alloc = allocate_proportionally(n_new, np.where(usable, n_per_stratum, 0))

        X_parts, y_parts = [], []
        X_cell = Xs[in_cell]
        y_cell = ys[in_cell] if ys is not None else None
        for stratum, n_g in zip(strata, alloc, strict=True):
            if n_g == 0:
                continue
            anchors = g_cell == stratum
            X_syn = self._interpolate(X_cell[anchors], Xs[context], ys_cls[context], c, int(n_g), rng)
            X_parts.append(X_syn)
            if y_cell is not None:
                if self.task_ == "regression":
                    y_parts.append(self._parent_targets(X_syn, X_cell[anchors], y_cell[anchors]))
                else:
                    y_parts.append(np.full(len(X_syn), c, dtype=ys.dtype))
        X_out = np.vstack(X_parts)
        y_out = np.concatenate(y_parts) if y_parts else np.empty(0)
        return X_out, y_out

    def _interpolate(
        self,
        X_anchor: np.ndarray,
        X_context: np.ndarray,
        y_context: np.ndarray,
        c: Any,
        n_new: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """Run the interpolator to get exactly ``n_new`` samples of class ``c`` from ``X_anchor``.

        Some interpolators (SVM-SMOTE, ADASYN, KMeans-SMOTE) only approximately
        produce the requested number; a few more are requested and a random
        subset is kept.

        Raises
        ------
        RuntimeError
            If the interpolator cannot produce enough samples.

        """
        n_anchor = len(X_anchor)
        X_in = np.vstack([X_anchor, X_context])
        y_in = np.concatenate([np.full(n_anchor, c, dtype=y_context.dtype), y_context])
        out, missing = [], n_new
        for attempt in range(6):
            sampler = self._configure(self.interpolator_, rng, n_anchor=n_anchor, n_total=len(X_in))
            if isinstance(sampler, RandomOverSampler):
                n_request = missing
            else:  # request a margin, doubled at every retry, and keep a random subset
                n_request = (missing + max(10, int(np.ceil(0.2 * missing)))) * 2**attempt
            sampler.set_params(sampling_strategy={c: n_anchor + n_request})
            try:
                X_res, y_res = sampler.fit_resample(X_in, y_in)
            except (ValueError, RuntimeError) as err:
                if "No samples will be generated" in str(err):  # ADASYN rounds small requests to zero: ask for more
                    continue
                raise type(err)(f"[ISI] {type(sampler).__name__} failed for class {c!r} ({n_anchor} anchors): {err}") from err
            X_syn = self._split_synthetic(X_in, y_in, X_res, y_res)
            if len(X_syn) > missing:
                X_syn = X_syn[np.sort(rng.choice(len(X_syn), missing, replace=False))]
            out.append(X_syn)
            missing -= len(X_syn)
            if missing == 0:
                return np.vstack(out)
        raise RuntimeError(
            f"[ISI] {type(self.interpolator_).__name__} produced {n_new - missing} of the {n_new} samples requested for "
            f"class {c!r}; use another interpolator."
        )

    @staticmethod
    def _split_synthetic(X: np.ndarray, y: np.ndarray, X_resampled: np.ndarray, y_resampled: np.ndarray) -> np.ndarray:
        """Return the synthetic samples appended by an over-sampler.

        Raises
        ------
        ValueError
            If the over-sampler does not return the original samples first,
            unchanged, as imblearn over-samplers do.

        """
        n_samples = len(X)
        if (
            len(X_resampled) < n_samples
            or not np.array_equal(X_resampled[:n_samples], X)
            or not np.array_equal(y_resampled[:n_samples], y)
        ):
            raise ValueError(
                "The interpolator must return the original samples first, unchanged, followed by the "
                "synthetic samples (as imblearn over-samplers do)."
            )
        return X_resampled[n_samples:]

    # ------------------------------------------------------------------ #
    # Under-sampling
    # ------------------------------------------------------------------ #
    def _undersample(
        self,
        Xs: np.ndarray,
        ys: np.ndarray,
        ys_cls: np.ndarray,
        groups: np.ndarray,
        c: Any,
        target: int,
        rng: np.random.RandomState,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reduce class ``c`` of one site to ``target`` samples, stratified by covariates.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Local indices of the kept real samples, and the features and
            targets of prototypes created by prototype-generation methods.

        """
        cell = np.flatnonzero(ys_cls == c)
        context = np.flatnonzero(ys_cls != c)
        strata, n_per_stratum = np.unique(groups[cell], return_counts=True)
        alloc = allocate_proportionally(target, n_per_stratum, capacity=n_per_stratum)
        kept, X_proto, y_proto = [], [], []
        for stratum, n_stratum, n_keep in zip(strata, n_per_stratum, alloc, strict=True):
            members = cell[groups[cell] == stratum]
            if n_keep == n_stratum:
                kept.append(members)
                continue
            if n_keep == 0:
                continue
            idx_in = np.concatenate([members, context])
            # neighbour-based under-samplers (NearMiss) look at both the class and the other classes
            sampler = self._configure(self.undersampler_, rng, n_anchor=min(len(members), len(context)), n_total=len(idx_in))
            sampler.set_params(sampling_strategy={c: int(n_keep)})
            try:
                X_res, y_res = sampler.fit_resample(Xs[idx_in], ys_cls[idx_in])
            except (ValueError, RuntimeError) as err:
                raise type(err)(f"[ISI] {type(sampler).__name__} failed for class {c!r}: {err}") from err
            if hasattr(sampler, "sample_indices_"):
                chosen = idx_in[sampler.sample_indices_]
                chosen = chosen[ys_cls[chosen] == c]
                kept.append(self._exact_count(chosen, members, int(n_keep), rng))
            else:  # prototype generation (e.g. ClusterCentroids)
                if self.task_ == "regression":
                    raise ValueError(
                        f"{type(sampler).__name__} creates new samples and cannot be used for regression; "
                        "use a selection method such as RandomUnderSampler or NearMiss."
                    )
                protos = X_res[y_res == c]
                X_proto.append(protos)
                y_proto.append(np.full(len(protos), c, dtype=ys.dtype))
        kept_idx = np.concatenate(kept) if kept else np.empty(0, dtype=int)
        X_p = np.vstack(X_proto) if X_proto else np.empty((0, Xs.shape[1]))
        y_p = np.concatenate(y_proto) if y_proto else np.empty(0, dtype=ys.dtype)
        return kept_idx, X_p, y_p

    @staticmethod
    def _exact_count(chosen: np.ndarray, members: np.ndarray, n_keep: int, rng: np.random.RandomState) -> np.ndarray:
        """Trim or top up (with unselected real samples) a selection to exactly ``n_keep`` samples."""
        if len(chosen) > n_keep:
            return np.sort(rng.choice(chosen, n_keep, replace=False))
        if len(chosen) < n_keep:
            rest = np.setdiff1d(members, chosen)
            return np.sort(np.concatenate([chosen, rng.choice(rest, n_keep - len(chosen), replace=False)]))
        return chosen

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #
    @staticmethod
    def _configure(template: SamplerMixin, rng: np.random.RandomState, n_anchor: int, n_total: int) -> SamplerMixin:
        """Clone a sampler with its own seed and neighbour counts that fit the data."""
        sampler = clone(template)
        params = sampler.get_params()
        new = {}
        if "random_state" in params:
            new["random_state"] = int(rng.randint(_SEED_MAX))
        for name in _ANCHOR_NEIGHBOR_PARAMS:
            if isinstance(params.get(name), int | np.integer) and params[name] > n_anchor - 1:
                new[name] = max(1, n_anchor - 1)
        for name in _ALL_NEIGHBOR_PARAMS:
            if isinstance(params.get(name), int | np.integer) and params[name] > n_total - 1:
                new[name] = max(1, n_total - 1)
        return sampler.set_params(**new) if new else sampler

    @staticmethod
    def _cell_labels(sites: np.ndarray, y_cls: np.ndarray) -> np.ndarray:
        """Integer label of each site-class cell."""
        site_idx = np.unique(sites, return_inverse=True)[1].ravel()
        cls_idx = np.unique(y_cls, return_inverse=True)[1].ravel()
        return site_idx * (cls_idx.max() + 1) + cls_idx

    def _group_labels(
        self,
        sites: np.ndarray,
        cat: np.ndarray | None,
        cont: np.ndarray | None,
        n_bins_cont_cov: int | None,
        binning_strategy_cont_cov: str,
    ) -> np.ndarray:
        """Covariate stratum of each sample (continuous covariates are binned within each site)."""
        groups = np.zeros(len(sites), dtype=np.int64)
        if cat is None and cont is None:
            return groups
        for site in np.unique(sites):
            mask = sites == site
            groups[mask] = self._create_group_labels(
                cat[mask] if cat is not None else None,
                cont[mask] if cont is not None else None,
                n_bins_cont_cov,
                binning_strategy_cont_cov,
            )
        return groups

    def _infer_task(self, y: np.ndarray) -> str:
        """Return the task: the ``task`` parameter, or inferred from the dtype of ``y``.

        Boolean, integer, string and object targets mean classification;
        anything else (floating point) means regression.
        """
        if self.task not in {"auto", "classification", "regression"}:
            raise ValueError(f"task must be 'auto', 'classification' or 'regression', got {self.task!r}")
        if self.task != "auto":
            return self.task
        return "classification" if y.dtype.kind in "biuUSO" else "regression"

    def _bin_target(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Bin a continuous target into ``n_bins`` classes.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Bin index of each sample (``0`` to ``n_bins - 1``) and bin edges.

        Raises
        ------
        ValueError
            If ``n_bins`` or ``binning_strategy`` is invalid.

        """
        if not isinstance(self.n_bins, int | np.integer) or self.n_bins < 2:
            raise ValueError(f"n_bins must be an integer >= 2 for regression, got {self.n_bins!r}")
        if self.binning_strategy == "uniform":
            bins = np.linspace(y.min(), y.max(), self.n_bins + 1)
        elif self.binning_strategy == "quantile":
            bins = np.quantile(y, np.linspace(0, 1, self.n_bins + 1))
        else:
            raise ValueError(f"binning_strategy must be 'uniform' or 'quantile', got {self.binning_strategy!r}")
        y_bins = np.clip(np.digitize(y, bins[1:-1]), 0, len(bins) - 2)
        return y_bins, bins

    def _create_group_labels(
        self,
        cat: np.ndarray | None,
        cont: np.ndarray | None,
        n_bins_cont_cov: int | None,
        binning_strategy_cont_cov: str = "quantile",
    ) -> np.ndarray:
        """Combine categorical values and binned continuous covariates into one stratum label.

        Returns
        -------
        np.ndarray
            Integer stratum of each sample.

        Raises
        ------
        ValueError
            If neither covariate is given.

        """
        if cat is None and cont is None:
            raise ValueError("At least one of 'cat' or 'cont' must be provided.")
        n_samples = len(cat) if cat is not None else len(cont)
        cat_labels = np.unique(cat, axis=0, return_inverse=True)[1].ravel() if cat is not None else None
        cont_labels = (
            self._resolve_continuous_covariate(cont, n_samples, n_bins_cont_cov, binning_strategy_cont_cov)
            if cont is not None
            else None
        )
        if cat_labels is not None and cont_labels is not None:
            return cat_labels * (cont_labels.max() + 1) + cont_labels
        return cat_labels if cat_labels is not None else cont_labels

    @staticmethod
    def _resolve_continuous_covariate(
        cont: np.ndarray, n_samples: int, n_bins_cont_cov: int | None, binning_strategy_cont_cov: str
    ) -> np.ndarray:
        """Bin each continuous covariate and combine the bins into one integer label (mixed radix).

        Returns
        -------
        np.ndarray
            Integer label of each sample.

        Raises
        ------
        ValueError
            If ``n_bins_cont_cov`` or ``binning_strategy_cont_cov`` is invalid.

        """
        if n_bins_cont_cov is None or n_bins_cont_cov < 2:
            raise ValueError(f"n_bins_cont_cov must be >= 2 when continuous covariates are provided. Got: {n_bins_cont_cov}")
        if binning_strategy_cont_cov not in {"quantile", "uniform"}:
            raise ValueError(f"binning_strategy_cont_cov must be 'quantile' or 'uniform'. Got: {binning_strategy_cont_cov}")
        labels = np.zeros(n_samples, dtype=np.int64)
        for col in cont.T:
            if np.all(col == col[0]):
                bins = np.zeros(n_samples, dtype=np.int64)
            else:
                if binning_strategy_cont_cov == "quantile":
                    edges = np.unique(np.percentile(col, np.linspace(0, 100, n_bins_cont_cov + 1)))
                else:
                    edges = np.linspace(col.min(), col.max(), n_bins_cont_cov + 1)
                bins = np.zeros(n_samples, dtype=np.int64) if len(edges) <= 2 else np.digitize(col, edges[1:-1])
            labels = labels * (bins.max() + 1) + bins
        return labels

    @staticmethod
    def _parent_targets(X_new: np.ndarray, X_anchor: np.ndarray, y_anchor: np.ndarray, k: int = 10) -> np.ndarray:
        """Interpolate the continuous targets of synthetic samples from their parents.

        For each synthetic sample ``x``, the parents are the two anchors
        ``x_a``, ``x_b`` whose segment passes closest to ``x``; with ``lam``
        the position of the projection of ``x`` on the segment, the target is
        ``y_a + lam * (y_b - y_a)``. Candidate segments join the nearest
        anchors of ``x`` to their own nearest anchors (as in SMOTE); when no
        candidate contains ``x`` exactly, all segments are searched for cells
        of up to 1000 anchors.

        Returns
        -------
        np.ndarray
            Targets of the synthetic samples.

        """
        n = len(X_anchor)
        y_anchor = y_anchor.astype(float)
        if n == 1:
            return np.full(len(X_new), y_anchor[0])
        n_cand = min(n, max(2 * k + 1, 15))
        cand = NearestNeighbors(n_neighbors=n_cand).fit(X_anchor).kneighbors(X_new, return_distance=False)
        nbrs = NearestNeighbors(n_neighbors=min(n, k + 1)).fit(X_anchor).kneighbors(X_anchor, return_distance=False)
        scale = max(1.0, float(np.mean(np.sum(X_anchor**2, axis=1))))
        y_new = np.empty(len(X_new))
        for i, x in enumerate(X_new):
            a = np.repeat(cand[i], nbrs.shape[1])
            b = nbrs[cand[i]].ravel()
            lam, res = IntraSiteInterpolation._project(x, X_anchor[a], X_anchor[b])
            best = int(np.argmin(res))
            if res[best] > 1e-12 * scale and n <= 1000:
                a, b = np.triu_indices(n, k=1)
                lam, res = IntraSiteInterpolation._project(x, X_anchor[a], X_anchor[b])
                best = int(np.argmin(res))
            y_new[i] = y_anchor[a[best]] + lam[best] * (y_anchor[b[best]] - y_anchor[a[best]])
        return y_new

    @staticmethod
    def _project(x: np.ndarray, A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Position (clipped to [0, 1]) and squared distance of ``x`` projected on the segments ``[A_i, B_i]``.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``lam`` and squared residual of each segment.

        """
        d = B - A
        norm2 = np.einsum("ij,ij->i", d, d)
        u = x - A
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = np.where(norm2 > 0, np.clip(np.einsum("ij,ij->i", u, d) / norm2, 0.0, 1.0), 0.0)
        r = u - lam[:, np.newaxis] * d
        return lam, np.einsum("ij,ij->i", r, r)

    def _fit_resample(self, X: npt.ArrayLike, y: npt.ArrayLike, **params: Any) -> tuple[npt.NDArray, npt.NDArray]:
        """Resample; ``sites`` must be passed as a keyword argument.

        Returns
        -------
        tuple
            Output of :meth:`fit_resample`.

        Raises
        ------
        TypeError
            If ``sites`` is not given.

        """
        if "sites" not in params:
            raise TypeError("IntraSiteInterpolation needs `sites`: call fit_resample(X, y, sites=sites).")
        return self.fit_resample(X, y, **params)

    def __sklearn_tags__(self) -> Tags:
        """Return sklearn compatibility tags."""
        tags = super().__sklearn_tags__()
        tags.estimator_type = "sampler"
        return tags
