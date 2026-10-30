"""Intra-site interpolation-based harmonization."""

import warnings
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import structlog
from imblearn.base import SamplerMixin
from imblearn.over_sampling import RandomOverSampler
from sklearn.base import BaseEstimator, clone
from sklearn.utils import Tags, check_random_state
from sklearn.utils.validation import check_array, check_consistent_length, check_is_fitted, check_X_y

from uniharmony._utils import validate_sites
from uniharmony.interpolation._utils import (
    allocate_proportionally,
    create_interpolator,
    effective_dimension,
    validate_all_classes_per_site,
    validate_covariates,
    variance_ratio,
)


__all__ = ["IntraSiteInterpolation"]

logger = structlog.get_logger(src="IntraSiteInterpolation")

# Number of synthetic samples drawn to estimate how well an interpolator preserves the variance of a cell.
_PILOT_MIN, _PILOT_MAX = 200, 2000
# Neighbour parameters of imblearn samplers that must not exceed the number of anchors (or of samples).
_ANCHOR_NEIGHBOR_PARAMS = ("k_neighbors", "n_neighbors")
_ALL_NEIGHBOR_PARAMS = ("m_neighbors",)
_SEED_MAX = np.iinfo(np.int32).max


class IntraSiteInterpolation(SamplerMixin, BaseEstimator):
    """Intra-Site Interpolation (ISI).

    ISI removes the association between site and target in a training set by
    **balancing the classes within every site through interpolation**.
    Minority classes are over-sampled by interpolating between real samples
    of the same class (and, optionally, the same covariate stratum) *of the
    same site*, so the synthetic samples carry the biological variability of
    that class and the effect of site of that site. ISI never removes a real
    sample.

    For each site ``s`` and class ``c`` with ``n_sc`` real samples, every
    class of the site is brought to the target ``T_s`` (the largest class of
    the site, of any site, or a given number, see ``balance_strategy``) by
    adding ``T_s - n_sc`` samples created with ``interpolator``. After
    resampling every class has ``T_s`` samples in site ``s``, so
    ``P(y | site)`` is uniform and the site no longer predicts the target.

    **How much interpolation is advisable.** Interpolated samples are less
    spread than real ones, so the more synthetic samples a class receives, the
    more its variance shrinks. For every class that needs over-sampling ISI
    measures this with a pilot batch of synthetic samples (``rho``, see
    ``variance_tolerance``) and derives the largest amplification that keeps
    the variance of the class within ``variance_tolerance`` of the real one
    (``safe_amplification_``). If balancing needs more than that, ISI warns.
    To limit interpolation to the safe amount and close the remaining
    imbalance by under-sampling the larger classes, wrap ISI in
    :class:`~uniharmony.interpolation.SiteBalancedResampler`.

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

    balance_strategy : {"per_site", "global_max"} or dict, optional (default "per_site")
        - ``"per_site"``: every site is balanced to its own largest class.
        - ``"global_max"``: every site is balanced towards the largest class
          of any site, so sites also get the same size. Small sites then need
          much more interpolation.
        - dict ``{site: n}``: every class of each site is brought to ``n``
          samples (``n`` must not be smaller than the largest class of the
          site, since ISI only adds samples).

    variance_tolerance : float, optional (default 0.2)
        Largest relative loss of the within-class variance considered safe
        (``eps``). The rule works as follows. The interpolator generates
        samples that are less spread than the real ones: for SMOTE each new
        sample lies on the segment between a real sample and one of its
        nearest neighbours of the same class, so it is pulled towards the
        inside of the class. The ratio between the variance of the synthetic
        samples and that of the real samples, ``rho`` (averaged over the
        features), is measured on a pilot batch; it is close to 1 when many
        samples densely cover the class (close neighbours) and falls to about
        2/3 when the samples are sparse (two independent points mixed with a
        uniform weight keep ``E[(1 - l)^2 + l^2] = 2/3`` of the variance), and
        to 1/6 with only two samples (one segment). A class made of ``n`` real
        and ``r * n`` synthetic samples keeps a fraction
        ``(1 + r * rho) / (1 + r)`` of its real variance, i.e. it loses
        ``r * (1 - rho) / (1 + r)``. Keeping this loss below ``eps`` gives the
        safe amplification::

            r* = eps / ((1 - rho) - eps)   if 1 - rho > eps, else no limit

        (``|1 - rho|`` is used, so samplers that inflate the variance are
        limited too.) For example, with ``rho = 0.6`` and ``eps = 0.2``,
        ``r* = 1``: at most one synthetic sample per real one. Duplication
        (``"random"``) keeps ``rho`` close to one and is not limited by this
        rule, although it adds no new variability.

    n_bins : int or array-like, optional (default 10)
        For regression: number of bins of the target, or the bin edges
        (non-decreasing; values outside are assigned to the first / last bin).

    binning_strategy : {"uniform", "quantile"}, optional (default "quantile")
        How the regression target is binned when ``n_bins`` is an integer:
        equal-width (``"uniform"``) or equal-frequency (``"quantile"``) bins.

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

    class_counts_ : dict
        ``{site: {class: n_real}}``, real samples per site and class.

    target_counts_ : dict
        ``{site: T_s}``, the number of samples per class in each site after
        resampling.

    target_count_ : int or None
        Largest class count of any site for ``balance_strategy="global_max"``,
        ``None`` otherwise.

    samples_created_ : dict
        ``{site: {class: n_created}}``. For regression, classes are bins.

    amplification_ : dict
        ``{site: {class: n_created / n_real}}``.

    variance_ratio_ : dict
        ``{site: {class: rho}}`` measured on the pilot batch (``nan`` for
        classes that did not need over-sampling).

    safe_amplification_ : dict
        ``{site: {class: r*}}`` from the variance rule (``inf`` when not
        limited; ``nan`` for classes that did not need over-sampling).

    effective_dim_ : float
        Effective number of dimensions of the data (participation ratio of
        the correlation matrix of the features, centered within each
        site-class cell; estimated from at most 2000 samples). Few samples
        per effective dimension mean that interpolation fills the space
        poorly (low ``rho``).

    n_features_in_ : int
        Number of features.

    bins_ : ndarray or None
        Bin edges of the regression target, ``None`` for classification.

    task_ : str
        ``"classification"`` or ``"regression"``.

    interpolator_ : imblearn over-sampler
        Template of the over-sampler used.

    See Also
    --------
    SiteBalancedResampler : Combine ISI (or within-site random over-sampling) with under-sampling.
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

    **Tune hyper-parameters with ISI inside the inner cross-validation.**
    Synthetic samples are interpolated between their parents, so a model
    validated on samples whose parents it was trained on looks better than it
    is. Estimators that tune themselves on the data they receive (``RidgeCV``
    and ``RidgeClassifierCV`` with generalised cross-validation,
    ``LogisticRegressionCV``, early stopping on a validation split, ...)
    therefore pick too little regularisation when fitted on ISI output. Put
    ISI in the pipeline that is tuned instead, so it is re-applied to every
    inner training split::

        pipe = Pipeline([("isi", isi), ("scale", StandardScaler()), ("ridge", Ridge())])
        search = GridSearchCV(pipe, {"ridge__alpha": np.logspace(-1, 5, 13)})
        search.fit(X, y, sites=sites)  # with metadata routing enabled

    **Strong imbalance with many features.** A model with more features
    than samples and moderate regularisation can fit the few real samples of
    a heavily over-sampled class one by one. Synthetic samples lie between
    these real samples and add little to that fit, so ISI then removes the
    site-target association only partly, like re-weighting or random
    over-sampling with the same amplification. The remaining shortcut grows
    with the amplification and shrinks with the regularisation;
    under-sampling does not have this problem. When ISI warns that classes
    exceed their safe amplification, consider
    ``SiteBalancedResampler(IntraSiteInterpolation(), max_amplification="auto")``
    (interpolate up to the safe amount, then under-sample the larger
    classes), and check the remaining shortcut, for instance by repeating
    the cross-validation after permuting the target within each site.

    **Local and kernel models.** Interpolated classes are denser and less
    dispersed than real ones. Nearest-neighbour models can recognise the
    synthetic class of a site by its density, and kernel models with a
    bandwidth estimated on the resampled data are affected too. For them,
    prefer sample weights, under-sampling or ``interpolator="random"``
    (duplication), and check the remaining shortcut.

    Every class that needs over-sampling must have at least two samples in
    its site (one for ``"random"``): a single sample cannot be interpolated.
    For regression, target bins with fewer samples in a site are left as
    they are, with a warning.

    Covariates (``categorical_covariate``, ``continuous_covariate``) restrict
    who is interpolated with whom: synthetic samples are interpolated between
    samples of the same class, site and covariate stratum, and they are spread
    over the strata in proportion to the real samples of that class, so
    ``P(covariates | class, site)`` is preserved.

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
        balance_strategy: str | Literal["per_site", "global_max"] | dict = "per_site",
        variance_tolerance: float = 0.2,
        n_bins: int | npt.ArrayLike = 10,
        binning_strategy: str | Literal["uniform", "quantile"] = "quantile",
        task: str | Literal["auto", "classification", "regression"] = "auto",
        n_bins_cont_cov: int | None = None,
        binning_strategy_cont_cov: str | Literal["uniform", "quantile"] = "quantile",
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.interpolator = interpolator
        self.interpolator_kwargs = interpolator_kwargs
        self.balance_strategy = balance_strategy
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
        """Balance the classes within every site by interpolation.

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
            class (or target bin) is missing from a site, a class that needs
            over-sampling has too few samples, or a parameter is invalid.

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
        self.n_features_in_ = X.shape[1]

        unique_sites = np.unique(sites)
        counts = self._class_counts(sites, y_cls)
        targets = self._site_targets(counts)
        self.target_count_ = max(max(cs.values()) for cs in counts.values()) if self.balance_strategy == "global_max" else None
        self._check_cell_sizes(counts, targets)
        self.class_counts_ = counts

        groups = self._group_labels(sites, cat_cov, cont_cov, n_bins_cc, strategy_cc)
        self.effective_dim_ = effective_dimension(X, groups=self._cell_labels(sites, y_cls))

        X_out, y_out, site_out, idx_out = [], [], [], []
        self.target_counts_, self.samples_created_ = {}, {}
        self.amplification_, self.variance_ratio_, self.safe_amplification_ = {}, {}, {}

        for site in unique_sites:
            in_site = np.flatnonzero(sites == site)
            X_site, y_site = self._resample_site(
                site, X[in_site], y[in_site], y_cls[in_site], groups[in_site], counts[site], targets[site], rng
            )
            X_out.append(X_site)
            y_out.append(y_site)
            site_out.append(np.full(len(X_site), site, dtype=sites.dtype))
            idx_out.append(np.concatenate([in_site, np.full(len(X_site) - len(in_site), -1)]))

        self.sites_resampled_ = np.concatenate(site_out)
        self.sample_indices_ = np.concatenate(idx_out)
        self.is_synthetic_ = self.sample_indices_ < 0
        if not getattr(self, "_skip_unsafe_warning", False):
            self._warn_unsafe_amplification()
        X_res, y_res = np.vstack(X_out), np.concatenate(y_out)
        if self.task_ == "classification":
            y_res = y_res.astype(y.dtype, copy=False)
        logger.debug(f"[ISI] {len(X)} samples -> {len(X_res)} ({int(self.is_synthetic_.sum())} synthetic)")
        return X_res, y_res

    def estimate_safe_amplification(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        sites: npt.ArrayLike,
        *,
        categorical_covariate: npt.ArrayLike | None = None,
        continuous_covariate: npt.ArrayLike | None = None,
    ) -> pd.DataFrame:
        """Measure, without resampling, how much interpolation each class of each site can take.

        For every site-class cell with enough samples to be interpolated, a
        pilot batch of synthetic samples is drawn and the variance ratio
        ``rho`` and the safe amplification ``r*`` (see ``variance_tolerance``)
        are computed, exactly as in :meth:`fit_resample`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features.

        y : array-like of shape (n_samples,)
            Target.

        sites : array-like of shape (n_samples,)
            Site of each sample.

        categorical_covariate, continuous_covariate : array-like or None, default=None
            Covariates, as in :meth:`fit_resample`.

        Returns
        -------
        pandas.DataFrame
            One row per site and class with ``n_real``, ``variance_ratio``
            and ``safe_amplification`` (``inf`` when not limited).

        """
        X, y, sites, y_cls, cat_cov, cont_cov = self._validate_input(
            X, y, sites, categorical_covariate, continuous_covariate, self.n_bins_cont_cov, self.binning_strategy_cont_cov
        )
        self._validate_params_values()
        rng = check_random_state(self.random_state)
        self.interpolator_ = self._resolve_interpolator()
        groups = self._group_labels(sites, cat_cov, cont_cov, self.n_bins_cont_cov, self.binning_strategy_cont_cov)
        rows = []
        for site, cs in self._class_counts(sites, y_cls).items():
            in_site = sites == site
            for c, n in cs.items():
                rho, r_safe = (
                    self._safe_amplification(X[in_site], y_cls[in_site], groups[in_site], c, n_needed=1, rng=rng)
                    if n >= self._min_anchors()
                    else (np.nan, np.nan)
                )
                rows.append({"site": site, "class": c, "n_real": n, "variance_ratio": rho, "safe_amplification": r_safe})
        return pd.DataFrame(rows)

    def summary(self) -> pd.DataFrame:
        """Return a per site-class report of the resampling.

        Returns
        -------
        pandas.DataFrame
            One row per site and class with the number of real, created and
            final samples, the amplification, the variance ratio of the
            interpolator (``rho``), the safe amplification derived from it and
            the number of real samples per effective dimension.

        """
        check_is_fitted(self, "target_counts_")
        rows = []
        for site in self.target_counts_:
            for c, n_created in self.samples_created_[site].items():
                n_real = self.class_counts_[site][c]
                rows.append(
                    {
                        "site": site,
                        "class": c,
                        "n_real": n_real,
                        "n_created": n_created,
                        "n_final": n_real + n_created,
                        "amplification": self.amplification_[site][c],
                        "variance_ratio": self.variance_ratio_[site][c],
                        "safe_amplification": self.safe_amplification_[site][c],
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
        strategy = self.balance_strategy
        if not isinstance(strategy, dict) and strategy not in {"per_site", "global_max"}:
            raise ValueError(f"balance_strategy must be 'per_site', 'global_max' or a dict {{site: n}}, got {strategy!r}")
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

    @staticmethod
    def _class_counts(sites: np.ndarray, y_cls: np.ndarray) -> dict:
        """Return ``{site: {class: n}}`` for the classes (regression bins) present in each site."""
        counts = {}
        for site in np.unique(sites):
            labels, n = np.unique(y_cls[sites == site], return_counts=True)
            counts[site] = {c: int(k) for c, k in zip(labels, n, strict=True)}
        return counts

    def _site_targets(self, counts: dict) -> dict:
        """Return the number of samples per class of each site after balancing, from ``balance_strategy``.

        Raises
        ------
        ValueError
            If a target given as a dict is missing for a site or smaller than the largest class of the site.

        """
        if self.balance_strategy == "global_max":
            top = max(max(cs.values()) for cs in counts.values())
            return dict.fromkeys(counts, top)
        if self.balance_strategy == "per_site":
            return {site: max(cs.values()) for site, cs in counts.items()}
        targets = {}
        for site, cs in counts.items():
            if site not in self.balance_strategy:
                raise ValueError(f"balance_strategy gives no target for site {site!r}.")
            target = int(self.balance_strategy[site])
            if target < max(cs.values()):
                raise ValueError(
                    f"Site {site!r}: the target ({target}) is smaller than its largest class ({max(cs.values())}). ISI "
                    "only adds samples; use SiteBalancedResampler to also remove samples."
                )
            targets[site] = target
        return targets

    def _check_cell_sizes(self, counts: dict, targets: dict) -> None:
        """Every class that needs over-sampling needs at least two samples in its site (one to be duplicated).

        Raises
        ------
        ValueError
            For classification, if a class that needs over-sampling has too few samples in a site.

        """
        min_n = self._min_anchors()
        small = [(site, c, n) for site, cs in counts.items() for c, n in cs.items() if n < min_n and n < targets[site]]
        if not small:
            return
        detail = "; ".join(f"site {site}: class {c} has {n} sample" for site, c, n in small)
        if self.task_ == "classification":
            raise ValueError(
                f"Every class needs at least {min_n} samples in every site to be interpolated (a single sample cannot be "
                f"interpolated). Too few samples: {detail}. Remove or merge these sites."
            )
        warnings.warn(
            f"Target bins with fewer than {min_n} samples in a site cannot be interpolated and are left as they are, so "
            f"the site still carries information about the target: {detail}. Use fewer bins (n_bins), or restrict the "
            "training data to the target range that all sites share.",
            UserWarning,
            stacklevel=3,
        )

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
        target: int,
        rng: np.random.RandomState,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bring every class of one site to ``target`` samples by interpolation and record its statistics.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Resampled features and targets of the site (real samples first,
            in their original order, then the new samples).

        """
        # Classes (regression bins) too small to interpolate are left as they are (see fit_resample).
        frozen = [c for c, n in counts.items() if n < self._min_anchors()]
        classes = [c for c in counts if c not in frozen]

        rhos, safe, X_new, y_new = {}, {}, [], []
        for c in classes:
            n_needed = target - counts[c]
            # safe amplification (variance rule) from all real samples of the site
            rhos[c], safe[c] = self._safe_amplification(Xs, ys_cls, groups, c, n_needed=n_needed, rng=rng)
            if n_needed > 0:
                X_c, y_c = self._oversample(Xs, ys, ys_cls, groups, c, n_needed, rng)
                X_new.append(X_c)
                y_new.append(y_c)
        logger.debug(f"[ISI] Site {site}: counts={counts}, rho={rhos}, safe={safe}, target={target}")

        self.target_counts_[site] = target
        self.samples_created_[site] = {c: (target - counts[c] if c in classes else 0) for c in counts}
        self.amplification_[site] = {c: self.samples_created_[site][c] / counts[c] for c in counts}
        self.variance_ratio_[site] = {c: rhos.get(c, np.nan) for c in counts}
        self.safe_amplification_[site] = {c: safe.get(c, np.nan) for c in counts}
        return np.vstack([Xs, *X_new]), np.concatenate([ys, *y_new])

    # ------------------------------------------------------------------ #
    # Amplification: variance rule
    # ------------------------------------------------------------------ #
    def _n_neighbors(self) -> int:
        """Return the number of neighbours the interpolator uses (``k_neighbors`` / ``n_neighbors``; 5 if unknown)."""
        params = self.interpolator_.get_params()
        for name in _ANCHOR_NEIGHBOR_PARAMS:
            if isinstance(params.get(name), int | np.integer):
                return int(params[name])
        return 5

    def _min_anchors(self) -> int:
        """Smallest number of real samples the interpolator needs in a cell (two to interpolate, one to duplicate)."""
        return 1 if isinstance(self.interpolator_, RandomOverSampler) else 2

    def _safe_amplification(
        self,
        Xs: np.ndarray,
        ys_cls: np.ndarray,
        groups: np.ndarray,
        c: Any,
        n_needed: int,
        rng: np.random.RandomState,
    ) -> tuple[float, float]:
        """Measure ``rho`` for class ``c`` of a site and derive the safe amplification ``r*``.

        A pilot batch of synthetic samples is drawn with the interpolator from
        all real samples of the site (nothing is removed). ``rho`` is the
        ratio between the variance of these samples and that of the real
        samples of the class; a class of ``n`` real and ``r * n`` synthetic
        samples loses ``r * |1 - rho| / (1 + r)`` of its variance, which stays
        below ``eps = variance_tolerance`` for ``r <= eps / (|1 - rho| - eps)``.

        Returns
        -------
        tuple[float, float]
            ``(rho, r*)``; both ``nan`` if the class needs no over-sampling,
            ``r* = inf`` if the variance loss stays below ``eps`` for any ``r``.

        """
        if n_needed <= 0:
            return np.nan, np.nan
        in_cell = ys_cls == c
        n_pilot = int(np.clip(2 * in_cell.sum(), _PILOT_MIN, _PILOT_MAX))
        X_pilot, _ = self._oversample(Xs, None, ys_cls, groups, c, n_pilot, rng)
        rho = variance_ratio(X_pilot, Xs[in_cell])
        if not np.isfinite(rho):
            return rho, np.inf
        loss_per_sample, eps = abs(1.0 - rho), self.variance_tolerance
        return rho, (np.inf if loss_per_sample <= eps else eps / (loss_per_sample - eps))

    def _warn_unsafe_amplification(self, advice: str | None = None, stacklevel: int = 3) -> None:
        """Warn, in one message, about the classes interpolated beyond their safe amplification.

        ``advice`` replaces the closing recommendation (used by ``SiteBalancedResampler``).
        """
        if advice is None:
            advice = (
                "Consider SiteBalancedResampler(IntraSiteInterpolation(), max_amplification='auto'), which limits "
                "interpolation to the safe amount and under-samples the larger classes, or excluding these sites."
            )
        rows = []
        for site, amps in self.amplification_.items():
            for c, r in amps.items():
                r_safe, rho = self.safe_amplification_[site][c], self.variance_ratio_[site][c]
                if np.isfinite(r_safe) and r > r_safe + 1e-12:
                    n_real = self.class_counts_[site][c]
                    kept = (1 + r * rho) / (1 + r)
                    rows.append(
                        f"  site {site}, class {c}: {n_real} real -> {n_real + self.samples_created_[site][c]} samples "
                        f"({r:.3g} synthetic per real sample, {r / (1 + r):.1%} of the class); safe up to {r_safe:.2g} per "
                        f"real sample (about {int(np.floor(n_real * r_safe))} synthetic samples); the class keeps about "
                        f"{kept:.0%} of its variance (rho={rho:.2f})."
                    )
        if rows:
            warnings.warn(
                f"IntraSiteInterpolation: {len(rows)} class(es) are too small, compared with the largest class of their "
                f"site, to be balanced by interpolation while keeping their variance (variance_tolerance="
                f"{self.variance_tolerance}). Most of their samples are synthetic and the class shrinks towards its "
                "centre:\n" + "\n".join(rows) + "\nWith many features, models can also fit the few real samples of "
                "these classes one by one, so part of the site-target association remains (see the Notes of "
                f"IntraSiteInterpolation). {advice}",
                UserWarning,
                stacklevel=stacklevel,
            )

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
                    y_parts.append(self._parent_targets(X_syn, X_cell[anchors], y_cell[anchors], k=self._n_neighbors()))
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
                raise type(err)(
                    f"[ISI] {type(sampler).__name__} failed for class {c!r} ({n_anchor} anchors): {err} "
                    "Use interpolator='smote', which works with any class of at least two samples."
                ) from err
            X_syn = self._split_synthetic(X_in, y_in, X_res, y_res)
            if len(X_syn) > missing:
                X_syn = X_syn[np.sort(rng.choice(len(X_syn), missing, replace=False))]
            out.append(X_syn)
            missing -= len(X_syn)
            if missing == 0:
                return np.vstack(out)
        raise RuntimeError(
            f"[ISI] {type(self.interpolator_).__name__} produced {n_new - missing} of the {n_new} samples requested for "
            f"class {c!r}. Use interpolator='smote', which always produces the requested number of samples."
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
        """Bin a continuous target into ``n_bins`` classes (or the bins given by edges).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Bin index of each sample (``0`` to ``n_bins - 1``) and bin edges.

        Raises
        ------
        ValueError
            If ``n_bins`` or ``binning_strategy`` is invalid.

        """
        if not isinstance(self.n_bins, int | np.integer):  # bin edges
            bins = np.asarray(self.n_bins, dtype=float)
            # non-decreasing: quantile edges of a discrete target repeat values (empty bins), as with an integer n_bins
            if bins.ndim != 1 or len(bins) < 3 or np.any(np.diff(bins) < 0) or bins[-1] <= bins[0]:
                raise ValueError(f"n_bins must be an integer >= 2 or non-decreasing bin edges (at least 3), got {self.n_bins!r}")
            return np.clip(np.digitize(y, bins[1:-1]), 0, len(bins) - 2), bins
        if self.n_bins < 2:
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
        ``y_a + lam * (y_b - y_a)``. All segments are searched for cells of up
        to 300 anchors; for larger cells, the candidates join the ``max(5k, 50)``
        nearest anchors of ``x`` to their own ``2k`` nearest anchors (SMOTE joins
        a sample to one of its ``k`` nearest neighbours). Distances
        are computed from inner products, so the cost does not grow with the
        number of features once the Gram matrices are formed.

        Returns
        -------
        np.ndarray
            Targets of the synthetic samples.

        """
        n = len(X_anchor)
        y_anchor = y_anchor.astype(float)
        if n == 1 or len(X_new) == 0:
            return np.full(len(X_new), y_anchor[0] if n else np.nan)
        G = X_anchor @ X_anchor.T
        sq = np.diag(G).copy()
        H = X_new @ X_anchor.T
        xx = np.einsum("ij,ij->i", X_new, X_new)
        y_new = np.empty(len(X_new))
        if n <= 300:
            a_all, b_all = np.triu_indices(n)  # includes a == b (duplicates)
            chunk = max(1, int(4e6 // len(a_all)))
            for start in range(0, len(X_new), chunk):
                rows = np.arange(start, min(start + chunk, len(X_new)))
                a = np.broadcast_to(a_all, (len(rows), len(a_all)))
                b = np.broadcast_to(b_all, (len(rows), len(b_all)))
                y_new[rows] = IntraSiteInterpolation._best_segment(rows, a, b, G, sq, H, xx, y_anchor)
            return y_new
        n_cand = min(n, max(5 * k, 50))
        d2 = xx[:, np.newaxis] - 2.0 * H + sq[np.newaxis, :]
        cand = np.argpartition(d2, n_cand - 1, axis=1)[:, :n_cand]
        D = sq[:, np.newaxis] + sq[np.newaxis, :] - 2.0 * G
        n_nbrs = min(n, 2 * k + 1)
        nbrs = np.argpartition(D, n_nbrs - 1, axis=1)[:, :n_nbrs]
        residual = np.empty(len(X_new))
        chunk = max(1, int(4e6 // (n_cand * n_nbrs)))
        for start in range(0, len(X_new), chunk):
            rows = np.arange(start, min(start + chunk, len(X_new)))
            a = np.repeat(cand[rows], n_nbrs, axis=1)
            b = nbrs[cand[rows]].reshape(len(rows), -1)
            y_new[rows], residual[rows] = IntraSiteInterpolation._best_segment(rows, a, b, G, sq, H, xx, y_anchor, True)
        # exact search over all segments for the few samples whose parents were not among the candidates
        missed = np.flatnonzero(residual > 1e-9 * max(1.0, float(np.mean(sq))))
        if len(missed) and n <= 3000:
            a_all, b_all = np.triu_indices(n)
            for i in missed:
                rows = np.array([i])
                y_new[i] = IntraSiteInterpolation._best_segment(
                    rows, a_all[np.newaxis], b_all[np.newaxis], G, sq, H, xx, y_anchor
                )[0]
        return y_new

    @staticmethod
    def _best_segment(
        rows: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        G: np.ndarray,
        sq: np.ndarray,
        H: np.ndarray,
        xx: np.ndarray,
        y_anchor: np.ndarray,
        return_residual: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Target interpolated on the closest of the candidate segments ``[a, b]`` of each synthetic sample.

        Returns
        -------
        np.ndarray or tuple[np.ndarray, np.ndarray]
            Interpolated target of each synthetic sample in ``rows`` (and the
            squared distance to its segment if ``return_residual``).

        """
        Hr = H[rows]
        ha = np.take_along_axis(Hr, a, axis=1)
        hb = np.take_along_axis(Hr, b, axis=1)
        gab = G[a, b]
        ud = hb - ha - gab + sq[a]  # (x - x_a) . (x_b - x_a)
        dd = sq[a] + sq[b] - 2.0 * gab  # |x_b - x_a|^2
        uu = xx[rows, np.newaxis] - 2.0 * ha + sq[a]  # |x - x_a|^2
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = np.where(dd > 0, np.clip(ud / dd, 0.0, 1.0), 0.0)
        res = uu - 2.0 * lam * ud + lam**2 * dd
        best = np.argmin(res, axis=1)
        idx = np.arange(len(rows))
        a_best, b_best, lam_best = a[idx, best], b[idx, best], lam[idx, best]
        y_best = y_anchor[a_best] + lam_best * (y_anchor[b_best] - y_anchor[a_best])
        return (y_best, res[idx, best]) if return_residual else y_best

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
