"""Within-site balancing by over-sampling (ISI) combined with under-sampling."""

import warnings
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import structlog
from imblearn.base import SamplerMixin
from sklearn.base import BaseEstimator, clone
from sklearn.utils import Tags, check_random_state
from sklearn.utils.validation import check_is_fitted

from uniharmony.interpolation._intra_site import IntraSiteInterpolation
from uniharmony.interpolation._utils import allocate_proportionally, create_undersampler


__all__ = ["SiteBalancedResampler"]

logger = structlog.get_logger()


class SiteBalancedResampler(SamplerMixin, BaseEstimator):
    """Balance the classes within every site by over-sampling *and* under-sampling.

    ``SiteBalancedResampler`` combines a within-site over-sampler,
    :class:`~uniharmony.interpolation.IntraSiteInterpolation` (ISI, with any
    interpolator, including ``"random"`` duplication), with an imblearn
    under-sampler. Over-sampling is limited to at most ``max_amplification``
    synthetic samples per real sample of a class; the remaining imbalance is
    closed by under-sampling the larger classes of the site, so both meet in
    the middle. As with ISI alone, every class then has the same number of
    samples in a site and the site no longer predicts the target.

    For each site ``s`` with ``n_sc`` real samples in class ``c``:

    1. The site target ``n_max`` is the largest class of the site, of any
       site, or a given number (the ``balance_strategy`` of the over-sampler).
    2. Each class below ``n_max`` gets a cap ``r_sc`` (``max_amplification``)
       and the site target becomes
       ``T_s = min(n_max, min_c floor(n_sc * (1 + r_sc)))``.
    3. Classes above ``T_s`` are reduced to ``T_s`` real samples with
       ``undersampler`` (stratified by covariates).
    4. The over-sampler brings the classes below ``T_s`` up to ``T_s``.

    ``max_amplification=0`` gives pure within-site under-sampling and
    ``max_amplification=None`` gives the over-sampler alone.

    Parameters
    ----------
    oversampler : IntraSiteInterpolation or None, optional (default None)
        Within-site over-sampler. ``None`` means
        ``IntraSiteInterpolation(random_state=random_state)`` (SMOTE). Its
        ``balance_strategy``, binning and covariate options are used for the
        whole procedure. Use ``IntraSiteInterpolation("random")`` for
        within-site duplication.

    undersampler : str or imblearn under-sampler, optional (default "cluster-centroids")
        Under-sampler for the classes above the capped site target. Strings:
        ``"cluster-centroids"`` (k-means on the class, keeping the real
        sample closest to each centroid, i.e.
        ``ClusterCentroids(voting="hard")``, so the kept samples cover the
        whole class), ``"nearmiss"`` (``"nearmiss-1"``), ``"nearmiss-2"``,
        ``"nearmiss-3"``, ``"instance-hardness"`` or ``"random"``. Any imblearn
        under-sampler that selects real samples and accepts a target count per
        class (``dict`` ``sampling_strategy``) can be passed. Cleaning methods
        (Tomek links, ENN, ...) cannot reach an exact count, and prototype
        generators (``ClusterCentroids(voting="soft")``) create new samples;
        both are rejected.

    undersampler_kwargs : dict or None, optional (default None)
        Keyword arguments for ``undersampler`` when it is given as a string.

    max_amplification : float, "auto", callable or None, optional (default "auto")
        Maximum number of synthetic samples per real sample of a class.

        - ``"auto"``: the safe amplification of each class from the variance
          rule of the over-sampler (see
          ``IntraSiteInterpolation.variance_tolerance``), measured on all real
          samples of the site before anything is removed.
        - float ``>= 0``: the same cap for every class (``0`` means pure
          under-sampling).
        - callable: ``f(X_cell) -> float`` returning the cap of a class from
          its real samples.
        - ``None``: no cap (the over-sampler alone; nothing is removed).

    random_state : int, RandomState instance or None, optional (default None)
        Seed for the under-sampler and for the pilot draws of
        ``max_amplification="auto"`` (and for the default over-sampler). A
        given over-sampler uses its own ``random_state``.

    Attributes
    ----------
    oversampler_ : IntraSiteInterpolation
        The fitted over-sampler (its ``summary()``, ``variance_ratio_``,
        ``effective_dim_``, ... describe the over-sampling step).

    undersampler_ : imblearn under-sampler
        Template of the under-sampler used.

    class_counts_ : dict
        ``{site: {class: n_real}}`` before resampling.

    target_counts_ : dict
        ``{site: T_s}``, the number of samples per class in each site after
        resampling.

    samples_removed_ : dict
        ``{site: {class: n_removed}}``.

    samples_created_ : dict
        ``{site: {class: n_created}}``.

    amplification_ : dict
        ``{site: {class: n_created / n_real}}``.

    amplification_cap_ : dict
        ``{site: {class: cap}}`` applied (``inf`` where not limited).

    variance_ratio_, safe_amplification_ : dict
        ``{site: {class: value}}`` measured before resampling for
        ``max_amplification="auto"`` (``nan`` otherwise, and for classes
        that did not need over-sampling).

    sites_resampled_ : ndarray of shape (n_samples_new,)
        Site of each resampled sample.

    sample_indices_ : ndarray of shape (n_samples_new,)
        Index in the input of each resampled sample, ``-1`` for synthetic
        samples.

    is_synthetic_ : ndarray of shape (n_samples_new,)
        Whether each resampled sample was created by the over-sampler.

    effective_dim_ : float
        Effective dimension of the data (see ``IntraSiteInterpolation``).

    task_ : str
        ``"classification"`` or ``"regression"``.

    bins_ : ndarray or None
        Bin edges of the regression target (computed on all samples and used
        for both steps), ``None`` for classification.

    n_features_in_ : int
        Number of features.

    See Also
    --------
    IntraSiteInterpolation : Within-site balancing by interpolation only.

    Notes
    -----
    The same rules as for ISI apply: resample training data only, and tune
    hyper-parameters with the resampler inside the inner cross-validation.

    With many features and a strong imbalance, up-weighting methods (ISI,
    random over-sampling, re-weighting) leave part of the site-target
    association in models that can fit individual samples, whereas
    under-sampling does not but discards data. The cap trades one for the
    other; check the remaining association by repeating the cross-validation
    after permuting the target within each site, and look at
    ``samples_removed_``: when most samples of a site must be removed, the
    imbalance is beyond what within-site balancing can repair without losing
    information.

    Examples
    --------
    >>> from uniharmony.interpolation import IntraSiteInterpolation, SiteBalancedResampler
    >>> sbr = SiteBalancedResampler(IntraSiteInterpolation("smote"), max_amplification="auto")
    >>> X_bal, y_bal = sbr.fit_resample(X, y, sites=sites)  # doctest: +SKIP

    """

    def __init__(
        self,
        oversampler: IntraSiteInterpolation | None = None,
        undersampler: str
        | Literal["random", "nearmiss", "nearmiss-1", "nearmiss-2", "nearmiss-3", "cluster-centroids", "instance-hardness"]
        | SamplerMixin = "cluster-centroids",
        undersampler_kwargs: dict | None = None,
        max_amplification: float | str | Literal["auto"] | Callable[[np.ndarray], float] | None = "auto",
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.oversampler = oversampler
        self.undersampler = undersampler
        self.undersampler_kwargs = undersampler_kwargs
        self.max_amplification = max_amplification
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
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Balance the classes within every site.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features.

        y : array-like of shape (n_samples,)
            Target (classes, or continuous values binned by the over-sampler).

        sites : array-like of shape (n_samples,)
            Site of each sample.

        categorical_covariate, continuous_covariate : array-like or None, default=None
            Covariates (see ``IntraSiteInterpolation.fit_resample``). Both
            steps are stratified by them.

        Returns
        -------
        X_resampled : numpy.ndarray of shape (n_samples_new, n_features)
            Resampled features, grouped by site.

        y_resampled : numpy.ndarray of shape (n_samples_new,)
            Resampled targets.

        Raises
        ------
        ValueError
            If a parameter is invalid or the data do not allow within-site
            balancing (see ``IntraSiteInterpolation``).

        """
        self._validate_params()
        os_ = IntraSiteInterpolation(random_state=self.random_state) if self.oversampler is None else clone(self.oversampler)
        rng = check_random_state(self.random_state)
        X, y, sites, y_cls, cat_cov, cont_cov = os_._validate_input(
            X, y, sites, categorical_covariate, continuous_covariate, os_.n_bins_cont_cov, os_.binning_strategy_cont_cov
        )
        os_._validate_params_values()
        os_.interpolator_ = os_._resolve_interpolator()
        self.undersampler_ = self._resolve_undersampler()
        self.task_, self.bins_, self.n_features_in_ = os_.task_, os_.bins_, X.shape[1]

        counts = os_._class_counts(sites, y_cls)
        site_max = os_._site_targets(counts)
        groups = os_._group_labels(sites, cat_cov, cont_cov, os_.n_bins_cont_cov, os_.binning_strategy_cont_cov)
        self.class_counts_ = counts
        self.target_counts_, self.samples_removed_, self.amplification_cap_ = {}, {}, {}
        self.variance_ratio_, self.safe_amplification_ = {}, {}

        kept = []
        for site in np.unique(sites):
            in_site = np.flatnonzero(sites == site)
            kept_site = self._balance_site(
                site, os_, X[in_site], y_cls[in_site], groups[in_site], counts[site], site_max[site], rng
            )
            kept.append(in_site[kept_site])
        kept = np.sort(np.concatenate(kept))

        # over-sample the kept samples up to the site targets, with the target bins of the full data
        os_.set_params(balance_strategy=dict(self.target_counts_))
        if self.task_ == "regression":
            os_.set_params(n_bins=self.bins_)
        os_._skip_unsafe_warning = True  # reported below, with advice that fits the wrapper
        cov = {}
        if categorical_covariate is not None:
            cov["categorical_covariate"] = np.asarray(categorical_covariate)[kept]
        if continuous_covariate is not None:
            cov["continuous_covariate"] = np.asarray(continuous_covariate)[kept]
        X_res, y_res = os_.fit_resample(X[kept], y[kept], sites=sites[kept], **cov)
        self.oversampler_ = os_
        if not isinstance(self.max_amplification, str):  # "auto" stays within the safe amount by design
            os_._warn_unsafe_amplification(
                advice="Lower `max_amplification` (or use 'auto', the safe amplification) to interpolate less and "
                "under-sample more.",
                stacklevel=3,
            )

        self.sites_resampled_ = os_.sites_resampled_
        self.sample_indices_ = np.where(os_.sample_indices_ >= 0, kept[np.maximum(os_.sample_indices_, 0)], -1)
        self.is_synthetic_ = os_.is_synthetic_
        self.samples_created_ = {s: dict(d) for s, d in os_.samples_created_.items()}
        for site, cs in counts.items():  # classes removed entirely do not appear in the over-sampler
            for c in cs:
                self.samples_created_[site].setdefault(c, 0)
        self.amplification_ = {s: {c: self.samples_created_[s][c] / n for c, n in cs.items()} for s, cs in counts.items()}
        self.effective_dim_ = os_.effective_dim_
        if self.task_ == "classification":
            y_res = y_res.astype(y.dtype, copy=False)
        n_removed = sum(sum(d.values()) for d in self.samples_removed_.values())
        logger.debug(f"[SBR] {len(X)} samples -> {len(X_res)} ({n_removed} removed, {int(self.is_synthetic_.sum())} synthetic)")
        return X_res, y_res

    def summary(self) -> pd.DataFrame:
        """Return a per site-class report of the resampling.

        Returns
        -------
        pandas.DataFrame
            One row per site and class with the number of real, removed,
            created and final samples, the amplification, the cap applied and,
            for ``max_amplification="auto"``, the variance ratio and safe
            amplification measured before resampling.

        """
        check_is_fitted(self, "target_counts_")
        rows = []
        for site, cs in self.class_counts_.items():
            for c, n_real in cs.items():
                rows.append(
                    {
                        "site": site,
                        "class": c,
                        "n_real": n_real,
                        "n_removed": self.samples_removed_[site][c],
                        "n_created": self.samples_created_[site][c],
                        "n_final": n_real - self.samples_removed_[site][c] + self.samples_created_[site][c],
                        "amplification": self.amplification_[site][c],
                        "amplification_cap": self.amplification_cap_[site][c],
                        "variance_ratio": self.variance_ratio_[site][c],
                        "safe_amplification": self.safe_amplification_[site][c],
                    }
                )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def _validate_params(self) -> None:
        """Validate the constructor parameters."""
        if self.oversampler is not None and not isinstance(self.oversampler, IntraSiteInterpolation):
            raise ValueError(
                f"oversampler must be an IntraSiteInterpolation instance (use interpolator='random' for within-site "
                f"duplication), got {self.oversampler!r}"
            )
        cap = self.max_amplification
        message = f"max_amplification must be 'auto', a number >= 0, a callable or None, got {cap!r}"
        if isinstance(cap, str):
            if cap != "auto":
                raise ValueError(message)
        elif cap is not None and not callable(cap) and (not np.isscalar(cap) or not cap >= 0):
            raise ValueError(message)

    def _resolve_undersampler(self) -> SamplerMixin:
        """Return the under-sampler template."""
        if isinstance(self.undersampler, str):
            return create_undersampler(self.undersampler, random_state=None, **(self.undersampler_kwargs or {}))
        if isinstance(self.undersampler, SamplerMixin):
            kind = getattr(self.undersampler, "_sampling_type", "under-sampling")
            if kind == "clean-sampling":
                raise ValueError(
                    f"{type(self.undersampler).__name__} is a cleaning method and cannot reach a target count per class. "
                    "Use RandomUnderSampler, NearMiss, ClusterCentroids or InstanceHardnessThreshold (apply cleaning "
                    "methods beforehand if needed)."
                )
            if kind != "under-sampling":
                raise ValueError(f"undersampler must be an under-sampler, got {self.undersampler!r}")
            return clone(self.undersampler)
        raise ValueError(f"Invalid undersampler: {self.undersampler!r}. Must be a string or an imblearn under-sampler.")

    # ------------------------------------------------------------------ #
    # Per-site target and under-sampling
    # ------------------------------------------------------------------ #
    def _cap(self, X_cell: np.ndarray, safe: float) -> float:
        """Amplification cap of a class, from ``max_amplification``."""
        cap = self.max_amplification
        if cap is None:
            return np.inf
        if isinstance(cap, str):  # "auto"
            return safe
        if callable(cap):
            return float(cap(X_cell))
        return float(cap)

    def _balance_site(
        self,
        site: Any,
        os_: IntraSiteInterpolation,
        Xs: np.ndarray,
        ys_cls: np.ndarray,
        groups: np.ndarray,
        counts: dict,
        n_max: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """Compute the capped target of a site and under-sample the classes above it.

        Returns
        -------
        np.ndarray
            Local indices of the real samples kept in the site.

        """
        # Regression bins too small to interpolate are left as they are by the over-sampler, so they do not limit
        # the target. A class too small to interpolate cannot be over-sampled (its safe amplification is 0).
        min_n = os_._min_anchors()
        frozen = [c for c, n in counts.items() if n < min_n] if self.task_ == "regression" else []
        classes = [c for c in counts if c not in frozen]
        rhos, safe, caps = {}, {}, {}
        for c in classes:
            if counts[c] < n_max and isinstance(self.max_amplification, str):
                if counts[c] >= min_n:
                    rhos[c], safe[c] = os_._safe_amplification(Xs, ys_cls, groups, c, n_needed=n_max - counts[c], rng=rng)
                else:
                    rhos[c], safe[c] = np.nan, 0.0
            caps[c] = self._cap(Xs[ys_cls == c], safe.get(c, np.nan)) if counts[c] < n_max else np.inf
        limits = [np.floor(counts[c] * (1 + caps[c])) for c in classes if np.isfinite(caps[c])]
        target = int(min([n_max, *limits]))

        kept = [np.flatnonzero(np.isin(ys_cls, frozen))]
        for c in classes:
            if counts[c] > target:
                kept.append(self._undersample(Xs, ys_cls, groups, c, target, rng))
            else:
                kept.append(np.flatnonzero(ys_cls == c))
        kept_idx = np.sort(np.concatenate(kept))

        kept_cls = ys_cls[kept_idx]
        self.target_counts_[site] = target
        self.samples_removed_[site] = {c: int(counts[c] - np.sum(kept_cls == c)) for c in counts}
        self.amplification_cap_[site] = {c: caps.get(c, np.inf) for c in counts}
        self.variance_ratio_[site] = {c: rhos.get(c, np.nan) for c in counts}
        self.safe_amplification_[site] = {c: safe.get(c, np.nan) for c in counts}
        n_removed = sum(self.samples_removed_[site].values())
        logger.debug(f"[SBR] Site {site}: counts={counts}, caps={caps}, target={target}, removed={n_removed}")
        if n_removed > 0.5 * sum(counts.values()):
            warnings.warn(
                f"Site {site}: with the amplification cap ({self._cap_description()}) its classes can only be brought to "
                f"{target} samples each, so {n_removed} of its {sum(counts.values())} real samples are removed by "
                f"under-sampling (class counts: {', '.join(f'{c}={n}' for c, n in counts.items())}). Consider excluding "
                "this site or relaxing `max_amplification`.",
                UserWarning,
                stacklevel=3,
            )
        return kept_idx

    def _cap_description(self) -> str:
        """Short text describing ``max_amplification`` for messages."""
        cap = self.max_amplification
        return "a custom function" if callable(cap) and not isinstance(cap, str) else f"max_amplification={cap!r}"

    def _undersample(
        self,
        Xs: np.ndarray,
        ys_cls: np.ndarray,
        groups: np.ndarray,
        c: Any,
        target: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """Reduce class ``c`` of one site to ``target`` real samples, stratified by covariates.

        Returns
        -------
        np.ndarray
            Local indices of the kept samples.

        Raises
        ------
        ValueError
            If the under-sampler creates new samples instead of selecting real ones.

        """
        cell = np.flatnonzero(ys_cls == c)
        context = np.flatnonzero(ys_cls != c)
        strata, n_per_stratum = np.unique(groups[cell], return_counts=True)
        alloc = allocate_proportionally(target, n_per_stratum, capacity=n_per_stratum)
        kept = []
        for stratum, n_stratum, n_keep in zip(strata, n_per_stratum, alloc, strict=True):
            members = cell[groups[cell] == stratum]
            if n_keep == n_stratum:
                kept.append(members)
                continue
            if n_keep == 0:
                continue
            idx_in = np.concatenate([members, context])
            # neighbour-based under-samplers (NearMiss) look at both the class and the other classes
            sampler = IntraSiteInterpolation._configure(
                self.undersampler_, rng, n_anchor=min(len(members), len(context)), n_total=len(idx_in)
            )
            sampler.set_params(sampling_strategy={c: int(n_keep)})
            try:
                X_res, y_res = sampler.fit_resample(Xs[idx_in], ys_cls[idx_in])
            except (ValueError, RuntimeError) as err:
                raise type(err)(f"[SBR] {type(sampler).__name__} failed for class {c!r}: {err}") from err
            chosen = self._selected_members(sampler, X_res, y_res, Xs, idx_in, members, c)
            if chosen is None:
                raise ValueError(
                    f"{type(sampler).__name__} creates new samples instead of selecting real ones; use a selection method "
                    "such as ClusterCentroids(voting='hard') (the default 'cluster-centroids'), NearMiss or "
                    "RandomUnderSampler."
                )
            kept.append(self._exact_count(chosen, members, int(n_keep), rng))
        return np.concatenate(kept) if kept else np.empty(0, dtype=int)

    @staticmethod
    def _selected_members(
        sampler: SamplerMixin,
        X_res: np.ndarray,
        y_res: np.ndarray,
        Xs: np.ndarray,
        idx_in: np.ndarray,
        members: np.ndarray,
        c: Any,
    ) -> np.ndarray | None:
        """Local indices of the real samples of class ``c`` kept by an under-sampler, ``None`` if it made new ones.

        Selection methods expose ``sample_indices_``; ``ClusterCentroids(voting="hard")`` returns copies of real
        samples, which are matched back to their rows.
        """
        if hasattr(sampler, "sample_indices_"):
            chosen = idx_in[sampler.sample_indices_]
            return chosen[np.isin(chosen, members)]
        lookup = {Xs[i].tobytes(): i for i in members}
        matched = [lookup.get(row.tobytes()) for row in X_res[y_res == c]]
        if all(m is not None for m in matched):
            return np.unique(np.asarray(matched, dtype=int))  # hard voting may pick the same sample twice
        return None

    @staticmethod
    def _exact_count(chosen: np.ndarray, members: np.ndarray, n_keep: int, rng: np.random.RandomState) -> np.ndarray:
        """Trim or top up (with unselected real samples) a selection to exactly ``n_keep`` samples."""
        if len(chosen) > n_keep:
            return np.sort(rng.choice(chosen, n_keep, replace=False))
        if len(chosen) < n_keep:
            rest = np.setdiff1d(members, chosen)
            return np.sort(np.concatenate([chosen, rng.choice(rest, n_keep - len(chosen), replace=False)]))
        return chosen

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
            raise TypeError("SiteBalancedResampler needs `sites`: call fit_resample(X, y, sites=sites).")
        return self.fit_resample(X, y, **params)

    def __sklearn_tags__(self) -> Tags:
        """Return sklearn compatibility tags."""
        tags = super().__sklearn_tags__()
        tags.estimator_type = "sampler"
        return tags
