"""Provide BARTharm transformer."""

# Adapted from:
# https://github.com/NeuroSML/BARTharm (commit 2614d52)
# with no license specified.
#
# The soft BART forests are translated from the SoftBart R package
# (https://github.com/theodds/SoftBART, licensed under GPL (>= 2)), see ``_softbart.py``.

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import numpy.typing as npt
import structlog
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import Tags, check_random_state
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import FLOAT_DTYPES, check_array, check_consistent_length, check_is_fitted

from ._softbart import ForestSnapshot, SoftBARTForest, quantile_normalize_bart


__all__ = ["BARTharm"]

logger = structlog.get_logger(src="BARTharm")

# Shape and rate of the inverse-gamma priors on the error variance and on the
# site variance scales (``alpha0`` and ``beta0`` in ``bartharm()``)
_PRIOR_SHAPE = 0.01
_PRIOR_RATE = 0.01
# Lower bound of the sampled variances (``pmax(..., 1e-12)`` in ``bartharm_inference()``)
_MIN_VARIANCE = 1e-12


class BARTharm(TransformerMixin, BaseEstimator):
    r"""Harmonize scanner effects using image quality metrics (IQMs).

    BARTharm [1]_ models every imaging-derived phenotype (feature) ``y`` as

    .. math::

        y = \mu(\text{IQMs}) + \tau(\text{biological covariates}) + \epsilon,
        \quad \epsilon \sim N(0, \sigma^2),

    where the scanner effect :math:`\mu` and the biological effect :math:`\tau`
    are soft Bayesian additive regression tree (SoftBART) ensembles [2]_ fitted
    jointly by Gibbs sampling. The harmonized feature is the observed feature
    minus the posterior estimate of the scanner effect, :math:`y - \hat\mu`,
    so no site labels are needed.

    With ``var_scaling=True`` the error variance is additionally allowed to
    differ between sites, :math:`\epsilon_i \sim N(0, \sigma^2 \delta_{s(i)}^2)`,
    and the harmonized feature is
    :math:`(y - \mu - \tau) / \delta_{s(i)} + \tau`, with the site scales
    :math:`\delta` normalized to a geometric mean of 1 in every draw.

    Each feature is z-scored before fitting and transformed back afterwards;
    the IQMs and biological covariates are quantile normalized to [0, 1].

    Parameters
    ----------
    n_iter : int, optional (default 5000)
        Number of Gibbs iterations.
    burn_in : int, optional (default 500)
        Number of initial iterations discarded from the posterior.
    thinning_interval : int, optional (default 2)
        Keep every ``thinning_interval``-th iteration.
    n_trees_mu : int, optional (default 200)
        Number of trees of the scanner effect (IQM) forest.
    n_trees_tau : int, optional (default 50)
        Number of trees of the biological effect forest.
    beta_mu, beta_tau : float, optional (default 2.0)
        Power of the tree depth prior, ``gamma * (1 + depth)^(-beta)``, of the
        scanner and biological forests. Larger values give shallower trees.
    gamma_mu, gamma_tau : float, optional (default 0.95)
        Base of the tree depth prior of the scanner and biological forests.
    var_scaling : bool, optional (default False)
        Whether to also harmonize site-specific error variances. Requires
        ``sites`` in :meth:`fit` and :meth:`transform`.
    posterior_summary : {"mean", "median"}, optional (default "mean")
        Summary of the posterior draws of the harmonized features.
    n_stored_draws : int, optional (default 200)
        Number of posterior draws (evenly spaced among the kept ones) whose
        forests are stored to harmonize new data in :meth:`transform`.
        Memory grows linearly with it.
    n_jobs : int or None, optional (default None)
        Number of features fitted in parallel. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context, ``-1`` means all processors.
    random_state : int, RandomState instance or None, optional (default None)
        Seed of the Gibbs sampler. Pass an int for reproducible results.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during :meth:`fit`.
    n_iqm_covariates_ : int
        Number of IQMs seen during :meth:`fit`.
    n_biological_covariates_ : int
        Number of biological covariates seen during :meth:`fit` (0 if none).
    feature_means_ : ndarray, shape (n_features,)
        Means used to z-score the features.
    feature_stds_ : ndarray, shape (n_features,)
        Standard deviations (``ddof=1``) used to z-score the features.
    rmse_ : ndarray, shape (n_features,)
        Root mean squared error, on the z-scored scale, of the posterior mean
        prediction :math:`\hat\mu + \hat\tau` on the training data.
    sigma_draws_ : ndarray, shape (n_features, n_kept_draws)
        Kept posterior draws of the error standard deviation (z-scored scale;
        with ``var_scaling=True``, not rescaled by the alignment of the site scales).
    sites_ : ndarray, shape (n_sites,)
        Fitted site labels. Only with ``var_scaling=True``.
    site_scales_ : ndarray, shape (n_features, n_sites)
        Posterior means of the site scales :math:`\delta` (geometric mean 1
        per draw). Only with ``var_scaling=True``.

    Notes
    -----
    The translation follows the BARTharm R code rather than the paper where the
    two differ (default number of trees and depth prior of :math:`\tau`, number
    of iterations and burn-in, and the order of the Gibbs updates).

    Differences to the R code:

    * :meth:`transform` harmonizes new samples with the stored posterior draws
      of the forests (the R code only harmonizes the samples it was fitted on).
      New IQMs and biological covariates are normalized by interpolating the
      empirical distribution of the training data. :meth:`fit_transform` uses
      all kept draws, so it can differ slightly from ``fit().transform()``.
    * The R code appends the integer site codes, not normalized, as an extra
      IQM column when site labels are given. As the forests only split in
      [0, 1], that column carries almost no information; it is not added here.
    * All saved draws after ``burn_in`` are kept. When ``burn_in`` is not a
      multiple of ``thinning_interval``, the R code also drops the last draw.
    * Missing values are not allowed (the R code drops incomplete rows).
    * Features with zero variance are returned unchanged (the R code returns NaN).

    Fitting is slow: every feature needs ``n_iter`` sweeps over
    ``n_trees_mu + n_trees_tau`` trees. Use ``n_jobs`` to fit features in
    parallel.

    References
    ----------
    .. [1] Prevot E, et al., (2025).
           BARTharm: MRI Harmonization Using Image Quality Metrics and Bayesian Non-parametric.
           bioRxiv. Published online 2025.
           doi:10.1101/2025.06.04.657792

    .. [2] Linero, A. R., & Yang, Y. (2018).
           Bayesian regression tree ensembles that adapt to smoothness and sparsity.
           Journal of the Royal Statistical Society: Series B, 80(5), 1087-1110.
           doi:10.1111/rssb.12293

    Examples
    --------
    >>> from uniharmony.iqm import BARTharm
    >>> harmonizer = BARTharm(random_state=0, n_jobs=-1)
    >>> X_harmonized = harmonizer.fit_transform(X, iqms, biological_covariates=age_sex)  # doctest: +SKIP
    >>> X_test_harmonized = harmonizer.transform(X_test, iqms_test)  # doctest: +SKIP

    """

    _parameter_constraints: dict = {  # noqa: RUF012
        "n_iter": [Interval(Integral, 1, None, closed="left")],
        "burn_in": [Interval(Integral, 0, None, closed="left")],
        "thinning_interval": [Interval(Integral, 1, None, closed="left")],
        "n_trees_mu": [Interval(Integral, 1, None, closed="left")],
        "n_trees_tau": [Interval(Integral, 1, None, closed="left")],
        "beta_mu": [Interval(Real, 0, None, closed="left")],
        "beta_tau": [Interval(Real, 0, None, closed="left")],
        "gamma_mu": [Interval(Real, 0, 1, closed="neither")],
        "gamma_tau": [Interval(Real, 0, 1, closed="neither")],
        "var_scaling": ["boolean"],
        "posterior_summary": [StrOptions({"mean", "median"})],
        "n_stored_draws": [Interval(Integral, 1, None, closed="left")],
        "n_jobs": [Integral, None],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        n_iter: int = 5000,
        burn_in: int = 500,
        thinning_interval: int = 2,
        n_trees_mu: int = 200,
        n_trees_tau: int = 50,
        beta_mu: float = 2.0,
        beta_tau: float = 2.0,
        gamma_mu: float = 0.95,
        gamma_tau: float = 0.95,
        var_scaling: bool = False,
        posterior_summary: str = "mean",
        n_stored_draws: int = 200,
        n_jobs: int | None = None,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.n_iter = n_iter
        self.burn_in = burn_in
        self.thinning_interval = thinning_interval
        self.n_trees_mu = n_trees_mu
        self.n_trees_tau = n_trees_tau
        self.beta_mu = beta_mu
        self.beta_tau = beta_tau
        self.gamma_mu = gamma_mu
        self.gamma_tau = gamma_tau
        self.var_scaling = var_scaling
        self.posterior_summary = posterior_summary
        self.n_stored_draws = n_stored_draws
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(
        self,
        X: npt.ArrayLike,
        iqm_covariates: npt.ArrayLike,
        biological_covariates: npt.ArrayLike | None = None,
        sites: npt.ArrayLike | None = None,
    ) -> "BARTharm":
        """Fit the BARTharm model to every feature.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The imaging-derived features to harmonize.
        iqm_covariates : array-like, shape (n_samples, n_iqm_covariates) or (n_samples,)
            The image quality metrics, which model the scanner effects.
        biological_covariates : array-like, shape (n_samples, n_biological_covariates) or (n_samples,) or None, \
optional (default None)
            The biological covariates (e.g., age, sex), which model the
            biological signal that must not be removed. Strongly recommended:
            without them, biological variation correlated with the IQMs can be
            attributed to the scanner. Include the variables that the IQMs are
            related to (e.g., a diagnosis associated with head motion).
            Categorical covariates must be numerically coded.
        sites : array-like, shape (n_samples,) or None, optional (default None)
            Site labels. Required if ``var_scaling=True``, ignored otherwise.

        Returns
        -------
        self : BARTharm
            The fitted transformer.

        """
        self._fit(X, iqm_covariates, biological_covariates, sites)
        return self

    def fit_transform(
        self,
        X: npt.ArrayLike,
        iqm_covariates: npt.ArrayLike,
        biological_covariates: npt.ArrayLike | None = None,
        sites: npt.ArrayLike | None = None,
    ) -> npt.NDArray:
        """Fit the model and harmonize the training data.

        The harmonized data are summarized over all kept posterior draws, as in
        the BARTharm R code.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The imaging-derived features to harmonize.
        iqm_covariates : array-like, shape (n_samples, n_iqm_covariates) or (n_samples,)
            The image quality metrics.
        biological_covariates : array-like, shape (n_samples, n_biological_covariates) or (n_samples,) or None, \
optional (default None)
            The biological covariates.
        sites : array-like, shape (n_samples,) or None, optional (default None)
            Site labels. Required if ``var_scaling=True``, ignored otherwise.

        Returns
        -------
        ndarray, shape (n_samples, n_features)
            The harmonized features.

        """
        return self._fit(X, iqm_covariates, biological_covariates, sites)

    def transform(
        self,
        X: npt.ArrayLike,
        iqm_covariates: npt.ArrayLike,
        biological_covariates: npt.ArrayLike | None = None,
        sites: npt.ArrayLike | None = None,
    ) -> npt.NDArray:
        """Harmonize (new) data with the stored posterior draws.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The imaging-derived features to harmonize.
        iqm_covariates : array-like, shape (n_samples, n_iqm_covariates) or (n_samples,)
            The image quality metrics.
        biological_covariates : array-like, shape (n_samples, n_biological_covariates) or (n_samples,) or None, \
optional (default None)
            The biological covariates. Only needed (and then required) if
            ``var_scaling=True`` and they were given in :meth:`fit`.
        sites : array-like, shape (n_samples,) or None, optional (default None)
            Site labels, all seen in :meth:`fit`. Required if ``var_scaling=True``.

        Returns
        -------
        ndarray, shape (n_samples, n_features)
            The harmonized features.

        """
        check_is_fitted(self)
        X = check_array(X, dtype=FLOAT_DTYPES, estimator=self)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but BARTharm was fitted with {self.n_features_in_} features.")
        X_iqm = self._iqm_normalizer.transform(
            _check_covariates(X, iqm_covariates, "iqm_covariates", self, self.n_iqm_covariates_)
        )
        X_bio = None
        site_codes = None
        if self.var_scaling:
            if self.n_biological_covariates_:
                if biological_covariates is None:
                    raise ValueError("biological_covariates were used in fit and are required with var_scaling=True.")
                X_bio = self._bio_normalizer.transform(
                    _check_covariates(X, biological_covariates, "biological_covariates", self, self.n_biological_covariates_)
                )
            site_codes = self._encode_sites(X, sites)

        X_std = (X - self.feature_means_) / np.where(self.feature_stds_ == 0.0, 1.0, self.feature_stds_)
        out = np.empty_like(X)
        for j in range(self.n_features_in_):
            if self.feature_stds_[j] == 0.0:
                out[:, j] = X[:, j]
                continue
            draws = [
                _harmonize_draw(X_std[:, j], mu.predict(X_iqm), tau, delta, site_codes)
                for mu, tau, delta in self._iter_stored_draws(j, X_bio)
            ]
            harmonized = _summarize(np.asarray(draws), self.posterior_summary)
            out[:, j] = harmonized * self.feature_stds_[j] + self.feature_means_[j]
        return out

    # ------------------------------------------------------------------

    def _fit(
        self,
        X: npt.ArrayLike,
        iqm_covariates: npt.ArrayLike,
        biological_covariates: npt.ArrayLike | None,
        sites: npt.ArrayLike | None,
    ) -> npt.NDArray:
        """Fit the model and return the harmonized training data."""
        self._validate_params()
        n_saved, n_burn_saved = self._check_sampler_settings()
        X, iqm, bio, site_codes = self._check_fit_inputs(X, iqm_covariates, biological_covariates, sites)

        self.n_features_in_ = X.shape[1]
        self.n_iqm_covariates_ = iqm.shape[1]
        self.n_biological_covariates_ = 0 if bio is None else bio.shape[1]
        self._iqm_normalizer = _QuantileNormalizer().fit(iqm)
        self._bio_normalizer = None if bio is None else _QuantileNormalizer().fit(bio)
        X_iqm = quantile_normalize_bart(iqm)
        X_bio = None if bio is None else quantile_normalize_bart(bio)

        self.feature_means_ = X.mean(axis=0)
        self.feature_stds_ = X.std(axis=0, ddof=1)
        constant = self.feature_stds_ == 0.0
        if np.any(constant):
            logger.warning(f"Features {np.flatnonzero(constant).tolist()} are constant and are returned unchanged.")
        X_std = (X - self.feature_means_) / np.where(constant, 1.0, self.feature_stds_)

        # Independent random streams per feature, so results do not depend on n_jobs
        entropy = check_random_state(self.random_state).randint(np.iinfo(np.int32).max)
        seeds = np.random.SeedSequence(entropy).spawn(self.n_features_in_)
        config = _SamplerConfig(
            n_iter=self.n_iter,
            thinning_interval=self.thinning_interval,
            n_burn_saved=n_burn_saved,
            n_saved=n_saved,
            n_trees_mu=self.n_trees_mu,
            n_trees_tau=self.n_trees_tau,
            beta_mu=self.beta_mu,
            beta_tau=self.beta_tau,
            gamma_mu=self.gamma_mu,
            gamma_tau=self.gamma_tau,
            posterior_summary=self.posterior_summary,
            n_stored_draws=self.n_stored_draws,
            n_sites=0 if site_codes is None else len(self.sites_),
        )
        logger.debug(f"Fitting BARTharm to {self.n_features_in_} features")
        fitted = np.flatnonzero(~constant)
        results: list[_FeatureResult | None] = [None] * self.n_features_in_
        for j, result in zip(
            fitted,
            Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_feature)(X_std[:, j], X_iqm, X_bio, site_codes, config, seeds[j]) for j in fitted
            ),
            strict=True,
        ):
            results[j] = result
        return self._collect_results(X, results, n_saved - n_burn_saved)

    def _check_sampler_settings(self) -> tuple[int, int]:
        """Return the number of saved draws and of saved draws discarded as burn-in."""
        n_saved = self.n_iter // self.thinning_interval
        n_burn_saved = self.burn_in // self.thinning_interval
        if n_saved < 1:
            raise ValueError(f"n_iter ({self.n_iter}) must be at least thinning_interval ({self.thinning_interval}).")
        if n_burn_saved >= n_saved:
            raise ValueError(
                f"No posterior draws left after burn-in: burn_in ({self.burn_in}) must be smaller than "
                f"n_iter ({self.n_iter}) by at least thinning_interval ({self.thinning_interval})."
            )
        return n_saved, n_burn_saved

    def _check_fit_inputs(
        self,
        X: npt.ArrayLike,
        iqm_covariates: npt.ArrayLike,
        biological_covariates: npt.ArrayLike | None,
        sites: npt.ArrayLike | None,
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray | None, npt.NDArray | None]:
        """Validate the inputs of fit; fit the site labels with ``var_scaling``."""
        X = check_array(X, dtype=FLOAT_DTYPES, ensure_min_samples=2, estimator=self)
        iqm = _check_covariates(X, iqm_covariates, "iqm_covariates", self)
        bio = None
        if biological_covariates is not None:
            bio = _check_covariates(X, biological_covariates, "biological_covariates", self)
        else:
            logger.warning(
                "No biological_covariates given: biological variation correlated with the IQMs "
                "may be attributed to the scanner and removed."
            )

        if hasattr(self, "sites_"):
            del self.sites_
        if not self.var_scaling:
            if sites is not None:
                logger.info("sites are only used with var_scaling=True and are ignored.")
            return X, iqm, bio, None

        if sites is None:
            raise ValueError("sites are required with var_scaling=True.")
        sites = _check_sites(X, sites, self)
        self.sites_, site_codes = np.unique(sites, return_inverse=True)
        if bio is not None:
            logger.warning(
                "With var_scaling=True the biological covariates are preserved and required at transform time. "
                "If you intend to build a machine learning (ML) model, make sure that you DO *NOT* use the "
                "ML model's target as a biological covariate, as this produces data leakage."
            )
        return X, iqm, bio, site_codes

    def _collect_results(self, X: npt.NDArray, results: list["_FeatureResult | None"], n_kept: int) -> npt.NDArray:
        """Store the posterior summaries of every feature and return the harmonized data."""
        self.rmse_ = np.zeros(self.n_features_in_)
        self.sigma_draws_ = np.zeros((self.n_features_in_, n_kept))
        self._mu_snapshots: list[list[ForestSnapshot]] = []
        self._tau_snapshots: list[list[ForestSnapshot] | None] = []
        self._stored_site_scales: list[npt.NDArray | None] = []
        if hasattr(self, "site_scales_"):
            del self.site_scales_
        if self.var_scaling:
            self.site_scales_ = np.ones((self.n_features_in_, len(self.sites_)))

        out = X.copy()
        for j, result in enumerate(results):
            if result is None:  # constant feature
                self._mu_snapshots.append([])
                self._tau_snapshots.append(None)
                self._stored_site_scales.append(None)
                continue
            logger.info(f"Feature {j}: prediction RMSE (z-scored) = {result.rmse:.4f}")
            out[:, j] = result.harmonized * self.feature_stds_[j] + self.feature_means_[j]
            self.rmse_[j] = result.rmse
            self.sigma_draws_[j] = result.sigma_draws
            self._mu_snapshots.append(result.mu_snapshots)
            self._tau_snapshots.append(result.tau_snapshots)
            self._stored_site_scales.append(result.stored_site_scales)
            if self.var_scaling:
                self.site_scales_[j] = result.site_scales
        return out

    def _encode_sites(self, X: npt.NDArray, sites: npt.ArrayLike | None) -> npt.NDArray:
        """Map site labels to the indices of the fitted sites."""
        if sites is None:
            raise ValueError("sites are required with var_scaling=True.")
        sites = _check_sites(X, sites, self)
        unseen = np.setdiff1d(np.unique(sites), self.sites_)
        if unseen.size:
            raise ValueError(f"sites {unseen.tolist()} were not seen in fit; their variance scales are unknown.")
        return np.searchsorted(self.sites_, sites)

    def _iter_stored_draws(self, feature: int, X_bio: npt.NDArray | None):
        """Yield ``(mu_snapshot, tau_prediction, site_scales)`` for the stored draws of a feature."""
        tau_snapshots = self._tau_snapshots[feature]
        site_scales = self._stored_site_scales[feature]
        for d, mu in enumerate(self._mu_snapshots[feature]):
            if site_scales is None:
                yield mu, None, None
            else:
                tau = 0.0 if tau_snapshots is None else tau_snapshots[d].predict(X_bio)
                yield mu, tau, site_scales[d]

    def __sklearn_is_fitted__(self) -> bool:
        """Check fitted status."""
        return hasattr(self, "_mu_snapshots")

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.estimator_type = "transformer"
        tags.target_tags.required = True
        tags.input_tags.two_d_array = True
        tags.non_deterministic = self.random_state is None
        return tags


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_covariates(
    X: npt.NDArray,
    covariates: npt.ArrayLike,
    name: str,
    estimator: BaseEstimator,
    n_columns: int | None = None,
) -> npt.NDArray:
    """Validate covariates and return them as a 2D float array."""
    if covariates is None:
        raise ValueError(f"{name} are required.")
    covariates = check_array(covariates, dtype=FLOAT_DTYPES, ensure_2d=False, estimator=estimator)
    if covariates.ndim == 1:
        covariates = covariates[:, np.newaxis]
    elif covariates.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D, got shape {covariates.shape}.")
    check_consistent_length(X, covariates)
    if n_columns is not None and covariates.shape[1] != n_columns:
        raise ValueError(f"{name} has {covariates.shape[1]} columns, but BARTharm was fitted with {n_columns}.")
    return covariates


def _check_sites(X: npt.NDArray, sites: npt.ArrayLike, estimator: BaseEstimator) -> npt.NDArray:
    """Validate site labels."""
    sites = check_array(sites, dtype=None, ensure_2d=False, estimator=estimator)
    if sites.ndim != 1:
        raise ValueError(f"sites must be 1D, got shape {sites.shape}.")
    check_consistent_length(X, sites)
    return sites


class _QuantileNormalizer:
    """Quantile normalization (``quantile_normalize_bart``) that can be applied to new data.

    New values are mapped by linear interpolation between the unique training
    values, which reproduces ``quantile_normalize_bart`` on the training data,
    and clipped to [0, 1].
    """

    def fit(self, X: npt.NDArray) -> "_QuantileNormalizer":
        self.unique_values_ = [np.unique(column) for column in X.T]
        return self

    def transform(self, X: npt.NDArray) -> npt.NDArray:
        out = np.zeros_like(X, dtype=float)
        for j, unique_values in enumerate(self.unique_values_):
            if len(unique_values) > 1:
                out[:, j] = np.interp(X[:, j], unique_values, np.linspace(0.0, 1.0, len(unique_values)))
        return out


@dataclass(frozen=True)
class _SamplerConfig:
    """Settings of the BARTharm Gibbs sampler."""

    n_iter: int
    thinning_interval: int
    n_burn_saved: int
    n_saved: int
    n_trees_mu: int
    n_trees_tau: int
    beta_mu: float
    beta_tau: float
    gamma_mu: float
    gamma_tau: float
    posterior_summary: str
    n_stored_draws: int
    n_sites: int  # 0 without variance scaling


@dataclass
class _FeatureResult:
    """Posterior summaries of one feature."""

    harmonized: npt.NDArray
    rmse: float
    sigma_draws: npt.NDArray
    site_scales: npt.NDArray | None
    mu_snapshots: list[ForestSnapshot]
    tau_snapshots: list[ForestSnapshot] | None
    stored_site_scales: npt.NDArray | None


def _harmonize_draw(
    y: npt.NDArray,
    mu: npt.NDArray,
    tau: npt.NDArray | float | None,
    site_scales: npt.NDArray | None,
    site_codes: npt.NDArray | None,
) -> npt.NDArray:
    """Harmonized feature of one posterior draw (``posterior_predict_bartharm[_noscaling]``)."""
    if site_scales is None:
        return y - mu
    return (y - mu - tau) / site_scales[site_codes] + tau


def _summarize(draws: npt.NDArray, summary: str) -> npt.NDArray:
    return np.median(draws, axis=0) if summary == "median" else np.mean(draws, axis=0)


def _stored_draw_indices(n_kept: int, n_stored: int) -> npt.NDArray:
    """Return the indices of ``n_stored`` evenly spaced draws among ``n_kept``."""
    return np.unique(np.round(np.linspace(0, n_kept - 1, min(n_stored, n_kept))).astype(int))


class _PosteriorAccumulator:
    """Collect the kept posterior draws of one feature."""

    def __init__(self, n_samples: int, n_kept: int, config: _SamplerConfig, store_tau: bool) -> None:
        self.n_kept = n_kept
        self.stored = set(_stored_draw_indices(n_kept, config.n_stored_draws).tolist())
        self.sum_prediction = np.zeros(n_samples)
        self.sum_harmonized = np.zeros(n_samples)
        self.harmonized_draws = np.empty((n_kept, n_samples)) if config.posterior_summary == "median" else None
        self.sigma_draws = np.empty(n_kept)
        self.scale_draws = np.empty((n_kept, config.n_sites)) if config.n_sites else None
        self.mu_snapshots: list[ForestSnapshot] = []
        self.tau_snapshots: list[ForestSnapshot] | None = [] if (config.n_sites and store_tau) else None
        self.stored_scales: list[npt.NDArray] = []

    def add(
        self,
        k: int,
        y: npt.NDArray,
        mu: npt.NDArray,
        tau: npt.NDArray,
        sigma: float,
        site_variances: npt.NDArray | None,
        site_codes: npt.NDArray | None,
        mu_forest: SoftBARTForest,
        tau_forest: SoftBARTForest | None,
    ) -> None:
        """Add the ``k``-th kept draw (``posterior_predict_bartharm[_noscaling]``)."""
        self.sum_prediction += mu + tau
        self.sigma_draws[k] = sigma
        scales = None
        if site_variances is not None:
            # align_deltas_posthoc: normalize the site scales to geometric mean 1
            scales = np.sqrt(site_variances)
            scales = scales / np.exp(np.mean(np.log(scales)))
            self.scale_draws[k] = scales
        harmonized = _harmonize_draw(y, mu, tau, scales, site_codes)
        if self.harmonized_draws is None:
            self.sum_harmonized += harmonized
        else:
            self.harmonized_draws[k] = harmonized
        if k in self.stored:
            self.mu_snapshots.append(mu_forest.snapshot())
            if self.tau_snapshots is not None:
                self.tau_snapshots.append(tau_forest.snapshot())
            if scales is not None:
                self.stored_scales.append(scales)

    def result(self, y: npt.NDArray) -> _FeatureResult:
        """Summarize the draws."""
        y_pred = self.sum_prediction / self.n_kept
        if self.harmonized_draws is None:
            harmonized = self.sum_harmonized / self.n_kept
        else:
            harmonized = np.median(self.harmonized_draws, axis=0)
        return _FeatureResult(
            harmonized=harmonized,
            rmse=float(np.sqrt(np.mean((y - y_pred) ** 2))),
            sigma_draws=self.sigma_draws,
            site_scales=None if self.scale_draws is None else self.scale_draws.mean(axis=0),
            mu_snapshots=self.mu_snapshots,
            tau_snapshots=self.tau_snapshots,
            stored_site_scales=None if self.scale_draws is None else np.asarray(self.stored_scales),
        )


def _sample_site_variances(
    residuals: npt.NDArray,
    site_masks: list[npt.NDArray],
    sigma: float,
    site_variances: npt.NDArray,
    rng: np.random.Generator,
) -> None:
    """Draw the site variance scales ``delta_s^2`` from their inverse-gamma full conditionals, in place."""
    for s, mask in enumerate(site_masks):
        n_site = int(mask.sum())
        if n_site > 0:  # sites without samples keep their value
            shape = _PRIOR_SHAPE + n_site / 2.0
            rate = _PRIOR_RATE + float(residuals[mask] @ residuals[mask]) / (2.0 * sigma**2)
            site_variances[s] = max(1.0 / rng.gamma(shape, 1.0 / rate), _MIN_VARIANCE)


def _make_forest(
    X: npt.NDArray | None,
    num_tree: int,
    beta: float,
    gamma: float,
    sigma: float,
    seed: np.random.SeedSequence,
) -> SoftBARTForest | None:
    """Make a forest with ``Hypers(..., normalize_Y = FALSE)`` and ``Opts(update_sigma = FALSE)``."""
    if X is None:
        return None
    forest = SoftBARTForest(X.shape[1], num_tree=num_tree, beta=beta, gamma=gamma, random_state=seed)
    forest.set_sigma(sigma)
    return forest


def _fit_feature(
    y: npt.NDArray,
    X_iqm: npt.NDArray,
    X_bio: npt.NDArray | None,
    site_codes: npt.NDArray | None,
    config: _SamplerConfig,
    seed: np.random.SeedSequence,
) -> _FeatureResult:
    """Run the BARTharm Gibbs sampler on one z-scored feature (``bartharm_inference``).

    Each iteration updates the scanner forest given the biological effect, the
    biological forest given the scanner effect, the site variance scales (with
    variance scaling, no reference site) and the error variance, in this order.
    """
    seed_mu, seed_tau, seed_main = seed.spawn(3)
    rng = np.random.default_rng(seed_main)
    n = y.shape[0]
    # Plain in-memory arrays (joblib may pass memory maps), so the forests can cache their leaf weights
    X_iqm = np.array(X_iqm, dtype=float)
    X_bio = None if X_bio is None else np.array(X_bio, dtype=float)

    sigma = float(np.std(y, ddof=1))
    mu_forest = _make_forest(X_iqm, config.n_trees_mu, config.beta_mu, config.gamma_mu, sigma, seed_mu)
    tau_forest = _make_forest(X_bio, config.n_trees_tau, config.beta_tau, config.gamma_tau, sigma, seed_tau)
    mu = np.zeros(n)
    tau = np.zeros(n)

    weights = None
    site_variances = None
    if config.n_sites:
        site_variances = np.ones(config.n_sites)
        site_masks = [site_codes == s for s in range(config.n_sites)]
        weights = np.ones(n)

    accumulator = _PosteriorAccumulator(n, config.n_saved - config.n_burn_saved, config, tau_forest is not None)
    for iteration in range(1, config.n_iter + 1):
        mu = mu_forest.do_gibbs(X_iqm, y - tau, weights)
        if tau_forest is not None:
            tau = tau_forest.do_gibbs(X_bio, y - mu, weights)
        residuals = y - mu - tau

        if site_variances is not None:
            _sample_site_variances(residuals, site_masks, sigma, site_variances, rng)
            weights = 1.0 / site_variances[site_codes]
            sse = float(np.sum(residuals**2 * weights))
        else:
            sse = float(residuals @ residuals)

        # Global error variance, inverse-gamma full conditional
        sigma2 = 1.0 / rng.gamma(_PRIOR_SHAPE + n / 2.0, 1.0 / (_PRIOR_RATE + 0.5 * sse))
        sigma = float(np.sqrt(max(sigma2, _MIN_VARIANCE)))
        for forest in (mu_forest, tau_forest):
            if forest is not None:
                forest.set_sigma(sigma)

        # Saved draw number iteration // thinning_interval; the first n_burn_saved are burn-in
        if iteration % config.thinning_interval == 0 and iteration // config.thinning_interval > config.n_burn_saved:
            k = iteration // config.thinning_interval - config.n_burn_saved - 1
            accumulator.add(k, y, mu, tau, sigma, site_variances, site_codes, mu_forest, tau_forest)
    return accumulator.result(y)
