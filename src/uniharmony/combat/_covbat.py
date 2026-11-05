"""Provide CovBat transformer.

Adapted from:
https://github.com/andy1764/CovBat_Harmonization
licensed under Artistic License 2.0.

CovBat harmonizes both mean/variance (via ComBat) and covariance
(via PCA + ComBat on scores) across sites. After CovBat,
machine-learning classifiers should no longer be able to detect site
membership from the covariance structure of the data.
"""

import numpy as np
import numpy.typing as npt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from uniharmony._verbose import get_logger

from ._base import BaseComBat
from ._neuro_combat import NeuroComBat


__all__ = ["CovBat"]

logger = get_logger(src="CovBat")


class CovBat(BaseComBat):
    """Harmonization of mean and covariance for multi-site data.

    CovBat extends ComBat with a covariance-harmonization step.  The
    algorithm is:

    1. **First ComBat** (with empirical Bayes) - removes site effects from
       mean and variance.  The data are *residualized* (mean left at zero).
    2. **PCA + Second ComBat** - principal components are computed on the
       residualized data and the leading scores are harmonized with a second
       ComBat step (without empirical Bayes according to the original paper).
        This removes site-specific covariance structure.
    3. **Back-projection** - the harmonized scores are projected back to the
       original feature space and the mean is restored.

    Parameters
    ----------
    site_column : str or int or None, default None
        Column of ``X`` holding the site of every sample. If None, the sites
        are passed with the ``sites`` keyword. See
        :class:`~uniharmony.combat.NeuroComBat`.

    categorical_columns : str or int or list or None, default None
        Columns of ``X`` holding categorical covariates to preserve.
        Alternative to the ``categorical_covariates`` keyword.

    continuous_columns : str or int or list or None, default None
        Columns of ``X`` holding continuous covariates to preserve.
        Alternative to the ``continuous_covariates`` keyword.

    ref_site : str or int or list or None, default None
        Reference site(s) of both ComBat steps. If None, every site is
        harmonized to the pooled mean, variance and covariance of all sites.
        If a single site, every site is harmonized to that site, whose data
        are left unchanged; if several sites, to their pooled distribution.
        See :class:`~uniharmony.combat.NeuroComBat`.

    std_var : bool, default True
        If ``True``, scale each feature to unit variance before PCA
        default value matches the original implementation.

    pct_var : float or None, default 0.95
        Proportion of variance (0-1) that the selected PCs must explain.
        Ignored when ``n_pc`` is not ``None``.
        default value matches the original implementation.

    n_pc : int or None, default None
        Exact number of principal components to harmonize.
        Overrides ``pct_var``.

    first_combat_eb : bool, default True
        Use empirical Bayes in the first ComBat step.

    first_combat_parametric : bool, default True
        Use parametric priors in the first ComBat step.

    mean_only : bool, default False
        Only adjust the mean (no scaling) in the first ComBat step, as
        ``mean.only`` of the original implementation.

    score_eb : bool, default False
        Use empirical Bayes when harmonizing PC scores.
        default value matches the original implementation.

    score_parametric : bool, default True
        Use parametric adjustments for the PC-score ComBat step.

    residualize : bool, default False
        If ``True``, the output is left mean-centered (the grand mean is not
        added back).
        default value matches the original implementation.

    var_epsilon : float, default 1e-8
        Small constant added to variances to avoid division by zero.

    delta_epsilon : float, default 1e-8
        Lower bound of the site scale factors, to avoid division by zero.

    tau_2_epsilon : float, default 1e-10
        Lower bound of the prior variance of the site location effects.

    max_iter : int, default 1000
        Maximum number of iterations of the parametric empirical Bayes solver.

    Attributes
    ----------
    sites_ : ndarray, shape (n_sites,)
        Site names seen during ``fit``.
    ref_sites_ : ndarray, shape (n_ref_sites,)
        The reference sites (all sites with ``ref_site=None``).
    n_pc_ : int
        Number of PCs selected for harmonization.
    n_iter_ : int
        Largest number of iterations of the parametric empirical Bayes solver
        of the two ComBat steps (0 if it is not used).
    n_features_in_ : int
        Number of columns of ``X`` seen during ``fit`` (metadata columns
        included).
    feature_names_in_ : ndarray of str
        Names of the columns seen during ``fit`` (metadata columns included).
        Only defined if ``X`` was a DataFrame with string column names.

    Notes
    -----
    Both ComBat steps divide the pooled residual sum of squares by
    :math:`n - S` (:math:`S` sites) instead of :math:`n` as R ``sva::ComBat``
    and ``neuroCombat`` do, so that the harmonized variance is unbiased. See
    :class:`~uniharmony.combat.NeuroComBat` for details.

    References
    ----------
    .. [1] Chen, A. A., Beer, J. C., Tustison, N. J., Cook, P. A.,
       Shinohara, R. T., Shou, H., & ADNI (2022).  Mitigating site effects
       in covariance for machine learning in neuroimaging data.
       *Human Brain Mapping*, 43(4), 1179-1195.
       https://doi.org/10.1002/hbm.25688

    """

    def __init__(
        self,
        site_column: str | int | None = None,
        categorical_columns: str | int | list | None = None,
        continuous_columns: str | int | list | None = None,
        ref_site: str | int | list | None = None,
        std_var: bool = True,
        pct_var: float | None = 0.95,
        n_pc: int | None = None,
        first_combat_eb: bool = True,
        first_combat_parametric: bool = True,
        mean_only: bool = False,
        score_eb: bool = False,
        score_parametric: bool = True,
        residualize: bool = False,
        var_epsilon: float = 1e-8,
        delta_epsilon: float = 1e-8,
        tau_2_epsilon: float = 1e-10,
        max_iter: int = 1000,
    ) -> None:
        self.site_column = site_column
        self.categorical_columns = categorical_columns
        self.continuous_columns = continuous_columns
        self.ref_site = ref_site
        self.std_var = std_var
        self.pct_var = pct_var
        self.n_pc = n_pc
        self.first_combat_eb = first_combat_eb
        self.first_combat_parametric = first_combat_parametric
        self.mean_only = mean_only
        self.score_eb = score_eb
        self.score_parametric = score_parametric
        self.residualize = residualize
        self.var_epsilon = var_epsilon
        self.delta_epsilon = delta_epsilon
        self.tau_2_epsilon = tau_2_epsilon
        self.max_iter = max_iter

    # Data keywords are requested by default when metadata routing is enabled
    __metadata_request__fit = {"sites": True, "categorical_covariates": True, "continuous_covariates": True}  # noqa: RUF012
    __metadata_request__transform = {"sites": True, "categorical_covariates": True, "continuous_covariates": True}  # noqa: RUF012

    def _solver_options(self) -> dict:
        return {
            "var_epsilon": self.var_epsilon,
            "delta_epsilon": self.delta_epsilon,
            "tau_2_epsilon": self.tau_2_epsilon,
            "max_iter": self.max_iter,
        }

    def fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike | None = None,
        *,
        sites: npt.ArrayLike | None = None,
        categorical_covariates: npt.ArrayLike | None = None,
        continuous_covariates: npt.ArrayLike | None = None,
    ) -> "CovBat":
        """Compute per-feature statistics to perform harmonization.

        Parameters
        ----------
        X : array-like or DataFrame, shape (n_samples, n_columns)
            The training input samples, with the site and covariate columns
            named in the constructor, if any.

        y : None
            Ignored. Present for scikit-learn API consistency.

        sites : array-like, shape (n_samples,) or None, default None
            Sites. Required unless ``site_column`` is set.

        categorical_covariates : array-like, shape (n_samples, n_categorical_covariates) or None, default None
            The categorical covariates to be preserved during harmonization
            (e.g., sex, disease). Alternative to ``categorical_columns``.

        continuous_covariates : array-like, shape (n_samples, n_continuous_covariates) or None, default None
            The continuous covariates to be preserved during harmonization
            (e.g., age, clinical scores). Alternative to ``continuous_columns``.

        Returns
        -------
        self : CovBat
            The fitted transformer.

        """
        logger.debug("Fitting")
        # ---- validate inputs ----
        data = self._validate_combat_input(
            X,
            reset=True,
            sites=sites,
            categorical_covariates=categorical_covariates,
            continuous_covariates=continuous_covariates,
        )
        X, sites = data.X, data.sites
        self.sites_ = np.unique(sites)
        self._set_reference_sites()
        covariates = {
            "categorical_covariates": data["categorical_covariates"],
            "continuous_covariates": data["continuous_covariates"],
        }

        # First combat
        # This is an adaptation of original ComBat.
        self._first_combat = _ResidualNeuroComBat(
            empirical_bayes=self.first_combat_eb,
            parametric_adjustments=self.first_combat_parametric,
            mean_only=self.mean_only,
            ref_site=self.ref_site,
            **self._solver_options(),
        )

        residualized = self._first_combat.fit_transform(X, sites=sites, **covariates)

        # ------------------------------------------------------------------
        # 2. Optional variance standardisation before PCA
        # ------------------------------------------------------------------
        if self.std_var:
            self._scaler = StandardScaler()
            pca_input = self._scaler.fit_transform(residualized)
        else:
            self._scaler = None
            pca_input = residualized

        # ------------------------------------------------------------------
        # 3. Fit PCA on residualized data
        # ------------------------------------------------------------------
        self._pca = PCA()
        self._pca.fit(pca_input)
        full_scores = self._pca.transform(pca_input)

        n_samples, n_features = pca_input.shape
        n_components = min(n_samples, n_features)
        # Warning about the behavior.
        if self.n_pc is not None and self.n_pc > 0 and self.pct_var is not None:
            logger.warning(
                f"Both n_pc ({self.n_pc}) and pct_var ({self.pct_var}) provided. "
                f"Using n_pc to determine number of components, ignoring pct_var."
            )
        if self.n_pc is not None and self.n_pc > 0:
            self.n_pc_ = min(self.n_pc, n_components)
        elif self.pct_var is not None:
            # Check the range of the explained variance
            if not 0 < self.pct_var < 1:
                raise ValueError(f"pct_var must be between 0 and 1, got {self.pct_var}")
            var_exp = np.cumsum(np.round(self._pca.explained_variance_ratio_, decimals=4))
            above = np.where(var_exp > self.pct_var)[0]
            self.n_pc_ = int(above[0]) + 1 if len(above) else n_components
        else:
            self.n_pc_ = n_components

        scores = full_scores[:, : self.n_pc_]
        logger.debug(f"Selected {self.n_pc_} / {n_components} PCs for covariance harmonization")

        # -----------------------------------------------------------------------------------
        # 4. Second ComBat (no covariates, no EB in the original implementation) on PC scores
        # -----------------------------------------------------------------------------------

        self._second_combat = NeuroComBat(
            empirical_bayes=self.score_eb,
            parametric_adjustments=self.score_parametric,
            ref_site=self.ref_site,
            **self._solver_options(),
        )
        self._second_combat.fit(scores, sites=sites)
        self.n_iter_ = max(self._first_combat.n_iter_, self._second_combat.n_iter_)

        return self

    def transform(
        self,
        X: npt.ArrayLike,
        *,
        sites: npt.ArrayLike | None = None,
        categorical_covariates: npt.ArrayLike | None = None,
        continuous_covariates: npt.ArrayLike | None = None,
    ) -> npt.NDArray:
        """Harmonize data.

        Parameters
        ----------
        X : array-like or DataFrame, shape (n_samples, n_columns)
            The data to be harmonized, with the same columns as in :meth:`fit`.
        sites : array-like, shape (n_samples,) or None, default None
            Sites, all seen in :meth:`fit`. Required unless ``site_column`` is set.
        categorical_covariates : array-like, shape (n_samples, n_categorical_covariates) or None, default None
            The categorical covariates, if given to :meth:`fit` as keyword.
        continuous_covariates : array-like, shape (n_samples, n_continuous_covariates) or None, default None
            The continuous covariates, if given to :meth:`fit` as keyword.

        Returns
        -------
        array, shape (n_samples, n_features)
            The harmonized features (without the metadata columns).

        Raises
        ------
        ValueError
            If a site was not seen during :meth:`fit`.

        """
        logger.debug("Transforming")

        check_is_fitted(self)
        data = self._validate_combat_input(
            X,
            reset=False,
            sites=sites,
            categorical_covariates=categorical_covariates,
            continuous_covariates=continuous_covariates,
        )
        X, sites = data.X, data.sites
        categorical_covariates = data["categorical_covariates"]
        continuous_covariates = data["continuous_covariates"]

        # ---- 1. First ComBat (residualize) ----
        residualized, standardized_mean = self._first_combat.residualize(
            X,
            sites=sites,
            categorical_covariates=categorical_covariates,
            continuous_covariates=continuous_covariates,
        )

        # ---- 2. Scale (using fitted scaler) ----
        if self._scaler is not None:
            pca_input = self._scaler.transform(residualized)
        else:
            pca_input = residualized

        # ---- 3. Project to PC space ----
        full_scores = self._pca.transform(pca_input)

        # ---- 4. Second ComBat on leading scores ----
        scores = full_scores[:, : self.n_pc_]
        scores_harmonized = self._second_combat.transform(scores, sites=sites)

        # ---- 5. Back-project to feature space ----
        full_scores_harmonized = full_scores.copy()
        full_scores_harmonized[:, : self.n_pc_] = scores_harmonized

        # Reconstruct: scores @ components + mean_
        reconstructed = full_scores_harmonized @ self._pca.components_
        reconstructed += self._pca.mean_

        if self._scaler is not None:
            reconstructed = self._scaler.inverse_transform(reconstructed)

        # ---- 6. Restore mean from first ComBat ----
        if self.residualize:
            return reconstructed
        return self._keep_reference_samples(reconstructed + standardized_mean, X, sites)

    # Overridden for check_is_fitted() usage
    def __sklearn_is_fitted__(self) -> bool:
        """Check fitted status."""
        return (
            hasattr(self, "_first_combat")
            and hasattr(self, "_scaler")
            and hasattr(self, "_pca")
            and hasattr(self, "_second_combat")
        )


class _ResidualNeuroComBat(NeuroComBat):
    """NeuroComBat that returns residualized data (mean is NOT added back).

    In the CovBat pipeline the first ComBat step must residualize the data:
    site effects are removed from mean and variance, but the grand mean
    (including covariate effects) is **not** restored.  The PCA step then
    operates on pure residuals, and the mean is added back only after the
    second ComBat / back-projection step.
    """

    def transform(
        self,
        X: npt.ArrayLike,
        *,
        sites: npt.ArrayLike | None = None,
        categorical_covariates: npt.ArrayLike | None = None,
        continuous_covariates: npt.ArrayLike | None = None,
    ) -> npt.NDArray:
        """Harmonize data without restoring the mean (see :meth:`residualize`)."""
        return self.residualize(
            X, sites=sites, categorical_covariates=categorical_covariates, continuous_covariates=continuous_covariates
        )[0]

    def residualize(
        self,
        X: npt.ArrayLike,
        *,
        sites: npt.ArrayLike | None = None,
        categorical_covariates: npt.ArrayLike | None = None,
        continuous_covariates: npt.ArrayLike | None = None,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Harmonize data without restoring the mean.

        The fitted state is not modified (the standardized mean is returned,
        not stored).

        Returns
        -------
        residuals : ndarray, shape (n_samples, n_features)
            The harmonized data in the original feature scale (pooled standard
            deviation reapplied) with the mean left at zero.
        standardized_mean : ndarray, shape (n_samples, n_features)
            The mean (grand mean and covariate effects) that ``CovBat`` adds
            back after the covariance step.

        """
        logger.debug("Transforming (residualize only)")

        check_is_fitted(self)
        data = self._validate_combat_input(
            X,
            reset=False,
            sites=sites,
            categorical_covariates=categorical_covariates,
            continuous_covariates=continuous_covariates,
        )
        X, sites = data.X, data.sites

        # neuroCombat convention: rows = features, columns = samples
        X = X.T
        n_samples = sites.shape[0]
        idx_per_site = [list(np.where(sites == s)[0].tolist()) for s in self.sites_]
        design = self.transform_design_matrix(
            sites=sites,
            categorical_covariates=data["categorical_covariates"],
            continuous_covariates=data["continuous_covariates"],
        )
        standardized_data, standardized_mean = self.transform_standardize(X=X, design=design, n_samples=n_samples)
        bayes_data = self.harmonize(
            data=standardized_data,
            mean=0,  # Pass mean=0 so that harmonize() does NOT add the mean back.
            idx_per_site=idx_per_site,
            epsilon=self.delta_epsilon,  # Small constant added to the delta* to avoid division by zero.
        )
        # Sample-major format (n_samples, n_features)
        return bayes_data.T, standardized_mean.T
