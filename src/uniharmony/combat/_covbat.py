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
import structlog
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import (
    FLOAT_DTYPES,
    check_array,
    check_is_fitted,
)

from ._base import BaseComBat
from ._neuro_combat import NeuroComBat


__all__ = ["CovBat"]

logger = structlog.get_logger()
logger.bind(src="CovBat")


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

    score_eb : bool, default False
        Use empirical Bayes when harmonizing PC scores.
        default value matches the original implementation.

    score_parametric : bool, default True
        Use parametric adjustments for the PC-score ComBat step.

    residualize : bool, default False
        If ``True``, the output is left mean-centered (the grand mean is not
        added back).
        default value matches the original implementation.


    Attributes
    ----------
    sites_ : ndarray, shape (n_sites,)
        Site names seen during ``fit``.
    n_pc_ : int
        Number of PCs selected for harmonization.

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
        std_var: bool = True,
        pct_var: float | None = 0.95,
        n_pc: int | None = None,
        first_combat_eb: bool = True,
        first_combat_parametric: bool = True,
        score_eb: bool = False,
        score_parametric: bool = True,
        residualize: bool = False,
    ) -> None:
        self.std_var = std_var
        self.pct_var = pct_var
        self.n_pc = n_pc
        self.first_combat_eb = first_combat_eb
        self.first_combat_parametric = first_combat_parametric
        self.score_eb = score_eb
        self.score_parametric = score_parametric
        self.residualize = residualize

    def fit(
        self,
        X: npt.ArrayLike,
        sites: npt.ArrayLike,
        categorical_covariates: npt.ArrayLike | None = None,
        continuous_covariates: npt.ArrayLike | None = None,
        var_epsilon: float = 1e-8,
        delta_epsilon: float = 1e-8,
        tau_2_epsilon: float = 1e-10,
        max_iter: int = 1000,
    ) -> "CovBat":
        """Compute per-feature statistics to perform harmonization.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.

        sites : array-like, shape (n_samples,)
            Sites.

        categorical_covariates : array-like, shape (n_samples, n_categorical_covariates) or None, optional (default None)
            The categorical covariates to be preserved during harmonization.
            (e.g., sex, disease).

        continuous_covariates : array-like, shape (n_samples, n_continuous_covariates) or None, optional (default None)
            The continuous covariates to be preserved during harmonization.
            (e.g., age, clinical scores).

        var_epsilon : float, optional (default 1e-8)
            Small constant to add to variance to avoid division by zero.

        delta_epsilon : float, optional (default 1e-8)
            Small constant to add to delta variance to avoid division by zero in full mode.
            This is only used if empirical_bayes=True and parametric_adjustments=True.

        tau_2_epsilon : float, optional (default 1e-10)
            Small constant to add to tau_2 variance to avoid division by zero in full mode.
            This is only used if empirical_bayes=True and parametric_adjustments=True.

        max_iter : int, optional (default 1000)
            Maximum number of iterations for the solver in full mode.
            This is only used if empirical_bayes=True and parametric_adjustments=True.

        """
        logger.debug("Fitting")
        # ---- validate inputs ----
        X, sites = self._check_X_sites(X, sites, estimator=self)

        # First combat
        # This is an adaptation of original ComBat.
        self._first_combat = _ResidualNeuroComBat(
            empirical_bayes=self.first_combat_eb,
            parametric_adjustments=self.first_combat_parametric,
        )

        residualized = self._first_combat.fit_transform(
            X=X,
            sites=sites,
            categorical_covariates=categorical_covariates,
            continuous_covariates=continuous_covariates,
            delta_epsilon=delta_epsilon,
        )

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
        )
        self._second_combat.fit(
            X=scores,
            sites=sites,
            categorical_covariates=None,
            continuous_covariates=None,
            var_epsilon=var_epsilon,
            delta_epsilon=delta_epsilon,
            tau_2_epsilon=tau_2_epsilon,
            max_iter=max_iter,
        )

        return self

    def transform(
        self,
        X: npt.ArrayLike,
        sites: npt.ArrayLike,
        categorical_covariates: npt.ArrayLike | None = None,
        continuous_covariates: npt.ArrayLike | None = None,
    ) -> npt.NDArray:
        """Harmonize data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data to be harmonized.
        sites : array-like, shape (n_samples,)
            Sites.
        categorical_covariates : array-like, shape (n_samples, n_categorical_covariates) or None, optional (default None)
            The categorical covariates to be preserved during harmonization.
            (e.g., sex, disease).
        continuous_covariates : array-like, shape (n_samples, n_continuous_covariates) or None, optional (default None)
            The continuous covariates to be preserved during harmonization.
            (e.g., age, clinical scores).

        Returns
        -------
        array, shape (n_samples, n_features)
            The array containing the harmonized data across sites.

        Raises
        ------
        ValueError
            If one or more site or sites is or are unseen.

        """
        logger.debug("Transforming")

        check_is_fitted(self)

        # ---- 1. First ComBat (residualize) ----
        residualized = self._first_combat.transform(
            X=X,
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
        scores_harmonized = self._second_combat.transform(
            X=scores,
            sites=sites,
            categorical_covariates=None,
            continuous_covariates=None,
        )

        # ---- 5. Back-project to feature space ----
        full_scores_harmonized = full_scores.copy()
        full_scores_harmonized[:, : self.n_pc_] = scores_harmonized

        # Reconstruct: scores @ components + mean_
        reconstructed = full_scores_harmonized @ self._pca.components_
        reconstructed += self._pca.mean_

        if self._scaler is not None:
            reconstructed = self._scaler.inverse_transform(reconstructed)

        # ---- 6. Restore mean from first ComBat ----
        if not self.residualize:
            # _standardized_mean was populated by _first_combat.transform
            # shape (n_features, n_samples) → transpose to (n_samples, n_features)
            standardized_mean = self._first_combat._standardized_mean
            reconstructed = reconstructed + standardized_mean.T

        return reconstructed

    def fit_transform(
        self,
        X: npt.ArrayLike,
        sites: npt.ArrayLike,
        categorical_covariates: npt.ArrayLike | None = None,
        continuous_covariates: npt.ArrayLike | None = None,
        **fit_params,
    ) -> npt.NDArray:
        """Fit to data, then transform it.

        Overrides ``BaseComBat.fit_transform`` so that fit-only parameters
        (e.g. ``var_epsilon``) are not forwarded to ``transform``.
        """
        return self.fit(
            X=X,
            sites=sites,
            categorical_covariates=categorical_covariates,
            continuous_covariates=continuous_covariates,
            **fit_params,
        ).transform(
            X=X,
            sites=sites,
            categorical_covariates=categorical_covariates,
            continuous_covariates=continuous_covariates,
        )

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
        sites: npt.ArrayLike,
        categorical_covariates: npt.ArrayLike | None = None,
        continuous_covariates: npt.ArrayLike | None = None,
        delta_epsilon: float = 1e-8,
    ) -> npt.NDArray:
        """Harmonize data without restoring the mean.

        The returned array is in the original feature scale (pooled standard
        deviation has been reapplied) but the mean is left at zero.  The
        ``_standardized_mean`` attribute is populated so that the caller
        (``CovBat``) can add it back later.
        """
        logger.debug("Transforming (residualize only)")

        # ---- input validation (mirrors NeuroComBat.transform) ----
        check_is_fitted(self)
        X, sites = self._check_X_sites(X, sites, estimator=self)

        if self._categorical_covariates_used:
            categorical_covariates = check_array(categorical_covariates, dtype=None, estimator=self)

        if self._continuous_covariates_used:
            continuous_covariates = check_array(continuous_covariates, dtype=FLOAT_DTYPES, estimator=self)

        # neuroCombat convention: rows = features, columns = samples
        X = X.T

        if not all(s in self.sites_ for s in np.unique(sites)):
            raise ValueError("One or more sites were not seen during fit.")

        n_samples = sites.shape[0]
        idx_per_site = [list(np.where(sites == s)[0].tolist()) for s in self.sites_]

        design = self.transform_design_matrix(
            sites=sites,
            categorical_covariates=categorical_covariates,
            continuous_covariates=continuous_covariates,
        )
        self._design_matrix_shape = design.shape

        standardized_data, self._standardized_mean = self.transform_standardize(X=X, design=design, n_samples=n_samples)

        bayes_data = self.harmonize(
            data=standardized_data,
            mean=0,  # Pass mean=0 so that harmonize() does NOT add the mean back.
            idx_per_site=idx_per_site,
            epsilon=delta_epsilon,  # Small constant added to the delta* to avoid division by zero.
        )

        # Return in sample-major format (n_samples, n_features)
        return bayes_data.T
