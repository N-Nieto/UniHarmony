"""Provider CovBat transformer."""

# Adapted from:
# https://github.com/andy1764/CovBat_Harmonization
# licensed under Artistic License 2.0 .

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
    """Harmonization of mean and covariance for multi-site imaging data.

    This transformer performs:

    1. ComBat without correcting mean in the harmonized data, then
    2. Reduce dimension by Principal Component Analysis (PCA), then
    3. Harmonize the variance of the principal components

    This process removes the batch effects from the data, which is not removed by standard ComBat.
    Multivariate pattern analysis (MVPA) cannot use covariance information to detect site differences on CovBat-harmonized data
    unlike that from ComBat-harmonized data.

    Parameters
    ----------
    copy : bool, optional (default True)
        Whether to copy objects when doing `check_array`.

    Attributes
    ----------
    sites_ : array, shape (n_samples,)
        Fitted site names.

    References
    ----------
    .. [1] Chen, A. A., et al. (2022).
           Mitigating site effects in covariance for machine learning in neuroimaging data.
           Human Brain Mapping, 43(4), 1179-1195.
           https://doi.org/10.1002/hbm.25688

    """

    def __init__(
        self,
        copy: bool = True,
    ) -> None:
        self.copy = copy

    def fit(
        self,
        X: npt.ArrayLike,
        sites: npt.ArrayLike,
        categorical_covariates: npt.ArrayLike | None = None,
        continuous_covariates: npt.ArrayLike | None = None,
        pct_var: float | None = 0.95,
        n_pc: int | None = None,
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
        pct_var : float or None, optional (default 0.95)
            Numeric between 0 and 1 indicating the percent of variation that is
            explained by the adjusted PCs.
        n_pc : positive int or None, optional (default None)
            If not None, then this specifies the number of PCs to adjust. Overrides ``pct_var``.
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

        # First combat
        self._first_combat = _FirstNeuroComBat()
        combat_data = self._first_combat.fit_transform(
            X=X,
            sites=sites,
            categorical_covariates=categorical_covariates,
            continuous_covariates=continuous_covariates,
            var_epsilon=var_epsilon,
            delta_epsilon=delta_epsilon,
            tau_2_epsilon=tau_2_epsilon,
            max_iter=max_iter,
        )
        # bmu = np.mean(combat_data, axis=0)

        # Standardize data before PCA
        self._scaler = StandardScaler()
        combat_data = self._scaler.fit_transform(combat_data)
        # PCA
        self._pca = PCA()
        self._pca.fit(combat_data)
        pc_components = self._pca.components_
        full_scores = self._pca.fit_transform(combat_data)

        var_exp = np.cumsum(np.round(self._pca.explained_variance_ratio_, decimals=4))
        if pct_var is not None:
            npc = np.min(np.where(var_exp > pct_var)) + 1
        if n_pc is not None and n_pc > 0:
            npc = n_pc
        scores = full_scores.loc[range(0, npc), :]

        # Second combat
        self._second_combat = NeuroComBat(
            empirical_bayes=False,
        )
        scores_combat = self._second_combat.fit(
            X=scores,
            sites=sites,
            categorical_covariates=None,
            continuous_covariates=None,
        )
        full_scores.loc[range(0, npc), :] = scores_combat

        x_covbat = combat_data - combat_data  # create pandas DataFrame to store output
        # x_covbat = x_covbat.add(bmu, axis='index')
        proj = np.dot(full_scores.T, pc_components).T
        x_covbat += self._scaler.inverse_transform(proj.T).T
        # x_covbat = x_covbat * np.dot(vpsq, np.ones((1, int(n_array)))) + stand_mean
        x_covbat += self._standardized_mean

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

        self._second_combat.transform(
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


class _FirstNeuroComBat(NeuroComBat):
    """Custom NeuroComBat for first step of CovBat."""

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

        # Validate input
        check_is_fitted(self)
        X, sites = self._check_X_sites(X, sites, copy=self.copy, estimator=self)

        if self._categorical_covariates_used:
            categorical_covariates = check_array(categorical_covariates, dtype=None, estimator=self)

        if self._continuous_covariates_used:
            continuous_covariates = check_array(continuous_covariates, dtype=FLOAT_DTYPES, estimator=self)

        # Transpose to conform to neuroCombat and original ComBat
        X = X.T

        new_data_sites_name = np.unique(sites)

        # Check all sites from new_data were seen
        if not all(s in self.sites_ for s in new_data_sites_name):
            raise ValueError("There is a site unseen during the fit method in the data.")

        n_samples = sites.shape[0]
        idx_per_site = [list(np.where(sites == s)[0].tolist()) for s in self.sites_]

        design = self.transform_design_matrix(
            sites=sites,
            categorical_covariates=categorical_covariates,
            continuous_covariates=continuous_covariates,
        )

        standardized_data, self._standardized_mean = self.transform_standardize(
            X=X,
            design=design,
            n_samples=n_samples,
        )

        bayes_data = self.harmonize(
            data=standardized_data,
            mean=0,  # don't add grand mean
            idx_per_site=idx_per_site,
        )

        return bayes_data.T
