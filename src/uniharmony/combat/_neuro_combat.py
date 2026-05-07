"""Provide NeuroComBat transformer."""

# Adapted from:
# https://github.com/Jfortin1/neuroCombat
# licensed under MIT license.
#
# Adapted from:
# https://github.com/Warvito/neurocombat_sklearn
# licensed under MIT license.

import numpy as np
import numpy.typing as npt
import structlog
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import Tags
from sklearn.utils.validation import (
    FLOAT_DTYPES,
    check_array,
    check_consistent_length,
    check_is_fitted,
)

from uniharmony._utils import (
    handle_near_zero_values,
    handle_negative_variance,
    minimum_samples_warning,
    solve_ordinary_least_squares,
    validate_sites,
)

from ._design_matrix_mixin import DesignMatrixMixin
from ._ls_mixin import LocationAndScaleMixin


__all__ = ["NeuroComBat"]

logger = structlog.get_logger()


class NeuroComBat(DesignMatrixMixin, LocationAndScaleMixin, TransformerMixin, BaseEstimator):
    """Harmonize scanner effects in multi-site imaging data.

    This transformer performs harmonization using a parametric empirical Bayes
    framework proposed in ComBat [1]_ and adapted to neuroimaging data
    here [2]_ .

    Parameters
    ----------
    empirical_bayes : bool, optional (default True)
        Whether to perform empirical Bayes.
    parametric_adjustments : bool, optional (default True)
        Whether to perform parametric adjustments.
    mean_only : bool, optional (default False)
        Whether to only adjust mean (no scaling).
    copy : bool, optional (default True)
        Whether to copy objects when doing `check_array`.

    Attributes
    ----------
    sites_ : array, shape (n_samples,)
        Fitted site names.

    References
    ----------
    .. [1] W. Evan Johnson and Cheng Li
           "Adjusting batch effects in microarray expression data using empirical
           Bayes methods."
           Biostatistics, 8(1):118-127, 2007.
           https://doi.org/10.1093/biostatistics/kxj037

    .. [2] Fortin, Jean-Philippe, et al.
           "Harmonization of cortical thickness measurements across scanners and
           sites."
           Neuroimage 167 (2018): 104-120.
           https://doi.org/10.1016/j.neuroimage.2017.11.024

    """

    def __init__(
        self,
        empirical_bayes: bool = True,
        parametric_adjustments: bool = True,
        mean_only: bool = False,
        copy: bool = True,
    ) -> None:
        self.empirical_bayes = empirical_bayes
        self.parametric_adjustments = parametric_adjustments
        self.mean_only = mean_only
        self.copy = copy

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
    ) -> "NeuroComBat":
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

        # Check that X and sites have correct shape and type
        X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES, estimator=self)
        sites = check_array(sites, copy=self.copy, dtype=None, ensure_2d=False, estimator=self)
        check_consistent_length(X, sites)
        validate_sites(sites)

        # Check that categorical_covariates and continuous_covariates have correct shape and type if they are not None.
        # Also, track whether they were used during fit to check during transform
        self._categorical_covariates_used = False
        if categorical_covariates is not None:
            self._categorical_covariates_used = True
            categorical_covariates = check_array(categorical_covariates, dtype=None, estimator=self)

        self._continuous_covariates_used = False
        if continuous_covariates is not None:
            self._continuous_covariates_used = True
            continuous_covariates = check_array(continuous_covariates, dtype=FLOAT_DTYPES, estimator=self)

        if self._categorical_covariates_used or self._continuous_covariates_used:
            logger.warning(
                "You specified categorical and/or continuous covariates to be preserved. "
                "If you intend to build a machine learning (ML) model,"
                "then make sure that you DO *NOT* preserve the ML model's target as covariate. "
                "You will be required to provide the covariate also at transform time, and this will produce data leakage. "
                "If you are performing a statistical analysis and want to preserve a variable of interest, "
                "then it is correct to specify it as covariate."
            )

        # Transpose to conform to neuroCombat and original ComBat
        X = X.T

        self.sites_, n_samples_per_site = np.unique(sites, return_counts=True)
        self._n_sites = len(self.sites_)
        n_samples = sites.shape[0]
        idx_per_site = [list(np.where(sites == s)[0].tolist()) for s in self.sites_]

        design = self.fit_design_matrix(
            sites=sites,
            categorical_covariates=categorical_covariates,
            continuous_covariates=continuous_covariates,
        )
        logger.debug("Standardizing data across features")
        standardized_data, _ = self._standardize_across_features(
            X,
            design,
            n_samples,
            n_samples_per_site,
            fitting=True,
            epsilon=var_epsilon,
        )

        self.fit_ls_model(
            data=standardized_data,
            design=design,
            idx_per_site=idx_per_site,
            epsilon=var_epsilon,
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

        X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES, estimator=self)
        sites = check_array(sites, copy=self.copy, dtype=None, ensure_2d=False, estimator=self)
        check_consistent_length(X, sites)

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
        n_samples_per_site = np.asarray([np.sum(sites == s) for s in self.sites_])
        idx_per_site = [list(np.where(sites == s)[0].tolist()) for s in self.sites_]

        design = self.transform_design_matrix(
            sites=sites,
            categorical_covariates=categorical_covariates,
            continuous_covariates=continuous_covariates,
        )
        logger.debug("Standardizing data across features")
        standardized_data, standardized_mean = self._standardize_across_features(
            X=X,
            design=design,
            n_samples=n_samples,
            n_samples_per_site=n_samples_per_site,
            fitting=False,
        )

        bayes_data = self.harmonize(
            standardized_data,
            standardized_mean,
            idx_per_site,
        )

        return bayes_data.T

    # Overridden to allow sites
    def fit_transform(
        self,
        X: npt.ArrayLike,
        sites: npt.ArrayLike,
        **fit_params,
    ) -> npt.NDArray:
        """Fit to data, then transform it.

        Fits transformer to `X` and `sites` with optional parameters
        `fit_params` and returns a transformed version of `X`.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input samples.
        sites : array-like, shape (n_samples, 1)
            Sites.
        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        array, shape (n_samples, n_features)
            Transformed array.

        """
        return self.fit(X, sites, **fit_params).transform(X, sites, **fit_params)

    def _standardize_across_features(
        self,
        X: npt.ArrayLike,
        design: npt.NDArray,
        n_samples: int,
        n_samples_per_site: npt.NDArray,
        fitting: bool = False,
        epsilon: float = 1e-8,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Standardization of the features.

        The magnitude of the features could create bias in the empirical
        Bayes estimates of the prior distribution. To avoid this, the features
        are standardized to all of them have similar overall mean and variance.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Features.
        design : array
            Design matrix.
        n_samples : int
            Sample count.
        n_samples_per_site : array
            Sample count per site.
        fitting : bool, optional (default False)
            Whether fitting or not.
        epsilon : float, optional (default 1e-8)
            Small constant to add to variance to avoid division by zero.

        Returns
        -------
        Standardized data : array, shape (n_features, n_samples)
            Standardized data.
        Standardized mean : array, shape (n_features, n_samples)
            Standardized mean used during the process.

        """
        if fitting:
            # =====================================================================
            # STEP 1: Fit OLS model to estimate site and covariate effects (fitting only)
            # =====================================================================
            # SOLVES: beta_hat = (X_design^T * X_design)^(-1) * X_design^T * X_data
            # This is Ordinary Least Squares (OLS) - finds coefficients that minimize residuals
            #
            # beta_hat structure: [site_intercepts | covariate_effects] per feature
            #   - Rows 0 to _n_sites-1: intercept for each site (location effect)
            #   - Rows _n_sites+: effects of categorical/continuous covariates
            # Solve OLS is the same as fitting a linear model: X = design @ beta_hat + error
            # The step preservs the biological signal by modeling it as part of the residuals (error term)

            gram_matrix = design.T @ design
            self._beta_hat = solve_ordinary_least_squares(gram_matrix, X, design)

            # =====================================================================
            # STEP 2: Compute weighted grand mean across sites
            # =====================================================================
            # PURPOSE: Create a reference mean representing the "average site"
            # This becomes our harmonization target - all sites will be aligned to this
            minimum_samples_warning(n_samples_per_site)

            # Weighted average: each site's intercept weighted by sample proportion
            site_weights = np.array(n_samples_per_site) / float(n_samples)
            self._grand_mean = site_weights.T @ self._beta_hat[: self._n_sites, :]

            # =====================================================================
            # STEP 3: Compute pooled residual variance
            # =====================================================================
            # PURPOSE: Estimate variance after removing site/covariate effects
            # This captures biological + noise variance, excluding batch effects
            X_predicted = (design @ self._beta_hat).T  # Shape: (n_features, n_samples)
            residuals = X - X_predicted
            if n_samples < 30:
                # Use sample variance for small datasets
                self._var_pooled = np.sum(residuals**2, axis=1, keepdims=True) / (n_samples - 1)
            else:
                # Population variance for larger datasets (matches original behavior)
                self._var_pooled = np.mean(residuals**2, axis=1, keepdims=True)

            # Handle near-zero variance features
            # Features with ~0 variance cause division by zero in standardization
            # This can happen with constant features or features with very small range
            self._var_pooled = handle_near_zero_values(self._var_pooled, epsilon=epsilon)
        # End Fitting

        # =====================================================================
        # STEP 4: Construct target mean for each sample (harmonization target)
        # =====================================================================
        # The standardized_mean represents what each sample's mean SHOULD be
        # after harmonization: grand_mean + covariate_effects (site effects REMOVED)
        # STRUCTURE: standardized_mean = grand_mean (site-harmonized) + covariate_adjustment
        # Component A: Grand mean replicated for all samples
        # Shape: (n_features, n_samples) - same target mean for all samples
        standardized_mean = self._grand_mean.T[:, np.newaxis] @ np.ones((1, n_samples))

        # Component B: Add covariate effects (preserved biological variation)
        # We create a modified design matrix with site columns zeroed out
        # This removes site-specific intercepts but keeps covariate columns
        design_covariates_only = design.copy()
        design_covariates_only[:, : self._n_sites] = 0  # Zero out site effect columns

        # Add covariate contributions: design_no_site @ beta_hat
        # Only covariate rows of beta_hat contribute since site columns are zeroed
        covariate_adjustment = (design_covariates_only @ self._beta_hat).T
        standardized_mean += covariate_adjustment

        # =====================================================================
        # STEP 5: Standardize data to common scale
        # =====================================================================
        # FORMULA: Z = (X - target_mean) / pooled_std
        #
        # RESULT:
        #   - Mean is centered relative to grand_mean + covariates (site effects removed)
        #   - Variance normalized to ~1 across all features
        #   - Features now on comparable scale for Empirical Bayes estimation

        # Make sure the variance is not negative due to numerical issues before taking sqrt
        self._var_pooled = handle_negative_variance(self._var_pooled)
        pooled_std = np.sqrt(self._var_pooled)
        standardized_data = (X - standardized_mean) / (pooled_std @ np.ones((1, n_samples)))

        # =====================================================================
        # STEP 6: Standardization stats for debugging
        # =====================================================================
        logger.debug("Standardization stats:")
        logger.debug(f"  Grand mean range: [{self._grand_mean.min():.4f}, {self._grand_mean.max():.4f}]")
        logger.debug(f"  Pooled std range: [{pooled_std.min():.4f}, {pooled_std.max():.4f}]")
        logger.debug(f"  Standardized data mean: {standardized_data.mean():.6f} (should be ~0)")
        logger.debug(f"  Standardized data std: {standardized_data.std():.4f} (should be ~1)")
        return standardized_data, standardized_mean

    # Overridden for check_is_fitted() usage
    def __sklearn_is_fitted__(self) -> bool:
        """Check fitted status."""
        return hasattr(self, "_gamma_star") and hasattr(self, "_delta_star")

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.estimator_type = "transformer"
        tags.target_tags.required = True
        tags.target_tags.two_d_labels = True
        tags.target_tags.positive_only = True
        tags.input_tags.two_d_array = True
        tags.input_tags.allow_nan = True
        return tags
