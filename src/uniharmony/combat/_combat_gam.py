"""Provide ComBatGAM transformer."""

# Adapted from:
# https://github.com/rpomponio/neuroHarmonize
# licensed under MIT license

import numpy as np
import numpy.typing as npt
import pandas as pd
import structlog
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import Tags
from sklearn.utils.validation import (
    FLOAT_DTYPES,
    check_array,
    check_consistent_length,
    check_is_fitted,
)
from statsmodels.gam.api import BSplines, GLMGam

from uniharmony._utils import (
    handle_near_zero_values,
    handle_negative_variance,
    minimum_samples_warning,
    validate_sites,
)

from ._design_matrix_mixin import DesignMatrixMixin
from ._ls_mixin import LocationAndScaleMixin


__all__ = ["ComBatGAM"]

logger = structlog.get_logger()


class ComBatGAM(DesignMatrixMixin, LocationAndScaleMixin, TransformerMixin, BaseEstimator):
    """Harmonize multi-site scanner effects controlling for non-linear age effects.

    This is an improvement on NeuroComBat allowing for non-linear effects to be controlled by Generalized Additive Models (GAMs).

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
    .. [1] Pomponio, R., Shou, H., Davatzikos, C., et al., (2019).
           "Harmonization of large MRI datasets for the analysis of brain imaging patterns throughout the lifespan."
           Neuroimage 208.
           https://doi.org/10.1016/j.neuroimage.2019.116450.

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
        smooth_covariates: npt.ArrayLike,
        smooth_covariates_bounds: tuple[float, float] | None = None,
        continuous_covariates: npt.ArrayLike | None = None,
        df: int = 10,
        degree: int = 3,
        var_epsilon: float = 1e-8,
        delta_epsilon: float = 1e-8,
        tau_2_epsilon: float = 1e-10,
        max_iter: int = 1000,
    ) -> "ComBatGAM":
        """Compute per-feature statistics to perform harmonization.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        sites : array-like, shape (n_samples,)
            Sites.
        smooth_covariates : array-like, shape (n_samples, n_smooth_covariates)
            The smooth, non-linear covariates. GAMs are used for optimal smoothing (e.g., age).
        smooth_covariates_bounds : tuple of float and float or None, optional (default None)
            Custom boundaries of the smoothing terms useful when holdout data covers different range than
            specify the bounds as (minimum, maximum). Currently not supported for models with multiple smooth covariates.
        continuous_covariates : array-like, shape (n_samples, n_continuous_covariates) or None, optional (default None)
            The continuous covariates to be preserved during harmonization
            (e.g., clinical scores).
        df : int, optional (default 10)
            Number of basis functions or degrees of freedom for BSplines. Default value used in the original implementation.
        degree : int, optional (default 3)
            Degree(s) of the spline for BSplines. Default value used in the original implementation.
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

        # Check smooth_covariates and its bounds if passed
        smooth_covariates = check_array(smooth_covariates, dtype=FLOAT_DTYPES, estimator=self)
        if smooth_covariates_bounds is None:
            smooth_covariates_bounds = (None, None)
        logger.info(
            "If you intend to build a machine learning (ML) model,"
            "then make sure that you DO *NOT* preserve the ML model's target as covariate. "
            "You will be required to provide the covariate also at transform time, and this will produce data leakage. "
            "If you are performing a statistical analysis and want to preserve a variable of interest, "
            "then it is correct to specify it as covariate."
        )

        # Check that continuous_covariates has correct shape and type if it is not None.
        # Also, track whether it was used during fit to check during transform
        self._continuous_covariates_used = False
        if continuous_covariates is not None:
            self._continuous_covariates_used = True
            continuous_covariates = check_array(continuous_covariates, dtype=FLOAT_DTYPES, estimator=self)

            logger.warning(
                "You specified continuous covariates to be preserved. "
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
            categorical_covariates=None,
            continuous_covariates=continuous_covariates,
        )
        # Setup design matrix for smoothing
        logger.debug("Setting up smoothing using B-Splines")
        # Create cubic spline basis for smooth covariates
        x_spline = smooth_covariates.copy()
        smooth_covariates_cols = smooth_covariates.shape[1]
        if smooth_covariates_cols == 1:
            self._bsplines = BSplines(
                x_spline,
                df=df,
                degree=degree,
                knot_kwds=[
                    {
                        "lower_bound": smooth_covariates_bounds[0],
                        "upper_bound": smooth_covariates_bounds[1],
                    }
                ],
            )
        else:
            self._bsplines = BSplines(
                x_spline,
                df=[df] * smooth_covariates_cols,
                degree=[degree] * smooth_covariates_cols,
            )
        # Construct formula and dataframe required for GAM
        formula = "y ~ "
        df_gam = {}
        # Set data from created design matrix
        for b in range(self._n_sites):
            v = f"x{b!s}"
            formula += f"{v} + "
            df_gam[v] = design[:, b]
        # Set data from continuous covariates
        if self._continuous_covariates_used:
            for c in range(continuous_covariates.shape[1]):
                v = f"c{c!s}"
                formula += f"{v} + "
                df_gam[v] = continuous_covariates[:, c].astype(float)
        # Complete formula
        formula = formula[:-2] + "- 1"
        logger.debug(f"Final formula for smoothing: {formula}")
        df_gam = pd.DataFrame(df_gam)
        # For matrix operations, a modified design matrix is required
        design = np.concatenate((df_gam, self._bsplines.basis), axis=1)

        logger.debug("Standardizing data across features")
        standardized_data, _ = self._standardize_across_features(
            X=X,
            design=design,
            n_samples=n_samples,
            n_samples_per_site=n_samples_per_site,
            smooth_term_cols=smooth_covariates_cols,
            smooth_formula=formula,
            df_gam=df_gam,
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
        smooth_covariates: npt.ArrayLike,
        continuous_covariates: npt.ArrayLike | None = None,
    ) -> npt.NDArray:
        """Harmonize data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data to be harmonized.
        sites : array-like, shape (n_samples,)
            Sites.
        smooth_covariates : array-like, shape (n_samples, n_smooth_covariates)
            The smooth, non-linear terms. GAMs are used for optimal smoothing (e.g., age).
        continuous_covariates : array-like, shape (n_samples, n_continuous_covariates) or None, optional (default None)
            The continuous covariates to be preserved during harmonization.
            (e.g., clinical scores).

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

        smooth_covariates = check_array(smooth_covariates, dtype=FLOAT_DTYPES, estimator=self)

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
            categorical_covariates=None,
            continuous_covariates=continuous_covariates,
        )
        # Setup design matrix for smoothing
        logger.debug("Setting up smoothing using B-Splines")
        # Create cubic spline basis for smooth covariates
        x_spline = smooth_covariates.copy()
        bs_basis = self._bsplines.transform(x_spline)
        # Construct dataframe required for GAM
        df_gam = {}
        # Set data from created design matrix
        for b in range(self._n_sites):
            v = f"x{b!s}"
            df_gam[v] = design[:, b]
        # Set data from continuous covariates
        if self._continuous_covariates_used:
            for c in range(continuous_covariates.shape[1]):
                v = f"c{c!s}"
                df_gam[v] = continuous_covariates[:, c].astype(float)
        df_gam = pd.DataFrame(df_gam)
        # For matrix operations, a modified design matrix is required
        design = np.concatenate((df_gam, bs_basis), axis=1)

        logger.debug("Standardizing data across features")
        standardized_data, standardized_mean = self._standardize_across_features(
            X=X,
            design=design,
            n_samples=n_samples,
            n_samples_per_site=n_samples_per_site,
            smooth_term_cols=None,
            smooth_formula=None,
            df_gam=None,
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
        smooth_covariates: npt.ArrayLike,
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
        smooth_covariates : array-like, shape (n_samples, n_smooth_terms)
            The smooth, non-linear covariates. GAMs are used for optimal smoothing (e.g., age).
        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        array, shape (n_samples, n_features)
            Transformed array.

        """
        return self.fit(X, sites, smooth_covariates, **fit_params).transform(X, sites, smooth_covariates, **fit_params)

    def _standardize_across_features(
        self,
        X: npt.ArrayLike,
        design: npt.NDArray,
        n_samples: int,
        n_samples_per_site: npt.NDArray,
        smooth_term_cols: int | None,
        smooth_formula: str | None,
        df_gam: pd.DataFrame | None,
        fitting: bool = False,
        epsilon: float = 1e-8,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Standardization of the features.

        The magnitude of the features could create bias in the empirical
        Bayes estimates of the prior distribution. To avoid this, the features
        are standardized to all of them have similar overall mean and variance.
        If smoothing is requested, ``beta_hat`` is calculated by smoothing with GAMs.

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
        smooth_term_cols : int or None
            Smoothing terms count.
        smooth_formula : str or None
            Smoothing formula.
        df_gam : pd.DataFrame or None
            Dataframe for GAM.
        bsplines : statsmodels.gam.api.BSplines or None
            BSplines for GAM.
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
            # STEP 1: Smoothing with GAMs
            # =====================================================================
            if X.shape[0] > 10:
                logger.info("Smoothing more than 10 variables may take several minutes of computation.")
            # Penalization weight (not the final weight)
            alpha = np.array([1.0] * smooth_term_cols)
            # Empty matrix for beta
            self._beta_hat = np.zeros((design.shape[1], X.shape[0]))
            # Estimate beta for each variable to be harmonized
            for i in range(0, X.shape[0]):
                df_gam.loc[:, "y"] = X[i, :]
                gam_bs = GLMGam.from_formula(smooth_formula, data=df_gam, smoother=self._bsplines, alpha=alpha)
                gam_bs.fit()
                # Optimal penalization weights alpha can be obtained through gcv/kfold
                # Note: kfold is faster, gcv is more robust
                gam_bs.alpha = gam_bs.select_penweight_kfold()[0]
                res_bs_optim = gam_bs.fit()
                self._beta_hat[:, i] = res_bs_optim.params

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
