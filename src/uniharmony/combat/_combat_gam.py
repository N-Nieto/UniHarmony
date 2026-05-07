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
from statsmodels.gam.api import BSplines

from uniharmony._utils import validate_sites

from ._design_matrix_mixin import DesignMatrixMixin
from ._ls_mixin import LocationAndScaleMixin
from ._standardization_mixin import StandardizationMixin


__all__ = ["ComBatGAM"]

logger = structlog.get_logger()


class ComBatGAM(DesignMatrixMixin, StandardizationMixin, LocationAndScaleMixin, TransformerMixin, BaseEstimator):
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
        check_consistent_length(smooth_covariates, sites)
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
            check_consistent_length(continuous_covariates, sites)

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

        standardized_data = self.fit_standardize(
            X=X,
            design=design,
            n_samples=n_samples,
            n_samples_per_site=n_samples_per_site,
            n_smooth_cols=smooth_covariates_cols,
            smooth_formula=formula,
            df_gam=df_gam,
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

        standardized_data, standardized_mean = self.transform_standardize(
            X=X,
            design=design,
            n_samples=n_samples,
        )

        bayes_data = self.harmonize(
            data=standardized_data,
            mean=standardized_mean,
            idx_per_site=idx_per_site,
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
