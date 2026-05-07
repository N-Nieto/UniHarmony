"""Provide StandardizationMixin."""

import numpy as np
import numpy.typing as npt
import pandas as pd
import structlog
from statsmodels.gam.api import GLMGam

from uniharmony._utils import (
    handle_near_zero_values,
    handle_negative_variance,
    minimum_samples_warning,
    solve_ordinary_least_squares,
)


__all__ = ["StandardizationMixin"]

logger = structlog.get_logger()
logger = logger.bind(src="StandardizationMixin")


class StandardizationMixin:
    """Mixin class to perform standardization of features."""

    def fit_standardize(
        self,
        X: npt.ArrayLike,
        design: npt.NDArray,
        n_samples: int,
        n_samples_per_site: npt.NDArray,
        n_smooth_cols: int | None,
        smooth_formula: str | None,
        df_gam: pd.DataFrame | None,
        epsilon: float = 1e-8,
    ) -> npt.NDArray:
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
        n_smooth_cols : int or None
            Smoothing terms count.
        smooth_formula : str or None
            Smoothing formula.
        df_gam : pd.DataFrame or None
            Dataframe for GAM.
        bsplines : statsmodels.gam.api.BSplines or None
            BSplines for GAM.
        epsilon : float, optional (default 1e-8)
            Small constant to add to variance to avoid division by zero.

        Returns
        -------
        array, shape (n_features, n_samples)
            Standardized data.

        """
        logger.debug("Standardizing data across features")
        if n_smooth_cols is not None:
            # Smoothing with GAMs
            if X.shape[0] > 10:
                logger.info("Smoothing more than 10 variables may take several minutes of computation.")
            # Penalization weight (not the final weight)
            alpha = np.array([1.0] * n_smooth_cols)
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
        else:
            # Fit OLS model
            gram_matrix = design.T @ design
            self._beta_hat = solve_ordinary_least_squares(gram_matrix, X, design)
        # Compute weighted grand mean across sites to create reference mean
        minimum_samples_warning(n_samples_per_site)
        site_weights = np.array(n_samples_per_site) / float(n_samples)
        self._grand_mean = site_weights.T @ self._beta_hat[: self._n_sites, :]

        # Compute pooled residual variance to estimate variance after removing
        # site / covariate effects
        X_predicted = (design @ self._beta_hat).T
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

        return self.transform_standardize(
            X=X,
            design=design,
            n_samples=n_samples,
        )[0]

    def transform_standardize(
        self,
        X: npt.ArrayLike,
        design: npt.NDArray,
        n_samples: int,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Standardize features on fitted standardization of input.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Features.
        design : array
            Design matrix.
        n_samples : int
            Sample count.

        Returns
        -------
        array, shape (n_features, n_samples)
            Standardized data.
        array, shape (n_features, n_samples)
            Standardized mean.

        """
        logger.debug("Standardizing new data using fitted data")
        # Construct target mean for each sample (harmonization target)
        standardized_mean = self._grand_mean.T[:, np.newaxis] @ np.ones((1, n_samples))

        # Add covariate effects (preserved biological variation)
        design_covariates_only = design.copy()
        design_covariates_only[:, : self._n_sites] = 0  # Zero out site effect columns

        # Add covariate contributions
        covariate_adjustment = (design_covariates_only @ self._beta_hat).T
        standardized_mean += covariate_adjustment

        # Standardize data to common scale
        # Make sure the variance is not negative due to numerical issues before taking sqrt
        self._var_pooled = handle_negative_variance(self._var_pooled)
        pooled_std = np.sqrt(self._var_pooled)
        standardized_data = (X - standardized_mean) / (pooled_std @ np.ones((1, n_samples)))

        logger.debug("Standardization stats:")
        logger.debug(f"  Grand mean range: [{self._grand_mean.min():.4f}, {self._grand_mean.max():.4f}]")
        logger.debug(f"  Pooled std range: [{pooled_std.min():.4f}, {pooled_std.max():.4f}]")
        logger.debug(f"  Standardized data mean: {standardized_data.mean():.6f} (should be ~0)")
        logger.debug(f"  Standardized data std: {standardized_data.std():.4f} (should be ~1)")

        return standardized_data, standardized_mean
