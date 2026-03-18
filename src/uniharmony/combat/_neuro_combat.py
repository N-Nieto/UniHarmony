"""Provide NeuroComBat transformer."""

# Adapted from:
# https://github.com/Jfortin1/neuroCombat/blob/master/neuroCombat/
# neuroCombat.py
# licensed under MIT license.
#
# Adapted from:
# https://github.com/Warvito/neurocombat_sklearn/blob/master/examples/
# machine_learning_example.py
# licensed under MIT license.

import math

import numpy as np
import numpy.typing as npt
import structlog
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
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
    validate_covariates,
    validate_sites,
)


__all__ = ["NeuroComBat"]

logger = structlog.get_logger()


class NeuroComBat(TransformerMixin, BaseEstimator):
    """Harmonize scanner effects in multi-site imaging data.

    This transformer performs harmonization using a parametric empirical Bayes
    framework proposed in ComBat[^1] and adapted to neuroimaging data
    here[^2].

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

    References
    ----------
    [^1]:
        W. Evan Johnson and Cheng Li
        "Adjusting batch effects in microarray expression data using empirical
        Bayes methods."
        Biostatistics, 8(1):118-127, 2007.
        https://doi.org/10.1093/biostatistics/kxj037

    [^2]:
        Fortin, Jean-Philippe, et al.
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
        sites : array-like, shape (n_samples, 1)
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

        # ######## Set up and check data ########
        # Check that X and sites have correct shape and type, and convert sites if they are strings
        X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES, estimator=self)
        if isinstance(next(iter(sites)), str):
            sites = self._convert_sites(sites)
        if np.asarray(sites).ndim == 1:
            sites = np.asarray(sites).reshape(-1, 1)
        sites = check_array(sites, copy=self.copy, estimator=self)

        check_consistent_length(X, sites)

        # Check that categorical_covariates and continuous_covariates have correct shape and type if they are not None.
        # Track of whether they were used during fit to check during transform
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
                "ComBat will require the covariate to be provided also at transform time, and this will produce data leakage. "
                "If you are performing a statistical analysis and want to preserve a variable of interest,"
                "then it is correct to specify it as covariate."
            )

        # Transpose to conform to neuroCombat and original ComBat
        X = X.T

        self._sites_names, n_samples_per_site = np.unique(sites, return_counts=True)
        self._n_sites = len(self._sites_names)
        n_samples = sites.shape[0]
        idx_per_site = [list(np.where(sites == idx)[0]) for idx in self._sites_names]

        logger.debug("Making design matrix")
        design = self._make_design_matrix(
            sites,
            categorical_covariates,
            continuous_covariates,
            fitting=True,
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
        logger.debug("Fitting L/S model")
        gamma_hat, delta_hat = self._fit_ls_model(
            standardized_data,
            design,
            idx_per_site,
        )

        # Get the gamma_star and delta_star adjustments to be applied to the data
        if self.empirical_bayes:
            if self.parametric_adjustments:
                logger.debug("Finding priors")
                gamma_bar, tau_2, a_prior, b_prior = self._find_priors(gamma_hat, delta_hat, delta_epsilon, tau_2_epsilon)
                logger.debug("Finding parametric adjustments")
                self._gamma_star, self._delta_star = self._find_parametric_adjustments(
                    standardized_data, idx_per_site, gamma_hat, delta_hat, gamma_bar, tau_2, a_prior, b_prior, max_iter
                )
            else:
                logger.debug("Finding non-parametric adjustments")
                self._gamma_star, self._delta_star = self._find_non_parametric_adjustments(
                    standardized_data,
                    idx_per_site,
                    gamma_hat,
                    delta_hat,
                )
        else:
            logger.debug("Finding L/S adjustments without empirical Bayes")
            self._gamma_star = np.asarray(gamma_hat)
            self._delta_star = np.asarray(delta_hat)

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
        sites : array-like, shape (n_samples, 1)
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
        if isinstance(next(iter(sites)), str):
            sites = self._convert_sites(sites)
        if np.asarray(sites).ndim == 1:
            sites = np.asarray(sites).reshape(-1, 1)
        sites = check_array(sites, copy=self.copy, estimator=self)

        check_consistent_length(X, sites)

        if self._categorical_covariates_used:
            categorical_covariates = check_array(categorical_covariates, dtype=None, estimator=self)

        if self._continuous_covariates_used:
            continuous_covariates = check_array(continuous_covariates, dtype=FLOAT_DTYPES, estimator=self)

        # Transpose to conform to neuroCombat and original ComBat
        X = X.T

        new_data_sites_name = np.unique(sites)

        # Check all sites from new_data were seen
        if not all(site_name in self._sites_names for site_name in new_data_sites_name):
            raise ValueError("There is a site unseen during the fit method in the data.")

        n_samples = sites.shape[0]
        n_samples_per_site = np.asarray([np.sum(sites == site_name) for site_name in self._sites_names])
        idx_per_site = [list(np.where(sites == site_name)[0]) for site_name in self._sites_names]
        logger.debug("Making design matrix")
        design = self._make_design_matrix(
            sites,
            categorical_covariates,
            continuous_covariates,
            fitting=False,
        )
        logger.debug("Standardizing data across features")
        standardized_data, standardized_mean = self._standardize_across_features(
            X,
            design,
            n_samples,
            n_samples_per_site,
            fitting=False,
        )
        logger.debug("Harmonizing data")
        bayes_data = self._adjust_data_final(
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

    def _make_design_matrix(
        self,
        sites: npt.ArrayLike,
        categorical_covariates: npt.ArrayLike | None,
        continuous_covariates: npt.ArrayLike | None,
        fitting: bool = False,
    ) -> npt.NDArray:
        """Create a design matrix for the linear model.

        The design matrix combines:
        1. One-hot encoded sites (full encoding, all columns kept)
        2. One-hot encoded categorical covariates (first category dropped per covariate)
        3. Continuous covariates (used as-is)

        This follows standard ANOVA coding where the design matrix is used to
        estimate site effects while controlling for covariates.

        Parameters
        ----------
        sites : array-like, shape (n_samples,) or (n_samples, 1)
            Site labels for each sample. Can be integers or strings.
        categorical_covariates : array-like, shape (n_samples, n_categorical) or None
            Categorical covariates to preserve (e.g., sex, disease status).
            Each column is treated as a separate categorical variable.
        continuous_covariates : array-like, shape (n_samples, n_continuous) or None
            Continuous covariates to preserve (e.g., age, clinical scores).
        fitting : bool, optional (default False)
            If True, fit encoders on the data and store them as attributes.
            If False, use previously fitted encoders (must call with fitting=True first).

        Returns
        -------
        design : ndarray, shape (n_samples, n_effects)
            The design matrix where:
            - First n_sites columns are site indicators
            - Next columns are categorical covariates (drop-first encoded)
            - Final columns are continuous covariates

        Raises
        ------
        ValueError
            If fitting=False but encoders haven't been fitted yet.
            If categorical_covariates shape changes between fit and transform.
        RuntimeError
            If encoder classes differ between fit and transform.

        Notes
        -----
        The drop-first encoding for categorical covariates avoids collinearity
        with the intercept (which is implicit in the site effects). This is
        standard practice in regression analysis.

        Examples
        --------
        >>> sites = np.array([[1], [1], [2], [2]])
        >>> sex = np.array([['M'], ['F'], ['M'], ['F']])
        >>> age = np.array([[25], [30], [35], [40]])
        >>> design = self._make_design_matrix(sites, sex, age, fitting=True)
        >>> design.shape
        (4, 4)  # 2 sites + 1 sex (drop-first) + 1 age

        """
        # =====================================================================
        # STEP 1: Validate inputs
        # =====================================================================
        sites = np.asarray(sites)
        sites = validate_sites(sites)
        n_samples = sites.shape[0]

        categorical_covariates = validate_covariates(categorical_covariates, n_samples, "categorical_covariates")
        continuous_covariates = validate_covariates(continuous_covariates, n_samples, "continuous_covariates")
        # Validate fitting state
        if not fitting and not hasattr(self, "_site_encoder"):
            raise ValueError("Must call _make_design_matrix with fitting=True before using fitting=False")

        # =====================================================================
        # STEP 2: Fit encoders (if fitting=True)
        # =====================================================================
        if fitting:
            # Fit site encoder
            self._site_encoder = OneHotEncoder(
                sparse_output=False,
                dtype=np.float64,
                handle_unknown="error",
            )
            self._site_encoder.fit(sites)
            logger.debug(f"Fitted site encoder: {len(self._site_encoder.categories_[0])} sites")

            # Fit categorical encoders if provided
            if categorical_covariates is not None:
                n_cat_covs = categorical_covariates.shape[1]
                self._categorical_encoders = []

                for i in range(n_cat_covs):
                    cat_encoder = OneHotEncoder(
                        sparse_output=False,
                        dtype=np.float64,
                        drop="first",
                        handle_unknown="error",
                    )
                    cat_col = categorical_covariates[:, i].reshape(-1, 1)
                    cat_encoder.fit(cat_col)
                    self._categorical_encoders.append(cat_encoder)

                    logger.debug(
                        f"Fitted categorical encoder {i}: "
                        f"{len(cat_encoder.categories_[0])} categories "
                        f"(dropped: {cat_encoder.categories_[0][0]})"
                    )

        # =====================================================================
        # STEP 3: Transform all features
        # =====================================================================
        design_parts = []

        # Transform sites
        sites_encoded = self._site_encoder.transform(sites)
        design_parts.append(sites_encoded)
        n_sites = sites_encoded.shape[1]
        logger.debug(f"Sites encoded: {n_samples} samples x {n_sites} sites")

        # Transform categorical covariates
        if categorical_covariates is not None:
            for i, cat_encoder in enumerate(self._categorical_encoders):
                cat_col = categorical_covariates[:, i].reshape(-1, 1)
                cat_encoded = cat_encoder.transform(cat_col)

                design_parts.append(cat_encoded)
                n_categories = len(cat_encoder.categories_[0])
                logger.debug(f"Categorical covariate {i} encoded: {n_categories} categories -> {cat_encoded.shape[1]} columns")

        # Add continuous covariates
        if continuous_covariates is not None:
            design_parts.append(continuous_covariates)
            logger.debug(f"Added {continuous_covariates.shape[1]} continuous covariates")

        # =====================================================================
        # STEP 4: Assemble design matrix
        # =====================================================================
        design = np.hstack(design_parts)

        # Final validation
        if design.shape[0] != n_samples:
            raise RuntimeError(f"Design matrix has {design.shape[0]} rows but expected {n_samples}")

        logger.debug(f"Design matrix shape: {design.shape}")

        return design

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

    def _fit_ls_model(
        self,
        standardized_data: npt.NDArray,
        design: npt.NDArray,
        idx_per_site: list[list[int]],
        epsilon: float = 1e-8,
    ) -> tuple[npt.NDArray, list]:
        """Fit Location and Scale (L/S) model to estimate site-specific batch effects.

        This method estimates the location (mean) and scale (variance) adjustments
        needed for each site. These are the "batch effects" that ComBat will remove.

        Parameters
        ----------
        standardized_data : array, shape (n_features, n_samples)
            Standardized data from _standardize_across_features.
            Note: Transposed shape (features x samples) compared to input X.
        design : array, shape (n_samples, n_effects)
            Design matrix containing site indicators and covariates.
        idx_per_site : list of list of int
            List where each element contains sample indices for that site.
            e.g., idx_per_site[0] = [0, 5, 10, 15] means samples 0,5,10,15 are from site 0.
        epsilon : float, optional (default 1e-8)
            Small constant to add to variance to avoid division by zero.

        Returns
        -------
        gamma_hat : array, shape (n_sites, n_features)
            Estimated location (mean) shift for each site and each feature.
            This is how much each site's mean differs from the grand mean.
        delta_hat : list of arrays, length n_sites
            Each array has shape (n_features,) containing estimated scale (variance)
            for each site and each feature. This is each site's variance.

        """
        # =====================================================================
        # STEP 1: Extract site-only design matrix (remove covariate columns)
        # =====================================================================
        # The first _n_sites columns of design are site indicator variables (one-hot encoded)
        # We only want site effects here, not covariate effects
        # site_design is n_samples x n_sites, where site_design[i,j] = 1 if sample i is from site j
        site_design = design[:, : self._n_sites]

        # =====================================================================
        # STEP 2: Estimate location parameters (gamma_hat) via OLS
        # =====================================================================
        # PURPOSE: Estimate how much each site's mean differs from the grand mean
        #
        # MODEL: standardized_data = site_design @ gamma_hat + error
        #
        # Since data is standardized, we expect gamma_hat to be close to 0
        # Non-zero values indicate site-specific location shifts (batch effects)
        #
        # SOLUTION: gamma_hat = (X_site^T @ X_site)^(-1) @ X_site^T @ standardized_data
        # This gives the OLS estimate of site means on the standardized scale
        gram_site = site_design.T @ site_design

        gamma_hat = solve_ordinary_least_squares(gram_site, standardized_data, site_design)

        # Result shape: (n_sites, n_features)
        # gamma_hat[j, k] = mean offset of site j for feature k

        # =====================================================================
        # STEP 3: Estimate scale parameters (delta_hat) per site
        # =====================================================================
        # Estimate each site's variance for each feature
        #
        # If sites have different variances, this captures it
        # delta_hat[j][k] = variance of feature k in site j
        #
        # NOTE: We compute this per-site using sample indices, not via OLS
        # This is because variance is a second-order moment estimated directly

        delta_hat = []
        for site_idx, site_idxs in enumerate(idx_per_site):
            if self.mean_only:
                # [MEAN-ONLY MODE] Assume equal variance across sites
                # Set all variances to 1 (already standardized)
                # This assumes batch effect is only in location, not scale
                delta_hat.append(np.repeat(1, standardized_data.shape[0]))
            else:
                # [FULL MODE] Estimate site-specific variances
                # Check minimum samples for variance estimation
                # With ddof=1, we need at least 2 samples, but more is better
                n_site_samples = len(site_idxs)
                if n_site_samples < 2:
                    logger.error(
                        f"Site {site_idx} has only {n_site_samples} sample(s). "
                        "Cannot estimate variance with ddof=1. Setting variance to 1."
                    )
                    delta_hat.append(np.ones(standardized_data.shape[0]))
                    continue
                elif n_site_samples < 16:
                    # Warn about small sample sizes for variance estimation
                    # Research shows ComBat becomes unstable with <16-32 samples per site
                    logger.warning(
                        f"Site {site_idx} has only {n_site_samples} samples. "
                        "Variance estimates may be unstable. Consider using mean_only=True "
                        "or collecting more data."
                    )

                # standardized_data[:, site_idxs] = all samples from this site
                # Shape: (n_features, n_samples_in_site)
                site_data = standardized_data[:, site_idxs]

                # axis=1 = compute variance across samples for each feature
                # ddof=1 for sample variance (unbiased estimator)
                site_var = np.var(site_data, axis=1, ddof=1)

                # Handle near-zero or negative variances
                # Numerical errors or constant features can cause this
                site_var = handle_near_zero_values(site_var, epsilon=epsilon)

                # delta_hat[site_idx][feature_idx] = variance of that feature in that site
                delta_hat.append(site_var)

        # Validate that we have the expected number of sites
        if len(delta_hat) != self._n_sites:
            raise ValueError(
                f"Mismatch in site count: expected {self._n_sites} sites, but delta_hat has {len(delta_hat)} entries."
            )

        # =====================================================================
        # STEP 4: Diagnostic logging of L/S estimates
        # =====================================================================
        # Add diagnostic logging in debug mode
        logger.debug("L/S Model estimates:")
        logger.debug(f"  Gamma hat shape: {gamma_hat.shape}")
        logger.debug(f"  Gamma hat range: [{gamma_hat.min():.4f}, {gamma_hat.max():.4f}]")
        for i, d in enumerate(delta_hat):
            logger.debug(f"  Site {i} delta range: [{d.min():.4f}, {d.max():.4f}]")
        return gamma_hat, delta_hat

    def _find_priors(
        self,
        gamma_hat: npt.NDArray,
        delta_hat: list[npt.NDArray],
        delta_epsilon: float = 1e-8,
        tau_2_epsilon: float = 1e-10,
    ) -> tuple[npt.NDArray, npt.NDArray, list[npt.NDArray], list[npt.NDArray]]:
        """Compute hyperparameters for the prior distributions of batch effects.

        This method estimates the hyperparameters for the Empirical Bayes priors:
        - Normal prior for location parameters (gamma): N(gamma_bar, tau_2)
        - Inverse-Gamma prior for scale parameters (delta): IG(a_prior, b_prior)

        These priors are estimated using method of moments from the observed L/S parameters.
        The priors allow ComBat to "borrow strength" across features, stabilizing estimates
        especially when sample sizes are small.

        Parameters
        ----------
        gamma_hat : array, shape (n_sites, n_features)
            Estimated location (mean) shifts for each site and feature from _fit_ls_model.
        delta_hat : list of arrays
            Estimated scale (variance) parameters for each site and feature.
            Each array has shape (n_features,).
        delta_epsilon : float, optional (default 1e-8)
            Small constant to add to variance to avoid numerical issues in inverse-gamma estimation.
        tau_2_epsilon : float, optional (default 1e-10)
            Small constant to add to tau_2 to avoid issues with near-zero variance in gamma_hat

        Returns
        -------
        gamma_bar : array, shape (n_sites,)
            Mean of the normal prior for location parameters per site.
            Represents the expected batch effect across features.
        tau_2 : array, shape (n_sites,)
            Variance of the normal prior for location parameters per site.
            Represents how much batch effects vary across features.
        a_prior : list of arrays or None
            Shape parameter of inverse-gamma prior for scale parameters per site.
            None if mean_only=True (no variance adjustment).
        b_prior : list of arrays or None
            Scale parameter of inverse-gamma prior for scale parameters per site.
            None if mean_only=True (no variance adjustment).

        """
        # =====================================================================
        # STEP 1: Handle zero/negative variances in delta_hat
        # =====================================================================
        # Explicit handling or zero or negative  variances
        # Zero or negative variances cause numerical issues in inverse-gamma estimation
        # We replace them with a small epsilon and log warnings
        delta_hat_clean = []
        for site_idx, site_deltas in enumerate(delta_hat):
            site_deltas = np.asarray(site_deltas)

            # Check for invalid values (zero, negative, NaN, Inf)
            invalid_mask = (site_deltas <= 0) | ~np.isfinite(site_deltas)

            if np.any(invalid_mask):
                n_invalid = np.sum(invalid_mask)
                logger.warning(
                    f"Site {site_idx}: {n_invalid} features have invalid variance values "
                    f"(<=0, NaN, or Inf). Setting to minimum variance (1e-8). "
                    "This may indicate constant features or numerical errors."
                )
                site_deltas = site_deltas.copy()
                site_deltas[invalid_mask] = delta_epsilon

            delta_hat_clean.append(site_deltas)

        # Step 2
        # Compute mean of gamma_hat across features (axis=1) for each site
        # This is the expected location effect (gamma_bar) for each site
        gamma_bar = np.mean(gamma_hat, axis=1)

        # Step 3
        # tau_2 represents how much location effects vary across features within each site
        # High tau_2 = heterogeneous batch effects across features
        # Low tau_2 = consistent batch effects across features (more pooling possible)
        tau_2 = np.var(gamma_hat, axis=1, ddof=1)

        # Handle near-zero tau_2 (no variation in batch effects)
        # This can happen if all features have identical batch effects (rare but possible)
        # or with very few features
        tau_2 = handle_near_zero_values(tau_2, epsilon=tau_2_epsilon)

        # =====================================================================
        # STEP 4: Compute inverse-gamma prior parameters for scale (delta)
        # =====================================================================
        # The inverse-gamma prior IG(a, b) has:
        #   Mean = b / (a - 1) for a > 1
        #   Variance = b^2 / ((a-1)^2 * (a-2)) for a > 2
        #
        # We estimate a and b using method of moments from the observed delta_hat values
        # a_prior and b_prior are estimated per feature (not per site) to allow
        # feature-specific variance pooling
        if self.mean_only:
            # [MEAN-ONLY MODE] No scale adjustment, so no need for inverse-gamma priors
            a_prior = None
            b_prior = None
        else:
            # [FULL MODE] Estimate inverse-gamma hyperparameters for each site
            # _aprior_fn and _bprior_fn implement method of moments estimation:
            #   a = 1 + mean^2 / variance (shape parameter)
            #   b = mean * (mean^2 / variance + 1) (scale parameter)

            # Initialize variables to store prior parameters for each site
            a_prior = []
            b_prior = []

            for _, site_deltas in enumerate(delta_hat_clean):
                # Compute both parameters at once for this site
                a_vals, b_vals = self._compute_inverse_gamma_priors(site_deltas)

                a_prior.append(a_vals)
                b_prior.append(b_vals)

            # Logger after calculating all priors to avoid cluttering logs with warnings for each site
            for site_idx, (a_vals, b_vals) in enumerate(zip(a_prior, b_prior, strict=False)):
                invalid_a = (a_vals <= 0) | ~np.isfinite(a_vals)
                invalid_b = (b_vals <= 0) | ~np.isfinite(b_vals)

                if np.any(invalid_a) or np.any(invalid_b):
                    n_bad = np.sum(invalid_a | invalid_b)
                    logger.warning(
                        f"Site {site_idx}: {n_bad} features have invalid prior parameters. "
                        "Clipping to valid range. This may indicate extreme variance values."
                    )
                    # Clip to valid ranges for inverse-gamma
                    a_vals = np.clip(a_vals, 1e-6, 1e6)
                    b_vals = np.clip(b_vals, 1e-8, 1e8)
                    a_prior[site_idx] = a_vals
                    b_prior[site_idx] = b_vals
                    logger.debug("Prior distribution parameters:")
                    logger.debug(f"  a_prior range: [{a_vals.min():.4f}, {a_vals.max():.4f}]")
                    logger.debug(f"  b_prior range: [{b_vals.min():.4f}, {b_vals.max():.4f}]")

        # =====================================================================
        # STEP 5: Diagnostic logging of prior parameters
        # =====================================================================
        logger.debug(f"  Gamma bar (mean location effect): {gamma_bar}")
        logger.debug(f"  Tau^2 (variance of location effects): {tau_2}")
        logger.debug(f"  Mean tau^2: {np.mean(tau_2):.6f} (higher = more heterogeneous effects)")

        return gamma_bar, tau_2, a_prior, b_prior

    def _find_parametric_adjustments(
        self,
        standardized_data: npt.NDArray,
        idx_per_site: list[list[int]],
        gamma_hat: npt.NDArray,
        delta_hat: list[npt.NDArray],
        gamma_bar: npt.NDArray,
        tau_2: npt.NDArray,
        a_prior: list[npt.NDArray],
        b_prior: list[npt.NDArray],
        max_iter: int = 1000,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Compute parametric empirical Bayes site/batch effect parameter estimates.

        This method applies parametric empirical Bayes shrinkage to the L/S model
        parameters (gamma_hat, delta_hat) to obtain adjusted parameters (gamma_star, delta_star).

        The shrinkage uses the prior distributions estimated in _find_priors:
        - Normal prior for gamma: shrinks site means toward gamma_bar
        - Inverse-Gamma prior for delta: shrinks site variances toward common variance

        For mean_only mode, only location (gamma) is adjusted, variance (delta) is fixed at 1.
        For full mode, both location and variance are adjusted via iterative solver.

        Parameters
        ----------
        standardized_data : array, shape (n_features, n_samples)
            Standardized data from _standardize_across_features.
        idx_per_site : list of list of int
            List of sample indices for each site.
        gamma_hat : array, shape (n_sites, n_features)
            Estimated location parameters from _fit_ls_model.
        delta_hat : list of arrays
            Estimated scale parameters from _fit_ls_model.
        gamma_bar : array, shape (n_sites,)
            Mean of normal prior for each site.
        tau_2 : array, shape (n_sites,)
            Variance of normal prior for each site.
        a_prior : list of arrays
            Shape parameters of inverse-gamma prior for each site.
        b_prior : list of arrays
            Scale parameters of inverse-gamma prior for each site.
        max_iter : int, optional (default 1000)
            Maximum number of iterations for the solver in full mode.

        Returns
        -------
        gamma_star : array, shape (n_sites, n_features)
            Adjusted (shrunken) location parameters for each site and feature.
        delta_star : array, shape (n_sites, n_features)
            Adjusted (shrunken) scale parameters for each site and feature.

        """
        gamma_star, delta_star = [], []
        for i, site_idxs in enumerate(idx_per_site):
            if self.mean_only:
                # Fix incorrect parameter passing from original code!
                # Original passes n=1, delta_star=1 which is wrong
                # Should use actual sample count and delta_hat[i]
                n_samples = len(site_idxs)

                # Shrink gamma_hat toward gamma_bar using prior precision
                gamma_adj = self._postmean(
                    gamma_hat[i],  # Observed site means
                    gamma_bar[i],  # Prior mean
                    n_samples,  # Number of samples in site
                    1.0,  # Fixed variance (standardized)
                    tau_2[i],  # Prior variance
                )
                gamma_star.append(gamma_adj)

                # Fixed variance = 1 (standardized scale)
                delta_star.append(np.ones(standardized_data.shape[0]))
            else:
                # [FULL MODE] Adjust both location and variance iteratively
                gamma_adj, delta_adj = self._iteration_solver(
                    standardized_data[:, site_idxs],
                    gamma_hat[i],
                    delta_hat[i],
                    gamma_bar[i],
                    tau_2[i],
                    a_prior[i],
                    b_prior[i],
                    max_iter=max_iter,
                )
                gamma_star.append(gamma_adj)
                delta_star.append(delta_adj)
        # Stack lists into arrays with proper shape checking
        gamma_star_arr = np.asarray(gamma_star)
        delta_star_arr = np.asarray(delta_star)

        expected_shape = (self._n_sites, standardized_data.shape[0])
        if gamma_star_arr.shape != expected_shape:
            raise ValueError(f"gamma_star shape {gamma_star_arr.shape} != expected {expected_shape}")
        if delta_star_arr.shape != expected_shape:
            raise ValueError(f"delta_star shape {delta_star_arr.shape} != expected {expected_shape}")
        return gamma_star_arr, delta_star_arr

    def _iteration_solver(
        self,
        standardized_data: npt.NDArray,
        gamma_hat: npt.NDArray,
        delta_hat: npt.NDArray,
        gamma_bar: npt.NDArray,
        tau_2: npt.NDArray,
        a_prior: npt.NDArray,
        b_prior: npt.NDArray,
        convergence: float = 0.0001,
        max_iter: int = 1000,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Solve iteratively the posterior mean and variance via Empirical Bayes.

        Parameters
        ----------
        standardized_data : array, shape (n_features, n_samples_site)
            Standardized data for a single site.
        gamma_hat : array, shape (n_features,)
            Initial location estimates.
        delta_hat : array, shape (n_features,)
            Initial scale estimates.
        gamma_bar : array, shape (n_features,)
            Prior mean for location.
        tau_2 : array, shape (n_features,)
            Prior variance for location.
        a_prior : array shape (n_features,)
            Prior shape for scale.
        b_prior : array, shape (n_features,)
            Prior scale for scale.
        convergence : float, optional (default 0.0001)
            Relative change threshold for convergence.
        max_iter : int, optional (default 1000)
            Maximum number of iterations.

        Returns
        -------
        gamma_star : ndarray, shape (n_features,)
            Converged posterior location estimates.
        delta_star : ndarray, shape (n_features,)
            Converged posterior scale estimates.

        """
        gamma_hat_new = None
        delta_hat_new = None
        # Handle missing data (NaN) by counting non-NaN samples per feature
        sample_size = np.sum(~np.isnan(standardized_data), axis=1)
        if np.any(sample_size == 0):
            raise ValueError("Some features have all NaN values for this site")

        gamma_hat_old = np.asarray(gamma_hat, dtype=np.float64).copy()
        delta_hat_old = np.asarray(delta_hat, dtype=np.float64).copy()

        # Validate inputs
        if np.any(delta_hat_old <= 0):
            logger.warning("_iteration_solver: Initial delta_hat <= 0, clipping")
            delta_hat_old = np.clip(delta_hat_old, 1e-8, None)

        change = 1.0
        count = 0

        while change > convergence:
            # Add iteration limit to void infinite loops in edge cases
            if count >= max_iter:
                logger.warning(
                    f"_iteration_solver: Did not converge after {max_iter} iterations. "
                    f"Final change: {change:.6f}. Using current estimates."
                )
                break

            # E-step: Update gamma given current delta
            gamma_hat_new = self._postmean(gamma_hat, gamma_bar, sample_size, delta_hat_old, tau_2)

            # M-step: Update delta given new gamma
            # Use broadcasting instead of explicit ones matrix
            gamma_expanded = gamma_hat_new[:, np.newaxis]

            with np.errstate(invalid="ignore"):
                residuals = standardized_data - gamma_expanded
                resid2 = np.square(residuals)
                sum_2 = np.nansum(resid2, axis=1)

            delta_hat_new = self._postvar(sum_2, sample_size, a_prior, b_prior)
            # Handle any numerical issues in delta_hat_new
            delta_hat_new = np.clip(delta_hat_new, 1e-8, None)

            # Convergence check
            with np.errstate(divide="ignore", invalid="ignore"):
                gamma_change = np.abs(gamma_hat_new - gamma_hat_old)
                gamma_rel_change = np.where(np.abs(gamma_hat_old) > 1e-10, gamma_change / np.abs(gamma_hat_old), gamma_change)

                delta_change = np.abs(delta_hat_new - delta_hat_old)
                delta_rel_change = np.where(np.abs(delta_hat_old) > 1e-10, delta_change / np.abs(delta_hat_old), delta_change)

                change = max(np.max(gamma_rel_change), np.max(delta_rel_change))

            gamma_hat_old = gamma_hat_new.copy()
            delta_hat_old = delta_hat_new.copy()
            count += 1

        logger.debug(f"_iteration_solver converged in {count} iterations (change={change:.6f})")
        # Final validation of outputs
        gamma_hat_new = np.asarray(gamma_hat_new, dtype=np.float64)
        delta_hat_new = np.asarray(delta_hat_new, dtype=np.float64)
        return gamma_hat_new, delta_hat_new

    def _find_non_parametric_adjustments(
        self,
        standardized_data: npt.NDArray,
        idx_per_site: list[list[int]],
        gamma_hat: npt.NDArray,
        delta_hat: list[npt.NDArray],
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Compute non-parametric empirical Bayes site/batch effect parameter estimates.

        Parameters
        ----------
        standardized_data : array, shape (n_features, n_samples)
            Standardized data.
        idx_per_site : list of list of int
            List of sample indices for each site.
        gamma_hat : array, shape (n_sites, n_features)
            Estimated location parameters.
        delta_hat : list of arrays
            Estimated scale parameters.

        Returns
        -------
        gamma_star : array, shape (n_sites, n_features)
            Adjusted location parameters via non-parametric EB.
        delta_star : array, shape (n_sites, n_features)
            Adjusted scale parameters via non-parametric EB.

        """
        gamma_star, delta_star = [], []

        for i, site_idxs in enumerate(idx_per_site):
            # Don't modify delta_hat in place
            if self.mean_only:
                site_delta = np.ones(standardized_data.shape[0])
            else:
                site_delta = np.asarray(delta_hat[i])

            gamma_adj, delta_adj = self._int_eprior(
                standardized_data[:, site_idxs],
                gamma_hat[i],
                site_delta,
            )

            gamma_star.append(gamma_adj)
            delta_star.append(delta_adj)

        # Convert to arrays with validation
        gamma_star_arr = np.asarray(gamma_star, dtype=np.float64)
        delta_star_arr = np.asarray(delta_star, dtype=np.float64)

        expected_shape = (self._n_sites, standardized_data.shape[0])
        if gamma_star_arr.shape != expected_shape:
            raise ValueError(f"gamma_star shape mismatch: {gamma_star_arr.shape} vs {expected_shape}")
        if delta_star_arr.shape != expected_shape:
            raise ValueError(f"delta_star shape mismatch: {delta_star_arr.shape} vs {expected_shape}")

        return gamma_star_arr, delta_star_arr

    def _int_eprior(
        self,
        standardized_data: npt.NDArray,
        gamma_hat: npt.NDArray,
        delta_hat: npt.NDArray,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Compute non-parametric empirical Bayes adjustments via kernel weighting.

        Parameters
        ----------
        standardized_data : array, shape (n_features, n_samples_site)
            Standardized data for a single site.
        gamma_hat : array, shape (n_features,)
            Location parameters for each feature.
        delta_hat : array, shape (n_features,)
            Scale parameters for each feature.

        Returns
        -------
        gamma_star : ndarray, shape (n_features,)
            Weighted location estimates.
        delta_star : ndarray, shape (n_features,)
            Weighted scale estimates.

        """
        n_features = standardized_data.shape[0]

        gamma_hat = np.asarray(gamma_hat, dtype=np.float64)
        delta_hat = np.asarray(delta_hat, dtype=np.float64)

        # Pre-allocate output arrays
        gamma_star = np.empty(n_features, dtype=np.float64)
        delta_star = np.empty(n_features, dtype=np.float64)

        # Precompute constants
        two_pi = 2.0 * math.pi

        for i in range(n_features):
            x = standardized_data[i, :]
            n = x.shape[0]

            # Leave-one-out: use all OTHER features' parameters
            g_other = np.delete(gamma_hat, i)
            d_other = np.delete(delta_hat, i)

            # Vectorized likelihood computation with broadcasting
            x_expanded = x[np.newaxis, :]
            g_expanded = g_other[:, np.newaxis]

            residuals = x_expanded - g_expanded
            resid2 = np.square(residuals)
            sum2 = np.sum(resid2, axis=1)

            # Use log-space for numerical stability
            with np.errstate(divide="ignore", invalid="ignore"):
                log_lh = -0.5 * n * np.log(two_pi * d_other) - sum2 / (2.0 * d_other)
                log_lh_max = np.max(log_lh)
                lh = np.exp(log_lh - log_lh_max)

            # Handle numerical issues
            if np.any(np.isnan(lh)) or np.any(np.isinf(lh)):
                logger.warning(
                    f"Feature {i}: Numerical issues in likelihood computation, replacing NaN/Inf with zeros"
                    "Please check data for constant features or extreme values."
                )
                lh = np.nan_to_num(lh, nan=0.0, posinf=0.0, neginf=0.0)

            lh_sum = np.sum(lh)
            if lh_sum < 1e-300:
                logger.warning(f"Feature {i}: All likelihoods near zero, using original estimate")
                gamma_star[i] = gamma_hat[i]
                delta_star[i] = delta_hat[i]
                continue

            # Compute weights and weighted averages for gamma and delta
            weights = lh / lh_sum

            # Weighted average of gamma and delta using the computed weights
            gamma_star[i] = np.sum(g_other * weights)
            delta_star[i] = np.sum(d_other * weights)

        return gamma_star, delta_star

    def _adjust_data_final(
        self,
        standardized_data: npt.NDArray,
        standardized_mean: npt.NDArray,
        idx_per_site: list[list[int]],
        epsilon: float = 1e-8,
    ):
        """Compute the final harmonized data by applying EB adjustments.

        This method applies the Empirical Bayes adjustments (gamma_star, delta_star)
        to remove site effects and then rescales the data back to original scale.

        The harmonization formula for each site j:
            bayes_data = (standardized_data - gamma_star[j]) / sqrt(delta_star[j])
            Then rescale: bayes_data * sqrt(var_pooled) + standardized_mean

        Parameters
        ----------
        standardized_data : array, shape (n_features, n_samples)
            Standardized data from _standardize_across_features.
        standardized_mean : array, shape (n_features, n_samples)
            Standardized mean used during standardization.
        idx_per_site : list of list of int
            List of sample indices for each site.
        epsilon : float, optional (default 1e-8)
            Small constant to add to delta_star to avoid division by zero.

        Returns
        -------
        bayes_data : ndarray, shape (n_features, n_samples)
            Harmonized data in original scale.

        Raises
        ------
        ValueError
            If gamma_star or delta_star are not fitted or have wrong shapes.
        RuntimeError
            If numerical issues are detected during harmonization.

        """
        # =====================================================================
        # STEP 1: Validate fitted attributes
        # =====================================================================
        if not hasattr(self, "_gamma_star") or not hasattr(self, "_delta_star"):
            raise RuntimeError("gamma_star and delta_star must be computed before adjustment. Call fit() first.")
        n_sites = self._n_sites
        var_pooled = self._var_pooled
        gamma_star = self._gamma_star
        delta_star = self._delta_star

        # Validate shapes
        expected_shape = (n_sites, standardized_data.shape[0])
        if gamma_star.shape != expected_shape:
            raise ValueError(f"gamma_star shape {gamma_star.shape} != expected {expected_shape}.")
        if delta_star.shape != expected_shape:
            raise ValueError(f"delta_star shape {delta_star.shape} != expected {expected_shape}.")
        if var_pooled.shape[0] != standardized_data.shape[0]:
            raise ValueError(f"var_pooled has {var_pooled.shape[0]} features but data has {standardized_data.shape[0]} features.")

        # =====================================================================
        # STEP 2: Apply EB adjustments per site
        # =====================================================================
        # Create output array (copy to avoid modifying input)
        bayes_data = standardized_data.copy()
        pooled_std = np.sqrt(var_pooled)
        if pooled_std.ndim == 1:
            pooled_std = pooled_std[:, np.newaxis]
        for j, site_idxs in enumerate(idx_per_site):
            if len(site_idxs) == 0:
                logger.warning(f"Site {j} has no samples")
                continue

            # Get EB parameters for this site
            site_gamma = gamma_star[j, :][:, np.newaxis]  # (n_features, 1)
            site_delta = np.maximum(delta_star[j, :], epsilon)[:, np.newaxis]  # (n_features, 1)

            # Apply harmonization formula
            bayes_data[:, site_idxs] = (bayes_data[:, site_idxs] - site_gamma) / np.sqrt(site_delta)

        # Rescale to original scale
        bayes_data = bayes_data * pooled_std + standardized_mean

        # Validate output
        if not np.all(np.isfinite(bayes_data)):
            raise RuntimeError("Harmonization produced non-finite values")
        return bayes_data

    def _compute_inverse_gamma_priors(self, delta_hat: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        """Calculate both shape (a) and scale (b) parameters for inverse-gamma prior.

        Computes both simultaneously since they share moment calculations.
        This is ~2x faster than calling separate functions.

        Parameters
        ----------
        delta_hat : array-like, shape (n_features,)
            Estimated variance parameters for a single site.

        Returns
        -------
        a_prior : ndarray, shape (n_features,)
            Shape parameter: a = (2*s2 + m^2) / s2
        b_prior : ndarray, shape (n_features,)
            Scale parameter: b = (m*s2 + m^3) / s2

        """
        delta_hat = np.asarray(delta_hat, dtype=np.float64)

        # Clean input
        if np.any(delta_hat <= 0) or not np.all(np.isfinite(delta_hat)):
            logger.warning("Invalid values in delta_hat, clipping to valid range and replacing non-finite values with median")
            delta_hat = np.clip(delta_hat, 1e-10, 1e10)
            delta_hat = np.where(np.isfinite(delta_hat), delta_hat, np.median(delta_hat))

        # Compute moments once
        m = np.mean(delta_hat)
        s2 = np.var(delta_hat, ddof=1)

        # Handle near-zero variance
        if s2 < 1e-10:
            logger.warning("Variance of delta_hat is near zero, using strong pooling for priors")
            return (
                np.full_like(delta_hat, 1e6),  # Large a = strong pooling
                np.full_like(delta_hat, m),  # b = mean
            )

        # Compute both parameters
        m2 = m * m
        a_prior = (2.0 * s2 + m2) / s2
        b_prior = m * (s2 + m2) / s2

        a_prior = np.clip(a_prior, 1e-6, 1e8, dtype=np.float32)
        b_prior = np.clip(b_prior, 1e-8, 1e8, dtype=np.float32)
        return a_prior, b_prior

    def _postmean(
        self,
        gamma_hat: npt.ArrayLike,
        gamma_bar: npt.ArrayLike,
        n: int | npt.NDArray,
        delta_star: float | npt.ArrayLike,
        tau_2: npt.ArrayLike,
    ) -> npt.NDArray:
        """Compute posterior mean (shrunken estimate) for location parameters.

        Formula: gamma_star = (tau_2 * n * gamma_hat + delta * gamma_bar) / (tau_2 * n + delta)

        Parameters
        ----------
        gamma_hat : array-like, shape (n_features,)
            Observed site means from L/S model.
        gamma_bar : array-like, shape (n_features,)
            Prior mean (expected batch effect across features).
        n : int or array-like
            Sample size for the site.
        delta_star : float or array-like
            Estimated variance (sampling precision).
        tau_2 : array-like, shape (n_features,)
            Prior variance (how much batch effects vary across features).

        Returns
        -------
        gamma_star : ndarray, shape (n_features,)
            Posterior mean (shrunken estimate) for each feature.

        """
        # Convert to arrays for broadcasting
        gamma_hat = np.asarray(gamma_hat)
        gamma_bar = np.asarray(gamma_bar)
        tau_2 = np.asarray(tau_2)
        n = np.asarray(n)
        delta_star = np.asarray(delta_star)

        # Compute posterior mean using precision-weighted average
        prior_precision = tau_2 * n

        # Handle numerical issues
        denominator = prior_precision + delta_star
        if np.any(denominator <= 0):
            logger.warning("_postmean: Non-positive denominator, clipping")
            denominator = np.clip(denominator, 1e-10, None)

        return (prior_precision * gamma_hat + delta_star * gamma_bar) / denominator

    def _postvar(
        self,
        sum_2: float | npt.ArrayLike,
        n: int | npt.ArrayLike,
        a_prior: npt.ArrayLike,
        b_prior: npt.ArrayLike,
    ) -> npt.NDArray:
        """Compute posterior variance estimate for scale parameters.

        Formula: delta_star = (0.5 * sum_2 + b_prior) / (n/2 + a_prior - 1)

        Parameters
        ----------
        sum_2 : float or array-like, shape (n_features,)
            Sum of squared deviations: sum((x - gamma_star)^2)
        n : int or array-like
            Sample size.
        a_prior : array-like, shape (n_features,)
            Shape parameter of inverse-gamma prior.
        b_prior : array-like, shape (n_features,)
            Scale parameter of inverse-gamma prior.

        Returns
        -------
        delta_star : ndarray, shape (n_features,)
            Posterior variance estimate for each feature.

        """
        # Convert to arrays for broadcasting
        sum_2 = np.asarray(sum_2)
        a_prior = np.asarray(a_prior)
        b_prior = np.asarray(b_prior)
        n = np.asarray(n)

        # Posterior update for inverse-gamma
        numerator = 0.5 * sum_2 + b_prior
        denominator = n / 2.0 + a_prior - 1.0

        # Handle numerical issues
        if np.any(denominator <= 0):
            logger.warning("_postvar: Non-positive denominator, using prior estimate only")
            with np.errstate(divide="ignore", invalid="ignore"):
                prior_mean = np.where(a_prior > 1, b_prior / (a_prior - 1), b_prior)
            return prior_mean

        result = numerator / denominator

        # Ensure positive variance
        return np.clip(result, 1e-8, None)

    def _convert_sites(self, s: list[str]) -> list[int]:
        """Convert sites to proper format."""
        ks = set(s)
        vs = list(range(1, len(ks) + 1))
        kvs = dict(zip(ks, vs, strict=True))
        return [kvs[k] for k in s]

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
        return tags
