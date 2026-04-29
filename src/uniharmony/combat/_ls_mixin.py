"""Provide LocationAndScaleMixin."""

import numpy as np
import numpy.typing as npt
import structlog

from uniharmony._utils import handle_near_zero_values, solve_ordinary_least_squares


__all__ = ["LocationAndScaleMixin"]

logger = structlog.get_logger()
logger = logger.bind(src="LocationAndScaleMixin")


class LocationAndScaleMixin:
    """Mixin class to provide Location and Scale (L/S) model to estimate site-specific batch effects.

    This method estimates the location (mean) and scale (variance) adjustments
    needed for each site. These are the "batch effects" that ComBat will remove.

    """

    def fit_ls_model(
        self,
        data: npt.NDArray,
        design: npt.NDArray,
        idx_per_site: list[list[int]],
        epsilon: float = 1e-8,
    ) -> tuple[npt.NDArray, list]:
        """Fit L/S model.

        Parameters
        ----------
        data : array, shape (n_features, n_samples)
            Standardized data.
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
        delta_hat : list of array
            Each array has shape (n_features,) containing estimated scale (variance)
            for each site and each feature. This is each site's variance.

        """
        logger.debug("Fitting L/S model")
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

        gamma_hat = solve_ordinary_least_squares(gram_site, data, site_design)

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
        for idx, site_idxs in enumerate(idx_per_site):
            if self.mean_only:
                # [MEAN-ONLY MODE] Assume equal variance across sites
                # Set all variances to 1 (already standardized)
                # This assumes batch effect is only in location, not scale
                delta_hat.append(np.repeat(1, data.shape[0]))
            else:
                # [FULL MODE] Estimate site-specific variances
                # Check minimum samples for variance estimation
                # With ddof=1, we need at least 2 samples, but more is better
                n_site_samples = len(site_idxs)
                if n_site_samples < 2:
                    logger.error(
                        f"Site {idx} has only {n_site_samples} sample(s). "
                        "Cannot estimate variance with ddof=1. Setting variance to 1."
                    )
                    delta_hat.append(np.ones(data.shape[0]))
                    continue
                elif n_site_samples < 16:
                    # Warn about small sample sizes for variance estimation
                    # Research shows ComBat becomes unstable with <16-32 samples per site
                    logger.warning(
                        f"Site {idx} has only {n_site_samples} samples. "
                        "Variance estimates may be unstable. Consider using mean_only=True "
                        "or collecting more data."
                    )

                site_data = data[:, site_idxs]

                # axis=1 = compute variance across samples for each feature
                # ddof=1 for sample variance (unbiased estimator)
                site_var = np.var(site_data, axis=1, ddof=1)

                # Handle near-zero or negative variances
                # Numerical errors or constant features can cause this
                site_var = handle_near_zero_values(site_var, epsilon=epsilon)

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
