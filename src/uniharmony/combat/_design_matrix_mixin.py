"""Provide DesignMatrixMixin."""

import numpy as np
import numpy.typing as npt
import structlog
from sklearn.preprocessing import OneHotEncoder


__all__ = ["DesignMatrixMixin"]

logger = structlog.get_logger()
logger = logger.bind(src="DesignMatrixMixin")


class DesignMatrixMixin:
    """Mixin class to construct design matrix for ComBat-based methods.

    Creates a design matrix for the linear model. The design matrix combines:
      1. One-hot encoded sites (full encoding, all columns kept)
      2. One-hot encoded categorical covariates (first category dropped per covariate) if provided.
      3. Continuous covariates (used as-is) if provided.

    This follows standard ANOVA coding where the design matrix is used to
    estimate site effects while controlling for covariates.

    Notes
    -----
    The drop-first encoding for categorical covariates avoids co-linearity
    with the intercept (which is implicit in the site effects). This is
    standard practice in regression analysis.

    """

    def fit_design_matrix(
        self,
        sites: npt.NDArray,
        categorical_covariates: npt.NDArray | None,
        continuous_covariates: npt.NDArray | None,
    ) -> npt.NDArray:
        """Fit encoders and make design matrix.

        Parameters
        ----------
        sites : array, shape (n_samples, 1)
            Site labels for each sample.
        categorical_covariates : array, shape (n_samples, n_categorical) or None
            Categorical covariates to preserve (e.g., sex, disease status).
            Each column is treated as a separate categorical variable.
        continuous_covariates : array, shape (n_samples, n_continuous) or None
            Continuous covariates to preserve (e.g., age, clinical scores).

        Returns
        -------
        ndarray, shape (n_samples, n_effects)
            The design matrix where:
            - First n_sites columns are site indicators
            - Next columns are categorical covariates (drop-first encoded) if provided
            - Final columns are continuous covariates if provided

        Examples
        --------
        >>> sites = np.array([[1], [1], [2], [2]])
        >>> sex = np.array([['M'], ['F'], ['M'], ['F']])
        >>> age = np.array([[25], [30], [35], [40]])
        >>> design = self.fit_design_matrix(sites, sex, age)
        >>> design.shape
        (4, 4)  # 2 sites + 1 sex (drop-first) + 1 age

        """
        logger.debug("Making design matrix")
        # Fit site encoder
        self._site_encoder = OneHotEncoder(
            sparse_output=False,
            dtype=float,
            handle_unknown="error",
        )
        self._site_encoder.fit(sites.reshape(-1, 1))
        logger.debug(f"Fitted site encoder: {len(self._site_encoder.categories_[0])} sites")

        # Fit categorical encoders if provided
        if categorical_covariates is not None:
            n_cat_covs = categorical_covariates.shape[1]
            self._categorical_encoders = []

            for i in range(n_cat_covs):
                cat_encoder = OneHotEncoder(
                    sparse_output=False,
                    dtype=float,
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

        return self.transform_design_matrix(
            sites=sites,
            categorical_covariates=categorical_covariates,
            continuous_covariates=continuous_covariates,
        )

    def transform_design_matrix(
        self,
        sites: npt.NDArray,
        categorical_covariates: npt.NDArray | None,
        continuous_covariates: npt.NDArray | None,
    ) -> npt.NDArray:
        """Transform using fitted encoders and make design matrix.

        Parameters
        ----------
        sites : array, shape (n_samples, 1)
            Site labels for each sample.
        categorical_covariates : array, shape (n_samples, n_categorical) or None
            Categorical covariates to preserve (e.g., sex, disease status).
            Each column is treated as a separate categorical variable.
        continuous_covariates : array, shape (n_samples, n_continuous) or None
            Continuous covariates to preserve (e.g., age, clinical scores).

        Returns
        -------
        ndarray, shape (n_samples, n_effects)
            The design matrix where:
            - First n_sites columns are site indicators
            - Next columns are categorical covariates (drop-first encoded) if provided
            - Final columns are continuous covariates if provided

        Raises
        ------
        ValueError
            If no site encoder is found.
        RuntimeError
            If encoder classes differ between fit and transform.

        Examples
        --------
        >>> sites = np.array([[1], [1], [2], [2]])
        >>> sex = np.array([['M'], ['F'], ['M'], ['F']])
        >>> age = np.array([[25], [30], [35], [40]])
        >>> design = self.transform_design_matrix(sites, sex, age)
        >>> design.shape
        (4, 4)  # 2 sites + 1 sex (drop-first) + 1 age

        """
        logger.debug("Making design matrix")

        if not hasattr(self, "_site_encoder"):
            raise ValueError("Must call fit_design_matrix")

        design_parts = []
        # Transform sites
        sites_encoded = self._site_encoder.transform(sites.reshape(-1, 1))
        design_parts.append(sites_encoded)
        n_samples = sites.shape[0]
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

        design = np.hstack(design_parts)

        # Final validation
        if design.shape[0] != n_samples:
            raise RuntimeError(f"Design matrix has {design.shape[0]} rows but expected {n_samples}")

        logger.debug(f"Design matrix shape: {design.shape}")

        return design
