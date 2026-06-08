"""Provide BARTharm transformer."""

# Adapted from:
# https://github.com/NeuroSML/BARTharm
# with no license specified.

import numpy as np
import numpy.typing as npt
import scipy.stats as stats
from sklearn.base import BaseEstimator, TransformerMixin


__all__ = ["BARTharm"]


class BARTharm(TransformerMixin, BaseEstimator):
    """Harmonize multi-site scanner effects using image quality metrics (IQM).

    Imaging-derived outcomes are harmonized by separating biological signal from scanner-related effects using
    Image Quality Metrics (IQMs) instead of Scanner IDs. It uses Bayesian Additive Regression Trees (BART) [1]_
    with Gibbs sampling to estimate scanner components (mu, from IQMs) and biological effects (tau, from biological covariates).

    References
    ----------
    .. [1] Prevot E, et al., (2025).
           BARTharm: MRI Harmonization Using Image Quality Metrics and Bayesian Non-parametric.
           bioRxiv. Published online 2025.
           doi:10.1101/2025.06.04.657792

    """

    def fit(
        self,
        X: npt.ArrayLike,
        biological_covariates: npt.ArrayLike | None = None,
        iqm_covariates: npt.ArrayLike | None = None,
    ) -> "BARTharm":
        """Compute per-feature statistics to perform harmonization.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        biological_covariates : array-like, shape (n_samples, n_biological_covariates) or None, optional (default None)
            The biological covariates to be preserved during harmonization.
        iqm_covariates : array-like, shape (n_samples, n_iqm_covariates) or None, optional (default None)
            The IQM covariates to be preserved during harmonization.

        """
        # Remove rows with nan

        # Quantile normalise biological covariates

        # Quantile normalise IQM covariates

        # Loop over features
        # Set up hyperparameters
        # Run Gibbs samples
        # Extract posteriors
        # Compute posterior mean prediction
        # RMSE
        # Compute harmonized feature by removing mu


def quantile_norm(arr):
    """Normalize the columns of ``arr`` to each have the same distribution.

    Given a 2D array of samples x features, quantile normalisation ensures all samples have the same
    spread of data (by construction).

    The data across each row are averaged to obtain an average column. Each
    column quantile is replaced with the corresponding quantile of the average
    column.

    Parameters
    ----------
    arr : array-like, shape (n_samples, n_features)
        The input data.

    Returns
    -------
    array, shape (n_samples, n_features)
        The normalized data.

    """
    # Column-wise, rank entries from lowest to highest
    ranks = np.apply_along_axis(stats.rankdata, 0, arr)
    # Convert ranks to integer indices from 0 to rows-1
    rank_idxs = ranks.astype(int) - 1
    # Compute quantiles
    quantiles = np.mean(np.sort(arr, axis=0), axis=1)
    # Index the quantiles for each rank with the ranks matrix
    return quantiles[rank_idxs]
