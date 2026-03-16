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

        """
        logger.debug("Fitting")

        X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES, estimator=self)
        if np.asarray(sites).ndim == 1:
            sites = np.asarray(sites).reshape(-1, 1)
        sites = check_array(sites, copy=self.copy, estimator=self)

        check_consistent_length(X, sites)

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
                "You specified categorical and / or continuous covariates to be preserved. "
                "If you intend to build a machine learning model, then make sure that you DO NOT preserve the covariates acting "
                "as the model's target, as this would be data leakage. "
                "Those should be passed in .transform() and not in .fit() ."
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
        )
        logger.debug("Fitting L/S model")
        gamma_hat, delta_hat = self._fit_ls_model(
            standardized_data,
            design,
            idx_per_site,
        )
        logger.debug("Finding priors")
        gamma_bar, tau_2, a_prior, b_prior = self._find_priors(
            gamma_hat,
            delta_hat,
        )
        if self.empirical_bayes:
            if self.parametric_adjustments:
                logger.debug("Finding parametric adjustments")
                self._gamma_star, self._delta_star = self._find_parametric_adjustments(
                    standardized_data,
                    idx_per_site,
                    gamma_hat,
                    delta_hat,
                    gamma_bar,
                    tau_2,
                    a_prior,
                    b_prior,
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
            design,
            standardized_mean,
            n_samples_per_site,
            n_samples,
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
        """Create a design matrix.

        It contains:

          * One-hot encoding of the sites [n_samples, n_sites]
          * One-hot encoding of each categorical covariates (removing
            the first column) [n_samples,
            (n_categorical_covivariate_names-1) * n_categorical_covariates]
          * Each continuous covariates

        Parameters
        ----------
        sites : array-like, shape (n_samples, 1)
            Sites.
        categorical_covariates : array-like, shape (n_samples, n_categorical_covariates) or None
            Categorical covariates.
        continuous_covariates : array-like, shape (n_samples, n_continuous_covariates) or None
            Continuous covariates.
        fitting : bool, optional (default False)
            Whether fitting or not.

        Returns
        -------
        array
            The design matrix.

        """
        design_list = []

        # Sites
        if fitting:
            self._site_encoder = OneHotEncoder(sparse_output=False)
            self._site_encoder.fit(sites)

        sites_design = self._site_encoder.transform(sites)
        design_list.append(sites_design)

        # Categorical covariates
        if categorical_covariates is not None:
            n_categorical_covariates = categorical_covariates.shape[1]

            if fitting:
                self._categorical_encoders = []

                for i in range(n_categorical_covariates):
                    cat_encoder = OneHotEncoder(sparse_output=False)
                    cat_encoder.fit(categorical_covariates[:, i][:, np.newaxis])
                    self._categorical_encoders.append(cat_encoder)

            for i in range(n_categorical_covariates):
                cat_encoder = self._categorical_encoders[i]
                cat_covariate_one_hot = cat_encoder.transform(categorical_covariates[:, i][:, np.newaxis])
                cat_covariate_design = cat_covariate_one_hot[:, 1:]
                design_list.append(cat_covariate_design)

        # Continuous covariates
        if continuous_covariates is not None:
            design_list.append(continuous_covariates)

        design = np.hstack(design_list)
        return design

    def _standardize_across_features(
        self,
        X: npt.ArrayLike,
        design: npt.NDArray,
        n_samples: int,
        n_samples_per_site: list[int],
        fitting: bool = False,
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
        n_samples_per_site : list of int
            Sample count per site.
        fitting : bool, optional (default False)
            Whether fitting or not.

        Returns
        -------
        array
            Standardized data.
        array
            Standardized mean used during the process.

        """
        if fitting:
            self._beta_hat = np.dot(np.dot(np.linalg.inv(np.dot(design.T, design)), design.T), X.T)

            # Standardization Model
            self._grand_mean = np.dot(
                (n_samples_per_site / float(n_samples)).T,
                self._beta_hat[: self._n_sites, :],
            )
            self._var_pooled = np.dot(
                ((X - np.dot(design, self._beta_hat).T) ** 2),
                np.ones((n_samples, 1)) / float(n_samples),
            )

        standardized_mean = np.dot(self._grand_mean.T[:, np.newaxis], np.ones((1, n_samples)))

        tmp = np.asarray(design.copy())
        tmp[:, : self._n_sites] = 0
        standardized_mean += np.dot(tmp, self._beta_hat).T

        standardized_data = (X - standardized_mean) / np.dot(np.sqrt(self._var_pooled), np.ones((1, n_samples)))

        return standardized_data, standardized_mean

    def _fit_ls_model(
        self,
        standardized_data: npt.NDArray,
        design: npt.NDArray,
        idx_per_site: list[list[int]],
    ) -> tuple[npt.NDArray, list]:
        """Location and scale (L/S) adjustments.

        Parameters
        ----------
        standardized_data : array
            Standardized data.
        design : array
            Design matrix.
        idx_per_site : list of list of int
            Index per site.

        Returns
        -------
        array
            Gamma hat.
        list
            Delta hat.

        """
        site_design = design[:, : self._n_sites]
        gamma_hat = np.dot(
            np.dot(
                np.linalg.inv(np.dot(site_design.T, site_design)),
                site_design.T,
            ),
            standardized_data.T,
        )

        delta_hat = []
        for site_idxs in idx_per_site:
            if self.mean_only:
                delta_hat.append(np.repeat(1, standardized_data.shape[0]))
            else:
                delta_hat.append(np.var(standardized_data[:, site_idxs], axis=1, ddof=1))

        return gamma_hat, delta_hat

    def _find_priors(
        self,
        gamma_hat: npt.NDArray,
        delta_hat: npt.ArrayLike,
    ) -> tuple[npt.ArrayLike, npt.ArrayLike, list, list]:
        """Compute a and b priors.

        Parameters
        ----------
        gamma_hat : array
            Gamma hat.
        delta_hat : array-like
            Delta hat.

        Returns
        -------
        array-like
            Gamma bar.
        array-like
            Tau 2.
        list
            a prior.
        list
            b prior.

        """
        delta_hat = list(map(_convert_zeroes, delta_hat))
        gamma_bar = np.mean(gamma_hat, axis=1)
        tau_2 = np.var(gamma_hat, axis=1, ddof=1)
        if self.mean_only:
            a_prior = None
            b_prior = None
        else:
            a_prior = list(map(_aprior_fn, delta_hat))
            b_prior = list(map(_bprior_fn, delta_hat))

        return gamma_bar, tau_2, a_prior, b_prior

    def _find_parametric_adjustments(
        self,
        standardized_data: npt.NDArray,
        idx_per_site: list[list[int]],
        gamma_hat: npt.NDArray,
        delta_hat: npt.ArrayLike,
        gamma_bar: npt.ArrayLike,
        tau_2: npt.ArrayLike,
        a_prior: list,
        b_prior: list,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Compute parametric empirical Bayes site/batch effect parameter estimates.

        Parameters
        ----------
        standardized_data : array
            Standardized data.
        idx_per_site : list of list of int
            Index per site.
        gamma_hat : array
            Gamma hat.
        delta_hat : array-like
            Delta hat.
        gamma_bar : array-like
            Gamma bar.
        tau_2 : array-like
            Tau 2.
        a_prior : list
            a prior.
        b_prior : list
            b prior.

        Returns
        -------
        array
            Gamma star.
        array
            Delta star.

        """
        gamma_star, delta_star = [], []
        for i, site_idxs in enumerate(idx_per_site):
            if self.mean_only:
                gamma_star.append(_postmean(gamma_hat[i], gamma_bar[i], 1, 1, tau_2[i]))
                delta_star.append(np.repeat(1, standardized_data.shape[0]))
            else:
                gamma_hat_adjust, delta_hat_adjust = self._iteration_solver(
                    standardized_data[:, site_idxs],
                    gamma_hat[i],
                    delta_hat[i],
                    gamma_bar[i],
                    tau_2[i],
                    a_prior[i],
                    b_prior[i],
                )
                gamma_star.append(gamma_hat_adjust)
                delta_star.append(delta_hat_adjust)

        return np.asarray(gamma_star), np.asarray(delta_star)

    def _iteration_solver(
        self,
        standardized_data: npt.NDArray,
        gamma_hat: npt.ArrayLike,
        delta_hat: npt.ArrayLike,
        gamma_bar: npt.ArrayLike,
        tau_2: npt.ArrayLike,
        a_prior: list,
        b_prior: list,
        convergence: float = 0.0001,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Find the parametric site/batch effect adjustments.

        Parameters
        ----------
        standardized_data : array
            Standardized data.
        gamma_hat : array-like
            Gamma hat.
        delta_hat : array-like
            Delta hat.
        gamma_bar : array-like
            Gamma bar.
        tau_2 : array-like
            Tau 2.
        a_prior : list
            a prior.
        b_prior : list
            b prior.
        convergence : float, optional (default 0.0001)
            Convergence threshold.

        Returns
        -------
        array
            Gamma hat adjusted.
        array
            Delta hat adjusted.

        """
        n = (1 - np.isnan(standardized_data)).sum(axis=1)
        gamma_hat_old = gamma_hat.copy()
        delta_hat_old = delta_hat.copy()

        change = 1
        count = 0

        while change > convergence:
            gamma_hat_new = _postmean(gamma_hat, gamma_bar, n, delta_hat_old, tau_2)
            sum_2 = (
                (
                    standardized_data
                    - np.dot(
                        gamma_hat_new[:, np.newaxis],
                        np.ones((1, standardized_data.shape[1])),
                    )
                )
                ** 2
            ).sum(axis=1)
            delta_hat_new = _postvar(sum_2, n, a_prior, b_prior)

            change = max(
                (abs(gamma_hat_new - gamma_hat_old) / gamma_hat_old).max(),
                (abs(delta_hat_new - delta_hat_old) / delta_hat_old).max(),
            )

            gamma_hat_old = gamma_hat_new
            delta_hat_old = delta_hat_new

            count += 1

        return gamma_hat_new, delta_hat_new

    def _find_non_parametric_adjustments(
        self,
        standardized_data: npt.NDArray,
        idx_per_site: list[list[int]],
        gamma_hat: npt.NDArray,
        delta_hat: npt.ArrayLike,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Compute non-parametric empirical Bayes site/batch effect parameter estimates.

        Parameters
        ----------
        standardized_data : array
            Standardized data.
        idx_per_site : list of list of int
            Index per site.
        gamma_hat : array
            Gamma hat.
        delta_hat : array-like
            Delta hat.

        Returns
        -------
        array
            Gamma star.
        array
            Delta star.

        """
        gamma_star, delta_star = [], []
        for i, site_idxs in enumerate(idx_per_site):
            if self.mean_only:
                delta_hat[i] = np.repeat(1, standardized_data.shape[0])
            gamma_hat_adjust, delta_hat_adjust = self._int_eprior(
                standardized_data[:, site_idxs],
                gamma_hat[i],
                delta_hat[i],
            )

            gamma_star.append(gamma_hat_adjust)
            delta_star.append(delta_hat_adjust)

        return np.asarray(gamma_star), np.asarray(delta_star)

    def _int_eprior(
        self,
        standardized_data: npt.NDArray,
        gamma_hat: npt.ArrayLike,
        delta_hat: npt.ArrayLike,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Find the non-parametric site/batch effect adjustments.

        Parameters
        ----------
        standardized_data : array
            Standardized data.
        gamma_hat : array-like
            Gamma hat.
        delta_hat : array-like
            Delta hat.

        Returns
        -------
        array
            Gamma hat adjusted.
        array
            Delta hat adjusted.

        """
        r = standardized_data.shape[0]
        gamma_star, delta_star = [], []
        for i in range(0, r, 1):
            g = np.delete(gamma_hat, i)
            d = np.delete(delta_hat, i)
            x = standardized_data[i, :]
            n = x.shape[0]
            j = np.repeat(1, n)
            a = np.repeat(x, g.shape[0])
            a = a.reshape(n, g.shape[0])
            a = np.transpose(a)
            b = np.repeat(g, n)
            b = b.reshape(g.shape[0], n)
            resid2 = np.square(a - b)
            sum2 = resid2.dot(j)
            lh = 1 / (2 * math.pi * d) ** (n / 2) * np.exp(-sum2 / (2 * d))
            lh = np.nan_to_num(lh)
            gamma_star.append(sum(g * lh) / sum(lh))
            delta_star.append(sum(d * lh) / sum(lh))

        return gamma_star, delta_star

    def _adjust_data_final(
        self,
        standardized_data: npt.NDArray,
        design: npt.NDArray,
        standardized_mean: npt.NDArray,
        n_samples_per_site: list[int],
        n_samples: int,
        idx_per_site: list[list[int]],
    ):
        """Compute the harmonized data.

        Parameters
        ----------
        standardized_data : array
            Standardized data.
        design : array
            Design matrix.
        standardized_mean : array
            Standardized mean.
        n_samples_per_site : list of int
            Sample count per site.
        n_samples : int
            Sample count.
        idx_per_site : list of list of int
            Index per site.

        Returns
        -------
        array
            Bayes data.

        """
        n_sites = self._n_sites
        var_pooled = self._var_pooled
        gamma_star = self._gamma_star
        delta_star = self._delta_star

        site_design = design[:, :n_sites]

        bayes_data = standardized_data

        for j, site_idxs in enumerate(idx_per_site):
            denominator = np.dot(
                np.sqrt(delta_star[j, :])[:, np.newaxis],
                np.ones((1, n_samples_per_site[j])),
            )
            numerator = bayes_data[:, site_idxs] - np.dot(site_design[site_idxs, :], gamma_star).T

            bayes_data[:, site_idxs] = numerator / denominator

        bayes_data = bayes_data * np.dot(np.sqrt(var_pooled), np.ones((1, n_samples))) + standardized_mean

        return bayes_data

    # Overridden for check_is_fitted() usage
    def __sklearn_is_fitted__(self) -> bool:
        """Check fitted status."""
        return hasattr(self, "_n_sites")

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.estimator_type = "transformer"
        tags.target_tags.required = True
        tags.target_tags.two_d_labels = True
        tags.target_tags.positive_only = True
        tags.input_tags.two_d_array = True
        return tags


def _convert_zeroes(x: npt.NDArray) -> npt.NDArray:
    """Convert zeroes.

    Parameters
    ----------
    x : array
        Input array.

    Returns
    -------
    array
        Output array.

    """
    x[x == 0] = 1
    return x


def _aprior_fn(delta_hat: npt.NDArray) -> float:
    """Calculate a prior.

    Parameters
    ----------
    delta_hat : array-like
        Delta hat.

    Returns
    -------
    array
        a prior.

    """
    m = np.mean(delta_hat)
    s2 = np.var(delta_hat, ddof=1, dtype=np.float32)
    return (2 * s2 + m**2) / s2


def _bprior_fn(delta_hat: npt.NDArray) -> float:
    """Calculate b prior.

    Parameters
    ----------
    delta_hat : array-like
        Delta hat.

    Returns
    -------
    array
        b prior.

    """
    m = np.mean(delta_hat)
    s2 = np.var(delta_hat, ddof=1, dtype=np.float32)
    return (m * s2 + m**3) / s2


def _postmean(
    gamma_hat: npt.ArrayLike,
    gamma_bar: npt.ArrayLike,
    n: int,
    delta_star: int,
    tau_2: npt.ArrayLike,
):
    """Postmean.

    Parameters
    ----------
    gamma_hat : array
        Gamma hat.
    gamma_bar : array-like
        Gamma bar.
    n : int
        Count.
    delta_star : list
        Delta star.
    tau_2 : array-like
        Tau 2.

    Returns
    -------
    float
        Postmean.

    """
    return (tau_2 * n * gamma_hat + delta_star * gamma_bar) / (tau_2 * n + delta_star)


def _postvar(sum_2: float, n: int, a_prior: list, b_prior: list) -> float:
    """Postvar.

    Parameters
    ----------
    sum_2 : float
        Sum squared.
    n : int
        Count.
    a_prior : list
        a prior.
    b_prior : list
        b prior.

    Returns
    -------
    float
        Postvar.

    """
    return (0.5 * sum_2 + b_prior) / (n / 2.0 + a_prior - 1.0)
