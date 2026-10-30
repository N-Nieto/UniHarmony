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
from sklearn.utils.validation import check_is_fitted

from uniharmony._utils import validate_sites

from ._base import BaseComBat


__all__ = ["NeuroComBat"]

logger = structlog.get_logger(src="NeuroComBat")  # lazy: honours later `uniharmony.verbosity` calls


class NeuroComBat(BaseComBat):
    r"""Harmonize scanner effects in multi-site imaging data.

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
        Whether to only adjust the mean (no scaling). The empirical Bayes
        estimate of each site effect uses the site sample size; this
        deliberately corrects R ``sva::ComBat`` and ``neuroCombat``, which use
        a sample size of 1 (see Notes).

    Attributes
    ----------
    sites_ : array, shape (n_samples,)
        Fitted site names.

    Notes
    -----
    This implementation follows the ComBat model of [1]_ and differs from
    R ``sva::ComBat`` (3.50.0) and ``neuroCombat`` (0.2.12) in two documented
    places. Both were found with simulations from the ComBat model and are
    tested in ``tests/combat/test_reference_conformance.py``:

    * **Pooled variance.** The residual sum of squares is divided by
      :math:`n - S` (:math:`S` sites) instead of :math:`n`. ComBat rescales each
      site with a variance computed with ``ddof=1``, so the harmonized data keep
      :math:`n - S` residual degrees of freedom. Dividing by :math:`n` shrinks
      the harmonized variance by :math:`(n - S)/n` (0.90 with 30 subjects in
      3 sites); dividing by :math:`n - S` makes it unbiased. Outside
      ``mean_only``, the output is the reference output with the deviations
      from the standardized mean multiplied by :math:`\sqrt{n/(n - S)}`.
    * **mean_only.** With the scale fixed to 1, the site mean
      :math:`\hat\gamma_{ig}` has sampling variance :math:`1/n_i`, so the
      posterior mean is
      :math:`\gamma^*_{ig} = (\tau_i^2 n_i \hat\gamma_{ig} + \bar\gamma_i) / (\tau_i^2 n_i + 1)`.
      The references use :math:`n_i = 1`, which leaves most of the
      feature-specific site effect in the data (a site classifier then
      recovers the site well above chance).

    See the "Implementations conformance" gallery for the derivations,
    the simulations, and a runtime and memory comparison.

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
    ) -> None:
        self.empirical_bayes = empirical_bayes
        self.parametric_adjustments = parametric_adjustments
        self.mean_only = mean_only

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

        # Validate input
        X, sites = self._check_X_sites(X, sites, estimator=self)
        validate_sites(sites)

        # Check that categorical_covariates and continuous_covariates have correct shape and type if they are not None.
        # Also, track whether they were used during fit to check during transform
        self._categorical_covariates_used = False
        if categorical_covariates is not None:
            self._categorical_covariates_used = True
            categorical_covariates = self._check_categorical_covariates(X, categorical_covariates)

        self._continuous_covariates_used = False
        if continuous_covariates is not None:
            self._continuous_covariates_used = True
            continuous_covariates = self._check_continuous_covariates(X, continuous_covariates, estimator=self)

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

        standardized_data = self.fit_standardize(
            X=X,
            design=design,
            n_samples=n_samples,
            n_samples_per_site=n_samples_per_site,
            n_smooth_cols=None,
            smooth_formula=None,
            df_gam=None,
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

    def fit_transform(
        self,
        X: npt.ArrayLike,
        sites: npt.ArrayLike,
        categorical_covariates: npt.ArrayLike | None = None,
        continuous_covariates: npt.ArrayLike | None = None,
        var_epsilon: float = 1e-8,
        delta_epsilon: float = 1e-8,
        tau_2_epsilon: float = 1e-10,
        max_iter: int = 1000,
    ) -> npt.NDArray:
        """Fit to data, then harmonize it.

        Equivalent to ``fit(...).transform(...)``: the covariates are used by
        both, the remaining options only by :meth:`fit`.

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

        Returns
        -------
        array, shape (n_samples, n_features)
            The harmonized data.

        """
        return self._fit_then_transform(
            X=X,
            sites=sites,
            categorical_covariates=categorical_covariates,
            continuous_covariates=continuous_covariates,
            var_epsilon=var_epsilon,
            delta_epsilon=delta_epsilon,
            tau_2_epsilon=tau_2_epsilon,
            max_iter=max_iter,
        )

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
        X, sites = self._check_X_sites(X, sites, estimator=self)

        if self._categorical_covariates_used:
            categorical_covariates = self._check_categorical_covariates(X, categorical_covariates)

        if self._continuous_covariates_used:
            continuous_covariates = self._check_continuous_covariates(X, continuous_covariates)

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
