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
from sklearn.utils.validation import check_is_fitted

from uniharmony._verbose import get_logger

from ._base import BaseComBat


__all__ = ["NeuroComBat"]

logger = get_logger(src="NeuroComBat")


class NeuroComBat(BaseComBat):
    r"""Harmonize scanner effects in multi-site imaging data.

    This transformer performs harmonization using a parametric empirical Bayes
    framework proposed in ComBat [1]_ and adapted to neuroimaging data
    here [2]_ .

    Parameters
    ----------
    site_column : str or int or None, optional (default None)
        Column of ``X`` holding the site of every sample (name for a
        DataFrame, position for an array). If None, the sites are passed to
        :meth:`fit` and :meth:`transform` with the ``sites`` keyword. Read the
        sites from a column to use the transformer in a scikit-learn
        :class:`~sklearn.pipeline.Pipeline`, where only ``X`` and ``y`` are
        split by the cross-validation folds.
    categorical_columns : str or int or list or None, optional (default None)
        Columns of ``X`` holding categorical covariates to preserve (e.g., sex,
        disease). Alternative to the ``categorical_covariates`` keyword.
    continuous_columns : str or int or list or None, optional (default None)
        Columns of ``X`` holding continuous covariates to preserve (e.g., age,
        clinical scores). Alternative to the ``continuous_covariates`` keyword.
    ref_site : str or int or list or None, optional (default None)
        Reference site(s) of the harmonization. If None, every site is
        harmonized to the pooled mean and variance of all sites. If a single
        site, every site is harmonized to the mean and variance of that site,
        whose data are left unchanged (R ``sva::ComBat`` ``ref.batch``). If
        several sites, every site (the reference sites included) is
        harmonized to the pooled mean and variance of the reference sites.
        The covariate effects are always estimated from all sites. See Notes.
    empirical_bayes : bool, optional (default True)
        Whether to perform empirical Bayes.
    parametric_adjustments : bool, optional (default True)
        Whether to perform parametric adjustments.
    mean_only : bool, optional (default False)
        Whether to only adjust the mean (no scaling). The empirical Bayes
        estimate of each site effect uses the site sample size; this
        deliberately corrects R ``sva::ComBat`` and ``neuroCombat``, which use
        a sample size of 1 (see Notes).
    var_epsilon : float, optional (default 1e-8)
        Small constant added to variances to avoid division by zero.
    delta_epsilon : float, optional (default 1e-8)
        Lower bound of the site scale factors, to avoid division by zero. Used
        with ``empirical_bayes=True`` and ``parametric_adjustments=True``.
    tau_2_epsilon : float, optional (default 1e-10)
        Lower bound of the prior variance of the site location effects. Used
        with ``empirical_bayes=True`` and ``parametric_adjustments=True``.
    max_iter : int, optional (default 1000)
        Maximum number of iterations of the parametric empirical Bayes
        solver. Used with ``empirical_bayes=True`` and
        ``parametric_adjustments=True``.

    Attributes
    ----------
    sites_ : array, shape (n_sites,)
        Fitted site names.
    ref_sites_ : array, shape (n_ref_sites,)
        The reference sites (all sites with ``ref_site=None``).
    n_iter_ : int
        Largest number of iterations of the parametric empirical Bayes solver
        over the sites (0 if the solver is not used).
    n_features_in_ : int
        Number of columns of ``X`` seen during ``fit`` (metadata columns
        included).
    feature_names_in_ : ndarray of str
        Names of the columns seen during ``fit`` (metadata columns included).
        Only defined if ``X`` was a DataFrame with string column names.

    Examples
    --------
    Sites and covariates as columns of a DataFrame (works in a scikit-learn
    :class:`~sklearn.pipeline.Pipeline` and with cross-validation):

    >>> combat = NeuroComBat(site_column="site", continuous_columns=["age"])
    >>> X_harmonized = combat.fit_transform(df)  # doctest: +SKIP

    Sites and covariates as keywords:

    >>> combat = NeuroComBat()
    >>> X_harmonized = combat.fit_transform(X, sites=sites, continuous_covariates=age)  # doctest: +SKIP

    Notes
    -----
    This implementation follows the ComBat model of [1]_ and differs from
    R ``sva::ComBat`` (3.50.0) and ``neuroCombat`` (0.2.12) in three documented
    places. They were found with simulations from the ComBat model and are
    tested against frozen neuroCombat output (identical to R ``sva::ComBat``)
    in ``tests/combat/test_reference_conformance.py``:

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

    * **Stopping rule of the parametric empirical Bayes solver.** The solver
      stops when the largest relative change of :math:`\gamma^*` and
      :math:`\delta^*` is below 1e-4, with the change divided by
      :math:`|\text{previous estimate}|`. The references divide by the signed
      estimate (``max(abs(g.new - g.old) / g.old, ...)``), so features with a
      negative location effect never block the stop and the output is not
      converged to the stated tolerance (differences up to about 1e-5 of the
      data scale). Run to convergence, both give the same solution.

    **Reference site** (``ref_site``). The target of the harmonization is the
    grand mean :math:`\hat\alpha = \sum_{i \in R} (n_i / n_R) \hat\beta_i`
    of the reference sites :math:`R` (all sites by default) and their pooled
    residual variance
    :math:`\hat\sigma^2 = \sum_{i \in R}\sum_j (y_{ij} - \hat y_{ij})^2 / (n_R - |R|)`.
    The covariate effects and the empirical Bayes priors still use all sites.
    With a single reference site, :math:`\hat\sigma^2` is its own sample
    variance, its estimated site effect is exactly
    :math:`\hat\gamma = 0, \hat\delta = 1` and is not shrunk, so its data are
    unchanged. With :math:`n_R` instead of :math:`n_R - 1` (R ``sva::ComBat``,
    ``neuroCombat``) the other sites would get the reference variance shrunk
    by :math:`(n_R - 1)/n_R`.

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
        site_column: str | int | None = None,
        categorical_columns: str | int | list | None = None,
        continuous_columns: str | int | list | None = None,
        ref_site: str | int | list | None = None,
        empirical_bayes: bool = True,
        parametric_adjustments: bool = True,
        mean_only: bool = False,
        var_epsilon: float = 1e-8,
        delta_epsilon: float = 1e-8,
        tau_2_epsilon: float = 1e-10,
        max_iter: int = 1000,
    ) -> None:
        self.site_column = site_column
        self.categorical_columns = categorical_columns
        self.continuous_columns = continuous_columns
        self.ref_site = ref_site
        self.empirical_bayes = empirical_bayes
        self.parametric_adjustments = parametric_adjustments
        self.mean_only = mean_only
        self.var_epsilon = var_epsilon
        self.delta_epsilon = delta_epsilon
        self.tau_2_epsilon = tau_2_epsilon
        self.max_iter = max_iter

    # Data keywords are requested by default when metadata routing is enabled
    __metadata_request__fit = {"sites": True, "categorical_covariates": True, "continuous_covariates": True}  # noqa: RUF012
    __metadata_request__transform = {"sites": True, "categorical_covariates": True, "continuous_covariates": True}  # noqa: RUF012

    def fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike | None = None,
        *,
        sites: npt.ArrayLike | None = None,
        categorical_covariates: npt.ArrayLike | None = None,
        continuous_covariates: npt.ArrayLike | None = None,
    ) -> "NeuroComBat":
        """Compute per-feature statistics to perform harmonization.

        Parameters
        ----------
        X : array-like or DataFrame, shape (n_samples, n_columns)
            The training input samples, with the site and covariate columns
            named in the constructor, if any.
        y : None
            Ignored. Present for scikit-learn API consistency.
        sites : array-like, shape (n_samples,) or None, optional (default None)
            Sites. Required unless ``site_column`` is set.
        categorical_covariates : array-like, shape (n_samples, n_categorical_covariates) or None, optional (default None)
            The categorical covariates to be preserved during harmonization
            (e.g., sex, disease). Alternative to ``categorical_columns``.
        continuous_covariates : array-like, shape (n_samples, n_continuous_covariates) or None, optional (default None)
            The continuous covariates to be preserved during harmonization
            (e.g., age, clinical scores). Alternative to ``continuous_columns``.

        Returns
        -------
        self : NeuroComBat
            The fitted transformer.

        """
        logger.debug("Fitting")

        # Validate input
        data = self._validate_combat_input(
            X,
            reset=True,
            sites=sites,
            categorical_covariates=categorical_covariates,
            continuous_covariates=continuous_covariates,
        )
        X, sites = data.X, data.sites
        categorical_covariates = data["categorical_covariates"]
        continuous_covariates = data["continuous_covariates"]

        if categorical_covariates is not None or continuous_covariates is not None:
            logger.warning(
                "You specified categorical and/or continuous covariates to be preserved. "
                "If you intend to build a machine learning (ML) model, "
                "then make sure that you DO *NOT* preserve the ML model's target as covariate. "
                "You will be required to provide the covariate also at transform time, and this will produce data leakage. "
                "If you are performing a statistical analysis and want to preserve a variable of interest, "
                "then it is correct to specify it as covariate."
            )

        # Transpose to conform to neuroCombat and original ComBat
        X = X.T

        self.sites_, n_samples_per_site = np.unique(sites, return_counts=True)
        self._set_reference_sites()
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
            epsilon=self.var_epsilon,
        )

        self.fit_ls_model(
            data=standardized_data,
            design=design,
            idx_per_site=idx_per_site,
            epsilon=self.var_epsilon,
            delta_epsilon=self.delta_epsilon,
            tau_2_epsilon=self.tau_2_epsilon,
            max_iter=self.max_iter,
        )

        return self

    def transform(
        self,
        X: npt.ArrayLike,
        *,
        sites: npt.ArrayLike | None = None,
        categorical_covariates: npt.ArrayLike | None = None,
        continuous_covariates: npt.ArrayLike | None = None,
    ) -> npt.NDArray:
        """Harmonize data.

        Parameters
        ----------
        X : array-like or DataFrame, shape (n_samples, n_columns)
            The data to be harmonized, with the same columns as in :meth:`fit`.
        sites : array-like, shape (n_samples,) or None, optional (default None)
            Sites, all seen in :meth:`fit`. Required unless ``site_column`` is set.
        categorical_covariates : array-like, shape (n_samples, n_categorical_covariates) or None, optional (default None)
            The categorical covariates, if given to :meth:`fit` as keyword.
        continuous_covariates : array-like, shape (n_samples, n_continuous_covariates) or None, optional (default None)
            The continuous covariates, if given to :meth:`fit` as keyword.

        Returns
        -------
        array, shape (n_samples, n_features)
            The harmonized features (without the metadata columns).

        Raises
        ------
        ValueError
            If a site was not seen during :meth:`fit`.

        """
        logger.debug("Transforming")

        # Validate input
        check_is_fitted(self)
        data = self._validate_combat_input(
            X,
            reset=False,
            sites=sites,
            categorical_covariates=categorical_covariates,
            continuous_covariates=continuous_covariates,
        )
        X, sites = data.X, data.sites

        # Transpose to conform to neuroCombat and original ComBat
        X = X.T

        n_samples = sites.shape[0]
        idx_per_site = [list(np.where(sites == s)[0].tolist()) for s in self.sites_]

        design = self.transform_design_matrix(
            sites=sites,
            categorical_covariates=data["categorical_covariates"],
            continuous_covariates=data["continuous_covariates"],
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
            epsilon=self.delta_epsilon,
        )

        return self._keep_reference_samples(bayes_data.T, data.X, sites)
