"""Provide ComBatGAM transformer."""

# Adapted from:
# https://github.com/rpomponio/neuroHarmonize
# licensed under MIT license

from typing import Self

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.utils import Tags, check_random_state
from sklearn.utils.validation import (
    check_is_fitted,
)
from statsmodels.gam.api import BSplines

from uniharmony._data import TransformOutput
from uniharmony._verbose import get_logger

from ._base import CONTINUOUS, SITES, SMOOTH, BaseComBat
from ._design_matrix_mixin import _as_columns


__all__ = ["ComBatGAM"]

logger = get_logger(src="ComBatGAM")


class ComBatGAM(BaseComBat):
    r"""Harmonize multi-site scanner effects controlling for non-linear age effects.

    This is an improvement on NeuroComBat allowing for non-linear effects to be controlled by Generalized Additive Models (GAMs).

    Parameters
    ----------
    site_column : str or int or None, optional (default None)
        Column of ``X`` holding the site of every sample. If None, the sites
        are passed with the ``sites`` keyword. See
        :class:`~uniharmony.combat.NeuroComBat`.
    smooth_columns : str or int or list or None, optional (default None)
        Columns of ``X`` holding the smooth covariates, whose non-linear effect
        is modelled with splines (e.g., age). Alternative to the
        ``smooth_covariates`` keyword; one of them is required.
    continuous_columns : str or int or list or None, optional (default None)
        Columns of ``X`` holding continuous covariates with a linear effect to
        preserve. Alternative to the ``continuous_covariates`` keyword.
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
    df : int, optional (default 10)
        Number of basis functions (degrees of freedom) of the B-splines.
        Default value used in the original implementation.
    degree : int, optional (default 3)
        Degree of the B-splines. Default value used in the original
        implementation.
    smooth_covariates_bounds : tuple of (float, float) or None, optional (default None)
        Range ``(minimum, maximum)`` covered by the splines, useful when new
        data cover a wider range than the training data. Only supported for a
        single smooth covariate. If None, the training range is used.
    var_epsilon : float, optional (default 1e-8)
        Small constant added to variances to avoid division by zero.
    delta_epsilon : float, optional (default 1e-8)
        Lower bound of the site scale factors, to avoid division by zero.
    tau_2_epsilon : float, optional (default 1e-10)
        Lower bound of the prior variance of the site location effects.
    max_iter : int, optional (default 1000)
        Maximum number of iterations of the parametric empirical Bayes solver.
    random_state : int, RandomState instance or None, optional (default None)
        Controls the shuffling of the k-fold cross-validation used to select
        the GAM penalty weights. Pass an int for reproducible results across
        multiple calls.

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

    Notes
    -----
    The ComBat steps follow Johnson et al. (2007) and differ from
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
    .. [1] Pomponio, R., Shou, H., Davatzikos, C., et al., (2019).
           "Harmonization of large MRI datasets for the analysis of brain imaging patterns throughout the lifespan."
           Neuroimage 208.
           https://doi.org/10.1016/j.neuroimage.2019.116450.

    """

    _metadata_specs = (SITES, SMOOTH, CONTINUOUS)

    def __init__(
        self,
        site_column: str | int | None = None,
        smooth_columns: str | int | list | None = None,
        continuous_columns: str | int | list | None = None,
        ref_site: str | int | list | None = None,
        empirical_bayes: bool = True,
        parametric_adjustments: bool = True,
        mean_only: bool = False,
        df: int = 10,
        degree: int = 3,
        smooth_covariates_bounds: tuple[float, float] | None = None,
        var_epsilon: float = 1e-8,
        delta_epsilon: float = 1e-8,
        tau_2_epsilon: float = 1e-10,
        max_iter: int = 1000,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.site_column = site_column
        self.smooth_columns = smooth_columns
        self.continuous_columns = continuous_columns
        self.ref_site = ref_site
        self.empirical_bayes = empirical_bayes
        self.parametric_adjustments = parametric_adjustments
        self.mean_only = mean_only
        self.df = df
        self.degree = degree
        self.smooth_covariates_bounds = smooth_covariates_bounds
        self.var_epsilon = var_epsilon
        self.delta_epsilon = delta_epsilon
        self.tau_2_epsilon = tau_2_epsilon
        self.max_iter = max_iter
        self.random_state = random_state

    # Data keywords are requested by default when metadata routing is enabled
    __metadata_request__fit = {"sites": True, "smooth_covariates": True, "continuous_covariates": True}  # noqa: RUF012
    __metadata_request__transform = {"sites": True, "smooth_covariates": True, "continuous_covariates": True}  # noqa: RUF012

    def fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike | None = None,
        *,
        sites: npt.ArrayLike | None = None,
        smooth_covariates: npt.ArrayLike | None = None,
        continuous_covariates: npt.ArrayLike | None = None,
    ) -> Self:
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
        smooth_covariates : array-like, shape (n_samples, n_smooth_covariates) or None, optional (default None)
            The smooth, non-linear covariates. GAMs are used for optimal
            smoothing (e.g., age). Required unless ``smooth_columns`` is set.
        continuous_covariates : array-like, shape (n_samples, n_continuous_covariates) or None, optional (default None)
            The continuous covariates to be preserved during harmonization
            (e.g., clinical scores). Alternative to ``continuous_columns``.

        Returns
        -------
        self : ComBatGAM
            The fitted transformer.

        """
        logger.debug("Fitting")

        # Validate input
        data = self._validate_combat_input(
            X,
            reset=True,
            required=("smooth_covariates",),
            sites=sites,
            smooth_covariates=smooth_covariates,
            continuous_covariates=continuous_covariates,
        )
        X, sites = data.X, data.sites
        smooth_covariates = data["smooth_covariates"]
        continuous_covariates = data["continuous_covariates"]
        smooth_covariates_bounds = self.smooth_covariates_bounds
        if smooth_covariates_bounds is None:
            smooth_covariates_bounds = (None, None)
        df, degree = self.df, self.degree

        logger.info(
            "If you intend to build a machine learning (ML) model, "
            "then make sure that you DO *NOT* preserve the ML model's target as covariate. "
            "You will be required to provide the covariate also at transform time, and this will produce data leakage. "
            "If you are performing a statistical analysis and want to preserve a variable of interest, "
            "then it is correct to specify it as covariate."
        )

        self._continuous_covariates_used = continuous_covariates is not None
        if self._continuous_covariates_used:
            logger.warning(
                "You specified continuous covariates to be preserved. "
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
            categorical_covariates=None,
            continuous_covariates=continuous_covariates,
        )
        # Setup design matrix for smoothing
        logger.debug("Setting up smoothing using B-Splines")
        # Create cubic spline basis for smooth covariates
        x_spline = _as_columns(smooth_covariates, "smooth_covariates").copy()
        smooth_covariates_cols = x_spline.shape[1]
        self._n_smooth_covariates = smooth_covariates_cols
        if smooth_covariates_cols > 1 and smooth_covariates_bounds != (None, None):
            raise ValueError("smooth_covariates_bounds is currently only supported for a single smooth covariate.")
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
            cont_covs = _as_columns(continuous_covariates, "continuous_covariates")
            for c in range(cont_covs.shape[1]):
                v = f"c{c!s}"
                formula += f"{v} + "
                df_gam[v] = cont_covs[:, c].astype(float)
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
            epsilon=self.var_epsilon,
            random_state=check_random_state(self.random_state),
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
        smooth_covariates: npt.ArrayLike | None = None,
        continuous_covariates: npt.ArrayLike | None = None,
    ) -> TransformOutput:
        """Harmonize data.

        Parameters
        ----------
        X : array-like or DataFrame, shape (n_samples, n_columns)
            The data to be harmonized, with the same columns as in :meth:`fit`.
        sites : array-like, shape (n_samples,) or None, optional (default None)
            Sites, all seen in :meth:`fit`. Required unless ``site_column`` is set.
        smooth_covariates : array-like, shape (n_samples, n_smooth_covariates) or None, optional (default None)
            The smooth covariates, if given to :meth:`fit` as keyword.
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
            smooth_covariates=smooth_covariates,
            continuous_covariates=continuous_covariates,
        )
        X, sites = data.X, data.sites
        smooth_covariates = data["smooth_covariates"]
        continuous_covariates = data["continuous_covariates"]

        # Transpose to conform to neuroCombat and original ComBat
        X = X.T

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
        x_spline = _as_columns(smooth_covariates, "smooth_covariates").copy()
        if x_spline.shape[1] != self._n_smooth_covariates:
            raise ValueError(
                f"smooth_covariates has {x_spline.shape[1]} columns, but {self._n_smooth_covariates} were seen during fit."
            )
        x_spline = self._clip_to_spline_range(x_spline)
        bs_basis = self._bsplines.transform(x_spline)
        # Construct dataframe required for GAM
        df_gam = {}
        # Set data from created design matrix
        for b in range(self._n_sites):
            v = f"x{b!s}"
            df_gam[v] = design[:, b]
        # Set data from continuous covariates
        if self._continuous_covariates_used:
            cont_covs = _as_columns(continuous_covariates, "continuous_covariates")
            for c in range(cont_covs.shape[1]):
                v = f"c{c!s}"
                df_gam[v] = cont_covs[:, c].astype(float)
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
            epsilon=self.delta_epsilon,
        )

        return self._keep_reference_samples(bayes_data.T, data.X, sites)

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        # The penalty weights are selected with shuffled k-fold CV, seeded by random_state
        tags.non_deterministic = self.random_state is None
        return tags

    def _clip_to_spline_range(self, x_spline: npt.NDArray) -> npt.NDArray:
        """Clip smooth covariates to the range covered by the fitted B-splines.

        B-splines cannot be evaluated outside their outermost knots (the
        training range, or ``smooth_covariates_bounds`` if given). Values
        outside that range are clipped to its boundary, so the smooth effect
        is extrapolated as a constant, and a warning is logged.

        Parameters
        ----------
        x_spline : array, shape (n_samples, n_smooth_covariates)
            Smooth covariates to transform.

        Returns
        -------
        array, shape (n_samples, n_smooth_covariates)
            The clipped smooth covariates.

        """
        x_spline = x_spline.astype(float, copy=True)
        for i, smoother in enumerate(self._bsplines.smoothers):
            lower, upper = smoother.knots.min(), smoother.knots.max()
            outside = (x_spline[:, i] < lower) | (x_spline[:, i] > upper)
            if np.any(outside):
                logger.warning(
                    f"{int(outside.sum())} sample(s) have smooth covariate {i} outside the fitted range "
                    f"[{lower:.4g}, {upper:.4g}]. They are clipped to the boundary (constant extrapolation). "
                    "Set smooth_covariates_bounds in the constructor to cover the expected range."
                )
                x_spline[:, i] = np.clip(x_spline[:, i], lower, upper)
        return x_spline
