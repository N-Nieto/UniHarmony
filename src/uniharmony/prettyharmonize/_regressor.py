"""Provide PrettYharmonizeRegressor."""

# Adapted from:
# https://github.com/juaml/PrettYharmonize
# licensed under AGPL-3.0.

from collections.abc import Mapping
from numbers import Integral
from typing import Any

import numpy as np
import numpy.typing as npt
import structlog
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from uniharmony.combat._available_models import ComBatModel

from ._base import BasePrettYharmonize, _subset
from ._prettyharmonize import TargetHarmonizer


__all__ = ["PrettYharmonizeRegressor"]

logger = structlog.get_logger(src="PrettYharmonize")

_ALPHAS = tuple(np.logspace(-3, 3, 13))


class PrettYharmonizeRegressor(RegressorMixin, BasePrettYharmonize):
    """Leakage-free ComBat harmonization for regression (PrettYharmonize).

    Every sample is harmonized once per *regression point* (pretended target
    values spread over the training range). A base model scores each
    pretended version and a stack model, trained on out-of-fold scores,
    predicts the target [1]_. The target of a predicted sample is never used.

    Parameters
    ----------
    harm_model : NeuroComBat, CovBat, ComBatGAM or None, optional (default None)
        Unfitted ComBat-based harmonization model with its parameters set.
        None means :class:`~uniharmony.combat.NeuroComBat`. The target is
        preserved as a continuous covariate (linear effect), or as smooth
        covariate (non-linear effect) with :class:`~uniharmony.combat.ComBatGAM`
        unless ``additional_covariates`` already has a smooth covariate.
    base_model : estimator or None, optional (default None)
        Regressor fitted on the harmonized data. None means
        ``Pipeline([("scaler", StandardScaler()), ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 13)))])``:
        ComBat returns data in the original feature units, while the ridge
        penalty assumes comparable feature scales.
    stack_model : estimator or None, optional (default None)
        Regressor mapping the score matrix to the target. None means
        ``RidgeCV(alphas=np.logspace(-3, 3, 13))``.
    cv : int, cross-validation splitter or iterable, optional (default 5)
        Inner cross-validation. An int is the number of folds of a shuffled
        :class:`~sklearn.model_selection.StratifiedKFold` stratified by site
        (see :class:`~uniharmony.prettyharmonize.PrettYharmonizeClassifier`).
    fit_final_model : bool, optional (default True)
        If True, refit ``harm_model`` on all the training data and
        ``base_model`` on the out-of-fold harmonized training data. If False,
        average the score matrices of the inner-fold models to predict.
    return_harmonized_data : bool, optional (default False)
        If True, store the out-of-fold harmonized training data in
        :attr:`harmonized_data_`.
    n_jobs : int or None, optional (default None)
        Number of inner folds fitted in parallel.
    random_state : int or None, optional (default None)
        Seed of the inner fold shuffling (used when ``cv`` is an int).
    regression_points : int, array-like or None, optional (default None)
        Pretended targets. None means 10 points and an int ``k`` means ``k``
        points, evenly spaced between the minimum and maximum training target.
    regression_search : bool, optional (default False)
        Instead of fixed points, search for each sample a self-consistent
        target ``t`` (``base_model(harmonize(x, t)) ≈ t``) by interval halving
        between the minimum and maximum training target (vectorized over
        samples). The score matrix then has a single column.
    regression_search_tol : float, optional (default 0.0)
        Stop the search for a sample when ``|t - prediction|`` is below this value.
    regression_search_max_iter : int, optional (default 20)
        Maximum number of interval halvings.

    Attributes
    ----------
    regression_points_ : ndarray
        Pretended targets used.
    y_min_, y_max_ : float
        Range of the training target.
    sites_, n_features_in_, score_matrix_, stack_model_, harm_model_, base_model_, fold_harm_models_, fold_base_models_, \
harmonized_data_
        See :class:`~uniharmony.prettyharmonize.PrettYharmonizeClassifier`.

    References
    ----------
    .. [1] Nieto, N., Eickhoff, S. B., Jung, C., Reuter, M., Diers, K.,
           Kelm, M., Lichtenberg, A., Raimondo, F., & Patil, K. R. (2026).
           Impact of leakage on data harmonization in machine learning
           pipelines in class imbalance across sites. Neurocomputing, 133146.
           https://doi.org/10.1016/j.neucom.2026.133146

    """

    def __init__(
        self,
        harm_model: ComBatModel | None = None,
        base_model: BaseEstimator | None = None,
        stack_model: BaseEstimator | None = None,
        cv: int | Any = 5,
        fit_final_model: bool = True,
        return_harmonized_data: bool = False,
        n_jobs: int | None = None,
        random_state: int | None = None,
        regression_points: int | npt.ArrayLike | None = None,
        regression_search: bool = False,
        regression_search_tol: float = 0.0,
        regression_search_max_iter: int = 20,
    ) -> None:
        super().__init__(
            harm_model=harm_model,
            base_model=base_model,
            stack_model=stack_model,
            cv=cv,
            fit_final_model=fit_final_model,
            return_harmonized_data=return_harmonized_data,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self.regression_points = regression_points
        self.regression_search = regression_search
        self.regression_search_tol = regression_search_tol
        self.regression_search_max_iter = regression_search_max_iter

    def _default_base_model(self) -> BaseEstimator:
        return Pipeline([("scaler", StandardScaler()), ("ridge", RidgeCV(alphas=_ALPHAS))])

    def _default_stack_model(self) -> BaseEstimator:
        return RidgeCV(alphas=_ALPHAS)

    def _prepare_target(self, y: npt.NDArray) -> npt.NDArray:
        try:
            y = np.asarray(y, dtype=float)
        except (TypeError, ValueError) as e:
            raise ValueError("y must be numeric for regression.") from e
        if not np.all(np.isfinite(y)):
            raise ValueError("y contains NaN or infinite values.")
        self.y_min_, self.y_max_ = float(y.min()), float(y.max())
        if self.y_min_ == self.y_max_:
            raise ValueError("y is constant.")
        if self.regression_search:
            self._check_search_params()
        self.regression_points_ = self._resolve_regression_points()
        return y

    def _check_search_params(self) -> None:
        if not isinstance(self.regression_search_max_iter, Integral) or self.regression_search_max_iter < 1:
            raise ValueError("regression_search_max_iter must be an integer >= 1.")
        if self.regression_search_tol < 0:
            raise ValueError("regression_search_tol must be >= 0.")

    def _resolve_regression_points(self) -> npt.NDArray:
        points: Any = self.regression_points
        if points is None:
            return np.linspace(self.y_min_, self.y_max_, 10)
        if isinstance(points, Integral):
            if points < 2:
                raise ValueError("regression_points must be >= 2.")
            return np.linspace(self.y_min_, self.y_max_, int(points))
        if np.ndim(points) == 0:
            raise TypeError(f"regression_points must be None, an int (number of points) or an array of values; got {points!r}.")
        points = np.asarray(points, dtype=float).ravel()
        if points.size == 0:
            raise ValueError("regression_points is empty.")
        if points.min() < self.y_min_ or points.max() > self.y_max_:
            logger.warning(
                f"Some regression_points are outside the training target range [{self.y_min_:g}, {self.y_max_:g}]; "
                "they extrapolate the harmonization model."
            )
        return points

    def _pretend_targets(self) -> list:
        return list(self.regression_points_)

    def _model_scores(self, model: Any, X: npt.NDArray) -> npt.NDArray:
        return np.asarray(model.predict(X), dtype=float).reshape(X.shape[0], -1)

    def _get_predictions(
        self, harmonizer: TargetHarmonizer, model: Any, X: npt.NDArray, sites: npt.NDArray, covariates: dict[str, npt.NDArray]
    ) -> npt.NDArray:
        if not self.regression_search:
            return super()._get_predictions(harmonizer, model, X, sites, covariates)
        return self._search(harmonizer, model, X, sites, covariates).reshape(-1, 1)

    def _search(
        self, harmonizer: TargetHarmonizer, model: Any, X: npt.NDArray, sites: npt.NDArray, covariates: dict[str, npt.NDArray]
    ) -> npt.NDArray:
        """Vectorized interval-halving search of a self-consistent target.

        For each sample, start from ``[y_min, y_max]``; at each step harmonize
        with both end points, predict, and move the end point whose prediction
        is farther from itself to the middle of the interval. Return the
        prediction of the better end point of the last step.
        """
        n_samples = X.shape[0]
        low, high = np.full(n_samples, self.y_min_), np.full(n_samples, self.y_max_)
        pred_low, pred_high = np.zeros(n_samples), np.zeros(n_samples)
        dist_low, dist_high = np.full(n_samples, np.inf), np.full(n_samples, np.inf)
        active = np.ones(n_samples, dtype=bool)
        for _ in range(self.regression_search_max_iter):
            if not active.any():
                break
            idx = np.flatnonzero(active)
            X_low, X_high = harmonizer.iter_pretend(X[idx], sites[idx], [low[idx], high[idx]], _subset(covariates, idx))
            pred_low[idx] = self._model_scores(model, X_low)[:, 0]
            pred_high[idx] = self._model_scores(model, X_high)[:, 0]
            dist_low[idx] = np.abs(low[idx] - pred_low[idx])
            dist_high[idx] = np.abs(high[idx] - pred_high[idx])
            middle = (low[idx] + high[idx]) / 2
            low_better = dist_low[idx] < dist_high[idx]
            current = np.where(low_better, dist_low[idx], dist_high[idx])
            high[idx] = np.where(low_better, middle, high[idx])
            low[idx] = np.where(low_better, low[idx], middle)
            active[idx] = current > self.regression_search_tol
        return np.where(dist_low < dist_high, pred_low, pred_high)

    # ``sites`` comes before ``sample_weight`` (as in ``fit``), unlike the scikit-learn signature.
    def score(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        sites: npt.ArrayLike | None = None,
        sample_weight: npt.ArrayLike | None = None,
        additional_covariates: Mapping[str, npt.ArrayLike] | None = None,
    ) -> float:
        """Return the coefficient of determination R².

        This method is used by scikit-learn model selection tools when no
        ``scoring`` is given; with metadata routing it is also the only scorer
        that receives ``sites``.

        Parameters
        ----------
        X, sites, additional_covariates
            See :meth:`predict`.
        y : array-like, shape (n_samples,)
            True target.
        sample_weight : array-like, shape (n_samples,) or None, optional (default None)
            Sample weights.

        Returns
        -------
        float
            R².

        """
        y_pred = self.predict(X, sites, additional_covariates)  # type: ignore[arg-type]
        return float(r2_score(y, y_pred, sample_weight=sample_weight))
