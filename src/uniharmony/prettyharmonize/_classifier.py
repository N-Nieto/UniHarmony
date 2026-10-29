"""Provide PrettYharmonizeClassifier."""

# Adapted from:
# https://github.com/juaml/PrettYharmonize
# licensed under AGPL-3.0.

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import type_of_target

from ._base import BasePrettYharmonize, DecisionPredictor, ProbabilisticPredictor


__all__ = ["PrettYharmonizeClassifier"]


class PrettYharmonizeClassifier(ClassifierMixin, BasePrettYharmonize):
    """Leakage-free ComBat harmonization for classification (PrettYharmonize).

    When the class balance differs across sites, ComBat must preserve the
    class as a covariate, which requires the label of the data being
    harmonized. PrettYharmonize [1]_ avoids this leakage by harmonizing every
    sample once per class, *pretending* it belongs to that class. A base model
    scores each pretended version and a stack model, trained on out-of-fold
    scores, predicts the class. The label of a predicted sample is never used.

    Parameters
    ----------
    harm_model : NeuroComBat, CovBat, ComBatGAM or None, optional (default None)
        Unfitted ComBat-based harmonization model with its parameters set,
        e.g., ``NeuroComBat(empirical_bayes=False)``. None means
        :class:`~uniharmony.combat.NeuroComBat`. The class is preserved as a
        categorical covariate (one-hot encoded continuous covariates for
        :class:`~uniharmony.combat.ComBatGAM`, which then needs a smooth
        covariate in ``additional_covariates``).
    base_model : estimator or None, optional (default None)
        Classifier fitted on the harmonized data. It must implement
        ``predict_proba`` or ``decision_function``. None means
        ``Pipeline([("scaler", StandardScaler()), ("logit", LogisticRegression())])``:
        ComBat returns data in the original feature units, while the L2
        penalty of the logistic regression assumes comparable feature scales.
    stack_model : estimator or None, optional (default None)
        Classifier mapping the score matrix to the class. None means
        ``LogisticRegression()``.
    cv : int, cross-validation splitter or iterable, optional (default 5)
        Inner cross-validation. An int is the number of folds of a shuffled
        :class:`~sklearn.model_selection.StratifiedKFold` stratified by
        site x class. A splitter is called as ``cv.split(X, y, groups=sites)``
        and an iterable yields ``(train, validation)`` index pairs. Each
        training split must keep at least 2 samples of the sites it is
        validated on and every class.
    fit_final_model : bool, optional (default True)
        If True, refit ``harm_model`` on all the training data (to harmonize
        new data) and ``base_model`` on the out-of-fold harmonized training
        data. If False, keep the models of the inner folds and average their
        score matrices to predict (no refit; ``cv`` times slower prediction).
    return_harmonized_data : bool, optional (default False)
        If True, store the out-of-fold harmonized training data in
        :attr:`harmonized_data_` (:meth:`fit_transform` always returns them).
    n_jobs : int or None, optional (default None)
        Number of inner folds fitted in parallel.
    random_state : int or None, optional (default None)
        Seed of the inner fold shuffling (used when ``cv`` is an int).

    Attributes
    ----------
    classes_ : ndarray, shape (n_classes,)
        Class labels.
    sites_ : ndarray, shape (n_sites,)
        Sites seen during fit.
    n_features_in_ : int
        Number of features seen during fit.
    score_matrix_ : ndarray, shape (n_samples, n_scores)
        Out-of-fold score matrix on which the stack model is trained: per
        pretended class, P(``classes_[1]``) for binary problems or all class
        probabilities for multi-class problems.
    stack_model_ : estimator
        Stack model fitted on :attr:`score_matrix_`.
    harm_model_ : ComBat-based model
        Harmonization model fitted on all the training data (true classes
        preserved). Only if ``fit_final_model=True``.
    base_model_ : estimator
        Base model fitted on the out-of-fold harmonized training data. Only if
        ``fit_final_model=True``.
    fold_harm_models_, fold_base_models_ : list
        Harmonization and base models of the inner folds. Only if
        ``fit_final_model=False``.
    harmonized_data_ : ndarray, shape (n_samples, n_features)
        Out-of-fold harmonized training data. Only if ``return_harmonized_data=True``.

    References
    ----------
    .. [1] Nieto, N., Eickhoff, S. B., Jung, C., Reuter, M., Diers, K.,
           Kelm, M., Lichtenberg, A., Raimondo, F., & Patil, K. R. (2026).
           Impact of leakage on data harmonization in machine learning
           pipelines in class imbalance across sites. Neurocomputing, 133146.
           https://doi.org/10.1016/j.neucom.2026.133146

    Examples
    --------
    >>> from uniharmony.combat import NeuroComBat
    >>> from uniharmony.datasets import make_multisite_classification
    >>> from uniharmony.prettyharmonize import PrettYharmonizeClassifier
    >>> X, y, sites = make_multisite_classification(
    ...     n_samples=400, balance_per_site=[[0.8, 0.2], [0.2, 0.8]], random_state=0
    ... )
    >>> clf = PrettYharmonizeClassifier(harm_model=NeuroComBat(), random_state=0)
    >>> clf = clf.fit(X, y, sites)
    >>> clf.predict(X[:5], sites[:5]).shape
    (5,)

    """

    _categorical_target = True

    def _default_base_model(self) -> BaseEstimator:
        return Pipeline([("scaler", StandardScaler()), ("logit", LogisticRegression())])

    def _default_stack_model(self) -> BaseEstimator:
        return LogisticRegression()

    def _prepare_target(self, y: npt.NDArray) -> npt.NDArray:
        target_type = type_of_target(y, input_name="y")
        if target_type not in ("binary", "multiclass"):
            raise ValueError(f"Unknown label type for classification: {target_type}. Use PrettYharmonizeRegressor instead.")
        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError("y must contain at least 2 classes.")
        return y_encoded

    def _strata(self, y: npt.NDArray, sites: npt.NDArray) -> npt.NDArray:
        # One stratum per (site, class) pair, so every fold keeps the class balance of every site.
        _, strata = np.unique(np.column_stack([sites.astype(str), y.astype(str)]), axis=0, return_inverse=True)
        return strata.ravel()

    def _check_fold(self, i_fold: int, y_train: npt.NDArray) -> None:
        missing = np.setdiff1d(np.arange(len(self.classes_)), y_train)
        if missing.size:
            raise ValueError(
                f"Inner fold {i_fold}: classes {self.classes_[missing].tolist()} are missing from the training split. "
                "Use fewer inner folds (cv)."
            )

    def _pretend_targets(self) -> list:
        return list(range(len(self.classes_)))

    def _model_scores(self, model: Any, X: npt.NDArray) -> npt.NDArray:
        if isinstance(model, ProbabilisticPredictor):
            proba = model.predict_proba(X)
            # Binary: P(class 1) suffices, as P(class 0) = 1 - P(class 1).
            return proba[:, [1]] if proba.shape[1] == 2 else proba
        if isinstance(model, DecisionPredictor):
            return np.asarray(model.decision_function(X)).reshape(X.shape[0], -1)
        raise TypeError(f"base_model must implement predict_proba or decision_function; {type(model).__name__} does not.")

    def _decode(self, y_encoded: npt.NDArray) -> npt.NDArray:
        return self.classes_[np.asarray(y_encoded, dtype=int)]

    def predict_proba(
        self,
        X: npt.ArrayLike,
        sites: npt.ArrayLike,
        additional_covariates: Mapping[str, npt.ArrayLike] | None = None,
    ) -> npt.NDArray:
        """Predict class probabilities without using the labels for harmonization.

        Parameters
        ----------
        X, sites, additional_covariates
            See :meth:`predict`.

        Returns
        -------
        array, shape (n_samples, n_classes)
            Class probabilities of the stack model (columns ordered as ``classes_``).

        """
        *_, scores = self._scores(X, sites, additional_covariates)
        if not isinstance(self.stack_model_, ProbabilisticPredictor):
            raise AttributeError(f"The stack model ({type(self.stack_model_).__name__}) does not implement predict_proba.")
        return self.stack_model_.predict_proba(scores)

    # ``sites`` comes before ``sample_weight`` (as in ``fit``), unlike the scikit-learn signature.
    def score(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        sites: npt.ArrayLike | None = None,
        sample_weight: npt.ArrayLike | None = None,
        additional_covariates: Mapping[str, npt.ArrayLike] | None = None,
    ) -> float:
        """Return the balanced accuracy.

        Balanced accuracy (the mean recall over classes) is not inflated by
        predicting the majority class, unlike accuracy. This method is used by
        scikit-learn model selection tools (e.g., ``cross_validate`` or
        ``GridSearchCV``) when no ``scoring`` is given; with metadata routing
        it is also the only scorer that receives ``sites``.

        Parameters
        ----------
        X, sites, additional_covariates
            See :meth:`predict`.
        y : array-like, shape (n_samples,)
            True labels.
        sample_weight : array-like, shape (n_samples,) or None, optional (default None)
            Sample weights.

        Returns
        -------
        float
            Balanced accuracy.

        """
        y_pred = self.predict(X, sites, additional_covariates)  # type: ignore[arg-type]
        return float(balanced_accuracy_score(y, y_pred, sample_weight=sample_weight))
