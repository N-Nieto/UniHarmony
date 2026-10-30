"""Provide the base class of the PrettYharmonize estimators."""

# Adapted from:
# https://github.com/juaml/PrettYharmonize
# licensed under AGPL-3.0.

import warnings
from collections.abc import Iterable, Mapping
from numbers import Integral
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np
import numpy.typing as npt
import structlog
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import FLOAT_DTYPES, check_array, check_consistent_length, check_is_fitted

from uniharmony._utils import minimum_samples_warning, validate_sites
from uniharmony.combat import NeuroComBat
from uniharmony.combat._available_models import ComBatModel, CovariateArguments, get_covariate_arguments

from ._prettyharmonize import TargetHarmonizer, _as_columns, target_argument


__all__ = ["BasePrettYharmonize"]

logger = structlog.get_logger(src="PrettYharmonize")

Split = tuple[npt.NDArray, npt.NDArray]


# Structural types, used to check the models and to let type checkers narrow
# ``BaseEstimator`` to models exposing the needed methods.
@runtime_checkable
class Predictor(Protocol):
    """Model with ``fit`` and ``predict``."""

    def fit(self, X: Any, y: Any) -> Any: ...

    def predict(self, X: Any) -> Any: ...


@runtime_checkable
class ProbabilisticPredictor(Protocol):
    """Model with ``predict_proba``."""

    def predict_proba(self, X: Any) -> npt.NDArray: ...


@runtime_checkable
class DecisionPredictor(Protocol):
    """Model with ``decision_function``."""

    def decision_function(self, X: Any) -> npt.NDArray: ...


def _check_samples(X: npt.ArrayLike, sites: npt.ArrayLike | None, estimator: BaseEstimator) -> tuple[npt.NDArray, npt.NDArray]:
    """Validate ``X`` (2-D, float) and ``sites`` (1-D, one per sample)."""
    if sites is None:
        raise ValueError("sites must be provided (one label per sample).")
    # scikit-learn annotates ``dtype`` as str only.
    X_checked = np.asarray(check_array(X, dtype=FLOAT_DTYPES, estimator=estimator))  # pyright: ignore[reportArgumentType]
    sites_checked = np.asarray(check_array(sites, dtype=None, ensure_2d=False, estimator=estimator))  # pyright: ignore[reportArgumentType]
    if sites_checked.ndim == 2 and sites_checked.shape[1] == 1:
        sites_checked = sites_checked.ravel()
    check_consistent_length(X_checked, sites_checked)
    return X_checked, sites_checked


def _unfitted(model: Any) -> Any:
    """Return an unfitted copy of ``model``."""
    return clone(model)


def _subset(covariates: dict[str, npt.NDArray], index: npt.NDArray) -> dict[str, npt.NDArray]:
    """Return the rows ``index`` of every covariate."""
    return {name: value[index] for name, value in covariates.items()}


class BasePrettYharmonize(BaseEstimator):
    """Base class of the PrettYharmonize estimators.

    Do not use this class directly; use
    :class:`~uniharmony.prettyharmonize.PrettYharmonizeClassifier` or
    :class:`~uniharmony.prettyharmonize.PrettYharmonizeRegressor`.

    PrettYharmonize [1]_ wraps a ComBat-based harmonization model
    (``harm_model``). The target is preserved by ``harm_model`` on training
    data, whose target is known, and *pretended* on the data to predict,
    whose target is unknown. Fitting has two levels:

    1. **Inner cross-validation.** In each inner fold, ``harm_model`` is fitted
       on the inner-training data with their target among its covariates and
       ``base_model`` is fitted on the harmonized inner-training data. Every
       validation sample is harmonized once per *pretended* target and scored
       by ``base_model``: these scores form its row of the *score matrix*.
       The validation samples are also harmonized with their true target,
       which gives the *out-of-fold harmonized* training data.
    2. **Stacking.** ``stack_model`` learns to predict the target from the
       score matrix.

    With ``fit_final_model=True``, ``harm_model`` is then refitted on all the
    training data (to harmonize new data) and ``base_model`` on the
    out-of-fold harmonized training data. Otherwise, the models of the inner
    folds are kept and their score matrices averaged at prediction.

    To predict, each sample is harmonized under every pretended target,
    scored by ``base_model`` and ``stack_model`` maps the scores to the
    prediction. The target of the predicted samples is never used.

    References
    ----------
    .. [1] Nieto, N., Eickhoff, S. B., Jung, C., Reuter, M., Diers, K.,
           Kelm, M., Lichtenberg, A., Raimondo, F., & Patil, K. R. (2026).
           Impact of leakage on data harmonization in machine learning
           pipelines in class imbalance across sites. Neurocomputing, 133146.
           https://doi.org/10.1016/j.neucom.2026.133146

    """

    _categorical_target: bool = False
    classes_: npt.NDArray  # set by the classifier

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
    ) -> None:
        self.harm_model = harm_model
        self.base_model = base_model
        self.stack_model = stack_model
        self.cv = cv
        self.fit_final_model = fit_final_model
        self.return_harmonized_data = return_harmonized_data
        self.n_jobs = n_jobs
        self.random_state = random_state

    # ------------------------------------------------------------------ #
    # Hooks implemented by the subclasses
    # ------------------------------------------------------------------ #
    def _default_base_model(self) -> BaseEstimator:
        raise NotImplementedError

    def _default_stack_model(self) -> BaseEstimator:
        raise NotImplementedError

    def _prepare_target(self, y: npt.NDArray) -> npt.NDArray:
        """Set the target-related fitted attributes and return the encoded target."""
        raise NotImplementedError

    def _pretend_targets(self) -> list:
        """Return the pretended (encoded) targets."""
        raise NotImplementedError

    def _model_scores(self, model: Any, X: npt.NDArray) -> npt.NDArray:
        """Return the output of ``base_model``, shape (n_samples, n_scores)."""
        raise NotImplementedError

    def _strata(self, y: npt.NDArray, sites: npt.NDArray) -> npt.NDArray:
        """Return the labels stratifying the default inner folds."""
        return sites

    def _check_fold(self, i_fold: int, y_train: npt.NDArray) -> None:
        """Extra checks on an inner training split."""

    def _decode(self, y_encoded: npt.NDArray) -> npt.NDArray:
        return y_encoded

    @property
    def _n_classes(self) -> int | None:
        return len(self.classes_) if self._categorical_target else None

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def _resolve_models(self) -> tuple[ComBatModel, Any, Any]:
        """Return unfitted copies of the harmonization, base and stack models."""
        harm_model = NeuroComBat() if self.harm_model is None else cast("ComBatModel", clone(self.harm_model))
        get_covariate_arguments(harm_model)  # raises TypeError if not a ComBat-based model
        models = []
        for name, model, default in (
            ("base_model", self.base_model, self._default_base_model),
            ("stack_model", self.stack_model, self._default_stack_model),
        ):
            if model is None:
                models.append(default())
            elif isinstance(model, Predictor):
                models.append(_unfitted(model))
            else:
                raise TypeError(f"{name} must be a scikit-learn estimator or None, got {type(model).__name__}.")
        return harm_model, models[0], models[1]

    def _check_inputs(
        self, X: npt.ArrayLike, sites: npt.ArrayLike | None, additional_covariates: Mapping[str, Any] | None, reset: bool
    ) -> tuple[npt.NDArray, npt.NDArray, dict[str, npt.NDArray]]:
        """Validate ``X``, ``sites`` and ``additional_covariates``."""
        X_checked, sites_checked = _check_samples(X, sites, self)
        covariates = self._check_covariates(additional_covariates, X_checked.shape[0], reset)
        if reset:
            self.n_features_in_ = X_checked.shape[1]
            return X_checked, sites_checked, covariates
        if X_checked.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_checked.shape[1]} features, but {type(self).__name__} was fitted with {self.n_features_in_}."
            )
        unseen = np.setdiff1d(sites_checked, self.sites_)
        if unseen.size:
            raise ValueError(f"Sites unseen during fit cannot be harmonized: {unseen.tolist()}.")
        return X_checked, sites_checked, covariates

    def _check_covariates(
        self, additional_covariates: Mapping[str, Any] | None, n_samples: int, reset: bool
    ) -> dict[str, npt.NDArray]:
        """Validate the additional covariates against the arguments of ``harm_model``.

        At fit, their names are stored in ``_covariate_names`` so that the same
        covariates are required to predict.
        """
        if additional_covariates is None:
            additional_covariates = {}
        if not isinstance(additional_covariates, Mapping):
            raise TypeError(
                "additional_covariates must be a dict mapping covariate arguments of harm_model to arrays, "
                f"e.g. {{'continuous_covariates': age}}; got {type(additional_covariates).__name__}."
            )
        # Covariates are passed as 2-D columns: some models (e.g., CovBat) do not accept 1-D arrays.
        covariates = {name: _as_columns(value) for name, value in additional_covariates.items() if value is not None}
        if reset:
            accepted = self._covariate_arguments.names
            unknown = sorted(set(covariates) - set(accepted))
            if unknown:
                raise ValueError(
                    f"{type(self._harm_model).__name__} does not accept the covariate argument(s) {unknown}; "
                    f"accepted: {list(accepted)}."
                )
            self._covariate_names = sorted(covariates)
        elif sorted(covariates) != self._covariate_names:
            raise ValueError(
                f"The model was fitted with additional covariates {self._covariate_names}; got {sorted(covariates)}."
            )
        for name, value in covariates.items():
            if value.ndim != 2 or value.shape[0] != n_samples:
                raise ValueError(
                    f"additional_covariates['{name}'] must have one row per sample ({n_samples}), got shape {value.shape}."
                )
        return covariates

    def _check_required_covariates(self, covariates: dict[str, npt.NDArray]) -> None:
        missing = [
            name for name in self._covariate_arguments.required if name not in covariates and name != self._target_argument
        ]
        if missing:
            raise ValueError(
                f"{type(self._harm_model).__name__} requires the covariate argument(s) {missing} "
                "in additional_covariates (e.g., age as smooth covariate)."
            )

    def _check_target_identifiable(self, y_encoded: npt.NDArray, sites: npt.NDArray) -> None:
        """Raise if the target is a linear combination of the site indicators."""
        site_columns = OneHotEncoder(sparse_output=False).fit_transform(sites.reshape(-1, 1))
        if self._categorical_target:
            target_columns = OneHotEncoder(sparse_output=False, drop="first").fit_transform(y_encoded.reshape(-1, 1))
        else:
            target_columns = y_encoded.reshape(-1, 1)
        design = np.hstack([site_columns, target_columns])
        if np.linalg.matrix_rank(design) < design.shape[1]:
            raise ValueError(
                "The target is confounded with the sites (e.g., constant within every site), so its effect "
                "cannot be separated from the site effect by the harmonization model."
            )

    def _harm_fit_options(self, y_encoded: npt.NDArray, covariates: dict[str, npt.NDArray]) -> dict[str, Any]:
        """Return the options of ``harm_model.fit`` needed inside the inner folds.

        The splines of a smooth covariate cannot extrapolate, so their range is
        set to the one of the whole training data: every inner fold can then
        harmonize its validation samples (and every pretended target).
        """
        arguments = self._covariate_arguments
        if arguments.smooth is None or arguments.smooth_bounds is None:
            return {}
        values = y_encoded if self._target_argument == arguments.smooth else covariates[arguments.smooth]
        values = _as_columns(values).astype(float)
        if values.shape[1] != 1:
            return {}  # bounds are only supported for a single smooth covariate
        return {arguments.smooth_bounds: (float(values.min()), float(values.max()))}

    # ------------------------------------------------------------------ #
    # Inner cross-validation
    # ------------------------------------------------------------------ #
    def _make_splits(self, X: npt.NDArray, y: npt.NDArray, sites: npt.NDArray) -> list[Split]:
        """Return the inner (train, validation) splits."""
        cv: Any = self.cv
        if isinstance(cv, Integral):
            if cv < 2:
                raise ValueError(f"cv must be >= 2 folds, got {cv}.")
            splitter = StratifiedKFold(n_splits=int(cv), shuffle=True, random_state=self.random_state)
            splits = list(splitter.split(X, self._strata(y, sites)))
        elif hasattr(cv, "split"):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="The groups parameter is ignored")
                splits = list(cv.split(X, y, groups=sites))
        elif isinstance(cv, Iterable):
            splits = list(cv)
        else:
            raise TypeError("cv must be an int, a splitter with a split method or an iterable of (train, test) indices.")
        self._check_splits(splits, y, sites)
        return splits

    def _check_splits(self, splits: list[Split], y: npt.NDArray, sites: npt.NDArray) -> None:
        if len(splits) == 0:
            raise ValueError("The inner cross-validation produced no splits.")
        covered = np.zeros(sites.shape[0], dtype=bool)
        for i_fold, (train, test) in enumerate(splits):
            covered[test] = True
            train_sites, counts = np.unique(sites[train], return_counts=True)
            count = dict(zip(train_sites.tolist(), counts.tolist(), strict=True))
            small = [s for s in np.unique(sites[test]).tolist() if count.get(s, 0) < 2]
            if small:
                raise ValueError(
                    f"Inner fold {i_fold}: site(s) {small} have fewer than 2 samples in the training split, "
                    "so the harmonization model cannot be fitted for them. Use fewer inner folds (cv) "
                    "or remove/merge very small sites."
                )
            self._check_fold(i_fold, y[train])
        if not covered.all():
            raise ValueError(
                f"The inner cross-validation must place every training sample in a validation fold; {np.sum(~covered)} never are."
            )

    def _new_harmonizer(self, harm_model: ComBatModel) -> TargetHarmonizer:
        categorical_argument = self._target_argument == self._covariate_arguments.categorical
        return TargetHarmonizer(harm_model, self._target_argument, self._n_classes, categorical_argument)

    def _get_predictions(
        self, harmonizer: TargetHarmonizer, model: Any, X: npt.NDArray, sites: npt.NDArray, covariates: dict[str, npt.NDArray]
    ) -> npt.NDArray:
        """Return the score matrix of one (harmonizer, base model) pair, shape (n_samples, n_scores)."""
        versions = harmonizer.iter_pretend(X, sites, self._pretend_targets(), covariates)
        return np.hstack([self._model_scores(model, X_pretended) for X_pretended in versions])

    def _fit_fold(self, harm_model, base_model, X, y, sites, covariates, options, train, test):
        """Fit the models of one inner fold and evaluate them on its validation samples."""
        cov_train, cov_test = _subset(covariates, train), _subset(covariates, test)
        harmonizer = self._new_harmonizer(harm_model)
        harmonizer.fit(X[train], y[train], sites[train], cov_train, options)
        model = _unfitted(base_model).fit(harmonizer.transform(X[train], y[train], sites[train], cov_train), y[train])
        scores = self._get_predictions(harmonizer, model, X[test], sites[test], cov_test)
        # Validation samples are training data, so their true target can be
        # preserved: this harmonizes them with a model that never saw them.
        X_test_harmonized = harmonizer.transform(X[test], y[test], sites[test], cov_test)
        return scores, X_test_harmonized, harmonizer, model

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #
    def _fit(self, X, y, sites, additional_covariates) -> npt.NDArray:
        # Forget the attributes of a previous fit with other settings.
        for name in ("harm_model_", "base_model_", "fold_harm_models_", "fold_base_models_", "harmonized_data_"):
            self.__dict__.pop(name, None)
        harm_model, base_model, stack_model = self._resolve_models()
        self._harm_model = harm_model
        self._covariate_arguments: CovariateArguments = get_covariate_arguments(harm_model)
        X, sites, covariates = self._check_inputs(X, sites, additional_covariates, reset=True)
        y = self._check_y(X, y)
        validate_sites(sites)
        self.sites_, n_samples_per_site = np.unique(sites, return_counts=True)
        minimum_samples_warning(n_samples_per_site)
        y_encoded = self._prepare_target(y)
        self._check_target_identifiable(y_encoded, sites)
        self._target_argument = target_argument(self._covariate_arguments, self._categorical_target, covariates)
        self._check_required_covariates(covariates)
        options = self._harm_fit_options(y_encoded, covariates)

        splits = self._make_splits(X, y_encoded, sites)
        logger.info(
            f"Fitting PrettYharmonize ({type(harm_model).__name__}) on {X.shape[0]} samples with {len(splits)} inner folds"
        )
        folds: list[tuple[npt.NDArray, npt.NDArray, TargetHarmonizer, Any]] = Parallel(n_jobs=self.n_jobs)(  # pyright: ignore[reportAssignmentType]
            delayed(self._fit_fold)(harm_model, base_model, X, y_encoded, sites, covariates, options, train, test)
            for train, test in splits
        )

        # Assemble the out-of-fold score matrix and harmonized data.
        self.score_matrix_ = np.full((X.shape[0], folds[0][0].shape[1]), np.nan)
        X_harmonized = np.full(X.shape, np.nan)
        for (_, test), (scores, X_test_harmonized, _, _) in zip(splits, folds, strict=True):
            self.score_matrix_[test] = scores
            X_harmonized[test] = X_test_harmonized
        logger.debug("Fitting the stack model on the score matrix")
        self.stack_model_ = _unfitted(stack_model).fit(self.score_matrix_, y_encoded)

        if self.fit_final_model:
            logger.debug("Fitting the final harmonization model (all training data) and base model (out-of-fold data)")
            harmonizer = self._new_harmonizer(harm_model)
            harmonizer.fit(X, y_encoded, sites, covariates, options)
            self.harm_model_ = harmonizer.harm_model
            self.base_model_ = _unfitted(base_model).fit(X_harmonized, y_encoded)
            self._harmonizers, self._base_models = [harmonizer], [self.base_model_]
        else:
            self._harmonizers = [harmonizer for *_, harmonizer, _ in folds]
            self._base_models = [model for *_, model in folds]
            self.fold_harm_models_ = [harmonizer.harm_model for harmonizer in self._harmonizers]
            self.fold_base_models_ = self._base_models
        if self.return_harmonized_data:
            self.harmonized_data_ = X_harmonized
        return X_harmonized

    @staticmethod
    def _check_y(X: npt.NDArray, y: npt.ArrayLike | None) -> npt.NDArray:
        if y is None:
            raise ValueError("y must be provided.")
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        if y.ndim != 1:
            raise ValueError(f"y must be 1-dimensional, got shape {y.shape}.")
        check_consistent_length(X, y)
        return y

    def fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        sites: npt.ArrayLike,
        additional_covariates: Mapping[str, npt.ArrayLike] | None = None,
    ) -> "BasePrettYharmonize":
        """Fit the harmonization, base and stack models.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The training target. It is always preserved by ``harm_model``.
        sites : array-like, shape (n_samples,)
            Sites.
        additional_covariates : dict or None, optional (default None)
            Other covariates preserved by ``harm_model``, keyed by its
            covariate argument, e.g., ``{"categorical_covariates": sex,
            "continuous_covariates": age}`` or ``{"smooth_covariates": age}``
            (:class:`~uniharmony.combat.ComBatGAM`). Each value has one row per
            sample. They must be known for the data to predict too.

        Returns
        -------
        object
            The fitted estimator.

        """
        self._fit(X, y, sites, additional_covariates)
        return self

    def fit_transform(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        sites: npt.ArrayLike,
        additional_covariates: Mapping[str, npt.ArrayLike] | None = None,
    ) -> npt.NDArray:
        """Fit the model and return the out-of-fold harmonized training data.

        Each training sample is harmonized with its **true** target (legitimate
        for training data) by the harmonization model of the inner fold where
        it was a validation sample. This differs from :meth:`transform`, which
        harmonizes new data with their *predicted* target.

        Parameters
        ----------
        X, y, sites, additional_covariates
            See :meth:`fit`.

        Returns
        -------
        array, shape (n_samples, n_features)
            The out-of-fold harmonized training data.

        """
        return self._fit(X, y, sites, additional_covariates)

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def _scores(
        self, X: npt.ArrayLike, sites: npt.ArrayLike | None, additional_covariates: Mapping[str, Any] | None
    ) -> tuple[npt.NDArray, npt.NDArray, dict[str, npt.NDArray], npt.NDArray]:
        """Validate the inputs and return them with their score matrix (averaged over the kept models)."""
        check_is_fitted(self, "stack_model_")
        X, sites, covariates = self._check_inputs(X, sites, additional_covariates, reset=False)
        matrices = [
            self._get_predictions(harmonizer, model, X, sites, covariates)
            for harmonizer, model in zip(self._harmonizers, self._base_models, strict=True)
        ]
        return X, sites, covariates, np.mean(matrices, axis=0)

    def predict(
        self,
        X: npt.ArrayLike,
        sites: npt.ArrayLike,
        additional_covariates: Mapping[str, npt.ArrayLike] | None = None,
    ) -> npt.NDArray:
        """Predict the target without using it for harmonization.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        sites : array-like, shape (n_samples,)
            Sites (must have been seen during fit).
        additional_covariates : dict or None, optional (default None)
            The same additional covariates as in :meth:`fit`, for these samples.

        Returns
        -------
        array, shape (n_samples,)
            The predictions.

        """
        *_, scores = self._scores(X, sites, additional_covariates)
        return self._decode(self.stack_model_.predict(scores))

    def transform(
        self,
        X: npt.ArrayLike,
        sites: npt.ArrayLike,
        additional_covariates: Mapping[str, npt.ArrayLike] | None = None,
    ) -> npt.NDArray:
        """Harmonize new data with their *predicted* target.

        Parameters
        ----------
        X, sites, additional_covariates
            See :meth:`predict`.

        Returns
        -------
        array, shape (n_samples, n_features)
            The harmonized data (averaged over the inner-fold harmonization
            models if ``fit_final_model=False``).

        """
        X, sites, covariates, scores = self._scores(X, sites, additional_covariates)
        y_pred = self.stack_model_.predict(scores)
        return np.mean([harmonizer.transform(X, y_pred, sites, covariates) for harmonizer in self._harmonizers], axis=0)
