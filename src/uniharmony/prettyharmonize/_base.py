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
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_consistent_length, check_is_fitted

from uniharmony._data import HarmonizationInputMixin, check_sites_seen, extract_metadata
from uniharmony._utils import minimum_samples_warning, validate_sites
from uniharmony._verbose import get_logger
from uniharmony.combat import NeuroComBat
from uniharmony.combat._available_models import ComBatModels, CovariateArguments, get_covariate_arguments

from ._prettyharmonize import TargetHarmonizer, _as_columns, target_argument


__all__ = ["BasePrettYharmonize"]

logger = get_logger(src="PrettYharmonize")

Split = tuple[npt.NDArray, npt.NDArray]

#: Data keywords, requested by default when metadata routing is enabled
_METADATA_REQUEST = {"sites": True, "categorical_covariates": True, "continuous_covariates": True, "smooth_covariates": True}


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
        harm_model: ComBatModels | None = None,
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
    def _resolve_models(self) -> tuple[ComBatModels, Any, Any]:
        """Return unfitted copies of the harmonization, base and stack models."""
        harm_model = NeuroComBat() if self.harm_model is None else cast("ComBatModels", clone(self.harm_model))
        get_covariate_arguments(harm_model)  # raises TypeError if not a ComBat-based model
        # The columns of X named in harm_model are read here; the internal copies get the data as keywords
        self._harm_specs = type(harm_model)._metadata_specs
        self._harm_columns = harm_model._metadata_columns()
        harm_model.set_params(**{spec.column_param: None for spec in self._harm_specs})
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
        self, X: npt.ArrayLike, metadata: Mapping[str, Any], reset: bool
    ) -> tuple[npt.NDArray, npt.NDArray, dict[str, npt.NDArray]]:
        """Validate ``X``, the sites and the covariates.

        The sites and covariates are read from the columns of ``X`` named in
        ``harm_model`` (``site_column``, ``*_columns``) or from the keywords
        ``metadata``. At fit, the names of the covariates are stored in
        ``_covariate_names`` so that the same covariates are required to predict.
        """
        accepted = {spec.keyword for spec in self._harm_specs}
        unknown = sorted(name for name, value in metadata.items() if value is not None and name not in accepted)
        if unknown:
            raise ValueError(
                f"{type(self._harm_model).__name__} does not accept the covariate argument(s) {unknown}; "
                f"accepted: {list(self._covariate_arguments.names)}."
            )
        values = {name: value for name, value in metadata.items() if name in accepted}
        data = extract_metadata(self, X, self._harm_specs, self._harm_columns, values, reset=reset)
        if data.sites is None:
            raise ValueError(
                "sites must be provided (one label per sample): set site_column in harm_model to read them from a "
                "DataFrame X, or pass sites=... ."
            )
        covariates = {name: value for name, value in data.metadata.items() if name != "sites" and value is not None}
        if reset:
            self._metadata_positions = data.metadata_positions
            self._covariate_names = sorted(covariates)
        else:
            if sorted(covariates) != self._covariate_names:
                raise ValueError(
                    f"The model was fitted with additional covariates {self._covariate_names}; got {sorted(covariates)}."
                )
            check_sites_seen(data.sites, self.sites_, type(self).__name__)
        return data.X, data.sites, covariates

    def _check_required_covariates(self, covariates: dict[str, npt.NDArray]) -> None:
        missing = [
            name for name in self._covariate_arguments.required if name not in covariates and name != self._target_argument
        ]
        if missing:
            raise ValueError(
                f"{type(self._harm_model).__name__} requires the covariate argument(s) {missing} "
                "(e.g., age as smooth covariate): pass it as keyword or name its columns in harm_model."
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
        if values.shape[1] != 1 or getattr(self._harm_model, arguments.smooth_bounds, None) is not None:
            return {}  # bounds are only supported for a single smooth covariate; user bounds are kept
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

    def _new_harmonizer(self, harm_model: ComBatModels) -> TargetHarmonizer:
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
    def _fit(self, X, y, metadata) -> npt.NDArray:
        # Forget the attributes of a previous fit with other settings.
        for name in ("harm_model_", "base_model_", "fold_harm_models_", "fold_base_models_", "harmonized_data_"):
            self.__dict__.pop(name, None)
        harm_model, base_model, stack_model = self._resolve_models()
        self._harm_model = harm_model
        self._covariate_arguments: CovariateArguments = get_covariate_arguments(harm_model)
        X, sites, covariates = self._check_inputs(X, metadata, reset=True)
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

    # Data keywords are requested by default when metadata routing is enabled
    __metadata_request__fit = _METADATA_REQUEST
    __metadata_request__predict = _METADATA_REQUEST
    __metadata_request__transform = _METADATA_REQUEST
    __metadata_request__score = _METADATA_REQUEST

    def fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        *,
        sites: npt.ArrayLike | None = None,
        categorical_covariates: npt.ArrayLike | None = None,
        continuous_covariates: npt.ArrayLike | None = None,
        smooth_covariates: npt.ArrayLike | None = None,
    ) -> "BasePrettYharmonize":
        """Fit the harmonization, base and stack models.

        The sites and covariates are read from the columns of ``X`` named in
        ``harm_model`` (e.g., ``NeuroComBat(site_column="site")``), or passed
        as keywords. Use the columns to use the estimator in a scikit-learn
        :class:`~sklearn.pipeline.Pipeline` or with cross-validation.

        Parameters
        ----------
        X : array-like or DataFrame, shape (n_samples, n_columns)
            The training input samples, with the metadata columns named in
            ``harm_model``, if any.
        y : array-like, shape (n_samples,)
            The training target. It is always preserved by ``harm_model``.
        sites : array-like, shape (n_samples,) or None, optional (default None)
            Sites. Required unless ``harm_model`` has a ``site_column``.
        categorical_covariates, continuous_covariates, smooth_covariates : array-like or None, optional (default None)
            Other covariates preserved by ``harm_model``, if it accepts them
            (e.g., ``smooth_covariates`` for :class:`~uniharmony.combat.ComBatGAM`).
            Each has one row per sample. They must be known for the data to
            predict too.

        Returns
        -------
        object
            The fitted estimator.

        """
        self._fit(X, y, _metadata(sites, categorical_covariates, continuous_covariates, smooth_covariates))
        return self

    def fit_transform(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        *,
        sites: npt.ArrayLike | None = None,
        categorical_covariates: npt.ArrayLike | None = None,
        continuous_covariates: npt.ArrayLike | None = None,
        smooth_covariates: npt.ArrayLike | None = None,
    ) -> npt.NDArray:
        """Fit the model and return the out-of-fold harmonized training data.

        Each training sample is harmonized with its **true** target (legitimate
        for training data) by the harmonization model of the inner fold where
        it was a validation sample. This differs from :meth:`transform`, which
        harmonizes new data with their *predicted* target.

        Parameters
        ----------
        X, y, sites, categorical_covariates, continuous_covariates, smooth_covariates
            See :meth:`fit`.

        Returns
        -------
        array, shape (n_samples, n_features)
            The out-of-fold harmonized training data.

        """
        return self._fit(X, y, _metadata(sites, categorical_covariates, continuous_covariates, smooth_covariates))

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def _scores(
        self, X: npt.ArrayLike, metadata: Mapping[str, Any]
    ) -> tuple[npt.NDArray, npt.NDArray, dict[str, npt.NDArray], npt.NDArray]:
        """Validate the inputs and return them with their score matrix (averaged over the kept models)."""
        check_is_fitted(self, "stack_model_")
        X, sites, covariates = self._check_inputs(X, metadata, reset=False)
        matrices = [
            self._get_predictions(harmonizer, model, X, sites, covariates)
            for harmonizer, model in zip(self._harmonizers, self._base_models, strict=True)
        ]
        return X, sites, covariates, np.mean(matrices, axis=0)

    def predict(
        self,
        X: npt.ArrayLike,
        *,
        sites: npt.ArrayLike | None = None,
        categorical_covariates: npt.ArrayLike | None = None,
        continuous_covariates: npt.ArrayLike | None = None,
        smooth_covariates: npt.ArrayLike | None = None,
    ) -> npt.NDArray:
        """Predict the target without using it for harmonization.

        Parameters
        ----------
        X : array-like or DataFrame, shape (n_samples, n_columns)
            The input samples, with the same columns as in :meth:`fit`.
        sites : array-like, shape (n_samples,) or None, optional (default None)
            Sites (must have been seen during fit), if given as keyword to :meth:`fit`.
        categorical_covariates, continuous_covariates, smooth_covariates : array-like or None, optional (default None)
            The same covariates as in :meth:`fit`, for these samples, if given
            as keywords to :meth:`fit`.

        Returns
        -------
        array, shape (n_samples,)
            The predictions.

        """
        *_, scores = self._scores(X, _metadata(sites, categorical_covariates, continuous_covariates, smooth_covariates))
        return self._decode(self.stack_model_.predict(scores))

    def transform(
        self,
        X: npt.ArrayLike,
        *,
        sites: npt.ArrayLike | None = None,
        categorical_covariates: npt.ArrayLike | None = None,
        continuous_covariates: npt.ArrayLike | None = None,
        smooth_covariates: npt.ArrayLike | None = None,
    ) -> npt.NDArray:
        """Harmonize new data with their *predicted* target.

        Parameters
        ----------
        X, sites, categorical_covariates, continuous_covariates, smooth_covariates
            See :meth:`predict`.

        Returns
        -------
        array, shape (n_samples, n_features)
            The harmonized data (averaged over the inner-fold harmonization
            models if ``fit_final_model=False``).

        """
        metadata = _metadata(sites, categorical_covariates, continuous_covariates, smooth_covariates)
        X, sites, covariates, scores = self._scores(X, metadata)
        y_pred = self.stack_model_.predict(scores)
        return np.mean([harmonizer.transform(X, y_pred, sites, covariates) for harmonizer in self._harmonizers], axis=0)

    def get_feature_names_out(self, input_features: npt.ArrayLike | None = None) -> npt.NDArray:
        """Get the names of the harmonized features (the metadata columns are not part of the output).

        Parameters
        ----------
        input_features : array-like of str or None, optional (default None)
            Names of the input columns. If None, ``feature_names_in_`` is used.

        Returns
        -------
        ndarray of str
            Names of the output features.

        """
        return HarmonizationInputMixin.get_feature_names_out(self, input_features)  # type: ignore[arg-type]


def _metadata(sites: Any, categorical_covariates: Any, continuous_covariates: Any, smooth_covariates: Any) -> dict[str, Any]:
    """Collect the data keywords."""
    return {
        "sites": sites,
        "categorical_covariates": categorical_covariates,
        "continuous_covariates": continuous_covariates,
        "smooth_covariates": smooth_covariates,
    }
