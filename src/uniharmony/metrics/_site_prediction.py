"""Site-prediction diagnostics for harmonization evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import structlog
from sklearn.base import BaseEstimator, ClassifierMixin, is_classifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import BaseCrossValidator, cross_validate
from sklearn.utils.validation import check_consistent_length, check_X_y


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


__all__ = [
    "evaluate_site_prediction",
]

logger = structlog.get_logger()


def evaluate_site_prediction(
    X: npt.NDArray,
    sites: npt.NDArray,
    model: ClassifierMixin | None = None,
    cv: BaseCrossValidator | int | Iterable | None = None,
    metrics: str | Callable | list[str | Callable] | None = "balanced_accuracy",
    n_jobs: int | None = None,
    return_estimator: bool = False,
    random_state: int | np.random.RandomState = 42,
) -> dict[str, Any]:
    """Assess how well site membership can be predicted from features.

    A standard diagnostic in multi-site harmonization is to check whether
    the acquisition site can still be predicted from the feature matrix
    ``X``. High predictability of ``sites`` indicates that a residual
    Effect of Site (EoS) is present in the data — either because it was
    never removed (raw data) or because a harmonization method failed to
    remove it fully (harmonized data). This function fits ``model`` to
    predict ``sites`` from ``X`` under cross-validation and reports the
    requested performance ``metrics`` on the held-out folds.

    Internally, this is a thin, validated wrapper around
    :func:`sklearn.model_selection.cross_validate`, so any scikit-learn
    classifier, cross-validation splitter, or scorer is supported.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix.

    sites : np.ndarray of shape (n_samples,)
        Site membership label for each sample. Can be numeric or string,
        and must contain at least 2 unique values.

    model : sklearn classifier, optional (default None)
        Classifier used to predict ``sites`` from ``X``. Any estimator
        following the sklearn classifier API (i.e. exposing ``fit`` and
        ``predict``) can be used. If None, a
        :class:`sklearn.linear_model.LogisticRegression` is used.

    cv : int, cross-validation generator or iterable, optional (default None)
        Cross-validation splitting strategy, passed directly to
        ``cross_validate``. Follows scikit-learn's default resolution: if
        None, 5-fold cross-validation is used, with
        :class:`~sklearn.model_selection.StratifiedKFold` selected
        automatically because ``model`` is a classifier and ``sites`` is
        categorical. See :func:`sklearn.model_selection.cross_validate`
        for the full resolution rules.

    metrics : str, callable, or list of str/callable, optional (default None)
        Metric(s) used to score the held-out predictions, in any format
        accepted by the ``scoring`` argument of ``cross_validate`` (a
        scorer name string, a scorer callable, or a list of either). If
        None, defaults to ``["balanced_accuracy"]`` .

    n_jobs : int, optional (default None)
        Number of jobs to run in parallel, passed to ``cross_validate``.

    return_estimator : bool, optional (default False)
        If True, the fitted estimator for each fold is included in the
        output under the ``"estimators"`` key.

    random_state : int or RandomState instance, optional (default 42)
        Controls the randomness of the default model
        (``LogisticRegression``). Ignored if ``model`` is provided.

    Returns
    -------
    dict
        Dictionary with one entry per requested metric (keyed by the
        scorer name, e.g. ``"balanced_accuracy"``). Each entry is itself a
        dictionary with keys:

        - ``"scores"`` : np.ndarray of shape (n_splits,), per-fold test scores.
        - ``"mean"`` : float, mean score across folds.
        - ``"std"`` : float, standard deviation of the score across folds.

        Also includes ``"fit_time"`` and ``"score_time"`` (arrays of shape
        (n_splits,)), and ``"estimators"`` (list of fitted models) if
        ``return_estimator`` is True.

    Raises
    ------
    ValueError
        If ``X`` and ``sites`` have mismatched lengths, or if ``sites``
        contains fewer than 2 unique values.
    TypeError
        If ``model`` is provided but does not follow the sklearn
        classifier API.

    Notes
    -----
    A site-prediction performance close to chance level (e.g.
    ``balanced_accuracy`` close to ``1 / n_sites``) suggests that
    site-specific information is not (or is no longer) recoverable from
    the features. Conversely, performance well above chance indicates
    that site information leaks through the features, e.g. because
    harmonization did not fully remove the EoS.

    Examples
    --------
    >>> from uniharmony.datasets import make_multisite_classification
    >>> X, y, sites = make_multisite_classification(n_sites=3, n_samples=300)
    >>> results = evaluate_site_prediction(X, sites)
    >>> sorted(results.keys())
    ['balanced_accuracy', 'fit_time', 'roc_auc_ovr', 'score_time']

    Using a custom model, cv scheme and a single metric:

    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.model_selection import StratifiedKFold
    >>> results = evaluate_site_prediction(
    ...     X,
    ...     sites,
    ...     model=RandomForestClassifier(random_state=0),
    ...     cv=StratifiedKFold(n_splits=3),
    ...     metrics="balanced_accuracy",
    ... )
    >>> round(results["balanced_accuracy"]["mean"], 2) >= 0.0
    True

    """
    X, sites = check_X_y(X, sites)
    check_consistent_length(X, sites)

    n_sites = len(np.unique(sites))
    if n_sites < 2:
        raise ValueError(f"sites must contain at least 2 unique values, got {n_sites}")

    logger.info(f"Evaluating site prediction across {n_sites} sites and {X.shape[0]} samples")

    resolved_model = _resolve_model(model, random_state)
    scoring = _resolve_metrics(metrics)

    logger.debug(f"Model: {resolved_model}")
    logger.debug(f"Scoring: {list(scoring.keys())}")
    logger.debug(f"CV scheme: {cv if cv is not None else 'sklearn default'}")

    cv_results = cross_validate(
        estimator=resolved_model,
        X=X,
        y=sites,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        return_estimator=return_estimator,
    )

    results: dict[str, Any] = {
        "fit_time": cv_results["fit_time"],
        "score_time": cv_results["score_time"],
    }
    for metric_name in scoring:
        scores = cv_results[f"test_{metric_name}"]
        results[metric_name] = {
            "scores": scores,
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
        }
        logger.debug(f"{metric_name}: {np.mean(scores):.4f} +- {np.std(scores):.4f}")

    if return_estimator:
        results["estimators"] = cv_results["estimator"]

    return results


def _resolve_model(
    model: ClassifierMixin | BaseEstimator | None,
    random_state: int | np.random.RandomState,
) -> BaseEstimator:
    """Resolve the classifier used to predict site membership.

    Parameters
    ----------
    model : sklearn classifier or None
        User-provided classifier. If None, a default
        ``LogisticRegression`` is built.
    n_sites : int
        Number of unique sites. Only used to log whether the default
        model will fit a binary or a multinomial problem; scikit-learn's
        ``LogisticRegression`` handles this distinction internally.
    random_state : int or RandomState instance
        Random state for the default classifier.

    Returns
    -------
    sklearn classifier
        The resolved, unfitted classifier (a clone if ``model`` was
        provided, to avoid mutating the caller's estimator).

    Raises
    ------
    TypeError
        If ``model`` does not follow the sklearn classifier API.

    """
    if model is None:
        logger.debug("No model provided, using LogisticRegression as default.")
        model = LogisticRegression(max_iter=1000, random_state=random_state)

    elif not (hasattr(model, "fit") and hasattr(model, "predict")) or not is_classifier(model):
        raise TypeError(f"model must be a sklearn-compatible classifier, got {type(model)}")

    return model


def _resolve_metrics(
    metrics: str | Callable | list[str | Callable] | None,
) -> dict[str, str | Callable]:
    """Resolve the metrics used to score site prediction into a scoring dict.

    Parameters
    ----------
    metrics : str, callable, list of str/callable, or None
        User-provided scoring specification.

    Returns
    -------
    dict[str, str or callable]
        Mapping from metric name to scorer, ready to pass as ``scoring``
        to ``cross_validate``. Using a dict (rather than a list) keeps the
        output keys stable regardless of whether a metric was passed as a
        scorer name string or as a callable.

    """
    if metrics is None:
        metrics = ["balanced_accuracy"]
    elif not isinstance(metrics, list):
        metrics = [metrics]

    scoring: dict[str, str | Callable] = {}
    for metric in metrics:
        name = metric if isinstance(metric, str) else getattr(metric, "__name__", str(metric))
        scoring[name] = metric
    return scoring
