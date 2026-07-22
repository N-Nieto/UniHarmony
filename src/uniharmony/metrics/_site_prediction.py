"""Site-prediction diagnostics for harmonization evaluation."""

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import numpy.typing as npt
import structlog
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import BaseCrossValidator, cross_validate
from sklearn.utils.validation import check_consistent_length, check_X_y


__all__ = [
    "evaluate_site_prediction",
]

logger = structlog.get_logger()


def evaluate_site_prediction(
    X: npt.NDArray,
    sites: npt.NDArray,
    model: BaseEstimator | None = None,
    cv: BaseCrossValidator | int | Iterable | None = None,
    metrics: str | Callable | list[str | Callable] | None = None,
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

    Internally, this is a validated wrapper around
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
        ``LogisticRegression`` natively dispatches between a binary and a
        multinomial fit depending on the number of unique values in
        ``sites``, so no manual binary/multiclass selection is required.
        If a fitted or unfitted classifier is provided, it is cloned
        before use, so the caller's estimator instance is never mutated.

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
        scorer name string, a scorer callable, or a list of either). Raw
        metric functions from ``sklearn.metrics`` (e.g.
        ``balanced_accuracy_score``) are not valid scorers on their own;
        wrap them with :func:`sklearn.metrics.make_scorer` first. If
        None, defaults to ``["balanced_accuracy", "roc_auc"]`` when
        ``sites`` has 2 unique values (binary site membership), or
        ``["balanced_accuracy", "roc_auc_ovr"]`` when it has more than 2
        (multi-class site membership).

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
        If ``X`` and ``sites`` have mismatched lengths, if ``sites``
        contains fewer than 2 unique values, if ``metrics`` resolves to
        an empty list, or if scoring fails on any fold (e.g. an AUC
        scorer that does not support multiclass targets is requested for
        a multi-site problem). Scoring failures are raised immediately
        rather than silently reported as ``NaN``, since silent failures
        would defeat the purpose of a diagnostic function.
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

    resolved_model = _resolve_model(model, n_sites, random_state)
    scoring = _resolve_metrics(metrics, n_sites)

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
        error_score="raise",
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
        logger.info(f"{metric_name}: {np.mean(scores):.4f} +- {np.std(scores):.4f}")

    if return_estimator:
        results["estimators"] = cv_results["estimator"]

    return results


def _resolve_model(
    model: BaseEstimator | None,
    n_sites: int,
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
        The resolved, unfitted classifier. A clone of ``model`` is
        returned when ``model`` is provided, so that the caller's
        estimator instance is never fitted or otherwise mutated in
        place.

    Raises
    ------
    TypeError
        If ``model`` does not follow the sklearn classifier API.

    """
    if model is None:
        problem_type = "binary" if n_sites == 2 else "multinomial"
        logger.info(f"No model provided, using LogisticRegression ({problem_type}) as default.")
        return LogisticRegression(max_iter=1000, random_state=random_state)

    if not (hasattr(model, "fit") and hasattr(model, "predict")) or not is_classifier(model):
        raise TypeError(f"model must be a sklearn-compatible classifier, got {type(model)}")

    return clone(model)


def _resolve_metrics(
    metrics: str | Callable | list[str | Callable] | None,
    n_sites: int,
) -> dict[str, str | Callable]:
    """Resolve the metrics used to score site prediction into a scoring dict.

    Parameters
    ----------
    metrics : str, callable, list of str/callable, or None
        User-provided scoring specification. If None, defaults to
        balanced accuracy plus an AUC variant appropriate for the number
        of sites (``"roc_auc"`` for 2 sites, ``"roc_auc_ovr"`` for more).
    n_sites : int
        Number of unique sites. Used to choose between ``"roc_auc"``
        (binary) and ``"roc_auc_ovr"`` (multi-class) when ``metrics`` is
        None.

    Returns
    -------
    dict[str, str or callable]
        Mapping from metric name to scorer, ready to pass as ``scoring``
        to ``cross_validate``. Using a dict (rather than a list) keeps the
        output keys stable regardless of whether a metric was passed as a
        scorer name string or as a callable.

    Raises
    ------
    ValueError
        If ``metrics`` resolves to an empty list.

    """
    if metrics is None:
        auc_metric = "roc_auc" if n_sites == 2 else "roc_auc_ovr"
        metrics = ["balanced_accuracy", auc_metric]
    elif not isinstance(metrics, list):
        metrics = [metrics]

    if len(metrics) == 0:
        raise ValueError("metrics must contain at least one scorer name or callable, got an empty list")

    scoring: dict[str, str | Callable] = {}
    for metric in metrics:
        scoring[_metric_name(metric)] = metric
    return scoring


def _metric_name(metric: str | Callable) -> str:
    """Derive a human-readable key for a metric or scorer.

    Parameters
    ----------
    metric : str or callable
        A scorer name, a metric function (with a ``__name__``), or a
        scorer object produced by :func:`sklearn.metrics.make_scorer`
        (which does not expose ``__name__`` directly, but wraps the
        original metric function in a private ``_score_func``
        attribute).

    Returns
    -------
    str
        ``metric`` itself if it is already a string; the wrapped metric
        function's name for a ``make_scorer`` object; the callable's
        ``__name__`` otherwise; or ``str(metric)`` as a last resort.

    """
    if isinstance(metric, str):
        return metric
    if hasattr(metric, "__name__"):
        return metric.__name__
    score_func = getattr(metric, "_score_func", None)
    if score_func is not None and hasattr(score_func, "__name__"):
        return score_func.__name__
    return str(metric)
