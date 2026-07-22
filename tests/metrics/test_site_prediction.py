"""Tests for ``uniharmony.metrics.site_prediction.evaluate_site_prediction``.

Written as plain functions with ``pytest.fixture`` and
``pytest.mark.parametrize`` (no test classes), per project convention.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.model_selection import KFold, StratifiedKFold

from uniharmony.metrics._site_prediction import evaluate_site_prediction


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def binary_site_data() -> tuple[np.ndarray, np.ndarray]:
    """Synthetic 2-site problem with a learnable signal."""
    X, sites = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=6,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=0,
    )
    return X, sites


@pytest.fixture
def multiclass_site_data() -> tuple[np.ndarray, np.ndarray]:
    """Synthetic 4-site problem with a learnable signal."""
    X, sites = make_classification(
        n_samples=400,
        n_features=12,
        n_informative=8,
        n_classes=4,
        n_clusters_per_class=1,
        random_state=0,
    )
    return X, sites


@pytest.fixture
def string_labeled_site_data() -> tuple[np.ndarray, np.ndarray]:
    """Binary problem with non-numeric (string) site labels."""
    X, sites_int = make_classification(
        n_samples=150,
        n_features=8,
        n_classes=2,
        random_state=1,
    )
    site_names = np.array(["site_a", "site_b"])
    sites = site_names[sites_int]
    return X, sites


@pytest.fixture
def single_site_data() -> tuple[np.ndarray, np.ndarray]:
    """Degenerate case: only one unique site label."""
    X, _ = make_classification(n_samples=50, n_features=5, random_state=0)
    sites = np.zeros(50, dtype=int)
    return X, sites


# --------------------------------------------------------------------------- #
# Basic / default behaviour
# --------------------------------------------------------------------------- #


def test_default_call_returns_expected_top_level_keys(binary_site_data):
    """With no overrides, the result exposes fit/score time and the default metric."""
    X, sites = binary_site_data
    results = evaluate_site_prediction(X, sites)

    assert "fit_time" in results
    assert "score_time" in results
    assert "balanced_accuracy" in results


@pytest.mark.parametrize(
    "data_fixture,expected_auc_key",
    [
        ("binary_site_data", "roc_auc"),
        ("multiclass_site_data", "roc_auc_ovr"),
    ],
)
def test_default_metrics_are_balanced_accuracy_and_auc(data_fixture, expected_auc_key, request):
    """Default metrics are balanced accuracy plus an AUC variant chosen by n_sites.

    Binary site membership (2 unique sites) resolves to ``"roc_auc"``;
    multi-class site membership (>2 sites) resolves to ``"roc_auc_ovr"``,
    since plain ``"roc_auc"`` does not support multiclass targets.
    """
    X, sites = request.getfixturevalue(data_fixture)
    results = evaluate_site_prediction(X, sites)

    metric_keys = set(results.keys()) - {"fit_time", "score_time", "estimators"}
    assert metric_keys == {"balanced_accuracy", expected_auc_key}


def test_metrics_none_matches_implicit_default(binary_site_data):
    """Passing ``metrics=None`` explicitly resolves to the same output as omitting it."""
    X, sites = binary_site_data
    results_default = evaluate_site_prediction(X, sites)
    results_none = evaluate_site_prediction(X, sites, metrics=None)

    assert set(results_default.keys()) == set(results_none.keys())


@pytest.mark.parametrize("data_fixture", ["binary_site_data", "multiclass_site_data"])
def test_result_dict_has_scores_mean_std_per_metric(data_fixture, request):
    """Each metric entry exposes per-fold scores plus aggregated mean/std."""
    X, sites = request.getfixturevalue(data_fixture)
    results = evaluate_site_prediction(X, sites, metrics="balanced_accuracy")

    metric_result = results["balanced_accuracy"]
    assert set(metric_result.keys()) == {"scores", "mean", "std"}
    assert isinstance(metric_result["scores"], np.ndarray)
    assert metric_result["mean"] == pytest.approx(float(np.mean(metric_result["scores"])))
    assert metric_result["std"] == pytest.approx(float(np.std(metric_result["scores"])))


# --------------------------------------------------------------------------- #
# Metrics parametrization
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "metrics_arg,expected_keys",
    [
        ("balanced_accuracy", {"balanced_accuracy"}),
        (["balanced_accuracy", "accuracy"], {"balanced_accuracy", "accuracy"}),
        ("roc_auc", {"roc_auc"}),
    ],
)
def test_metrics_argument_shapes(binary_site_data, metrics_arg, expected_keys):
    """String and list-of-strings metric specs are supported."""
    X, sites = binary_site_data
    results = evaluate_site_prediction(X, sites, metrics=metrics_arg)

    metric_keys = set(results.keys()) - {"fit_time", "score_time", "estimators"}
    assert metric_keys == expected_keys


def test_raw_metric_function_is_rejected_by_cross_validate(binary_site_data):
    """A bare ``sklearn.metrics`` function (e.g. ``balanced_accuracy_score``).

    is NOT a valid scorer and must be wrapped with ``make_scorer`` first.
    ``cross_validate``'s ``scoring`` expects a callable with signature
    ``scorer(estimator, X, y)``, not a metric function with signature
    ``metric(y_true, y_pred)``. Passing the metric function directly fails.
    """
    X, sites = binary_site_data
    with pytest.raises(ValueError, match="looks like it is a metric function"):
        evaluate_site_prediction(X, sites, metrics=balanced_accuracy_score)


def test_make_scorer_wrapped_callable_gets_friendly_key(binary_site_data):
    """A ``make_scorer``-wrapped metric is accepted."""
    X, sites = binary_site_data
    scorer = make_scorer(balanced_accuracy_score)
    results = evaluate_site_prediction(X, sites, metrics=scorer)

    metric_keys = set(results.keys()) - {"fit_time", "score_time", "estimators"}
    assert metric_keys == {"balanced_accuracy_score"}


def test_empty_metrics_list_raises_clear_error(binary_site_data):
    """An empty metrics list is now rejected by our own validation."""
    X, sites = binary_site_data
    with pytest.raises(ValueError, match="metrics must contain at least one"):
        evaluate_site_prediction(X, sites, metrics=[])


def test_roc_auc_on_multiclass_raises_instead_of_silent_nan(multiclass_site_data):
    """Requesting ``roc_auc`` (binary-only) for a >2-site."""
    X, sites = multiclass_site_data
    with pytest.raises(ValueError, match="multi_class"):
        evaluate_site_prediction(X, sites, metrics="roc_auc")


def test_roc_auc_ovr_on_multiclass_is_valid_and_not_nan(multiclass_site_data):
    """The multiclass-safe scorer name works and yields a finite score."""
    X, sites = multiclass_site_data
    results = evaluate_site_prediction(X, sites, metrics="roc_auc_ovr")

    assert not np.isnan(results["roc_auc_ovr"]["mean"])
    assert 0.0 <= results["roc_auc_ovr"]["mean"] <= 1.0


# --------------------------------------------------------------------------- #
# Model resolution
# --------------------------------------------------------------------------- #


def test_default_model_is_logistic_regression(binary_site_data):
    """test_default_model_is_logistic_regression."""
    X, sites = binary_site_data
    results = evaluate_site_prediction(X, sites, return_estimator=True)

    assert all(isinstance(est, LogisticRegression) for est in results["estimators"])


def test_custom_classifier_is_used(binary_site_data):
    """test_custom_classifier_is_used."""
    X, sites = binary_site_data
    results = evaluate_site_prediction(
        X,
        sites,
        model=RandomForestClassifier(n_estimators=20, random_state=0),
        return_estimator=True,
    )

    assert all(isinstance(est, RandomForestClassifier) for est in results["estimators"])


def test_passing_a_regressor_raises_type_error(binary_site_data):
    """A model with fit/predict but that is not a classifier must be rejected."""
    X, sites = binary_site_data
    with pytest.raises(TypeError, match="classifier"):
        evaluate_site_prediction(X, sites, model=LinearRegression())


@pytest.mark.parametrize("bad_model", ["not_a_model", 42, object()])
def test_passing_a_non_estimator_raises_type_error(binary_site_data, bad_model):
    """Objects without fit/predict must be rejected with a clean TypeError."""
    X, sites = binary_site_data
    with pytest.raises(TypeError, match="classifier"):
        evaluate_site_prediction(X, sites, model=bad_model)


def test_user_supplied_model_object_is_not_mutated(binary_site_data):
    """The caller's estimator instance should remain unfitted after the call.

    ``_resolve_model`` explicitly clones ``model`` via ``sklearn.base.clone``
    before it is used, so the original object passed in is never fitted or
    otherwise mutated in place — independent of whatever ``cross_validate``
    does internally.
    """
    X, sites = binary_site_data
    user_model = LogisticRegression(max_iter=1000)
    results = evaluate_site_prediction(X, sites, model=user_model, return_estimator=True)

    assert not hasattr(user_model, "coef_")
    assert all(est is not user_model for est in results["estimators"])


# --------------------------------------------------------------------------- #
# CV resolution
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_splits", [3, 5])
def test_custom_cv_controls_number_of_folds(binary_site_data, n_splits):
    """test_custom_cv_controls_number_of_folds."""
    X, sites = binary_site_data
    results = evaluate_site_prediction(
        X,
        sites,
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0),
    )

    assert len(results["balanced_accuracy"]["scores"]) == n_splits


def test_default_cv_uses_five_folds(binary_site_data):
    """No explicit ``cv`` should resolve to sklearn's default of 5 folds."""
    X, sites = binary_site_data
    results = evaluate_site_prediction(X, sites)

    assert len(results["balanced_accuracy"]["scores"]) == 5


def test_non_stratified_cv_is_accepted(binary_site_data):
    """Non-classifier-specific splitters (e.g. plain KFold) still work."""
    X, sites = binary_site_data
    results = evaluate_site_prediction(X, sites, cv=KFold(n_splits=4))

    assert len(results["balanced_accuracy"]["scores"]) == 4


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #


def test_single_site_raises_value_error(single_site_data):
    """test_single_site_raises_value_error."""
    X, sites = single_site_data
    with pytest.raises(ValueError, match="at least 2 unique values"):
        evaluate_site_prediction(X, sites)


def test_mismatched_lengths_raise_value_error(binary_site_data):
    """test_mismatched_lengths_raise_value_error."""
    X, sites = binary_site_data
    with pytest.raises(ValueError):
        evaluate_site_prediction(X, sites[:-10])


def test_string_site_labels_are_supported(string_labeled_site_data):
    """``sites`` need not be numeric; string labels should work end-to-end."""
    X, sites = string_labeled_site_data
    results = evaluate_site_prediction(X, sites)

    assert "balanced_accuracy" in results
    assert not np.isnan(results["balanced_accuracy"]["mean"])


# --------------------------------------------------------------------------- #
# return_estimator flag
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("return_estimator", [True, False])
def test_return_estimator_flag_controls_estimators_key(binary_site_data, return_estimator):
    """test_return_estimator_flag_controls_estimators_key."""
    X, sites = binary_site_data
    results = evaluate_site_prediction(X, sites, return_estimator=return_estimator)

    assert ("estimators" in results) is return_estimator


def test_return_estimator_count_matches_number_of_folds(binary_site_data):
    """test_return_estimator_count_matches_number_of_folds."""
    X, sites = binary_site_data
    n_splits = 4
    results = evaluate_site_prediction(
        X,
        sites,
        cv=StratifiedKFold(n_splits=n_splits),
        return_estimator=True,
    )

    assert len(results["estimators"]) == n_splits


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #


def test_same_random_state_gives_reproducible_default_model_scores(binary_site_data):
    """Using the same ``random_state`` (and no custom model) should be deterministic."""
    X, sites = binary_site_data
    cv = StratifiedKFold(n_splits=5, shuffle=False)

    results_a = evaluate_site_prediction(X, sites, cv=cv, random_state=123)
    results_b = evaluate_site_prediction(X, sites, cv=cv, random_state=123)

    np.testing.assert_allclose(
        results_a["balanced_accuracy"]["scores"],
        results_b["balanced_accuracy"]["scores"],
    )
