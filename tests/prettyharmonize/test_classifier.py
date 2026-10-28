"""Tests for PrettYharmonizeClassifier."""

import numpy as np
import pytest
import sklearn
import structlog
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.svm import LinearSVC

from uniharmony.combat import ComBatGAM, CovBat, NeuroComBat
from uniharmony.datasets import make_multisite_classification
from uniharmony.prettyharmonize import PrettYharmonizeClassifier


def _split(X, y, sites):
    return train_test_split(X, y, sites, test_size=0.25, random_state=0, stratify=sites)


def test_binary(clf_data) -> None:
    """Binary classification: shapes, probabilities and accuracy."""
    X, y, sites = clf_data
    X_train, X_test, y_train, y_test, s_train, s_test = _split(X, y, sites)
    clf = PrettYharmonizeClassifier(random_state=0).fit(X_train, y_train, s_train)
    assert clf.predict(X_test, s_test).shape == y_test.shape
    proba = clf.predict_proba(X_test, s_test)
    assert proba.shape == (len(y_test), 2)
    assert_allclose(proba.sum(axis=1), 1)
    assert_array_equal(clf.classes_, [0, 1])
    assert clf.score_matrix_.shape == (len(y_train), 2)  # P(class 1) per pretended class
    assert not np.isnan(clf.score_matrix_).any()
    assert clf.n_features_in_ == X.shape[1]
    assert not hasattr(clf, "harmonized_data_")
    assert clf.score(X_test, y_test, s_test) > 0.7


def test_score_is_balanced_accuracy(clf_data) -> None:
    """The score is the balanced accuracy (not inflated by the majority class)."""
    X, y, sites = clf_data
    X_train, X_test, y_train, y_test, s_train, s_test = _split(X, y, sites)
    clf = PrettYharmonizeClassifier(random_state=0).fit(X_train, y_train, s_train)
    assert clf.score(X_test, y_test, s_test) == balanced_accuracy_score(y_test, clf.predict(X_test, s_test))


def test_string_labels(clf_data) -> None:
    """String labels give the same model as integer labels."""
    X, y, sites = clf_data
    labels = np.array(["control", "patient"])[y]
    a = PrettYharmonizeClassifier(random_state=0).fit(X, labels, sites)
    b = PrettYharmonizeClassifier(random_state=0).fit(X, y, sites)
    assert_array_equal(a.classes_, ["control", "patient"])
    assert_array_equal(a.predict(X, sites), a.classes_[b.predict(X, sites)])
    assert_allclose(a.predict_proba(X, sites), b.predict_proba(X, sites))


def test_multiclass() -> None:
    """Multi-class problems use all class probabilities per pretended class."""
    X, y, sites = make_multisite_classification(
        n_sites=3, n_samples=450, n_features=8, n_classes=3, site_effect_type="location+scale", random_state=0
    )
    clf = PrettYharmonizeClassifier(random_state=0).fit(X, y, sites)
    assert clf._target_argument == "categorical_covariates"
    assert clf.score_matrix_.shape == (450, 9)  # 3 pretended classes x 3 probabilities
    assert clf.predict_proba(X, sites).shape == (450, 3)
    assert clf.score(X, y, sites) > 0.5


def test_additional_covariates(clf_data, rng) -> None:
    """Additional covariates are preserved next to the class and required at prediction time."""
    X, y, sites = clf_data
    sex = rng.choice(["F", "M"], len(y))
    age = rng.uniform(20, 80, len(y))
    covariates = {"categorical_covariates": sex, "continuous_covariates": age}
    clf = PrettYharmonizeClassifier(random_state=0).fit(X, y, sites, covariates)
    assert clf._covariate_names == ["categorical_covariates", "continuous_covariates"]
    assert clf.predict(X, sites, covariates).shape == y.shape
    categorical = np.column_stack([y.astype(object), sex.astype(object)])  # the class is the first column
    expected = NeuroComBat().fit(X, sites, categorical_covariates=categorical, continuous_covariates=age)
    assert_allclose(
        clf.harm_model_.transform(X, sites, categorical_covariates=categorical, continuous_covariates=age),
        expected.transform(X, sites, categorical_covariates=categorical, continuous_covariates=age),
    )
    with pytest.raises(ValueError, match="fitted with additional covariates"):
        clf.predict(X, sites, {"continuous_covariates": age})


def test_additional_covariates_errors(clf_data, rng) -> None:
    """Additional covariates are validated against the arguments of harm_model."""
    X, y, sites = clf_data
    age = rng.uniform(20, 80, len(y))
    with pytest.raises(ValueError, match="does not accept the covariate argument"):
        PrettYharmonizeClassifier().fit(X, y, sites, {"smooth_covariates": age})
    with pytest.raises(TypeError, match="must be a dict"):
        PrettYharmonizeClassifier().fit(X, y, sites, age)
    with pytest.raises(ValueError, match="one row per sample"):
        PrettYharmonizeClassifier().fit(X, y, sites, {"continuous_covariates": age[:-1]})
    with pytest.raises(ValueError, match="requires the covariate"):
        PrettYharmonizeClassifier(harm_model=ComBatGAM()).fit(X, y, sites)


def test_reproducible_and_parallel(clf_data) -> None:
    """Results are reproducible and do not depend on n_jobs."""
    X, y, sites = clf_data
    a = PrettYharmonizeClassifier(random_state=3).fit(X, y, sites)
    b = PrettYharmonizeClassifier(random_state=3, n_jobs=2).fit(X, y, sites)
    assert_allclose(a.score_matrix_, b.score_matrix_)
    assert_allclose(a.predict_proba(X, sites), b.predict_proba(X, sites))


def test_folds_stratified_by_site_and_class(clf_data) -> None:
    """Inner folds keep the site x class proportions."""
    X, y, sites = clf_data
    clf = PrettYharmonizeClassifier(random_state=0)
    clf.classes_ = np.unique(y)
    splits = clf._make_splits(X, y, sites)
    assert len(splits) == 5
    for train, _ in splits:
        for s in np.unique(sites):
            for c in (0, 1):
                full = np.mean((sites == s) & (y == c))
                part = np.mean((sites[train] == s) & (y[train] == c))
                assert abs(full - part) < 0.03


def test_custom_cv(clf_data) -> None:
    """Custom splitters, lists of splits and integer cv work."""
    X, y, sites = clf_data
    cv = KFold(3, shuffle=True, random_state=0)
    a = PrettYharmonizeClassifier(cv=cv).fit(X, y, sites)
    splits = list(cv.split(X))
    b = PrettYharmonizeClassifier(cv=splits).fit(X, y, sites)
    assert_allclose(a.score_matrix_, b.score_matrix_)
    b.fit(X, y, sites)  # a list of splits can be reused
    assert PrettYharmonizeClassifier(cv=3, random_state=0).fit(X, y, sites).score_matrix_.shape == (len(y), 2)


@pytest.mark.filterwarnings("ignore:The least populated class")
def test_cv_errors(clf_data) -> None:
    """Invalid inner cross-validation raises informative errors."""
    X, y, sites = clf_data
    idx = np.arange(len(y))
    with pytest.raises(ValueError, match="never are"):
        PrettYharmonizeClassifier(cv=[(idx[:150], idx[150:200])]).fit(X, y, sites)
    with pytest.raises(TypeError, match="cv must be"):
        PrettYharmonizeClassifier(cv=3.5).fit(X, y, sites)
    with pytest.raises(ValueError, match="cv must be >= 2"):
        PrettYharmonizeClassifier(cv=1).fit(X, y, sites)
    small = sites.copy()
    small[:2] = 99
    with pytest.raises(ValueError, match="fewer than 2 samples"):
        PrettYharmonizeClassifier(random_state=0).fit(X, y, small)


def test_unseen_site(clf_data) -> None:
    """Unseen or missing sites raise an error."""
    X, y, sites = clf_data
    clf = PrettYharmonizeClassifier(random_state=0).fit(X, y, sites)
    with pytest.raises(ValueError, match="Sites unseen"):
        clf.predict(X[:5], np.full(5, 99))
    with pytest.raises(ValueError, match="sites must be provided"):
        clf.predict(X[:5], None)  # type: ignore[arg-type]


def test_models(clf_data) -> None:
    """User models are cloned; decision_function models work."""
    X, y, sites = clf_data
    pred = LogisticRegression(C=0.5)
    clf = PrettYharmonizeClassifier(base_model=pred, stack_model=LogisticRegression()).fit(X, y, sites)
    assert not hasattr(pred, "coef_")  # cloned, not fitted in place
    assert clf.base_model_.C == 0.5
    svc = PrettYharmonizeClassifier(base_model=LinearSVC()).fit(X, y, sites)  # decision_function only
    assert svc.score_matrix_.shape == (len(y), 2)
    with pytest.raises(TypeError, match="scikit-learn estimator"):
        PrettYharmonizeClassifier(base_model="logit").fit(X, y, sites)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="predict_proba or decision_function"):
        PrettYharmonizeClassifier(base_model=Ridge()).fit(X, y, sites)


def test_out_of_fold_harmonized_data(clf_data) -> None:
    """fit_transform returns each sample harmonized (true class) by the fold model that did not see it."""
    X, y, sites = clf_data
    splits = list(KFold(3, shuffle=True, random_state=0).split(X))
    clf = PrettYharmonizeClassifier(cv=splits, return_harmonized_data=True)
    X_oof = clf.fit_transform(X, y, sites)
    assert_allclose(clf.harmonized_data_, X_oof)
    train, test = splits[0]
    fold = NeuroComBat().fit(X[train], sites[train], categorical_covariates=y[train])
    assert_allclose(X_oof[test], fold.transform(X[test], sites[test], categorical_covariates=y[test]))
    # The final base model is fitted on the out-of-fold data; harm_model_ on all the data.
    base = clone(clf.base_model_).fit(X_oof, y)
    assert_allclose(clf.base_model_.predict_proba(X), base.predict_proba(X))
    full = NeuroComBat().fit(X, sites, categorical_covariates=y)
    assert_allclose(
        clf.harm_model_.transform(X, sites, categorical_covariates=y), full.transform(X, sites, categorical_covariates=y)
    )


def test_transform_uses_predicted_class(clf_data) -> None:
    """The transform harmonizes new data with the predicted class."""
    X, y, sites = clf_data
    clf = PrettYharmonizeClassifier(random_state=0).fit(X, y, sites)
    y_pred = clf.predict(X, sites)
    assert_allclose(clf.transform(X, sites), clf.harm_model_.transform(X, sites, categorical_covariates=y_pred))


def test_without_final_model(clf_data) -> None:
    """With fit_final_model=False, the inner-fold models are kept and their score matrices averaged."""
    X, y, sites = clf_data
    X_train, X_test, y_train, _, s_train, s_test = _split(X, y, sites)
    final = PrettYharmonizeClassifier(cv=3, random_state=0).fit(X_train, y_train, s_train)
    folds = PrettYharmonizeClassifier(cv=3, random_state=0, fit_final_model=False).fit(X_train, y_train, s_train)
    assert_allclose(final.score_matrix_, folds.score_matrix_)  # same inner folds
    assert not hasattr(folds, "harm_model_") and not hasattr(folds, "base_model_")
    assert len(folds.fold_harm_models_) == len(folds.fold_base_models_) == 3
    matrices = [
        np.column_stack(
            [
                model.predict_proba(harm.transform(X_test, s_test, categorical_covariates=np.full(len(s_test), c)))[:, 1]
                for c in (0, 1)
            ]
        )
        for harm, model in zip(folds.fold_harm_models_, folds.fold_base_models_, strict=True)
    ]
    assert_allclose(folds.predict_proba(X_test, s_test), folds.stack_model_.predict_proba(np.mean(matrices, axis=0)))
    assert folds.transform(X_test, s_test).shape == X_test.shape
    # Refitting with another setting removes the attributes of the previous fit.
    folds.set_params(fit_final_model=True).fit(X_train, y_train, s_train)
    assert hasattr(folds, "harm_model_") and not hasattr(folds, "fold_harm_models_")


def test_harm_model_parameters_are_used(clf_data) -> None:
    """The harmonization model is cloned with its parameters and fitted with the class."""
    X, y, sites = clf_data
    harm_model = NeuroComBat(empirical_bayes=False)
    clf = PrettYharmonizeClassifier(harm_model=harm_model, random_state=0).fit(X, y, sites)
    assert not hasattr(harm_model, "sites_")  # cloned, not fitted in place
    assert clf.harm_model_.empirical_bayes is False
    expected = NeuroComBat(empirical_bayes=False).fit(X, sites, categorical_covariates=y)
    assert_allclose(
        clf.harm_model_.transform(X, sites, categorical_covariates=y), expected.transform(X, sites, categorical_covariates=y)
    )


def test_other_combat_models(clf_data, rng) -> None:
    """Any registered ComBat-based model can be wrapped (CovBat, ComBatGAM)."""
    X, y, sites = clf_data
    covbat = PrettYharmonizeClassifier(harm_model=CovBat(), random_state=0).fit(X, y, sites)
    assert covbat._target_argument == "categorical_covariates"
    assert covbat.score(X, y, sites) > 0.7
    # ComBatGAM has no categorical covariates: the class goes in as one-hot continuous covariate.
    age = rng.uniform(20, 80, len(y))
    gam = PrettYharmonizeClassifier(harm_model=ComBatGAM(), cv=3, random_state=0)
    gam.fit(X, y, sites, {"smooth_covariates": age})
    assert gam._target_argument == "continuous_covariates"
    assert gam.predict(X, sites, {"smooth_covariates": age}).shape == y.shape
    with pytest.raises(ValueError, match="additional covariates"):
        gam.predict(X, sites)
    with pytest.raises(ValueError, match="cannot extrapolate"):
        gam.predict(X[:3], sites[:3], {"smooth_covariates": np.full(3, 200.0)})


def test_harm_model_errors(clf_data) -> None:
    """Unregistered harmonization models and site-confounded targets raise informative errors."""
    X, y, sites = clf_data
    with pytest.raises(TypeError, match="ComBat-based model"):
        PrettYharmonizeClassifier(harm_model=LogisticRegression()).fit(X, y, sites)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="confounded with the sites"):
        PrettYharmonizeClassifier().fit(X, (sites == sites[0]).astype(int), sites)


def test_no_leakage_warning(clf_data) -> None:
    """The ComBat target-leakage warning is not emitted: PrettYharmonize never uses the predicted samples' target."""
    X, y, sites = clf_data
    with structlog.testing.capture_logs() as logs:
        PrettYharmonizeClassifier(cv=3, random_state=0).fit(X, y, sites)
    assert not any("DO *NOT* preserve" in entry["event"] for entry in logs)


def test_clone_and_get_params() -> None:
    """The estimator follows the scikit-learn parameter API."""
    clf = PrettYharmonizeClassifier(harm_model=NeuroComBat(mean_only=True), cv=4, random_state=1, n_jobs=2)
    params = clone(clf).get_params()
    assert params["cv"] == 4
    assert params["harm_model__mean_only"] is True


def test_errors(clf_data) -> None:
    """Invalid inputs raise informative errors."""
    X, y, sites = clf_data
    with pytest.raises(ValueError, match="Unknown label type"):
        PrettYharmonizeClassifier().fit(X, y + 0.5, sites)
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        PrettYharmonizeClassifier().fit(X, y, sites[:-1])
    clf = PrettYharmonizeClassifier().fit(X, y, sites)
    with pytest.raises(ValueError, match="features"):
        clf.predict(X[:, :3], sites)


def test_metadata_routing_cross_validate(clf_data) -> None:
    """Works inside cross_validate with metadata routing."""
    X, y, sites = clf_data
    with sklearn.config_context(enable_metadata_routing=True):
        clf = PrettYharmonizeClassifier(cv=3, random_state=0).set_fit_request(sites=True).set_score_request(sites=True)
        result = cross_validate(clf, X, y, cv=KFold(3, shuffle=True, random_state=0), params={"sites": sites})
    assert np.all(result["test_score"] > 0.6)


def test_keeps_signal_under_site_target_dependence() -> None:
    """PrettYharmonize keeps the signal that ComBat without the target removes.

    With strong site-target dependence, removing the target from ComBat destroys the
    signal; PrettYharmonize keeps it (Nieto et al., 2026).
    """
    X, y, sites = make_multisite_classification(
        n_sites=2,
        n_samples=600,
        n_features=10,
        balance_per_site=[[0.9, 0.1], [0.1, 0.9]],
        signal_strength=0.5,
        site_effect_type="location",
        random_state=0,
    )
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sites, test_size=0.3, random_state=0, stratify=np.char.add(sites.astype(str), y.astype(str))
    )
    pretty = PrettYharmonizeClassifier(random_state=0).fit(X_train, y_train, s_train)
    auc_pretty = roc_auc_score(y_test, pretty.predict_proba(X_test, s_test)[:, 1])
    combat = NeuroComBat().fit(X_train, s_train)
    model = LogisticRegression().fit(combat.transform(X_train, s_train), y_train)
    auc_no_target = roc_auc_score(y_test, model.predict_proba(combat.transform(X_test, s_test))[:, 1])
    assert auc_pretty > auc_no_target + 0.05
