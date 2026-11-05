"""Harmonization to reference site(s): ``ref_site`` of the ComBat-based models.

* ``ref_site=None``: every site is harmonized to the pooled mean and variance
  of all sites.
* one site: every site is harmonized to that site's mean and variance; its
  data are unchanged.
* several sites: every site (the reference sites included) is harmonized to
  the pooled mean and variance of the reference sites.

Without empirical Bayes and covariates, ComBat's output has a closed form: the
mean of every harmonized site is the mean of the reference samples, and its
variance (``ddof=1``) is the pooled within-site variance of the reference
sites, ``RSS_ref / (n_ref - S_ref)``. The tests check it exactly, and the
behaviour of every model and estimator option. Conformance with neuroCombat's
``ref_batch`` is tested in ``test_reference_conformance.py``.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline

from uniharmony import verbosity_context
from uniharmony.combat import ComBatGAM, CovBat, NeuroComBat
from uniharmony.prettyharmonize import PrettYharmonizeClassifier


SITES = np.array(["a", "b", "c", "d"])


@pytest.fixture(autouse=True)
def _quiet():
    """Silence the logs and warnings of the estimators."""
    with verbosity_context("error"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.fixture
def data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Four sites of different sizes with location and scale site effects, and an age effect."""
    rng = np.random.default_rng(0)
    sizes = [30, 45, 60, 25]
    sites = np.repeat(SITES, sizes)
    site = np.searchsorted(SITES, sites)
    shift = rng.normal(0, 1.5, (4, 6))[site]
    scale = rng.uniform(0.5, 2.0, (4, 6))[site]
    age = rng.uniform(20, 80, len(sites))
    X = 5 + shift + scale * rng.normal(size=(len(sites), 6)) + 0.03 * age[:, None]
    return X, sites, age[:, None]


def _reference_statistics(X: np.ndarray, sites: np.ndarray, reference: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Mean of the reference samples and pooled within-site variance of the reference sites."""
    in_ref = np.isin(sites, reference)
    mean = X[in_ref].mean(axis=0)
    rss = sum(((X[sites == s] - X[sites == s].mean(axis=0)) ** 2).sum(axis=0) for s in reference)
    return mean, rss / (in_ref.sum() - len(reference))


# ---------------------------------------------------------------------------
# Closed form (no empirical Bayes, no covariates)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reference", [["b"], ["a", "c"], ["a", "b", "d"]])
def test_closed_form_without_empirical_bayes(data, reference: list[str]) -> None:
    """Every harmonized site has the mean and the variance of the reference pool."""
    X, sites, _ = data
    ref_site = reference[0] if len(reference) == 1 else reference
    harmonized = NeuroComBat(ref_site=ref_site, empirical_bayes=False).fit_transform(X, sites=sites)
    mean, variance = _reference_statistics(X, sites, reference)
    for s in SITES:
        np.testing.assert_allclose(harmonized[sites == s].mean(axis=0), mean, atol=1e-10)
        np.testing.assert_allclose(harmonized[sites == s].var(axis=0, ddof=1), variance, rtol=1e-10)


def test_pooled_is_the_reference_of_all_sites(data) -> None:
    """ref_site listing every site gives the default pooled harmonization."""
    X, sites, age = data
    for params in ({}, {"empirical_bayes": False}, {"parametric_adjustments": False}, {"mean_only": True}):
        pooled = NeuroComBat(**params).fit_transform(X, sites=sites, continuous_covariates=age)
        all_sites = NeuroComBat(ref_site=list(SITES), **params).fit_transform(X, sites=sites, continuous_covariates=age)
        np.testing.assert_allclose(all_sites, pooled, atol=1e-10)


# ---------------------------------------------------------------------------
# A single reference site is unchanged, in every model and mode
# ---------------------------------------------------------------------------

MODES = [
    {},
    {"empirical_bayes": False},
    {"parametric_adjustments": False},
    {"mean_only": True},
]


def _models(params: dict) -> list:
    return [
        NeuroComBat(ref_site="b", **params),
        ComBatGAM(ref_site="b", random_state=0, **params),
        CovBat(
            ref_site="b",
            first_combat_eb=params.get("empirical_bayes", True),
            first_combat_parametric=params.get("parametric_adjustments", True),
            mean_only=params.get("mean_only", False),
        ),
    ]


def _covariates(model, age: np.ndarray) -> dict:
    return {"smooth_covariates": age} if isinstance(model, ComBatGAM) else {"continuous_covariates": age}


@pytest.mark.parametrize("params", MODES, ids=["eb", "no_eb", "non_parametric", "mean_only"])
def test_single_reference_site_is_unchanged(data, params: dict) -> None:
    """The reference site keeps its data, in fit_transform and for new subjects."""
    X, sites, age = data
    in_ref = sites == "b"
    for model in _models(params):
        covariates = _covariates(model, age)
        harmonized = model.fit_transform(X, sites=sites, **covariates)
        np.testing.assert_array_equal(harmonized[in_ref], X[in_ref], err_msg=type(model).__name__)
        assert not np.allclose(harmonized[~in_ref], X[~in_ref])
        # New subjects of the reference site are unchanged too
        new = X[in_ref][:5] + 0.1
        new_covariates = {name: value[in_ref][:5] for name, value in covariates.items()}
        np.testing.assert_array_equal(model.transform(new, sites=np.full(5, "b"), **new_covariates), new)


def test_reference_site_unchanged_before_restoring_the_input(data) -> None:
    """The harmonized values of a single reference site equal its input up to rounding (the copy is cosmetic)."""
    X, sites, age = data
    model = NeuroComBat(ref_site="b").fit(X, sites=sites, continuous_covariates=age)
    in_ref = sites == "b"
    data_std, mean, idx_per_site = _standardized(model, X, sites, age)
    # harmonize() is the step before the copy of the input in transform()
    raw = model.harmonize(data=data_std, mean=mean, idx_per_site=idx_per_site).T
    np.testing.assert_allclose(raw[in_ref], X[in_ref], rtol=0, atol=1e-10 * np.abs(X).max())


def _standardized(model, X, sites, age):
    design = model.transform_design_matrix(sites=sites, categorical_covariates=None, continuous_covariates=age)
    data, mean = model.transform_standardize(X=X.T, design=design, n_samples=len(sites))
    idx_per_site = [list(np.flatnonzero(sites == s)) for s in model.sites_]
    return data, mean, idx_per_site


def test_other_sites_get_the_reference_distribution(data) -> None:
    """With a single reference site, the other sites move towards its mean and variance (with empirical Bayes)."""
    X, sites, _ = data
    harmonized = NeuroComBat(ref_site="b").fit_transform(X, sites=sites)
    mean, variance = _reference_statistics(X, sites, ["b"])
    for s in ("a", "c", "d"):
        before = np.abs(X[sites == s].mean(axis=0) - mean).mean()
        after = np.abs(harmonized[sites == s].mean(axis=0) - mean).mean()
        assert after < 0.2 * before
        assert np.abs(np.log(harmonized[sites == s].var(axis=0, ddof=1) / variance)).mean() < 0.3


def test_several_reference_sites_are_harmonized_to_their_pool(data) -> None:
    """With several reference sites, the reference sites themselves are harmonized (to their pooled distribution)."""
    X, sites, _ = data
    harmonized = NeuroComBat(ref_site=["a", "c"]).fit_transform(X, sites=sites)
    assert not np.allclose(harmonized[sites == "a"], X[sites == "a"])
    mean, _ = _reference_statistics(X, sites, ["a", "c"])
    np.testing.assert_allclose(harmonized[np.isin(sites, ["a", "c"])].mean(axis=0), mean, atol=0.15)


# ---------------------------------------------------------------------------
# Parameter handling
# ---------------------------------------------------------------------------


def test_ref_sites_attribute_and_order(data) -> None:
    """ref_sites_ lists the reference sites in the order of sites_; the order of ref_site does not matter."""
    X, sites, _ = data
    one = NeuroComBat(ref_site=["c", "a"]).fit(X, sites=sites)
    other = NeuroComBat(ref_site=["a", "c"]).fit(X, sites=sites)
    np.testing.assert_array_equal(one.ref_sites_, ["a", "c"])
    np.testing.assert_array_equal(one.transform(X, sites=sites), other.transform(X, sites=sites))
    np.testing.assert_array_equal(NeuroComBat().fit(X, sites=sites).ref_sites_, SITES)


def test_numeric_sites(data) -> None:
    """Numeric site labels; a label given as string is matched too."""
    X, sites, _ = data
    codes = np.searchsorted(SITES, sites)
    by_int = NeuroComBat(ref_site=1).fit_transform(X, sites=codes)
    by_str = NeuroComBat(ref_site="1").fit_transform(X, sites=codes)
    np.testing.assert_array_equal(by_int, by_str)
    np.testing.assert_array_equal(by_int[codes == 1], X[codes == 1])


@pytest.mark.parametrize(
    ("ref_site", "match"),
    [("z", "not one of the sites"), (["a", "z"], "not one of the sites"), (["a", "a"], "twice"), ([], "at least one")],
)
def test_invalid_ref_site(data, ref_site, match: str) -> None:
    """Unknown, repeated or empty reference sites raise informative errors."""
    X, sites, _ = data
    for model in (NeuroComBat(ref_site=ref_site), CovBat(ref_site=ref_site)):
        with pytest.raises(ValueError, match=match):
            model.fit(X, sites=sites)


def test_ref_site_is_a_parameter(data) -> None:
    """ref_site is a constructor parameter (get_params, set_params, clone) of every ComBat model."""
    for model in (NeuroComBat(ref_site="a"), CovBat(ref_site=["a", "b"]), ComBatGAM(ref_site="a")):
        assert clone(model).get_params()["ref_site"] == model.ref_site
        assert model.set_params(ref_site=None).ref_site is None


# ---------------------------------------------------------------------------
# Pipelines and PrettYharmonize
# ---------------------------------------------------------------------------


def test_reference_site_in_a_pipeline(data) -> None:
    """With the sites as a column, ref_site works in cross-validation (the folds are stratified on the sites)."""
    X, sites, age = data
    df = pd.DataFrame(X, columns=[f"f{j}" for j in range(X.shape[1])]).assign(site=sites, age=age[:, 0])
    y = (X[:, 0] + np.random.default_rng(1).normal(size=len(X)) > 5).astype(int)
    pipe = make_pipeline(NeuroComBat(site_column="site", continuous_columns="age", ref_site="b"), LogisticRegression())
    cv = StratifiedKFold(3, shuffle=True, random_state=0).split(df, sites)
    scores = cross_validate(pipe, df, y, cv=list(cv), scoring="roc_auc", error_score="raise")
    assert np.all(np.isfinite(scores["test_score"]))


def test_prettyharmonize_with_reference_site(data) -> None:
    """PrettYharmonize keeps the ref_site of its harmonization model."""
    X, sites, _ = data
    y = np.random.default_rng(2).integers(0, 2, len(X))
    clf = PrettYharmonizeClassifier(harm_model=NeuroComBat(ref_site="b"), cv=3, random_state=0).fit(X, y, sites=sites)
    np.testing.assert_array_equal(clf.harm_model_.ref_sites_, ["b"])
    assert clf.predict_proba(X, sites=sites).shape == (len(X), 2)
