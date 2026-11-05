"""The harmonizers and samplers in scikit-learn, imbalanced-learn and julearn pipelines.

The sites and covariates are given in one of two ways (see ``uniharmony._data``):

* as **columns** of a DataFrame ``X``, named in the constructor: the folds of
  every scikit-learn tool split them with ``X``, so they reach ``transform``
  and every scorer;
* as **keywords** of ``fit`` / ``transform`` (direct calls, NumPy users,
  metadata routing).

The two paths must give the same result. The column path must work with every
scikit-learn model selection tool, scorers included.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import sklearn
from imblearn.pipeline import Pipeline as ImbPipeline
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from uniharmony import verbosity_context
from uniharmony.combat import ComBatGAM, CovBat, NeuroComBat
from uniharmony.interpolation import InterSiteMatchedInterpolation, IntraSiteInterpolation, SiteBalancedResampler
from uniharmony.iqm import BARTharm
from uniharmony.ot import OptimalTransportDomainAdaptation
from uniharmony.prettyharmonize import PrettYharmonizeClassifier


FEATURES = ["f0", "f1", "f2", "f3"]
CV = StratifiedKFold(3, shuffle=True, random_state=0)


@pytest.fixture(autouse=True)
def _quiet():
    """Silence the covariate warnings of the harmonizers."""
    with verbosity_context("error"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.fixture
def frame() -> tuple[pd.DataFrame, np.ndarray]:
    """Three sites with location/scale site effects, an age effect, a sex effect and a class effect.

    The class balance differs across sites, so the site predicts the class.
    The columns are interleaved (features, site, age, sex), with a custom index.
    """
    rng = np.random.default_rng(0)
    n_per_site = 80
    sites = np.repeat(["a", "b", "c"], n_per_site)
    p_class = np.select([sites == "a", sites == "b"], [0.25, 0.5], 0.75)
    y = (rng.uniform(size=len(sites)) < p_class).astype(int)
    age = rng.uniform(20, 80, len(sites))
    sex = rng.choice(np.array(["F", "M"]), len(sites))
    shift = np.select([sites == "a", sites == "b"], [0.0, 1.5], -1.0)[:, None]
    scale = np.select([sites == "a", sites == "b"], [1.0, 1.6], 0.7)[:, None]
    X = 0.03 * age[:, None] + 0.4 * (sex == "M")[:, None] + 0.8 * y[:, None] + shift + scale * rng.normal(size=(len(sites), 4))
    df = pd.DataFrame(
        {"f0": X[:, 0], "site": sites, "f1": X[:, 1], "age": age, "f2": X[:, 2], "sex": sex, "f3": X[:, 3]},
        index=pd.Index([f"sub-{i:03d}" for i in range(len(sites))], name="subject"),
    )
    return df, y


# ---------------------------------------------------------------------------
# Column path = keyword path
# ---------------------------------------------------------------------------

# (estimator with column names, the same estimator for the keyword path, keyword data from the frame)
EQUIVALENCE_CASES = {
    "NeuroComBat": (
        NeuroComBat(site_column="site", categorical_columns="sex", continuous_columns="age"),
        NeuroComBat(),
        {"sites": "site", "categorical_covariates": ["sex"], "continuous_covariates": ["age"]},
    ),
    "CovBat": (
        CovBat(site_column="site", categorical_columns=["sex"], continuous_columns=["age"]),
        CovBat(),
        {"sites": "site", "categorical_covariates": ["sex"], "continuous_covariates": ["age"]},
    ),
    "ComBatGAM": (
        ComBatGAM(site_column="site", smooth_columns="age", random_state=0),
        ComBatGAM(random_state=0),
        {"sites": "site", "smooth_covariates": ["age"]},
    ),
    "OTDA": (
        OptimalTransportDomainAdaptation(ref_site="a", site_column="site"),
        OptimalTransportDomainAdaptation(ref_site="a"),
        {"sites": "site"},
    ),
}


def _keywords(df: pd.DataFrame, spec: dict) -> dict:
    return {name: df[columns].to_numpy() for name, columns in spec.items()}


@pytest.mark.parametrize("name", list(EQUIVALENCE_CASES))
def test_column_path_equals_keyword_path(frame, name: str) -> None:
    """Reading the metadata from columns gives the keyword result (fit, transform, fit_transform)."""
    df, _ = frame
    by_column, by_keyword, spec = EQUIVALENCE_CASES[name]
    metadata_columns = {c for columns in spec.values() for c in np.atleast_1d(columns)}
    df = df[[c for c in df.columns if c in FEATURES or c in metadata_columns]]
    X = df[FEATURES].to_numpy()
    expected = sklearn.clone(by_keyword).fit(X, **_keywords(df, spec)).transform(X, **_keywords(df, spec))
    assert_allclose(sklearn.clone(by_column).fit(df).transform(df), expected, rtol=1e-12, atol=1e-12)
    assert_allclose(sklearn.clone(by_column).fit_transform(df), expected, rtol=1e-12, atol=1e-12)
    fitted = sklearn.clone(by_column).fit(df)
    assert fitted.n_features_in_ == df.shape[1]
    assert_array_equal(fitted.feature_names_in_, df.columns)
    assert list(fitted.get_feature_names_out()) == [c for c in df.columns if c not in metadata_columns]


def test_bartharm_column_path_equals_keyword_path(frame) -> None:
    """BARTharm reads the IQMs, biological covariates and sites from columns."""
    df, _ = frame
    df = df.drop(columns="sex").assign(snr=np.linspace(0, 1, len(df)))
    params = {"n_iter": 20, "burn_in": 4, "n_trees_mu": 4, "n_trees_tau": 3, "n_stored_draws": 3, "random_state": 0}
    by_column = BARTharm(iqm_columns="snr", biological_columns="age", site_column="site", var_scaling=True, **params)
    keywords = {"iqm_covariates": df[["snr"]], "biological_covariates": df[["age"]], "sites": df["site"]}
    by_keyword = BARTharm(var_scaling=True, **params).fit(df[FEATURES], **keywords)
    assert_allclose(by_column.fit(df).transform(df), by_keyword.transform(df[FEATURES], **keywords))


def test_prettyharmonize_reads_the_columns_named_in_harm_model(frame) -> None:
    """PrettYharmonize reads the sites and covariates from the columns named in its harm_model."""
    df, y = frame
    by_column = PrettYharmonizeClassifier(harm_model=NeuroComBat(site_column="site", continuous_columns="age"), random_state=0)
    by_keyword = PrettYharmonizeClassifier(random_state=0)
    keywords = {"sites": df["site"].to_numpy(), "continuous_covariates": df[["age"]].to_numpy()}
    df = df.drop(columns="sex")
    by_column.fit(df, y)
    by_keyword.fit(df[FEATURES], y, **keywords)
    assert_allclose(by_column.predict_proba(df), by_keyword.predict_proba(df[FEATURES], **keywords))
    assert list(by_column.get_feature_names_out()) == FEATURES
    # The fitted harmonization model receives the data as keywords
    assert by_column.harm_model_.site_column is None


# ---------------------------------------------------------------------------
# Data handler: names, types, checks
# ---------------------------------------------------------------------------


def test_numpy_with_column_positions(frame) -> None:
    """With a NumPy array, the metadata columns are given by position."""
    df, _ = frame
    array = df[["f0", "f1", "age", "f2", "f3"]].to_numpy()
    sites = df["site"].map({"a": 0, "b": 1, "c": 2}).to_numpy()
    array = np.column_stack([sites, array])
    by_position = NeuroComBat(site_column=0, continuous_columns=[3]).fit_transform(array)
    expected = NeuroComBat().fit_transform(array[:, [1, 2, 4, 5]], sites=sites, continuous_covariates=array[:, [3]])
    assert_allclose(by_position, expected, rtol=1e-12, atol=1e-12)


def test_column_names_need_a_dataframe(frame) -> None:
    """Column names with a NumPy array raise a TypeError that explains the alternatives."""
    df, _ = frame
    with pytest.raises(TypeError, match="pandas DataFrame"):
        NeuroComBat(site_column="site").fit(df[FEATURES].to_numpy())


def test_julearn_renamed_columns_are_matched(frame) -> None:
    """A column renamed by julearn (``name__:type:__type``) is found by its plain name."""
    df, _ = frame
    df = df.drop(columns="sex")
    renamed = df.rename(columns={"site": "site__:type:__site", "age": "age__:type:__confound"})
    assert_allclose(
        NeuroComBat(site_column="site", continuous_columns="age").fit_transform(renamed),
        NeuroComBat(site_column="site", continuous_columns="age").fit_transform(df),
    )


@pytest.mark.parametrize(
    ("estimator", "kwargs", "match"),
    [
        (NeuroComBat(site_column="site"), {"sites": "x"}, "given twice"),
        (NeuroComBat(site_column="site", categorical_columns="site"), {}, "used both as"),
        (NeuroComBat(site_column="nope"), {}, "not in X"),
        (NeuroComBat(site_column=["site", "sex"]), {}, "single column"),
        (NeuroComBat(), {}, "needs sites"),
        (NeuroComBat(site_column="site", continuous_columns="sex"), {}, "could not convert|Unable to convert|must be numeric"),
    ],
)
def test_invalid_metadata(frame, estimator, kwargs, match: str) -> None:
    """Invalid metadata specifications raise informative errors."""
    df, _ = frame
    if "sites" in kwargs:
        kwargs = {"sites": df["site"]}
    with pytest.raises((ValueError, KeyError, TypeError), match=match):
        estimator.fit(df if estimator.get_params()["site_column"] is not None else df[FEATURES], **kwargs)


def test_no_feature_left(frame) -> None:
    """At least one feature must remain once the metadata columns are removed."""
    df, _ = frame
    with pytest.raises(ValueError, match="No feature left"):
        NeuroComBat(site_column="site", continuous_columns="age").fit(df[["site", "age"]])


def test_missing_metadata_is_rejected(frame) -> None:
    """Missing sites or covariates are rejected."""
    df, _ = frame
    df = df.drop(columns="sex")
    with_nan = df.copy()
    with_nan.loc[with_nan.index[3], "age"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        NeuroComBat(site_column="site", continuous_columns="age").fit(with_nan)
    with_nan = df.astype({"site": object})
    with_nan.loc[with_nan.index[3], "site"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        NeuroComBat(site_column="site").fit(with_nan)


def test_transform_needs_the_columns_of_fit(frame) -> None:
    """The transform method checks the columns against fit (names and order)."""
    df, _ = frame
    df = df.drop(columns=["sex", "age"])
    combat = NeuroComBat(site_column="site").fit(df)
    with pytest.raises(ValueError, match="feature names"):
        combat.transform(df[df.columns[::-1]])


def test_set_output_pandas_keeps_index_and_names(frame) -> None:
    """With set_output(transform="pandas") the harmonized features keep their names and the index of X."""
    df, _ = frame
    out = (
        NeuroComBat(site_column="site", continuous_columns="age")
        .set_output(transform="pandas")
        .fit_transform(df.assign(sex=(df["sex"] == "M").astype(float)))
    )
    assert list(out.columns) == ["f0", "f1", "f2", "sex", "f3"]
    assert out.index.equals(df.index)


def test_unseen_site_message_explains_cross_validation(frame) -> None:
    """A test fold with a site absent from training gets an error that suggests stratified folds."""
    df, _ = frame
    df = df.drop(columns=["sex", "age"])
    combat = NeuroComBat(site_column="site").fit(df[df.site != "c"])
    with pytest.raises(ValueError, match=r"not seen during fit.*StratifiedKFold"):
        combat.transform(df)


# ---------------------------------------------------------------------------
# scikit-learn pipelines and model selection (column path)
# ---------------------------------------------------------------------------

HARMONIZERS = {
    "NeuroComBat": NeuroComBat(site_column="site", continuous_columns="age"),
    "CovBat": CovBat(site_column="site", continuous_columns="age"),
    "ComBatGAM": ComBatGAM(site_column="site", smooth_columns="age", random_state=0),
    "OTDA": OptimalTransportDomainAdaptation(ref_site="a", site_column="site", ot_method="sinkhorn", reg=1.0),
}


def _pipeline(harmonizer) -> Pipeline:
    return make_pipeline(sklearn.clone(harmonizer), StandardScaler(), LogisticRegression())


@pytest.mark.parametrize("name", list(HARMONIZERS))
def test_cross_validate_with_scorers(frame, name: str) -> None:
    """cross_validate with explicit scorers gives finite scores (sites split with X, scorers see them)."""
    df, y = frame
    X = df.drop(columns="sex")
    scores = cross_validate(
        _pipeline(HARMONIZERS[name]), X, y, cv=CV, scoring=["roc_auc", "balanced_accuracy"], error_score="raise"
    )
    assert np.all(np.isfinite(scores["test_roc_auc"]))
    assert scores["test_roc_auc"].mean() > 0.5


@pytest.mark.parametrize("name", list(HARMONIZERS))
def test_cross_val_predict(frame, name: str) -> None:
    """cross_val_predict harmonizes every test fold with its own sites."""
    df, y = frame
    proba = cross_val_predict(_pipeline(HARMONIZERS[name]), df.drop(columns="sex"), y, cv=CV, method="predict_proba")
    assert proba.shape == (len(y), 2)
    assert np.all(np.isfinite(proba))


def test_grid_search_with_scorer(frame) -> None:
    """GridSearchCV over harmonizer and model parameters, with a scorer, gives a finite best score."""
    df, y = frame
    pipe = Pipeline([("combat", NeuroComBat(site_column="site")), ("scale", StandardScaler()), ("clf", LogisticRegression())])
    search = GridSearchCV(
        pipe,
        {"combat__empirical_bayes": [True, False], "combat__var_epsilon": [1e-8, 1e-6], "clf__C": [0.1, 1.0]},
        cv=CV,
        scoring="roc_auc",
        error_score="raise",
    )
    search.fit(df.drop(columns=["age", "sex"]), y)
    assert np.isfinite(search.best_score_)
    assert np.all(np.isfinite(search.cv_results_["mean_test_score"]))


def test_unshuffled_kfold_on_sorted_sites_fails_clearly(frame) -> None:
    """An unshuffled KFold on data sorted by site puts a whole site in a test fold: the error says so."""
    df, y = frame
    with pytest.raises(ValueError, match="not seen during fit"):
        cross_validate(_pipeline(HARMONIZERS["NeuroComBat"]), df.drop(columns="sex"), y, cv=KFold(3), error_score="raise")


def test_metadata_routing_keyword_path(frame) -> None:
    """With metadata routing, the keyword path works for the default score (the data keywords are requested by default)."""
    df, y = frame
    X = df[FEATURES].to_numpy()
    with sklearn.config_context(enable_metadata_routing=True):
        pipe = make_pipeline(NeuroComBat(), StandardScaler(), LogisticRegression())
        scores = cross_validate(pipe, X, y, cv=CV, params={"sites": df["site"].to_numpy()}, error_score="raise")
    by_column = cross_validate(_pipeline(NeuroComBat(site_column="site")), df[[*FEATURES, "site"]], y, cv=CV)
    assert_allclose(scores["test_score"], by_column["test_score"])


# ---------------------------------------------------------------------------
# imbalanced-learn: sampler followed by a harmonizer
# ---------------------------------------------------------------------------

SAMPLERS = {
    "ISI": IntraSiteInterpolation(site_column="site", passthrough_columns=["age", "sex"], random_state=0),
    "SBR": SiteBalancedResampler(site_column="site", passthrough_columns=["age", "sex"], random_state=0),
    "ISMI": InterSiteMatchedInterpolation(site_column="site", passthrough_columns=["age", "sex"], random_state=0),
}


@pytest.mark.parametrize("name", list(SAMPLERS))
def test_sampler_returns_metadata_of_synthetic_samples(frame, name: str) -> None:
    """The column path returns X with its metadata columns, filled in for the synthetic samples."""
    df, y = frame
    sampler = sklearn.clone(SAMPLERS[name])
    out, _ = sampler.fit_resample(df, y)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == list(df.columns)
    assert out.dtypes.to_dict() == df.dtypes.to_dict()
    assert_array_equal(out["site"].to_numpy(), sampler.sites_resampled_)
    real = ~sampler.is_synthetic_
    assert_array_equal(out.loc[real].to_numpy(), df.to_numpy()[sampler.sample_indices_[real]])
    assert sampler.is_synthetic_.any()
    synthetic = out.loc[sampler.is_synthetic_]
    assert synthetic.notna().all().all()
    assert synthetic["age"].between(df["age"].min(), df["age"].max()).all()
    assert set(synthetic["sex"]) <= {"F", "M"}
    # Same features as the keyword path
    keyword = sklearn.clone(SAMPLERS[name]).set_params(site_column=None, passthrough_columns=None)
    X_res, _ = keyword.fit_resample(df[FEATURES].to_numpy(), y, sites=df["site"].to_numpy())
    assert_allclose(out[FEATURES].to_numpy(), X_res)


def test_isi_parents_reproduce_the_synthetic_samples(frame) -> None:
    """The parents found for each SMOTE sample reproduce it, so its covariates are interpolated consistently."""
    df, y = frame
    isi = IntraSiteInterpolation(site_column="site", passthrough_columns=["age"], random_state=0)
    out, _ = isi.fit_resample(df.drop(columns="sex"), y)
    weights = isi._parent_weights(len(df))
    X = df[FEATURES].to_numpy()
    assert_allclose(weights @ X, out.loc[isi.is_synthetic_, FEATURES].to_numpy(), atol=1e-8)
    assert_allclose(weights @ df["age"].to_numpy(), out.loc[isi.is_synthetic_, "age"].to_numpy())
    # Parents come from the site of the synthetic sample
    parents = weights.tocoo()
    assert_array_equal(df["site"].to_numpy()[parents.col], out.loc[isi.is_synthetic_, "site"].to_numpy()[parents.row])


def test_ismi_parents_reproduce_the_synthetic_samples(frame) -> None:
    """ISMI interpolates the continuous metadata with the alpha of the features."""
    df, y = frame
    ismi = InterSiteMatchedInterpolation(k="average", site_column="site", passthrough_columns="age", random_state=0)
    out, _ = ismi.fit_resample(df.drop(columns="sex"), y)
    synthetic = out.loc[ismi.is_synthetic_]
    # Features and age are the same convex combinations of the input rows
    X_age = df[[*FEATURES, "age"]].to_numpy()
    # One block of synthetic samples per (site, other sites) pair, rows numbered within the block
    offset = np.cumsum([0] + [int(rows.max()) + 1 for rows, _, _ in ismi._parent_entries])
    weights = np.zeros((len(synthetic), len(df)))
    for start, (rows, parents, values) in zip(offset, ismi._parent_entries, strict=False):
        np.add.at(weights, (rows + start, parents), values)
    assert_allclose(weights @ X_age, synthetic[[*FEATURES, "age"]].to_numpy(), atol=1e-10)


@pytest.mark.parametrize("name", list(SAMPLERS))
def test_sampler_then_harmonizer_in_imblearn_pipeline(frame, name: str) -> None:
    """A sampler followed by a harmonizer in an imbalanced-learn Pipeline, evaluated with a scorer."""
    df, y = frame
    pipe = ImbPipeline(
        [
            ("sampler", sklearn.clone(SAMPLERS[name])),
            ("combat", NeuroComBat(site_column="site", categorical_columns="sex", continuous_columns="age")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression()),
        ]
    )
    scores = cross_validate(pipe, df, y, cv=CV, scoring="roc_auc", error_score="raise")
    assert np.all(np.isfinite(scores["test_score"]))


# ---------------------------------------------------------------------------
# julearn
# ---------------------------------------------------------------------------


def test_julearn_step(frame) -> None:
    """A harmonizer is a julearn pipeline step; julearn's typed column names are matched."""
    julearn = pytest.importorskip("julearn")
    df, y = frame
    data = df.drop(columns="sex").assign(target=y)
    creator = julearn.PipelineCreator(problem_type="classification", apply_to="continuous")
    creator.add(NeuroComBat(site_column="site", continuous_columns="age"), apply_to=["continuous", "site", "confound"])
    creator.add("zscore")
    creator.add("logit")
    scores = julearn.run_cross_validation(
        X=[*FEATURES, "site", "age"],
        y="target",
        data=data,
        X_types={"continuous": FEATURES, "site": ["site"], "confound": ["age"]},
        model=creator,
        cv=CV,
        scoring="roc_auc",
    )
    assert np.all(np.isfinite(scores["test_score"]))
    assert scores["test_score"].mean() > 0.6
