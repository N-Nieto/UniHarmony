# 🔗 Using UniHarmony in pipelines

A harmonizer needs per-sample **metadata** — the **site** of every sample and,
often, **covariates** whose effect must be preserved — both when it is fitted
*and* when it transforms new data: the site effect of a test subject can only
be removed if its site is known.

scikit-learn only moves `X` and `y` through a {class}`~sklearn.pipeline.Pipeline`,
and only `X` and `y` are split by the folds of its model selection tools
({func}`~sklearn.model_selection.cross_validate`,
{class}`~sklearn.model_selection.GridSearchCV`,
{func}`~sklearn.model_selection.cross_val_predict`, the scorers, ...).
UniHarmony therefore accepts the metadata in two ways.

## Two ways to give the sites and covariates

### 1. As columns of `X` (recommended in pipelines)

Name the metadata columns in the constructor and pass a pandas DataFrame:

```python
from uniharmony.combat import NeuroComBat

combat = NeuroComBat(site_column="site", categorical_columns=["sex"], continuous_columns=["age"])
X_harmonized = combat.fit(df_train).transform(df_test)
```

- The columns are read from `X`, checked, and **removed from the output**: the
  transformer returns the harmonized features only.
- The folds of every scikit-learn tool split the sites together with the
  features, so the test sites always reach `transform` and every scorer.
- The sites cannot be forgotten or misaligned at prediction time.
- With a NumPy array, give column **positions** instead of names
  (`site_column=0`).

### 2. As keywords of `fit` and `transform`

```python
combat = NeuroComBat()
combat.fit(X_train, sites=sites_train, continuous_covariates=age_train)
X_harmonized = combat.transform(X_test, sites=sites_test, continuous_covariates=age_test)
```

This is convenient for direct calls on NumPy arrays. The keywords are
keyword-only: they can never land in the slot of `y`.

### Rules shared by every estimator

| | |
|---|---|
| Signature | `fit(X, y=None, *, sites=None, <covariate keywords>)`, `transform(X, *, sites=None, ...)`. `y` is the second argument, as in scikit-learn; harmonizers that do not use it ignore it. |
| One source per metadata | Giving the same metadata as a column and as a keyword raises an error. |
| Settings | All numerical options (`var_epsilon`, `max_iter`, `df`, `degree`, `smooth_covariates_bounds`, `ref_site` of the ComBat models and of OTDA, `batch_size`, ...) are constructor parameters, so `GridSearchCV` can tune them and `clone` copies them. |
| `fit_transform` | Equals `fit(X, y, **metadata).transform(X, **metadata)`. |
| Feature names | `n_features_in_` / `feature_names_in_` describe the full input (metadata columns included); `get_feature_names_out()` lists the harmonized features, so `set_output(transform="pandas")` returns a DataFrame with the feature names and the index of `X`. |
| Checks | Missing or infinite values, inconsistent lengths, unseen sites, and covariates given to `fit` but not to `transform` (or the reverse) raise informative errors. |
| julearn | A column renamed by julearn (`site__:type:__site`) is found by its plain name (`site_column="site"`). |

The column and keyword parameters of each estimator:

| Estimator | Constructor (columns) | `fit` / `transform` (keywords) |
|---|---|---|
| {class}`~uniharmony.combat.NeuroComBat`, {class}`~uniharmony.combat.CovBat` | `site_column`, `categorical_columns`, `continuous_columns` | `sites`, `categorical_covariates`, `continuous_covariates` |
| {class}`~uniharmony.combat.ComBatGAM` | `site_column`, `smooth_columns`, `continuous_columns` | `sites`, `smooth_covariates`, `continuous_covariates` |
| {class}`~uniharmony.ot.OptimalTransportDomainAdaptation` | `site_column` (and `ref_site`, `use_labels`) | `sites` |
| {class}`~uniharmony.iqm.BARTharm` | `iqm_columns`, `biological_columns`, `site_column` | `iqm_covariates`, `biological_covariates`, `sites` |
| {class}`~uniharmony.prettyharmonize.PrettYharmonizeClassifier` / `Regressor` | the column parameters of its `harm_model` | the keywords of its `harm_model` |
| {class}`~uniharmony.interpolation.IntraSiteInterpolation`, {class}`~uniharmony.interpolation.SiteBalancedResampler`, {class}`~uniharmony.interpolation.InterSiteMatchedInterpolation` | `site_column`, `categorical_columns`, `continuous_columns`, `passthrough_columns` | `sites`, `categorical_covariates`, `continuous_covariates` |

## scikit-learn pipelines

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pipe = make_pipeline(NeuroComBat(site_column="site"), StandardScaler(), LogisticRegression())
cv = StratifiedKFold(5, shuffle=True, random_state=0)
scores = cross_validate(pipe, df, y, cv=cv, scoring=["roc_auc", "balanced_accuracy"])

search = GridSearchCV(pipe, {"neurocombat__empirical_bayes": [True, False], "logisticregression__C": [0.1, 1, 10]},
                      cv=cv, scoring="roc_auc")
search.fit(df, y)
```

The harmonizer must come before any step that does not know about the metadata
columns (or those steps must skip them, e.g. with a
{class}`~sklearn.compose.ColumnTransformer`).

### Why not metadata routing?

scikit-learn can route extra arguments (`sklearn.set_config(enable_metadata_routing=True)`;
UniHarmony's estimators request their data keywords by default). Tested with
scikit-learn 1.9, routing is not enough for model evaluation:

| Tool | Keywords + metadata routing | Columns of `X` |
|---|---|---|
| `Pipeline.fit` / `predict` with the keywords | works | works |
| `cross_validate` with the default `score` | works | works |
| `cross_validate(scoring="roc_auc")`, any scorer | **NaN scores** (scorers call `predict` without the sites; the error becomes `error_score=nan` with a warning only) | works |
| `GridSearchCV(scoring=...)` | **`best_score_` is NaN** | works |
| `cross_val_predict` | **error** (no routing to `predict`) | works |
| imbalanced-learn sampler followed by a harmonizer | the harmonizer cannot get the sites of the synthetic samples | works |

The column path works with every tool, so it is the recommended way to use
UniHarmony in pipelines. The keywords remain for direct calls.

## Samplers followed by a harmonizer (imbalanced-learn)

The samplers need an {class}`imblearn.pipeline.Pipeline` (scikit-learn's
`Pipeline` has no `fit_resample`). With the column path they return `X` with
its metadata columns, filled in for the synthetic samples, so a harmonizer can
follow them:

```python
from imblearn.pipeline import Pipeline
from uniharmony.interpolation import IntraSiteInterpolation

pipe = Pipeline([
    ("isi", IntraSiteInterpolation(site_column="site", passthrough_columns=["age"])),
    ("combat", NeuroComBat(site_column="site", continuous_columns=["age"])),
    ("scale", StandardScaler()),
    ("clf", LogisticRegression()),
])
```

For a synthetic sample:

- the **site** column holds the site it was created in;
- **continuous** covariates (and floating-point `passthrough_columns`) are
  interpolated with the weights of its features: for SMOTE-type samplers the
  two parents and the position on their segment are recovered from the
  features, for {class}`~uniharmony.interpolation.InterSiteMatchedInterpolation`
  the matched samples and `alpha` are used directly;
- **categorical** covariates (and other `passthrough_columns`) are copied from
  the parent with the largest weight.

`passthrough_columns` are carried to the output without being used for
resampling, e.g. the covariates that only the harmonizer needs.

## julearn

A harmonizer is a step of a julearn
[`PipelineCreator`](https://juaml.github.io/julearn/); give it the column types
it needs with `apply_to`. julearn renames the columns (`site__:type:__site`);
the plain names given to the harmonizer are matched:

```python
from julearn import PipelineCreator, run_cross_validation

creator = PipelineCreator(problem_type="classification", apply_to="continuous")
creator.add(NeuroComBat(site_column="site", continuous_columns="age"), apply_to=["continuous", "site", "confound"])
creator.add("zscore")
creator.add("logit")
scores = run_cross_validation(
    X=[*features, "site", "age"], y="target", data=df,
    X_types={"continuous": features, "site": ["site"], "confound": ["age"]},
    model=creator, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc",
)
```

## Cross-validation and sites

A harmonizer can only harmonize sites it has seen during `fit`. Every site of a
test fold must therefore also be in the training folds:

- **Shuffle the folds.** With an integer `cv`, julearn and some scikit-learn
  tools use an unshuffled `KFold`. On data sorted by site, a whole site then
  lands in one test fold, and harmonization fails with an error that says so.
  Pass a shuffled splitter, e.g. `StratifiedKFold(5, shuffle=True)`.
- **Stratify on the sites** (or on site × target) to keep every site in every
  training fold, e.g. `StratifiedKFold(5, shuffle=True).split(X, sites)`.
- To evaluate on **unseen sites** (leave-site-out), a site-wise harmonizer
  such as ComBat is not applicable: it has no estimate for the new site.

## Migrating from UniHarmony 0.0.5

Version 0.0.6 changes the calling convention of every harmonizer and sampler:

| 0.0.5 | 0.0.6 |
|---|---|
| `NeuroComBat().fit(X, sites, categorical_covariates=c)` | `NeuroComBat().fit(X, sites=sites, categorical_covariates=c)` or `NeuroComBat(site_column="site", categorical_columns="c").fit(df)` |
| `fit(X, sites, var_epsilon=..., max_iter=...)` | `NeuroComBat(var_epsilon=..., max_iter=...)` |
| `ComBatGAM().fit(X, sites, age, df=5, smooth_covariates_bounds=(0, 100))` | `ComBatGAM(df=5, smooth_covariates_bounds=(0, 100)).fit(X, sites=sites, smooth_covariates=age)` |
| `OptimalTransportDomainAdaptation().fit(X, sites, ref_site="A", y=labels)` | `OptimalTransportDomainAdaptation(ref_site="A", use_labels=True).fit(X, labels, sites=sites)` |
| `otda.transform(X, sites, batch_size=64)` | `OptimalTransportDomainAdaptation(..., batch_size=64)`, `otda.transform(X, sites=sites)` |
| `BARTharm().fit(X, iqms, bio, sites)` | `BARTharm().fit(X, iqm_covariates=iqms, biological_covariates=bio, sites=sites)` |
| `PrettYharmonizeClassifier().fit(X, y, sites, {"continuous_covariates": age})` | `PrettYharmonizeClassifier().fit(X, y, sites=sites, continuous_covariates=age)` |
| `clf.score(X, y, sites)` | `clf.score(X, y, sites=sites)` |
| `isi.fit_resample(X, y, sites, categorical_covariate=sex)` | `isi.fit_resample(X, y, sites=sites, categorical_covariates=sex)` |
| `ismi.fit_resample(..., allow_nan=True)` | `InterSiteMatchedInterpolation(allow_nan=True)` |
| `isi.fit_resample(..., n_bins_cont_cov=3)` | `IntraSiteInterpolation(n_bins_cont_cov=3)` |
