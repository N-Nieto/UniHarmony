"""
Harmonization as a julearn pipeline step
========================================

`julearn <https://juaml.github.io/julearn/>`_ builds scikit-learn pipelines
from typed columns: every column of the data gets a type (``continuous``,
``confound``, ``site``, ...) and every step is applied to some types. A
UniHarmony harmonizer is a julearn step: it receives the feature, site and
covariate columns and returns the harmonized features.

julearn renames the columns to ``<name>__:type:__<type>``; the harmonizer
finds them by their plain names (``site_column="site"``).

Terms used below:

* **X_types**: the type of every column of ``X``.
* **apply_to**: the column types a step receives.
"""

# %%
# Imports
# -------

import warnings

import pandas as pd
from julearn import PipelineCreator, run_cross_validation
from sklearn.model_selection import KFold, StratifiedKFold

from uniharmony import verbosity
from uniharmony.combat import NeuroComBat
from uniharmony.datasets import make_multisite_classification


verbosity("error")

# %%
# Data
# ----

X, y, sites, covariates = make_multisite_classification(
    n_sites=3,
    n_samples=300,
    n_features=5,
    balance_per_site=[[0.7, 0.3], [0.5, 0.5], [0.3, 0.7]],
    site_effect_type="location+scale",
    covariates=["age"],
    random_state=2,
)
features = [f"feature_{j}" for j in range(X.shape[1])]
data = pd.DataFrame(X, columns=features).assign(site=sites, age=covariates["age"], target=y)

# %%
# The pipeline
# ------------
#
# The harmonizer receives the features (``continuous``), the site and the age
# (``confound``); it returns the harmonized features, which the next steps
# (``zscore``, ``logit``) receive.

creator = PipelineCreator(problem_type="classification", apply_to="continuous")
creator.add(NeuroComBat(site_column="site", continuous_columns="age"), apply_to=["continuous", "site", "confound"])
creator.add("zscore")
creator.add("logit")

# %%
# Cross-validation
# ----------------
#
# Every site of a test fold must also be in the training folds. With an integer
# ``cv``, julearn uses an **unshuffled** ``KFold``, which here (data sorted by
# site) would put each site in its own test fold. Pass a shuffled, stratified
# splitter instead:

scores = run_cross_validation(
    X=[*features, "site", "age"],
    y="target",
    data=data,
    X_types={"continuous": features, "site": ["site"], "confound": ["age"]},
    model=creator,
    cv=StratifiedKFold(5, shuffle=True, random_state=0),
    scoring="roc_auc",
)
print(f"ROC AUC: {scores['test_score'].mean():.3f} ± {scores['test_score'].std():.3f}")

# %%
# What goes wrong with unshuffled folds: each test fold is a site that the
# harmonizer has never seen, and it says so.

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    unshuffled = run_cross_validation(
        X=[*features, "site", "age"],
        y="target",
        data=data,
        X_types={"continuous": features, "site": ["site"], "confound": ["age"]},
        model=creator,
        cv=KFold(3),
        scoring="roc_auc",
    )
print("Scores with KFold(3):", unshuffled["test_score"].to_numpy())
message = next(str(w.message) for w in caught if "not seen during fit" in str(w.message))
print(message[message.index("NeuroComBat: sites") :][:330], "...")

###############################################################################
# .. admonition:: Take-home message
#
#    Add the harmonizer to a julearn ``PipelineCreator`` with the column types
#    it needs (``apply_to``) and name its columns by their plain names. Use a
#    shuffled, stratified splitter so that every test site is a training site.
