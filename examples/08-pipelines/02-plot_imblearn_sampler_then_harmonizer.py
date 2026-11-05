"""
Sampler followed by a harmonizer (imbalanced-learn)
===================================================

When the class balance differs across sites, ComBat-type harmonizers face a
dilemma: without the class as a covariate they remove part of the class signal
with the site effect; with it they need the class of the test subjects
(leakage). Balancing the classes within every site first, with
:class:`~uniharmony.interpolation.IntraSiteInterpolation` (ISI), removes the
dependence between site and class, so the harmonizer can be fitted without the
class.

The two steps form an imbalanced-learn :class:`~imblearn.pipeline.Pipeline`
(scikit-learn's pipeline cannot resample). The harmonizer needs the site, and
here the age, of every **resampled** sample, including the synthetic ones that
ISI creates. Metadata routing cannot provide them (it only routes the metadata
of the input rows). With the sites and covariates as **columns** of ``X``,
the sampler returns them for every resampled sample:

* the site column holds the site the synthetic sample was created in;
* continuous columns are interpolated between the two parents of the sample,
  with the same weights as its features;
* other columns are copied from the closer parent.

Terms used below:

* **synthetic sample**: a sample created by interpolation between two real
  samples (its **parents**) of the same site and class.
* **passthrough column**: a column that is not a feature and not used by the
  sampler, but carried to its output (here age, needed by the harmonizer).
"""

# %%
# Imports
# -------

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler

from uniharmony import verbosity
from uniharmony.combat import NeuroComBat
from uniharmony.datasets import make_multisite_classification
from uniharmony.interpolation import IntraSiteInterpolation


verbosity("error")
sns.set_theme(style="whitegrid")

# %%
# Data
# ----

X, y, sites, covariates = make_multisite_classification(
    n_sites=3,
    n_samples=300,
    n_features=5,
    balance_per_site=[[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]],
    site_effect_type="location+scale",
    covariates=["age"],
    random_state=1,
)
features = [f"feature_{j}" for j in range(X.shape[1])]
df = pd.DataFrame(X, columns=features).assign(site=sites, age=covariates["age"])
pd.crosstab(df["site"], y, margins=True)

# %%
# ISI warns when a class needs more interpolation than keeps its variance (see
# the "IntraSiteInterpolation advance usage" example). The warning is silenced
# here to keep the output short; ``SiteBalancedResampler`` is the remedy.

warnings.filterwarnings("ignore", message="IntraSiteInterpolation: .* too small")

# %%
# What the sampler returns
# ------------------------
#
# ``site_column`` names the site column; ``passthrough_columns`` the columns
# that must follow the samples. The output has the columns of the input.

isi = IntraSiteInterpolation(site_column="site", passthrough_columns=["age"], random_state=0)
df_balanced, y_balanced = isi.fit_resample(df, y)
print(pd.crosstab(df_balanced["site"], y_balanced, margins=True))
df_balanced[isi.is_synthetic_].head()

# %%
# The parents of every synthetic sample are recovered from its features (SMOTE
# puts it on the segment between them). The parent weights reproduce the
# synthetic features, and give its age:

weights = isi._parent_weights(len(df))
synthetic = df_balanced[isi.is_synthetic_]
print("Largest feature error of the parents:", np.abs(weights @ X - synthetic[features].to_numpy()).max())

fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(df["age"], df["feature_0"], s=12, alpha=0.6, label="real")
ax.scatter(synthetic["age"], synthetic["feature_0"], s=12, alpha=0.6, label="synthetic")
ax.set(xlabel="age", ylabel="feature_0", title="Synthetic samples keep the age-feature relation")
ax.legend()
plt.show()

# %%
# The pipeline
# ------------
#
# ISI balances the training folds only; NeuroComBat is fitted on the balanced
# training data (with age as covariate, without the class) and harmonizes the
# test fold with its own sites and ages.

pipe = Pipeline(
    [
        ("isi", IntraSiteInterpolation(site_column="site", passthrough_columns=["age"], random_state=0)),
        ("combat", NeuroComBat(site_column="site", continuous_columns=["age"])),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression()),
    ]
)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
scores = cross_validate(pipe, df, y, cv=cv, scoring="roc_auc")
print(f"ISI + NeuroComBat, ROC AUC: {scores['test_score'].mean():.3f} ± {scores['test_score'].std():.3f}")

# %%
# For comparison, NeuroComBat without ISI (and without the class as covariate):

baseline = Pipeline(
    [("combat", NeuroComBat(site_column="site", continuous_columns=["age"])), ("scale", StandardScaler()), ("clf", LogisticRegression())]
)
scores_baseline = cross_validate(baseline, df, y, cv=cv, scoring="roc_auc")
print(f"NeuroComBat alone, ROC AUC: {scores_baseline['test_score'].mean():.3f} ± {scores_baseline['test_score'].std():.3f}")

###############################################################################
# .. admonition:: Take-home message
#
#    With the sites and covariates as columns, a sampler returns them for every
#    resampled sample (synthetic samples included), so a harmonizer can follow
#    it in an imbalanced-learn pipeline. Use ``passthrough_columns`` for the
#    columns that only later steps need. ``SiteBalancedResampler`` and
#    ``InterSiteMatchedInterpolation`` accept the same column parameters.
