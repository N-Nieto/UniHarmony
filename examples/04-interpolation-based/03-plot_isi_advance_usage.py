"""
IntraSiteInterpolation advance usage
====================================

Interpolation cannot create information: a handful of samples cannot be interpolated into thousands. ISI therefore
limits how many synthetic samples each class of each site receives (``max_amplification``) and closes the remaining
imbalance with an under-sampler (``undersampler``), so both meet in the middle.
"""

# %%
# Imports
# -------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

from uniharmony import verbosity
from uniharmony.datasets import make_multisite_classification
from uniharmony.interpolation import IntraSiteInterpolation


sns.set_theme(style="whitegrid")
verbosity("warning")

X, y, sites = make_multisite_classification(
    n_sites=3,
    n_samples=[400, 400, 400],
    n_features=5,
    balance_per_site=[[0.8, 0.2], [0.5, 0.5], [0.3, 0.7]],
    random_state=23,
)

# %%
# How much interpolation is allowed?
# ----------------------------------
#
# With ``max_amplification="auto"`` (default), ISI measures for every site and class how much of the real variance the
# synthetic samples keep (``variance_ratio``) and caps the amplification so that the class keeps at least 90% of its
# variance (``variance_tolerance=0.1``). ``summary()`` reports what happened in every site and class.

isi = IntraSiteInterpolation(interpolator="smote", random_state=42)
X_bal, y_bal = isi.fit_resample(X, y, sites=sites)
isi.summary().round(2)

# %%
# Different caps give different trade-offs between interpolation and under-sampling:
# ``None`` only over-samples, ``0`` only under-samples.

rows = []
for cap in [None, "auto", 1.0, 0]:
    isi_cap = IntraSiteInterpolation(interpolator="smote", max_amplification=cap, random_state=42)
    _, y_cap = isi_cap.fit_resample(X, y, sites=sites)
    report = isi_cap.summary()
    rows.append(
        {
            "max_amplification": str(cap),
            "real samples kept": int((report.n_real - report.n_removed).sum()),
            "synthetic samples": int(report.n_created.sum()),
        }
    )
df = pd.DataFrame(rows).melt(id_vars="max_amplification", var_name="samples", value_name="n")
plt.figure(figsize=[10, 6])
plt.title("Composition of the balanced training set")
sns.barplot(df, x="max_amplification", y="n", hue="samples")

# %%
# Global balancing
# ----------------
#
# ``balance_strategy="global_max"`` balances every site towards the largest class of any site, so sites also get
# similar sizes (within the amplification cap).

isi_global = IntraSiteInterpolation(balance_strategy="global_max", max_amplification=None, random_state=42)
_, y_global = isi_global.fit_resample(X, y, sites=sites)
print(f"Global target count: {isi_global.target_count_}, per site: {isi_global.target_counts_}")

# %%
# Choosing the samplers
# ---------------------
#
# Any imblearn over-sampler can be used to interpolate, and any under-sampler that accepts a target count per class to
# remove samples.

isi_nm = IntraSiteInterpolation(interpolator="borderline-smote", undersampler="nearmiss", max_amplification=1.0)
X_nm, y_nm = isi_nm.fit_resample(X, y, sites=sites)
print(isi_nm.target_counts_)

# %%
# Covariates
# ----------
#
# Covariates restrict who is interpolated with whom: synthetic samples are interpolated between samples of the same
# class, site and covariate stratum (here sex and three age bins), and are spread over the strata like the real samples
# of their class.

rng = np.random.default_rng(54)
sex = rng.integers(0, 2, len(y))
age = rng.normal(50, 10, len(y))
isi_cov = IntraSiteInterpolation(n_bins_cont_cov=3, random_state=42)
X_cov, y_cov = isi_cov.fit_resample(X, y, sites=sites, categorical_covariate=sex, continuous_covariate=age)

# The covariates of the real samples kept can be recovered with ``sample_indices_``
real = ~isi_cov.is_synthetic_
sex_kept = sex[isi_cov.sample_indices_[real]]

# %%
# Use on training data only
# -------------------------
#
# ISI must only see the training folds. Put it in an ``imblearn`` pipeline and route ``sites`` with scikit-learn
# metadata routing.

with sklearn.config_context(enable_metadata_routing=True):
    isi_cv = IntraSiteInterpolation(random_state=0).set_fit_resample_request(sites=True)
    pipe = Pipeline([("isi", isi_cv), ("clf", LogisticRegression())])
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    scores = cross_validate(pipe, X, y, cv=cv, params={"sites": sites}, scoring="roc_auc")
print(f"CV AUC with ISI inside the folds: {scores['test_score'].mean():.3f}")

###############################################################################
# .. admonition:: Take-home message
#
#    ISI interpolates only as much as the data can support and under-samples the rest, so every site ends up
#    class-balanced without inventing more data than the real samples can span.
