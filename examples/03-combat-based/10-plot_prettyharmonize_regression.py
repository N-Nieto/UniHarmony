"""
PrettYharmonize for regression
==============================

:class:`~uniharmony.prettyharmonize.PrettYharmonizeRegressor` harmonizes every
sample once per *regression point* (pretended target values spread over the
training range), or searches a self-consistent target per sample
(``regression_search=True``). Here sites cover different age ranges, so
ComBat must preserve age, which is also the target to predict.
"""

# %%
# Imports
# -------
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from uniharmony import verbosity
from uniharmony.combat import NeuroComBat
from uniharmony.prettyharmonize import PrettYharmonizeRegressor


verbosity("error")
sns.set_theme(style="whitegrid")

# %%
# Data: three sites with different age ranges
# -------------------------------------------
# Each feature depends linearly on age, plus a site location/scale effect.
rng = np.random.default_rng(0)
n_samples, n_features, n_sites = 600, 20, 3
sites = rng.permutation(np.arange(n_samples) % n_sites)
age = rng.uniform(20, 50, n_samples) + 15 * sites  # site 0: 20-50, site 1: 35-65, site 2: 50-80
weights = rng.normal(scale=0.05, size=n_features)
location = rng.normal(scale=1.5, size=(n_sites, n_features))
scale = np.exp(rng.normal(scale=0.4, size=(n_sites, n_features)))
X = np.outer(age, weights) + location[sites] + scale[sites] * rng.normal(size=(n_samples, n_features))

X_train, X_test, age_train, age_test, sites_train, sites_test = train_test_split(
    X, age, sites, test_size=0.3, random_state=0, stratify=sites
)

# %%
# PrettYharmonize
# ---------------
reg = PrettYharmonizeRegressor(harm_model=NeuroComBat(), regression_points=10, random_state=0)
reg.fit(X_train, age_train, sites_train)
print("pretended ages:", reg.regression_points_.round(1))
pred = reg.predict(X_test, sites_test)

search = PrettYharmonizeRegressor(regression_search=True, random_state=0).fit(X_train, age_train, sites_train)
pred_search = search.predict(X_test, sites_test)


# %%
# ComBat without age (leakage-free, removes the age signal)
# ---------------------------------------------------------
def ridge():
    """Regression model."""
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))


combat = NeuroComBat().fit(X_train, sites_train)
pred_no_target = ridge().fit(combat.transform(X_train, sites_train), age_train).predict(combat.transform(X_test, sites_test))

for name, p in [("ComBat without age", pred_no_target), ("PrettYharmonize", pred), ("PrettYharmonize (search)", pred_search)]:
    print(f"{name:26s} MAE = {mean_absolute_error(age_test, p):5.2f} years")

# %%
# Predicted vs true age
# ---------------------
fig, ax = plt.subplots(figsize=(5, 4.5))
ax.scatter(age_test, pred_no_target, s=14, alpha=0.6, marker="s", label="ComBat without age")
ax.scatter(age_test, pred, s=14, alpha=0.8, label="PrettYharmonize")
ax.plot([20, 80], [20, 80], color="gray", lw=1, ls=":")
ax.set_xlabel("True age")
ax.set_ylabel("Predicted age")
ax.legend()
plt.tight_layout()

# %%
