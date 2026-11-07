"""
Multisite data generation with custom covariates
================================================

A :class:`~uniharmony.datasets.Covariate` sets the distribution of a
covariate in every site and its effect on the features. Here the two sites
recruit very different populations, so part of the difference between the
sites is biology (age and sex), not a site effect: the covariates are
confounded with the site.
"""

# %%
# Imports
# -------

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from uniharmony import verbosity
from uniharmony.datasets import Covariate, CovariateSiteDistribution, make_multisite_classification


sns.set_theme(style="whitegrid")
verbosity("warning")

# %%
# Two different populations
# -------------------------
#
# Site 0 recruits children (age 10) who are mostly of sex 1; site 1 recruits
# older adults (age 70) who are mostly of sex 0. Age lowers every feature by
# 0.5 within-class SD per SD of age, and sex 1 raises them by 0.4 SD; these
# effects are the same in both sites.

covariates = [
    Covariate(
        name="age",
        site_distributions=[
            CovariateSiteDistribution(loc=10.0, scale=1.0),
            CovariateSiteDistribution(loc=70.0, scale=1.0),
        ],
        effect=-0.5,
    ),
    Covariate(
        name="sex",
        site_distributions=[
            CovariateSiteDistribution(probs=[0.1, 0.9]),
            CovariateSiteDistribution(probs=[0.9, 0.1]),
        ],
        effect=0.4,
    ),
]
X, y, sites, truth = make_multisite_classification(
    signal={"blobs": 1.0}, site_effect={"location": 1}, covariates=covariates, return_ground_truth=True
)
covars = truth["covariates"]
print(f"X has {X.shape[0]} examples and {X.shape[1]} features")

df = pd.DataFrame({"Age": covars["age"], "Feature 1": X[:, 0], "Sex": covars["sex"], "Site": sites})
plt.figure(figsize=[10, 6])
plt.title("Feature 1 against age, by sex and site")
sns.scatterplot(df, y="Feature 1", x="Age", hue="Sex", style="Site", alpha=0.7)
plt.show()

# %%
# What makes the sites differ?
# ----------------------------
#
# The ground truth splits the difference between the site means of each
# feature into the part due to the covariates and the part due to the site
# effect. Here most of it is biology: a harmonizer that removed the whole
# difference would also remove the effects of age and sex. Give such
# covariates to the harmonizer (e.g. ``NeuroComBat(...).fit(X, sites=sites,
# continuous_covariates=age, categorical_covariates=sex)``) so that it keeps
# them.


def site_difference(values):
    """Mean of site 1 minus mean of site 0, per feature, in within-class SD."""
    return (values[sites == 1].mean(axis=0) - values[sites == 0].mean(axis=0)) / truth["feature_scale"]


decomposition = pd.DataFrame(
    {
        "total": site_difference(X),
        "covariates (age, sex)": site_difference(truth["covariate_effect"]),
        "site effect": site_difference(truth["site_effect"]),
    }
).round(2)
decomposition.index.name = "feature"
print(decomposition)
