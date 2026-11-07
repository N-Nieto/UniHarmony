"""
ComBatGAM with covariates imbalanced across sites
=================================================

When age and sex are distributed differently in each site, their effects on
the features make the sites differ, on top of the site effect. Harmonizing
without them treats these differences as site effects and removes them;
giving them to ComBatGAM (age as a smooth term, sex as a linear term) keeps
them.

The simulator's ground truth gives the true age slope and sex difference,
and the data without site effect (``site_free``), to check each result.
"""

# %%
# Imports
# -------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from uniharmony import verbosity
from uniharmony.combat import ComBatGAM, NeuroComBat
from uniharmony.datasets import Covariate, CovariateSiteDistribution, make_multisite_classification


verbosity("error")
sns.set_theme(style="whitegrid")

# %%
# Data generation
# ---------------
#
# Site 0 is younger (mean age 40) and mostly of sex 1; site 1 is older (mean
# age 70) and mostly of sex 0. In both sites, age lowers the features and sex
# 1 raises them (in units of the features' within-class SD).

covariates = [
    Covariate(
        name="age",
        site_distributions=[
            CovariateSiteDistribution(loc=40.0, scale=10),
            CovariateSiteDistribution(loc=70.0, scale=10),
        ],
        effect=-0.5,
    ),
    Covariate(
        name="sex",
        site_distributions=[
            CovariateSiteDistribution(probs=[0.1, 0.9]),
            CovariateSiteDistribution(probs=[0.9, 0.1]),
        ],
        effect=0.8,
    ),
]
X, y, sites, truth = make_multisite_classification(
    n_features=2,
    signal={"blobs": 1.0},
    site_effect={"location": 2},
    covariates=covariates,
    return_ground_truth=True,
    random_state=0,
)
covars = truth["covariates"]
age, sex = covars["age"], covars["sex"]

###############################################################################
# .. caution::
#
#    The harmonizers are fitted on the whole dataset to illustrate their
#    effect. In a machine-learning evaluation, fit them on the training data
#    only.

# %%
# Harmonization
# -------------

harmonized = {
    "Raw": X,
    "NeuroComBat, no covariates": NeuroComBat().fit_transform(X, sites=sites),
    "ComBatGAM, age + sex": ComBatGAM(random_state=0).fit_transform(
        X, sites=sites, smooth_covariates=age, continuous_covariates=sex
    ),
}

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True, sharey=True)
for ax, (name, data) in zip(axes, harmonized.items(), strict=True):
    df = pd.DataFrame({"Age": age, "Feature 1": data[:, 0], "Sex": sex, "Site": sites})
    sns.scatterplot(df, x="Age", y="Feature 1", hue="Sex", style="Site", alpha=0.6, s=15, ax=ax)
    ax.set_title(name)
plt.tight_layout()
plt.show()

# %%
# Are the effects of age and sex kept?
# ------------------------------------
#
# A linear model ``Feature 1 ~ age + sex`` estimates the age slope and the sex
# difference. On the data without site effect, it recovers the true values.
# On the raw data, the estimates also depend on the site effect, which is
# confounded with age and sex: in this draw it happens to bias them little,
# but over 12 random seeds their median absolute error is 0.41, against 0.04
# for ComBatGAM. Harmonizing without the covariates removes most of the
# effects; ComBatGAM with the covariates keeps them and is closest to the
# data without site effect (in all 12 seeds).

design = np.column_stack([np.ones_like(age), age, sex])
rows = {}
for name, data in {"Without site effect (truth)": truth["site_free"], **harmonized}.items():
    _, slope, difference = np.linalg.lstsq(design, data[:, 0], rcond=None)[0]
    rmse = np.sqrt(((data - truth["site_free"]) ** 2).mean(axis=0) / truth["feature_scale"] ** 2).mean()
    rows[name] = {"Age slope (per 10 years)": 10 * slope, "Sex difference": difference, "RMSE to truth (SD)": rmse}
print(pd.DataFrame(rows).T.round(2).to_string())

###############################################################################
# .. admonition:: Take-home message
#
#    Covariates that differ between sites are confounded with the site
#    effect. Give them to the harmonizer, otherwise their effects are removed
#    together with the site effect.
