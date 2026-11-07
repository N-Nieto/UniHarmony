"""
ComBatGAM preserves a nonlinear age effect
==========================================

When the sites differ in age, age and site are confounded: part of the
difference between sites is biology. A harmonizer that does not know about
age removes these differences as if they were site effects. Giving age as a
covariate keeps its effect; ComBatGAM models it with a smooth function, so it
also keeps a nonlinear effect that a linear covariate (NeuroComBat) only
approximates.

The simulator returns the data without site effect (``site_free``), so we can
measure exactly how close each harmonization gets to it.
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


verbosity("warning")
sns.set_theme(style="whitegrid")

# %%
# Data
# ----
#
# Three sites recruit young, middle-aged and old participants. Age decreases
# every feature, faster and faster with age (the same in every site), and
# each site adds its own location and scale effect.

age = Covariate(
    name="age",
    site_distributions=[
        CovariateSiteDistribution(loc=30, scale=6, clip=(18, 90)),
        CovariateSiteDistribution(loc=50, scale=6, clip=(18, 90)),
        CovariateSiteDistribution(loc=70, scale=6, clip=(18, 90)),
    ],
    effect=-0.8,  # in units of the feature's within-class SD
    effect_shape=lambda years: np.exp(years / 20),  # accelerating decline
)
X, y, sites, truth = make_multisite_classification(
    n_sites=3,
    n_samples=900,
    n_features=4,
    signal={"blobs": 0.5},
    site_effect={"location": 2, "scale": 3},
    covariates=[age],
    return_ground_truth=True,
    random_state=0,
)
covariates = truth["covariates"]
ages = covariates["age"]

###############################################################################
# .. caution::
#
#    The harmonizers are fitted on the whole dataset to illustrate their
#    effect. In a machine-learning evaluation, fit them on the training data
#    only.

# %%
# Harmonization with and without age
# ----------------------------------

harmonized = {
    "Raw": X,
    "NeuroComBat, no covariate": NeuroComBat().fit_transform(X, sites=sites),
    "NeuroComBat, linear age": NeuroComBat().fit_transform(X, sites=sites, continuous_covariates=ages),
    "ComBatGAM, smooth age": ComBatGAM(random_state=0).fit_transform(X, sites=sites, smooth_covariates=ages),
}

# %%
# The true age curve of feature 1 (signal mean plus age effect) is drawn in
# black. Without age, ComBat aligns the three sites and flattens the curve:
# the age differences between sites are removed as site effects. With a
# linear age term, the trend is kept, but the part of the curve that a line
# cannot follow is partly absorbed into the site effects. ComBatGAM's smooth
# term follows the curve most closely.

order = np.argsort(ages)
true_curve = truth["signal"][:, 0].mean() + truth["covariate_effect"][order, 0]
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharex=True, sharey=True)
for ax, (name, data) in zip(axes, harmonized.items(), strict=True):
    df = pd.DataFrame({"Age": ages, "Feature 1": data[:, 0], "Site": sites})
    sns.scatterplot(df, x="Age", y="Feature 1", hue="Site", palette="deep", s=12, alpha=0.6, ax=ax)
    ax.plot(ages[order], true_curve, color="black", linewidth=2, label="true age effect")
    ax.set_title(name)
axes[-1].legend(loc="lower left", fontsize=8)
plt.tight_layout()
plt.show()

# %%
# Distance to the data without site effect
# ----------------------------------------
#
# The root-mean-square difference between the harmonized data and
# ``truth["site_free"]``, in units of the features' within-class SD. Removing
# the site effect while ignoring age leaves the data *further* from the truth
# than not harmonizing at all.

target = truth["site_free"]
rmse = pd.Series(
    {name: np.sqrt(((data - target) ** 2).mean(axis=0) / truth["feature_scale"] ** 2).mean() for name, data in harmonized.items()},
    name="RMSE to site-free data (SD)",
)
print(rmse.round(3).to_string())

###############################################################################
# .. admonition:: Take-home message
#
#    When the sites differ in a covariate that affects the features, give the
#    covariate to the harmonizer. Without it, harmonization removes biology
#    together with the site effect. For nonlinear effects such as age,
#    ComBatGAM's smooth term keeps the shape of the effect.
