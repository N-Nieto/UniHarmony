"""
Analyzing ComBatGAM behavior with imbalance across sites
=========================================================
"""

# %%
# Imports
# -------

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from uniharmony.datasets import make_multisite_classification
from uniharmony.datasets import Covariate, CovariateSiteDistribution

from uniharmony import verbosity
verbosity("warning")
from uniharmony.combat import ComBatGAM

sns.set_theme(style="whitegrid")

# %%
# Data generation
# ---------------

covars = [Covariate(
    name="age",
    site_distributions=[
        CovariateSiteDistribution(loc=40.0, scale=10, clip=None),
        CovariateSiteDistribution(loc=70.0, scale=10, clip=None)],
    x_correlation=0.1),
    Covariate(
    name="sex",
    site_distributions=[
        CovariateSiteDistribution(probs=[0.1,0.9]),
        CovariateSiteDistribution(probs=[0.9,0.1]),

    ],
    x_correlation=0.2)]


X, y, sites, covars = make_multisite_classification(n_features=2, signal_type="blobs", covariates=covars)

age = covars["age"]
sex = covars["sex"]
df = pd.DataFrame({"Class": y, "Site": sites, "Age": age,
                   "Feature1":X[:,0], "sex": sex})

plt.figure(figsize=[10, 6])
plt.title("Features vs age/sex distribution")
sns.scatterplot(df, y="Feature1", x="Age", hue="sex", style="Site")
plt.grid(axis="y", color="black", alpha=0.5, linestyle="--")
#########################################################################


# %%
# Preserving the target as covariate
# ----------------------------------

###############################################################################
# .. caution::
#
#    This is also wrong in ML context, where you don't have access to the full
#    dataset but may be a good option for statistical analysis.

# %%

combat_gam = ComBatGAM()
# This is the key line: we need to include the target variable as a covariate
# to preserve its relationship with the features during harmonization.
combat_gam.fit(X, sites, smooth_covariates=y)
X_harmonized = combat_gam.transform(X, sites, smooth_covariates=y)

df = pd.DataFrame({"Class": y, "Site": sites, "Age": age,
                   "Feature1":X_harmonized[:,0], "sex": sex})

plt.figure(figsize=[10, 6])
plt.title("Features vs age/sex distribution")
sns.scatterplot(df, y="Feature1", x="Age", hue="sex", style="Site")
plt.grid(axis="y", color="black", alpha=0.5, linestyle="--")



# %%
