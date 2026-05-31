"""
Multisite data generation with covariates
=========================================
"""

# %%
# Imports
# -------

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from uniharmony import verbosity
from uniharmony.datasets import make_multisite_classification
from uniharmony.datasets import Covariate, CovariateSiteDistribution

sns.set_theme(style="whitegrid")
verbosity("warning")

covars = [Covariate(
    name="age",
    site_distributions=[
        CovariateSiteDistribution(loc=10.0, scale=1.0, clip=None),
        CovariateSiteDistribution(loc=70.0, scale=1.0, clip=None)],
    x_correlation=0.9),
    Covariate(
    name="sex",
    site_distributions=[
        CovariateSiteDistribution(probs=[0.1,0.9]),
        CovariateSiteDistribution(probs=[0.9,0.1]),

    ],
    x_correlation=0.2)]


X, y, sites, covars = make_multisite_classification(signal_type="blobs",covariates=covars)
df = pd.DataFrame({"Class": y, "Site": sites, "Age": covars["age"], "Feature1":X[:,0], "sex": covars["sex"]})
print(f"X has {X.shape[0]} examples and {X.shape[1]} features")

plt.figure(figsize=[10, 6])
plt.title("Class and site distribution")
sns.scatterplot(df, y="Feature1", x="Age", hue="sex", style="Site")
plt.grid(axis="y", color="black", alpha=0.5, linestyle="--")


# %%
