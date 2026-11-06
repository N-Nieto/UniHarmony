"""
Multisite data generation with covariates
=========================================

``make_multisite_classification`` can also simulate covariates, such as age
and sex, whose effect on the features is the same in every site. The presets
``"age"`` and ``"sex"`` give a ready-made example.
"""

# %%
# Imports
# -------

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from uniharmony import verbosity
from uniharmony.datasets import make_multisite_classification


sns.set_theme(style="whitegrid")
verbosity("warning")

# %%
# Data with the age and sex presets
# ---------------------------------
#
# * ``"age"``: mean 50 and SD 20 in every site, with a linear effect of 0.5
#   (in units of each feature's within-class SD, per SD of age) on every
#   feature;
# * ``"sex"``: 50 % of each sex in every site; sex 1 is 0.3 SD higher on every
#   feature.
#
# The effect of each feature is the preset value times a weight drawn from
# U(0.5, 1.5); ``return_ground_truth=True`` returns the values used.

X, y, sites, truth = make_multisite_classification(covariates=["age", "sex"], return_ground_truth=True)
covars = truth["covariates"]
print(f"X has {X.shape[0]} examples and {X.shape[1]} features")
coefficients = pd.DataFrame(
    {
        "age (per SD of age)": truth["covariate_coefficients"]["age"],
        "sex (1 vs 0)": truth["covariate_coefficients"]["sex"][1],
    }
).round(2)
coefficients.index.name = "feature"
print(coefficients.head())

# %%
# Feature 1 increases with age and is higher for sex 1, in both sites. The
# two sites are also shifted by the site effect.

df = pd.DataFrame({"Age": covars["age"], "Feature 1": X[:, 0], "Sex": covars["sex"], "Site": sites})
plt.figure(figsize=[10, 6])
plt.title("Feature 1 against age, by sex and site")
sns.scatterplot(df, y="Feature 1", x="Age", hue="Sex", style="Site", alpha=0.7)
plt.show()
