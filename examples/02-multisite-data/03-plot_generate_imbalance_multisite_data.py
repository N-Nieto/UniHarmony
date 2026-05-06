"""
Generate imbalance multisite data
=================================

This example shows how to generate an unbalanced multisite dataset
using the ``balance_per_site`` parameter of the ``make_multisite_classification`` function.
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
# Data generation
# ---------------
# Let's start with the function as default, this will create a 2 site balanced problem.

# %%

X, y, sites = make_multisite_classification()
df = pd.DataFrame({"Class": y, "Site": sites})

general_balance = len(y[y == 1]) / len(y)
y_site_0 = y[sites == 0]
y_site_1 = y[sites == 1]
site_0_balance = len(y_site_0[y_site_0 == 1]) / len(y_site_0)
site_1_balance = len(y_site_1[y_site_1 == 1]) / len(y_site_1)

print(
    "The class distribution is balanced across sites and in general \n"
    f"General balance: {general_balance:.2f} \n"
    f"site 0 balance: {site_0_balance:.2f} \n"
    f"site 1 balance: {site_1_balance:.2f}"
)

plt.figure(figsize=[10, 6])
plt.title("Class and site distribution")
sns.countplot(df, x="Class", hue="Site")
plt.grid(axis="y", color="black", alpha=0.5, linestyle="--")

###############################################################################
# Let's now create a site imbalance problem. That means that, while the total number of examples per class is imbalance, the classes are not equally distributed by site.

# %%

X, y, sites = make_multisite_classification(balance_per_site=[0.3, 0.7])
df = pd.DataFrame({"Class": y, "Site": sites})
general_balance = len(y[y == 1]) / len(y)
y_site_0 = y[sites == 0]
y_site_1 = y[sites == 1]
site_0_balance = len(y_site_0[y_site_0 == 1]) / len(y_site_0)
site_1_balance = len(y_site_1[y_site_1 == 1]) / len(y_site_1)

print(
    "The class distribution is imbalanced across sites but balanced in general \n"
    f"General balance: {general_balance:.2f} \n"
    f"site 0 balance: {site_0_balance:.2f} \n"
    f"site 1 balance: {site_1_balance:.2f}"
)

plt.figure(figsize=[10, 6])
plt.title("Class and site distribution")
sns.countplot(df, x="Class", hue="Site")
plt.grid(axis="y", color="black", alpha=0.5, linestyle="--")


# %%

X, y, sites = make_multisite_classification(balance_per_site=[0.3, 0.3])
df = pd.DataFrame({"Class": y, "Site": sites})
general_balance = len(y[y == 1]) / len(y)
y_site_0 = y[sites == 0]
y_site_1 = y[sites == 1]
site_0_balance = len(y_site_0[y_site_0 == 1]) / len(y_site_0)
site_1_balance = len(y_site_1[y_site_1 == 1]) / len(y_site_1)

print(
    "The class are imbalanced in general, but have the same imbalance across sites\n"
    f"General balance: {general_balance:.2f} \n"
    f"site 0 balance: {site_0_balance:.2f} \n"
    f"site 1 balance: {site_1_balance:.2f}"
)

plt.figure(figsize=[10, 6])
plt.title("Class and site distribution")
sns.countplot(df, x="Class", hue="Site")
plt.grid(axis="y", color="black", alpha=0.5, linestyle="--")
