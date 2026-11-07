"""
Multisite data generation
=========================
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

# We will start with the function as default
X, y, sites = make_multisite_classification()
df = pd.DataFrame({"Class": y, "Site": sites})
print(f"X has {X.shape[0]} examples and {X.shape[1]} features")

plt.figure(figsize=[10, 6])
plt.title("Class and site distribution")
sns.countplot(df, x="Class", hue="Site")
plt.grid(axis="y", color="black", alpha=0.5, linestyle="--")


# %%

# Let's increase the number of sites.
# Note that the total number of samples is the same, but the distribution changed.
fig, axes = plt.subplots(1, 3, figsize=[15, 5])

for idx, n_sites in enumerate([2, 4, 10]):
    X, y, sites = make_multisite_classification(n_sites=n_sites)
    print(f"For n_sites {n_sites}, X has {X.shape[0]} examples and {X.shape[1]} features")

    df_plot = pd.DataFrame(
        {
            "Class": y,
            "Site": sites,
            "Feature 1": X[:, 0],
            "Feature 2": X[:, 1],
        }
    )
    sns.countplot(df_plot, x="Class", hue="Site", ax=axes[idx])
    plt.title("Class and site distribution")

    axes[idx].set_title(f"Class and site distribution with {n_sites} sites")
    axes[idx].grid(axis="y", color="black", alpha=0.5, linestyle="--")

plt.tight_layout()


# %%
# Plotting
# --------

fig, axes = plt.subplots(1, 3, figsize=[15, 5])

for idx, n_classes in enumerate([2, 4, 10]):
    X, y, sites = make_multisite_classification(n_classes=n_classes)
    print(f"For n_classes {n_classes}, X has {X.shape[0]} examples and {X.shape[1]} features")

    df_plot = pd.DataFrame(
        {
            "Class": y,
            "Site": sites,
            "Feature 1": X[:, 0],
            "Feature 2": X[:, 1],
        }
    )
    sns.countplot(df_plot, x="Class", hue="Site", ax=axes[idx])
    axes[idx].set_title(f"Class and site distribution with {n_classes} classes")
    axes[idx].grid(axis="y", color="black", alpha=0.5, linestyle="--")

plt.tight_layout()


# %%
# Variations
# ----------
#
# Changing the class balance
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# As default, the created problem is balanced accross sites, so all classes has the same number of examples in each site.
# We can control that using the ``balance_per_site`` argument of the function.

# %%

# In a binary classification problem, you need to set which
# proportion of class 1 is present in that site.
# Thus, the number of elements in the list must match the number of sites.
fig, axes = plt.subplots(1, 3, figsize=[15, 5])
balance_per_site_list = [[0.5, 0.5], [0.3, 0.7], [0.1, 0.9]]
for idx, balance_per_site in enumerate(balance_per_site_list):
    X, y, sites = make_multisite_classification(balance_per_site=balance_per_site)
    print(f"For n_classes {n_classes}, X has {X.shape[0]} examples and {X.shape[1]} features")

    df_plot = pd.DataFrame(
        {
            "Class": y,
            "Site": sites,
            "Feature 1": X[:, 0],
            "Feature 2": X[:, 1],
        }
    )
    sns.countplot(df_plot, x="Class", hue="Site", ax=axes[idx])
    axes[idx].set_title(f"Balance per classes {balance_per_site} ")
    axes[idx].grid(axis="y", color="black", alpha=0.5, linestyle="--")

plt.tight_layout()


# Now that we changed the balance, we have different occurence in different sites. Note that the total number of classes is still balanced, is the distribution accross sites that changed.


# %%
# Changing the signal type
# ^^^^^^^^^^^^^^^^^^^^^^^^
# The signal type is used to generate a ``base`` problem, using sklearn functions.
# From this generated problem, ``uniharmony`` adds an Effect of Site and noise on top.

fig, axes = plt.subplots(1, 5, figsize=[20, 7])

for idx, signal_type in enumerate(["linear", "moons", "circles", "blobs", "gaussian_quantiles"]):
    X, y, sites = make_multisite_classification(
        n_features=2,
        signal={signal_type: 1.0},
        site_effect={"location": 0},
        noise_strength=0.1,
    )

    df_plot = pd.DataFrame(
        {
            "Class": y,
            "Site": sites,
            "Feature 1": X[:, 0],
            "Feature 2": X[:, 1],
        }
    )

    sns.scatterplot(df_plot, x="Feature 1", y="Feature 2", hue="Class", ax=axes[idx])
    axes[idx].set_title(f"Signal Type: {signal_type}")
    axes[idx].grid(axis="y", color="black", alpha=0.5, linestyle="--")

plt.tight_layout()


# %%

fig, axes = plt.subplots(1, 3, figsize=[15, 5])

for idx, signal_st in enumerate([0, 1, 5]):
    X, y, sites = make_multisite_classification(
        n_features=2,
        signal={"linear": signal_st},
        site_effect={"location": 0},
        noise_strength=0.1,
    )

    df_plot = pd.DataFrame(
        {
            "Class": y,
            "Site": sites,
            "Feature 1": X[:, 0],
            "Feature 2": X[:, 1],
        }
    )

    sns.scatterplot(df_plot, x="Feature 1", y="Feature 2", hue="Class", ax=axes[idx])
    axes[idx].set_title(f"Signal Strength: {signal_st}")
    axes[idx].grid(axis="y", color="black", alpha=0.5, linestyle="--")

plt.tight_layout()


# %%
# Effect of Site effect
# ---------------------

fig, axes = plt.subplots(1, 3, figsize=[15, 5])

for idx, site_st in enumerate([0, 1, 5]):
    X, y, sites = make_multisite_classification(
        n_sites=4,
        n_features=2,
        noise_strength=0,
        signal={"circles": 1.0},
        site_effect={"location": site_st},
    )

    df_plot = pd.DataFrame(
        {
            "Class": y,
            "Site": sites,
            "Feature 1": X[:, 0],
            "Feature 2": X[:, 1],
        }
    )

    sns.scatterplot(df_plot, x="Feature 1", y="Feature 2", hue="Site", ax=axes[idx])
    axes[idx].set_title(f"Effect of Site Strength: {site_st}")
    axes[idx].grid(axis="y", color="black", alpha=0.5, linestyle="--")

plt.tight_layout()


# %%
# Combining site effects
# ----------------------
#
# ``site_effect`` is a dict with the strength of each type. With
# ``{"location": ..., "scale": ...}``, each site's features are multiplied around the global mean by a factor
# exp(U(-0.1 s, 0.1 s)) (scale) and shifted by an offset drawn from
# U(-s/2, s/2) within-class SDs (location), so a strong shift can be combined
# with a mild change of spread, or the reverse.

fig, axes = plt.subplots(1, 3, figsize=[15, 5], sharex=True, sharey=True)

for idx, (location, scale) in enumerate([(4, 0), (0, 10), (4, 3)]):
    X, y, sites = make_multisite_classification(
        n_sites=3,
        n_features=2,
        noise_strength=0,
        signal={"circles": 1.0},
        site_effect={"location": location, "scale": scale},
    )
    df_plot = pd.DataFrame({"Site": sites, "Feature 1": X[:, 0], "Feature 2": X[:, 1]})
    sns.scatterplot(df_plot, x="Feature 1", y="Feature 2", hue="Site", ax=axes[idx], s=10)
    axes[idx].set_title(f"location strength {location}, scale strength {scale}")

plt.tight_layout()


# %%
# Heterogeneous site effects
# --------------------------
#
# With ``site_effect_heterogeneity=h``, the site effect of every sample is
# multiplied by its own intensity exp(h z - h^2/2) (mean 1, ``z`` drawn per
# sample and shared by its features): a sample is moved further or less far
# in the direction of its site's offset. The sites stretch along their offset,
# a site-specific covariance that a per-feature mean and variance correction
# does not remove.

fig, axes = plt.subplots(1, 3, figsize=[15, 5], sharex=True, sharey=True)

for idx, heterogeneity in enumerate([0, 0.3, 0.8]):
    X, y, sites = make_multisite_classification(
        n_sites=3,
        n_features=2,
        noise_strength=0,
        signal={"blobs": 0.1},
        site_effect={"location": 8},
        site_effect_heterogeneity=heterogeneity,
    )
    df_plot = pd.DataFrame({"Site": sites, "Feature 1": X[:, 0], "Feature 2": X[:, 1]})
    sns.scatterplot(df_plot, x="Feature 1", y="Feature 2", hue="Site", ax=axes[idx], s=10)
    axes[idx].set_title(f"Heterogeneity {heterogeneity}")

plt.tight_layout()


# %%
# Site-by-class interaction
# -------------------------
#
# The ``"class_interaction"`` type gives each class of each site its own extra
# offset: the site effect differs between classes, so the class signal itself
# differs between sites.

fig, axes = plt.subplots(1, 2, figsize=[10, 5], sharex=True, sharey=True)

for idx, interaction in enumerate([0, 3]):
    X, y, sites = make_multisite_classification(
        n_sites=2,
        n_features=2,
        noise_strength=0,
        signal={"blobs": 2},
        site_effect={"location": 3, "class_interaction": interaction},
    )
    df_plot = pd.DataFrame({"Site": sites, "Class": y, "Feature 1": X[:, 0], "Feature 2": X[:, 1]})
    sns.scatterplot(df_plot, x="Feature 1", y="Feature 2", hue="Site", style="Class", ax=axes[idx], s=10)
    axes[idx].set_title(f"Site-by-class interaction {interaction}")

plt.tight_layout()


# %%
# Noise effect
# ------------

verbosity("warning")

fig, axes = plt.subplots(1, 3, figsize=[15, 5])

for idx, noise_st in enumerate([0.01, 0.1, 0.5]):
    X, y, sites = make_multisite_classification(
        n_features=2,
        signal={"linear": 1},
        site_effect={"location": 0},
        noise_strength=noise_st,
    )

    df_plot = pd.DataFrame(
        {
            "Class": y,
            "Site": sites,
            "Feature 1": X[:, 0],
            "Feature 2": X[:, 1],
        }
    )

    sns.scatterplot(df_plot, x="Feature 1", y="Feature 2", hue="Class", ax=axes[idx])
    axes[idx].set_title(f"Noise Strength: {noise_st}")
    axes[idx].grid(axis="y", color="black", alpha=0.5, linestyle="--")
    axes[idx].set_xlim(-5, 5)
    axes[idx].set_ylim(-5, 5)

plt.tight_layout()


# %%

fig, axes = plt.subplots(1, 3, figsize=[15, 5])

for idx, strength in enumerate([0, 1, 50]):
    X, y, sites = make_multisite_classification(
        n_sites=3, n_features=2, signal={"blobs": 4}, site_effect={"location": strength}, random_state=23
    )

    df_plot = pd.DataFrame(
        {
            "Class": y,
            "Site": sites,
            "Feature 1": X[:, 0],
            "Feature 2": X[:, 1],
        }
    )

    sns.scatterplot(df_plot, x="Feature 1", y="Feature 2", hue="Site", ax=axes[idx])
    axes[idx].set_title(f"Signal Strength: {strength}")
    axes[idx].grid(axis="y", color="black", alpha=0.5, linestyle="--")


plt.tight_layout()
