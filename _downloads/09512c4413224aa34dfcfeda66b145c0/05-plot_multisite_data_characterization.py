"""
Characterize a multisite problem
================================
"""

###############################################################################
# The first step before applying any harmonization technique is to understand and characterize our data

# %%
# Imports
# -------

import matplotlib.pyplot as plt
import seaborn as sns

from uniharmony import verbosity
from uniharmony.datasets import (
    get_multisite_data_statistics,
    make_multisite_classification,
    print_statistics_summary,
)
from uniharmony.plot import plot_features_by_site


sns.set_theme(style="whitegrid")
verbosity("warning")


# %%
# Data generation
# ---------------
# Let's use the multisite data generator to simulate some data

print("Generating example data...")
X, y, sites = make_multisite_classification(
    n_sites=5,
    n_samples=1000,
    n_features=10,
    n_classes=3,
    random_state=42,
)

print("\n" + "=" * 60)


# %%
# Now let's compute some statistics

print("Computing statistics...")
print("=" * 60)

# Compute statistics
stats = get_multisite_data_statistics(
    X=X,
    y=y,
    sites=sites,
    feature_names=[f"feat_{i}" for i in range(X.shape[1])],
)
verbosity("info")
# Print summary
print_statistics_summary(stats)
verbosity("warning")


# %%

# Same plot individual points overlay
fig2, ax2 = plot_features_by_site(
    X,
    sites,
    figsize=(14, 7),
    rotation=45,
    show_points=True,
    title="All Features by Site (with individual points)",
)
plt.show()
