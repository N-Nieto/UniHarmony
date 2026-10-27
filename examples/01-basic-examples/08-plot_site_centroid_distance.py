"""
Measure the Effect of Site with centroid distances
==================================================

If the Effect of Site (EoS) has been removed, the per-site mean vectors
(centroids) should only differ by sampling noise.
``site_centroid_distance`` turns this idea into a number: a pseudo-F that
compares the spread between site centroids with the spread within sites,
tested against a permutation null.

Here we compute it before and after NeuroComBat, and show the centroids on
PCA and t-SNE projections.
"""

# %%
# Imports
# -------
from uniharmony import verbosity

verbosity("warning")  # before importing the harmonizers, so their loggers are silenced too

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

from uniharmony.combat import NeuroComBat
from uniharmony.datasets import make_multisite_classification
from uniharmony.metrics import site_centroid_distance

sns.set_theme(style="whitegrid")
random_state = 42

# %%
# Data and harmonization
# ----------------------
# Four sites with a location EoS. NeuroComBat is fitted on the training
# set and the metric is computed on the test set: ComBat equalizes the site
# means of the data it was fitted on, so the training data would always look
# perfectly harmonized.
X, y, sites = make_multisite_classification(
    n_sites=4,
    n_samples=1200,
    n_features=20,
    site_effect_strength=1.5,
    random_state=random_state,
)
X_train, X_test, y_train, y_test, sites_train, sites_test = train_test_split(
    X, y, sites, test_size=0.5, stratify=sites, random_state=random_state
)

combat = NeuroComBat().fit(X_train, sites_train)
data = {"Raw": X_test, "NeuroComBat": combat.transform(X_test, sites_test)}

# %%
# Site centroid distance
# ----------------------
# ``statistic`` is the pseudo-F and ``null_mean`` its value expected by
# chance. ``r2`` is the share of the variance explained by site.
results = {name: site_centroid_distance(X_data, sites_test, y=y_test) for name, X_data in data.items()}
summary = pd.DataFrame(results).T[["statistic", "null_mean", "z_score", "p_value", "r2"]]
print(summary.astype(float).round(3))

# %%
# Centroids on 2D projections
# ---------------------------
# The PCA is fitted on the raw data and reused for the harmonized data, so
# both panels share the same axes. Since PCA is linear, the centroid of the
# projected samples is the projection of the true centroid. t-SNE is not
# linear and does not preserve distances between clusters: its centroids
# are only a visual aid. The numbers in the titles always come from the full
# feature space.
pca = PCA(n_components=2).fit(X_test)
embeddings = {
    "PCA": {name: pca.transform(X_data) for name, X_data in data.items()},
    "t-SNE": {name: TSNE(random_state=random_state).fit_transform(X_data) for name, X_data in data.items()},
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for row, (method, projections) in enumerate(embeddings.items()):
    for col, (name, X_2d) in enumerate(projections.items()):
        ax = axes[row, col]
        df = pd.DataFrame({"Component 1": X_2d[:, 0], "Component 2": X_2d[:, 1], "Site": sites_test})
        sns.scatterplot(df, x="Component 1", y="Component 2", hue="Site", palette="tab10", alpha=0.4, s=15, ax=ax)
        centroids = df.groupby("Site").mean()
        sns.scatterplot(
            centroids, x="Component 1", y="Component 2", hue="Site", palette="tab10",
            s=250, marker="X", edgecolor="black", linewidth=1.5, legend=False, ax=ax,
        )
        result = results[name]
        ax.set_title(f"{method}, {name}\nF = {result['statistic']:.2f} (chance {result['null_mean']:.2f}), p = {result['p_value']:.3f}")
plt.tight_layout()

# %%
# Why not compute the metric on the t-SNE embedding?
# --------------------------------------------------
# The same metric computed on the 2D t-SNE coordinates gives a different
# answer, which also changes with the perplexity. Use the projections to
# look at the data and the full feature space to measure it.
for perplexity in [5, 30, 100]:
    X_2d = TSNE(perplexity=perplexity, random_state=random_state).fit_transform(data["NeuroComBat"])
    result = site_centroid_distance(X_2d, sites_test, y=y_test, n_permutations=200)
    print(f"t-SNE (perplexity {perplexity:>3}) on harmonized data: F = {result['statistic']:.2f}, p = {result['p_value']:.3f}")
print(f"Full feature space on harmonized data:      F = {results['NeuroComBat']['statistic']:.2f}")

plt.show()
