"""
Characterise a multisite problem with MAREoS
============================================
"""

# %%
# Imports
# -------

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE

from uniharmony import verbosity
from uniharmony.datasets import load_MAREoS
from uniharmony.plot import plot_2d_components_by_value, plot_2d_projection


sns.set_theme(style="whitegrid")
verbosity("warning")

# %%
# Data generation
# ---------------
# Let's load the MAREoS datasets, which simulates several datasets with and without Effects of Site (EoS)

# Initialize a tSNE object
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000, learning_rate="auto")
# Load the MAREoS dataset
datasets = load_MAREoS()
print(datasets.keys())

# %%
# Now let's play with tSNE and the plotting helper functions

# EoS signal
dataset = datasets["eos_simple1"]
X = dataset["X"]
y = dataset["y"]
sites = dataset["sites"]
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000, learning_rate="auto")
X_tsne = tsne.fit_transform(X)
tsne_df_eos = pd.DataFrame({"comp1": X_tsne[:, 0], "comp2": X_tsne[:, 1], "site": sites, "target": y})

# True signal
dataset = datasets["true_simple1"]
X = dataset["X"]
y = dataset["y"]
sites = dataset["sites"]
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000, learning_rate="auto")
X_tsne = tsne.fit_transform(X)
tsne_df_true = pd.DataFrame({"comp1": X_tsne[:, 0], "comp2": X_tsne[:, 1], "site": sites, "target": y})

# Initialize figure
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Plot 1: EoS By site
ax1 = axes[0, 0]
plot_2d_components_by_value(tsne_df_eos, "site", "tSNE", ax1)

# Plot 2: EoS By target
ax2 = axes[1, 0]
plot_2d_components_by_value(tsne_df_eos, "target", "tSNE", ax2)

# # Plot 3: True Signal By site
ax3 = axes[0, 1]
plot_2d_components_by_value(tsne_df_true, "site", "tSNE", ax3)

# Plot 4: True Signal By target
ax4 = axes[1, 1]
plot_2d_components_by_value(tsne_df_true, "target", "tSNE", ax4)


###############################################################################
# We see that, for the EoS signal, the main tSNE components are related with the sites, which are also realted with the targets.
# On the other hand, there is not a clear relationship between the sites nor the target for the True signal.
#
# Now let's use the ``plot_tsne`` funtion which can simplify the code and will allowd us a fast and simple exploration

# %%

# EoS signal
dataset = datasets["eos_simple2"]
X = dataset["X"]
y = dataset["y"]
sites = dataset["sites"]
plot_2d_projection(X, y, sites, tsne)

# True signal
dataset = datasets["true_simple2"]
X = dataset["X"]
y = dataset["y"]
sites = dataset["sites"]
plot_2d_projection(X, y, sites, tsne)


# %%

# EoS signal
dataset = datasets["eos_simple2"]
X = dataset["X"]
y = dataset["y"]
sites = dataset["sites"]
plot_2d_projection(X, y, sites, tsne)

# True signal
dataset = datasets["true_simple2"]
X = dataset["X"]
y = dataset["y"]
sites = dataset["sites"]
plot_2d_projection(X, y, sites, tsne)


# %%

# EoS signal
dataset = datasets["eos_interaction1"]
X = dataset["X"]
y = dataset["y"]
sites = dataset["sites"]
plot_2d_projection(X, y, sites, tsne)


# True Signal
dataset = datasets["true_interaction1"]
X = dataset["X"]
y = dataset["y"]
sites = dataset["sites"]
plot_2d_projection(X, y, sites, tsne)


# %%

# EoS signal
dataset = datasets["eos_interaction2"]
X = dataset["X"]
y = dataset["y"]
sites = dataset["sites"]
plot_2d_projection(X, y, sites, tsne)


# True signal
dataset = datasets["true_interaction2"]
X = dataset["X"]
y = dataset["y"]
sites = dataset["sites"]
plot_2d_projection(X, y, sites, tsne)
