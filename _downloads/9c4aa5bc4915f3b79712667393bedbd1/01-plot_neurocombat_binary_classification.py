"""
Binary classification with NeuroComBat
======================================
"""

# %%
# Imports
# -------

from uniharmony import verbosity
verbosity("warning")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from uniharmony.datasets import make_multisite_classification
from uniharmony.plot import plot_decision_boundary_2d
from uniharmony.combat import NeuroComBat

sns.set_theme(style="whitegrid")

# %%
# Data generation
# ---------------
X, y, sites = make_multisite_classification(
    n_features=2,
    signal_strength=0.5,
    site_effect_strength=4,
)
random_state = 42
clf = LogisticRegression(random_state=random_state)
X_train, X_test, y_train, y_test, sites_train, sites_test = train_test_split(X, y, sites, test_size=0.3, random_state=42)

clf.fit(X=X_train, y=y_train)
y_pred = clf.predict_proba(X_test)[:,1]
combat = NeuroComBat()
X_harmonized = combat.fit_transform(X_train, sites_train)
clf.fit(X=X_harmonized, y=y_train)
X_test_harmonized = combat.transform(X_test, sites_test)
y_pred_harm = clf.predict_proba(X_test_harmonized)[:,1]


scores = roc_auc_score(y_test, y_pred)
scores_harmonized = roc_auc_score(y_test, y_pred_harm)

df_orig = pd.DataFrame(X_test, columns=["Feature 1", "Feature 2"])
df_orig["Site"] = sites_test
df_orig["Target"] = y_test

df_harm = pd.DataFrame(X_test_harmonized, columns=["Feature 1", "Feature 2"])
df_harm["Site"] = sites_test
df_harm["Target"] = y_test

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
sns.scatterplot(data=df_orig, x="Feature 1", y="Feature 2", hue="Target", style="Site", alpha=0.6, ax=axes[0])
axes[0].set_title(f"Original data by site. AUC {scores:0.3}")
clf.fit(X_train, y_train)
plot_decision_boundary_2d(axes[0], clf)

sns.scatterplot(data=df_harm, x="Feature 1", y="Feature 2", hue="Target", style="Site", alpha=0.6, ax=axes[1])
axes[1].set_title(f"Harmonized data by site. AUC {scores_harmonized:0.3}")
plt.tight_layout()
clf.fit(X_harmonized, y_train)

plot_decision_boundary_2d(axes[1], clf)

# %%
