"""
Evaluate site predictability
=============================

Diagnose the Effect of Site (EoS) in a multi-site dataset by checking how
well the acquisition site can be predicted from the features, before and
after a simple location-shift is removed.
"""

# %%
# Imports
# -------
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from uniharmony import verbosity
from uniharmony.datasets import make_multisite_classification
from uniharmony.metrics import evaluate_site_prediction

verbosity("error")
random_state = 42

# %%
# Data generation
# ----------------
# Simulate a 2-site classification problem with a strong location-based
# EoS, so that ``sites`` should be easy to recover from the raw features.
X, _, sites = make_multisite_classification(
    n_sites=5,
    n_classes=2,                # irrelevant for this example
    site_effect_strength=2,
    random_state=random_state,
)

# %%
# Default usage
# -------------
# With no arguments beyond ``X`` and ``sites``, the function uses a
# ``LogisticRegression`` model, scikit-learn's default 5-fold
# ``StratifiedKFold`` cross-validation, and reports balanced accuracy and
# (multi-class, one-vs-rest) AUC.
results = evaluate_site_prediction(X, sites)

print("In the results, the fitting and score time are returned, which is standard for sklearn")
print(results.keys())

mean = results["balanced_accuracy"]["mean"]
std = results["balanced_accuracy"]["std"]
print(f"{"Balanced Accuracy"}: {mean:.3f} +- {std:.3f}")


# %%
# Customizing model, cv scheme and metrics
# -----------------------------------------
# Any sklearn-compatible classifier, cv splitter, and metric(s) can be
# supplied explicitly.
model = RandomForestClassifier(n_estimators=50, random_state=random_state)
cv_scheme = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
metrics = ["balanced_accuracy", "roc_auc_ovo", "roc_auc_ovr", "f1_macro"]

results_rf = evaluate_site_prediction(
    X,
    sites,
    model= model,
    cv=cv_scheme,
    metrics=["balanced_accuracy", "roc_auc_ovo", "roc_auc_ovr", "f1_macro"],
)
print(f"RandomForest balanced_accuracy: {results_rf['balanced_accuracy']['mean']:.3f}")
print(f"RandomForest roc_auc_ovr: {results_rf['roc_auc_ovr']['mean']:.3f}")
print(f"RandomForest roc_auc_ovo: {results_rf['roc_auc_ovo']['mean']:.3f}")
print(f"RandomForest f1_macro: {results_rf['f1_macro']['mean']:.3f}")

# %%
