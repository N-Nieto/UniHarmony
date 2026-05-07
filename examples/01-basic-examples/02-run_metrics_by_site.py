"""Compute metrics by site."""

# %%
# Imports
# -------

import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from uniharmony import verbosity
from uniharmony.datasets import make_multisite_classification
from uniharmony.metrics import report_metrics_by_site


sns.set_theme(style="whitegrid")
verbosity("error")

clf = LogisticRegression()


# %%
# Data generation
# ---------------

X, y, sites = make_multisite_classification(n_sites=10, signal_strength=0.5)

X_train, X_test, y_train, y_test, sites_train, sites_test = train_test_split(X, y, sites)

# %%
# Metrics by site report
# ----------------------

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_scores = clf.predict_proba(X_test)[:, 0]
metrics = report_metrics_by_site(y_test, y_pred, sites_test, balanced_accuracy_score)

# for key in metrics.keys():
#     print(f"For site {key}: bACC {metrics[key]:.4}")

# %%

# Compute metrics but now request the overall
metrics = report_metrics_by_site(y_test, y_pred, sites_test, balanced_accuracy_score, overall_performance=True)

# Compute the metric outside the function to compare.
bacc = balanced_accuracy_score(y_true=y_test, y_pred=y_pred)

# Overall comparison.
print(f"Overall bACC: {bacc}")
# The overall performance is also stored in the metrics if requested.
print(f"Overall bACC: {metrics['balanced_accuracy_score']['overall']}")

###############################################################################
# If requested, the function also computes the overall performance and stores it as another entry in the dictionary.

# %%

# Single metric (simplest case)
metrics = report_metrics_by_site(y_test, y_scores, sites_test, accuracy_score)
print(metrics)

# Single metric with kwargs
metrics = report_metrics_by_site(y_test, y_scores, sites_test, f1_score, metric_kwargs={"threshold": 0.5, "average": "macro"})
print(metrics)

# Multiple metrics
metrics = report_metrics_by_site(
    y_test,
    y_scores,
    sites_test,
    metrics=[roc_auc_score, accuracy_score],
    metric_kwargs=[{}, {"threshold": 0.5}],
)
print(metrics)

# With overall performance
metrics = report_metrics_by_site(
    y_test,
    y_scores,
    sites_test,
    metrics=[accuracy_score, roc_auc_score],
    overall_performance=True,
)
print(metrics)
