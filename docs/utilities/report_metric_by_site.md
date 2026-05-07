(report_metric_by_site-utilities)=
# Site-Stratified Metrics

A robust module for computing machine learning metrics stratified by site (e.g., hospital, data center, geographic region). Features automatic binarization of continuous scores when discrete predictions are required, and supports computing one or many metrics in a single call.

## Features

- **Multi-Metric Computation** — Compute many metrics across all sites in one call, each with individual kwargs
- **Robust Input Validation** — Comprehensive type and shape checking with clear error messages
- **Automatic Binarization** — When continuous scores (`y_score`) are passed to a prediction-based metric, they are automatically thresholded (default: 0.5)
- **Flexible Site Identifiers** — Supports integer, string, or mixed site labels with automatic type normalization
- **Overall Performance** — Optionally include aggregate metrics across all sites

---

## Quick Start

```python
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from uniharmony.metrics import report_metrics_by_site

# Sample data: 6 samples across 3 sites
y_true = np.array([0, 1, 0, 1, 0, 1])
y_scores = np.array([0.1, 0.9, 0.2, 0.4, 0.3, 0.8])
sites = np.array(["A", "A", "B", "B", "C", "C"])

# Single metric (pass a callable directly)
results = report_metrics_by_site(y_true, y_scores, sites, accuracy_score)
# {'accuracy_score': {'A': 1.0, 'B': 0.5, 'C': 1.0}}

# Single metric with custom threshold
results = report_metrics_by_site(
    y_true, y_scores, sites, accuracy_score, metric_kwargs={"threshold": 0.3}
)

# Score-based metric (no binarization needed)
results = report_metrics_by_site(y_true, y_scores, sites, roc_auc_score)
# {'roc_auc_score': {'A': 1.0, 'B': 0.5, 'C': 1.0}}

# Multiple metrics with mixed score/pred requirements
results = report_metrics_by_site(
    y_true,
    y_scores,
    sites,
    metrics=[roc_auc_score, accuracy_score, f1_score],
    metric_kwargs=[
        {},                       # roc_auc_score: uses scores directly
        {"threshold": 0.5},       # accuracy_score: binarizes at 0.5
        {"threshold": 0.5, "average": "macro"},  # f1_score: binarizes + kwargs
    ],
    overall_performance=True,
)
# {
#   'roc_auc_score': {'overall': 0.833, 'A': 1.0, 'B': 0.5, 'C': 1.0},
#   'accuracy_score': {'overall': 0.833, 'A': 1.0, 'B': 0.5, 'C': 1.0},
#   'f1_score': {'overall': 0.833, 'A': 1.0, 'B': 0.5, 'C': 1.0}
# }
```

---

## API Reference

### `report_metrics_by_site`

Compute one or more metrics stratified by site.

```python
def report_metrics_by_site(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sites: np.ndarray,
    metrics: Callable | Sequence[Callable],
    metric_kwargs: dict[str, Any] | Sequence[dict[str, Any]] | None = None,
    overall_performance: bool = False,
    skip_empty_sites: bool = True,
) -> dict[str, dict[str | int, float]]
```

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `y_true` | `np.ndarray` | Ground-truth (correct) target values |
| `y_pred` | `np.ndarray` | Estimated targets, probability scores, or decision function outputs |
| `sites` | `np.ndarray` | Site identifiers for stratification (strings or integers) |
| `metrics` | `Callable \| Sequence[Callable]` | Metric function or list of metric functions to compute (e.g., from `sklearn.metrics`). Pass a single callable for one metric, or a sequence for multiple. |
| `metric_kwargs` | `dict \| Sequence[dict] \| None` | Keyword arguments for metrics (see below) |
| `overall_performance` | `bool` | Include `"overall"` key with aggregate metric |
| `skip_empty_sites` | `bool` | Skip sites with no samples (if `False`, raises error) |

**`metric_kwargs` Formats**

| Format | Behavior |
|--------|----------|
| `None` | No kwargs for any metric (empty dicts) |
| `dict` | Same kwargs applied to **all** metrics |
| `Sequence[dict]` | `metric_kwargs[i]` passed to `metrics[i]` |

**Returns**

`dict[str, dict[str | int, float]]` — Dictionary mapping metric names to site-wise results. Each inner dictionary maps site identifiers to metric values. When a single metric is passed, the result contains one top-level key (the metric's `__name__`).

**Auto-Binarization**

If `y_pred` contains continuous scores but the metric requires discrete predictions, the scores are automatically binarized:

- Values >= `threshold` (default: 0.5, from kwargs) -> 1
- Values < `threshold` -> 0

The `threshold` key is consumed during binarization and not passed to the metric itself.

**Examples**

```python
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# Single metric (simplest case)
report_metrics_by_site(y_true, y_scores, sites, accuracy_score)
# {'accuracy_score': {'A': 1.0, 'B': 0.5}}

# Single metric with kwargs
report_metrics_by_site(
    y_true, y_scores, sites, f1_score,
    metric_kwargs={"threshold": 0.5, "average": "macro"}
)

# Multiple metrics
report_metrics_by_site(
    y_true, y_scores, sites,
    metrics=[roc_auc_score, accuracy_score],
    metric_kwargs=[{}, {"threshold": 0.5}],
)

# With overall performance
report_metrics_by_site(
    y_true, y_scores, sites,
    metrics=[accuracy_score, roc_auc_score],
    overall_performance=True,
)
# {'accuracy_score': {'overall': 0.75, 'A': 1.0, ...},
#  'roc_auc_score': {'overall': 0.82, 'A': 1.0, ...}}
```

---

## Prediction-Type Handling

The module maintains registries of metrics and their input requirements:

| Registry | Metrics | Behavior with Scores |
|----------|---------|---------------------|
| `METRICS_REQUIRING_Y_SCORE` | `roc_auc_score`, `average_precision_score`, `roc_curve`, `precision_recall_curve`, `det_curve`, `brier_score_loss`, `log_loss` | Uses scores directly |
| `METRICS_REQUIRING_Y_PRED` | `accuracy_score`, `balanced_accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `fbeta_score`, `jaccard_score`, `cohen_kappa_score`, `matthews_corrcoef`, `hamming_loss`, `zero_one_loss`, `confusion_matrix`, `classification_report` | Auto-binarizes scores at `threshold` (default: 0.5) |

### How It Works

```python
y_true = np.array([0, 1, 0, 1])
y_scores = np.array([0.2, 0.8, 0.3, 0.9])

# Score-based metric: uses y_scores directly
report_metrics_by_site(y_true, y_scores, sites, roc_auc_score)

# Prediction-based metric: auto-binarizes y_scores -> [0, 1, 0, 1]
report_metrics_by_site(y_true, y_scores, sites, accuracy_score)
# Equivalent to: accuracy_score(y_true, (y_scores >= 0.5).astype(int))

# Custom threshold: binarizes at 0.3 -> [0, 1, 1, 1]
report_metrics_by_site(
    y_true, y_scores, sites, accuracy_score, metric_kwargs={"threshold": 0.3}
)
```

### Extending for Custom Metrics

Register custom metrics by adding them to the appropriate registry:

```python
from uniharmony.metrics import METRICS_REQUIRING_Y_SCORE, METRICS_REQUIRING_Y_PRED

# Custom score-based metric
def my_auc(y_true, y_score):
    ...
METRICS_REQUIRING_Y_SCORE.add("my_auc")

# Custom prediction-based metric
def my_accuracy(y_true, y_pred):
    ...
METRICS_REQUIRING_Y_PRED.add("my_accuracy")
```
