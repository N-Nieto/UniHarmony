"""Provide statistics and basic information about multisite data."""

from typing import Any

import numpy as np
import structlog
from sklearn.utils.validation import (
    check_array,
    check_consistent_length,
    check_X_y,
)


__all__ = [
    "get_site_data_statistics",
    "print_statistics_summary",
]

logger = structlog.get_logger()


def get_site_data_statistics(
    X: np.ndarray,
    y: np.ndarray,
    sites: np.ndarray,
    feature_names: list[str] | None = None,
    compute_comprehensive: bool = True,
) -> dict[str, Any]:
    """Compute comprehensive statistics for multi-site dataset.

    Calculates overall statistics, per-site statistics, per-class statistics,
    and inter-site/inter-class relationships.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix containing the input data
    y : np.ndarray of shape (n_samples,)
        Target labels (can be binary or multi-class)
    sites : np.ndarray of shape (n_samples,)
        Site labels indicating which site each sample belongs to
    feature_names : list of str or None, optional
        Names of features for more readable output. If None, uses indices.
    compute_comprehensive : bool, default=True
        Whether to compute comprehensive statistics including correlations
        and distribution metrics. Set to False for faster computation.

    Returns
    -------
    stats : dict
        Dictionary containing comprehensive data statistics with keys:
        - 'overall': Overall dataset statistics
        - 'site_statistics': Statistics per site
        - 'class_statistics': Statistics per class
        - 'correlations': Various correlation metrics
        - 'metadata': Additional metadata about the computation

    Raises
    ------
    ValueError
        If input arrays have incompatible shapes or invalid values
    TypeError
        If inputs are not numpy arrays

    Examples
    --------
    >>> X, y, sites = make_multisite_classification(n_sites=3, n_samples=100)
    >>> stats = get_site_data_statistics(X, y, sites)
    >>> print(stats['overall']['n_samples'])
    100

    """
    # Check data.
    X, y = check_X_y(X, y)
    sites = check_array(sites, dtype=None, ensure_2d=False)
    check_consistent_length(X, y, sites)

    # Get basic dimensions
    n_samples, n_features = X.shape
    unique_sites = np.unique(sites)
    unique_classes = np.unique(y)
    n_sites = len(unique_sites)
    n_classes = len(unique_classes)

    # Initialize statistics dictionary
    stats = {
        "overall": {},
        "site_statistics": {},
        "class_statistics": {},
        "correlations": {},
        "metadata": {
            "feature_names": feature_names,
            "n_features": n_features,
            "n_samples": n_samples,
            "n_sites": n_sites,
            "n_classes": n_classes,
            "unique_sites": unique_sites.tolist(),
            "unique_classes": unique_classes.tolist(),
        },
    }

    # Overall statistics
    logger.info(f"Computing statistics for {n_samples} samples, {n_features} features, {n_sites} sites, {n_classes} classes")

    stats["overall"] = _compute_overall_statistics(X, y, sites, feature_names)

    # Site-specific statistics
    stats["site_statistics"] = _compute_site_statistics(X, y, sites, unique_sites, feature_names)

    # Class-specific statistics
    stats["class_statistics"] = _compute_class_statistics(X, y, unique_classes, feature_names)

    # Correlation and relationship statistics
    if compute_comprehensive:
        stats["correlations"] = _compute_correlation_statistics(X, y, sites, unique_sites, unique_classes)

    return stats


def _compute_overall_statistics(
    X: np.ndarray,
    y: np.ndarray,
    site_labels: np.ndarray,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute overall dataset statistics.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target labels
    site_labels : np.ndarray
        Site labels
    feature_names : list or None
        Names of features

    Returns
    -------
    dict
        Overall statistics

    """
    n_samples, n_features = X.shape
    unique_sites = np.unique(site_labels)
    unique_classes = np.unique(y)

    # Class distribution
    class_counts = np.bincount(y)
    class_distribution = {f"class_{i}": count for i, count in enumerate(class_counts)}

    # Site distribution
    site_counts = np.bincount(site_labels.astype(int))
    site_distribution = {f"site_{i}": count for i, count in enumerate(site_counts)}

    # Feature statistics
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    feature_means = {name: float(val) for name, val in zip(feature_names, X.mean(axis=0), strict=True)}
    feature_stds = {name: float(val) for name, val in zip(feature_names, X.std(axis=0), strict=True)}
    feature_ranges = {name: {"min": float(X[:, i].min()), "max": float(X[:, i].max())} for i, name in enumerate(feature_names)}

    return {
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "n_sites": len(unique_sites),
        "n_classes": len(unique_classes),
        "unique_sites": unique_sites.tolist(),
        "unique_classes": unique_classes.tolist(),
        "class_distribution": class_distribution,
        "site_distribution": site_distribution,
        "overall_class_balance": {f"class_{i}": float(count / n_samples) for i, count in enumerate(class_counts)},
        "feature_statistics": {
            "means": feature_means,
            "stds": feature_stds,
            "ranges": feature_ranges,
        },
        "dataset_entropy": float(_compute_dataset_entropy(y)),
        "site_label_entropy": float(_compute_dataset_entropy(site_labels)),
    }


def _compute_site_statistics(
    X: np.ndarray,
    y: np.ndarray,
    site_labels: np.ndarray,
    unique_sites: np.ndarray,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute statistics for each site.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target labels.
    site_labels : np.ndarray
        Site labels.
    unique_sites : np.ndarray
        Unique site identifiers.
    feature_names : list or None
        Names of features.

    Returns
    -------
    dict
        Site statistics.

    """
    n_features = X.shape[1]
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    site_stats = {}

    for site in unique_sites:
        logger.info(f"  Processing site {int(site)}...")

        site_mask = site_labels == site
        X_site = X[site_mask]
        y_site = y[site_mask]

        # Basic counts
        n_site_samples = int(np.sum(site_mask))

        # Class distribution in this site
        class_counts = np.bincount(y_site.astype(int))
        class_distribution = {f"class_{i}": int(count) for i, count in enumerate(class_counts)}

        # Class balance percentages
        class_balance = {
            f"class_{i}": float(count / n_site_samples) if n_site_samples > 0 else 0.0 for i, count in enumerate(class_counts)
        }

        # Feature statistics for this site
        feature_means = {name: float(val) for name, val in zip(feature_names, X_site.mean(axis=0), strict=True)}
        feature_stds = {name: float(val) for name, val in zip(feature_names, X_site.std(axis=0), strict=True)}

        # Distance from global means
        global_means = X.mean(axis=0)
        site_means = X_site.mean(axis=0)
        feature_distance_from_global = {
            name: float(abs(site_mean - global_mean))
            for name, site_mean, global_mean in zip(feature_names, site_means, global_means, strict=True)
        }

        site_stats[f"site_{int(site)}"] = {
            "n_samples": n_site_samples,
            "class_counts": class_distribution,
            "class_balance": class_balance,
            "feature_means": feature_means,
            "feature_stds": feature_stds,
            "feature_distance_from_global": feature_distance_from_global,
            "site_entropy": float(_compute_dataset_entropy(y_site)),
        }

    return site_stats


def _compute_class_statistics(
    X: np.ndarray,
    y: np.ndarray,
    unique_classes: np.ndarray,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute statistics for each class.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target labels.
    unique_classes : np.ndarray
        Unique class identifiers.
    feature_names : list or None
        Names of features.

    Returns
    -------
    dict
        Class statistics.

    """
    n_features = X.shape[1]
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    class_stats = {}

    for class_label in unique_classes:
        logger.info(f"  Processing class {int(class_label)}...")

        class_mask = y == class_label
        X_class = X[class_mask]

        # Basic counts
        n_class_samples = int(np.sum(class_mask))

        # Feature statistics for this class
        feature_means = {name: float(val) for name, val in zip(feature_names, X_class.mean(axis=0), strict=True)}
        feature_stds = {name: float(val) for name, val in zip(feature_names, X_class.std(axis=0), strict=True)}

        class_stats[f"class_{int(class_label)}"] = {
            "n_samples": n_class_samples,
            "feature_means": feature_means,
            "feature_stds": feature_stds,
        }

    return class_stats


def _compute_correlation_statistics(
    X: np.ndarray,
    y: np.ndarray,
    site_labels: np.ndarray,
    unique_sites: np.ndarray,
    unique_classes: np.ndarray,
) -> dict[str, Any]:
    """Compute correlation and relationship statistics.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target labels
    site_labels : np.ndarray
        Site labels
    unique_sites : np.ndarray
        Unique site identifiers
    unique_classes : np.ndarray
        Unique class identifiers

    Returns
    -------
    dict
        Correlation statistics

    """
    correlations = {}

    # Site-target correlation (for binary classification only)
    if len(unique_classes) == 2:
        try:
            site_target_corr = np.corrcoef(site_labels, y)[0, 1]
            correlations["site_target_correlation"] = float(site_target_corr)
        except (ValueError, RuntimeWarning):
            correlations["site_target_correlation"] = None

    # Feature correlations with target (for binary classification)
    if len(unique_classes) == 2:
        feature_target_correlations = []
        for i in range(X.shape[1]):
            try:
                corr = np.corrcoef(X[:, i], y)[0, 1]
                feature_target_correlations.append(float(corr))
            except (ValueError, RuntimeWarning):
                feature_target_correlations.append(None)
        correlations["feature_target_correlations"] = feature_target_correlations

    # Inter-site feature mean correlations
    n_sites = len(unique_sites)
    if n_sites > 1:
        site_means = np.array([X[site_labels == site].mean(axis=0) for site in unique_sites])

        # Correlation matrix between site means
        site_mean_corr_matrix = np.corrcoef(site_means)
        correlations["inter_site_feature_correlation_matrix"] = site_mean_corr_matrix.tolist()

        # Average inter-site correlation
        mask = ~np.eye(n_sites, dtype=bool)
        avg_inter_site_corr = site_mean_corr_matrix[mask].mean()
        correlations["avg_inter_site_correlation"] = float(avg_inter_site_corr)

    # Between-class separation (for all classification)
    if len(unique_classes) > 1:
        class_separation = {}
        for i in range(len(unique_classes)):
            for j in range(i + 1, len(unique_classes)):
                mean_i = X[y == unique_classes[i]].mean(axis=0)
                mean_j = X[y == unique_classes[j]].mean(axis=0)
                distance = np.linalg.norm(mean_i - mean_j)
                key = f"class_{int(unique_classes[i])}_vs_class_{int(unique_classes[j])}"
                class_separation[key] = float(distance)
        correlations["class_separation_distances"] = class_separation

    return correlations


def _compute_dataset_entropy(labels: np.ndarray) -> float:
    """Compute entropy of label distribution.

    Parameters
    ----------
    labels : np.ndarray
        Array of labels

    Returns
    -------
    float
        Entropy of the label distribution

    """
    if len(labels) == 0:
        return 0.0

    _, counts = np.unique(labels, return_counts=True)
    probs = counts / len(labels)

    # Avoid log(0) by only considering non-zero probabilities
    entropy = -np.sum(probs * np.log2(probs + 1e-10))

    return entropy


def print_statistics_summary(stats: dict[str, Any], max_features: int = 5) -> None:
    """Print a human-readable summary of statistics.

    Parameters
    ----------
    stats : dict
        Statistics dictionary from get_site_data_statistics
    max_features : int, default=5
        Maximum number of features to show in summary

    """
    logger.info("=" * 60)
    logger.info("DATASET STATISTICS SUMMARY")
    logger.info("=" * 60)

    # Overall statistics
    overall = stats["overall"]
    logger.info("\nOVERALL:")
    logger.info(f"  Samples: {overall['n_samples']}")
    logger.info(f"  Features: {overall['n_features']}")
    logger.info(f"  Sites: {overall['n_sites']}")
    logger.info(f"  Classes: {overall['n_classes']}")

    # Class distribution
    logger.info("\nCLASS DISTRIBUTION:")
    for class_key, count in overall["class_distribution"].items():
        percentage = overall["overall_class_balance"].get(class_key, 0) * 100
        logger.info(f"  {class_key}: {count} samples ({percentage:.1f}%)")

    # Site distribution
    logger.info("\nSITE DISTRIBUTION:")
    for site_key, count in overall["site_distribution"].items():
        percentage = count / overall["n_samples"] * 100
        logger.info(f"  {site_key}: {count} samples ({percentage:.1f}%)")

    # Site statistics summary
    logger.info("\nSITE STATISTICS (summary):")
    for site_key, site_data in stats["site_statistics"].items():
        logger.info(f"  {site_key}:")
        logger.info(f"    Samples: {site_data['n_samples']}")
        logger.info(f"    Class distribution: {site_data['class_counts']}")

    # Show first few feature statistics
    if "feature_statistics" in overall:
        logger.info(f"\nFEATURE STATISTICS (first {max_features} features):")
        feature_names = list(overall["feature_statistics"]["means"].keys())[:max_features]
        for name in feature_names:
            mean = overall["feature_statistics"]["means"][name]
            std = overall["feature_statistics"]["stds"][name]
            logger.info(f"  {name}: mean={mean:.4f}, std={std:.4f}")

    # Correlation summary
    if stats.get("correlations"):
        logger.info("\nCORRELATIONS:")
        if "site_target_correlation" in stats["correlations"]:
            corr = stats["correlations"]["site_target_correlation"]
            if corr is not None:
                logger.info(f"  Site-Target Correlation: {corr:.4f}")

        if "avg_inter_site_correlation" in stats["correlations"]:
            corr = stats["correlations"]["avg_inter_site_correlation"]
            logger.info(f"  Average Inter-Site Correlation: {corr:.4f}")

    logger.info("=" * 60)
