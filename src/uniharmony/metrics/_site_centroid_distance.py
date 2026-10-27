"""Site centroid distance diagnostic for harmonization evaluation."""

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import structlog
from sklearn.covariance import ledoit_wolf_shrinkage, shrunk_covariance
from sklearn.utils import check_random_state
from sklearn.utils.validation import FLOAT_DTYPES, check_array, check_consistent_length

from uniharmony._utils import validate_sites


__all__ = [
    "site_centroid_distance",
]

logger = structlog.get_logger()

_METRICS = ("euclidean", "standardized", "mahalanobis")


def site_centroid_distance(
    X: npt.ArrayLike,
    sites: npt.ArrayLike,
    y: npt.ArrayLike | None = None,
    metric: str = "mahalanobis",
    n_permutations: int = 1000,
    random_state: int | np.random.RandomState | None = 42,
) -> dict[str, Any]:
    """Quantify how far apart the site centroids are, relative to chance.

    If no Effect of Site (EoS) is left in ``X``, the site centroids (the
    per-site mean vectors) differ only by sampling noise. This function
    measures the between-site spread of the centroids relative to the
    within-site spread, as a pseudo-F statistic::

        F = [tr(M B) / (n_cells - n_strata)] / [tr(M W) / (n_samples - n_cells)]

    where ``B`` and ``W`` are the between-site and within-site scatter
    matrices and ``M`` is set by ``metric``. Without an EoS, ``F`` is close
    to its permutation null mean (about 1 for the ``"euclidean"`` and
    ``"standardized"`` metrics). The null is obtained by permuting the site
    labels (within the classes of ``y``, if given).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix. To evaluate a harmonization method, pass held-out
        data harmonized with a model fitted on training data (see Notes).
    sites : array-like of shape (n_samples,)
        Site label of each sample. Every site needs at least 2 samples.
    y : array-like of shape (n_samples,), optional (default None)
        Categorical biological variable (e.g. diagnosis or sex) whose
        effect must not be counted as EoS. If given, sites are compared
        within each class of ``y``: class means are removed, the scatter
        is computed per site-class cell, and labels are permuted within
        classes. Use it when the class proportions differ between sites.
    metric : {"mahalanobis", "standardized", "euclidean"}, optional (default "mahalanobis")
        Geometry used to measure the distances (choice of ``M``):

        - ``"mahalanobis"``: inverse of the pooled within-site covariance.
          Sensitive to EoS along low-variance directions of correlated
          features. The within-site correlation matrix is shrunk towards
          the identity (Ledoit-Wolf), so the result does not depend on the
          feature units; with full shrinkage it equals ``"standardized"``
          and without shrinkage ``F`` is the Lawley-Hotelling trace divided
          by ``(n_cells - n_strata) * n_features``. Costs one
          ``n_features x n_features`` inversion per permutation.
        - ``"standardized"``: each feature divided by its pooled
          within-site variance. ``F`` is the mean univariate ANOVA F over
          features.
        - ``"euclidean"``: features as given. ``F`` is the
          Calinski-Harabasz index of the site partition.

    n_permutations : int, optional (default 1000)
        Number of site-label permutations for the null distribution. Use 0
        to skip the permutation test.
    random_state : int, RandomState instance or None, optional (default 42)
        Seed of the permutations.

    Returns
    -------
    dict
        - ``"statistic"`` : float, the pseudo-F.
        - ``"r2"`` : float, share of the (``M``-weighted) variance explained
          by site, in [0, 1].
        - ``"null_mean"``, ``"null_std"`` : float, mean and standard
          deviation of the permutation null of ``F``.
        - ``"z_score"`` : float, ``(F - null_mean) / null_std``.
        - ``"p_value"`` : float, permutation p-value,
          ``(1 + #{F_perm >= F}) / (1 + n_permutations)``.
        - ``"null_distribution"`` : np.ndarray of shape (n_permutations,).
        - ``"pairwise_distance"`` : pd.DataFrame of shape (n_sites, n_sites),
          bias-corrected distance between each pair of site centroids.

        The null entries are NaN when ``n_permutations < 2``.

    Raises
    ------
    ValueError
        If ``metric`` is unknown, ``n_permutations`` is negative, fewer
        than 2 sites are given, a site has fewer than 2 samples, or no
        class of ``y`` contains more than one site.

    Notes
    -----
    The squared distance between two estimated centroids is biased upwards
    by sampling noise: ``E||m_a - m_b||^2 = ||mu_a - mu_b||^2 + tr(S_a)/n_a
    + tr(S_b)/n_b``. Small sites therefore look far away even without EoS.
    ``"pairwise_distance"`` subtracts this term (the unbiased estimator of
    the linear-kernel MMD) and clips negative values to 0.

    With ``metric="mahalanobis"``, a pairwise distance ``D`` is a
    multivariate standardized mean difference. For two Gaussian sites
    with a common covariance, the best balanced accuracy of predicting
    the site from ``X`` is ``Phi(D / 2)``: 0.69 for ``D = 1`` and 0.84 for
    ``D = 2`` (compare with :func:`evaluate_site_prediction`).

    The metric only sees site *means*: differences in variance or
    covariance between sites are not measured. Methods like ComBat equalize
    the site means of the data they were fitted on by construction, so
    compute this metric on held-out data. Do not compute it on t-SNE or
    UMAP embeddings: they do not preserve distances between clusters.

    Examples
    --------
    >>> from uniharmony.datasets import make_multisite_classification
    >>> from uniharmony.metrics import site_centroid_distance
    >>> X, y, sites = make_multisite_classification(n_sites=3, n_samples=300)
    >>> result = site_centroid_distance(X, sites, y=y, n_permutations=200)
    >>> result["p_value"] < 0.05
    True

    """
    if metric not in _METRICS:
        raise ValueError(f"metric must be one of {_METRICS}, got {metric!r}")
    if n_permutations < 0:
        raise ValueError(f"n_permutations must be >= 0, got {n_permutations}")

    X = check_array(X, dtype=FLOAT_DTYPES)
    site_labels, site_codes, strata = _encode_labels(X, sites, y)
    n_strata, n_sites = strata.max() + 1, len(site_labels)

    # Drop constant features, then remove the class means so that only differences within classes count as EoS.
    X = X[:, X.var(axis=0) > 0]
    X = X - _group_means(X, strata, n_strata)[0][strata]
    cells = strata * n_sites + site_codes
    n_cells = len(np.unique(cells))
    df_between, df_within = n_cells - n_strata, X.shape[0] - n_cells
    if df_between == 0:
        raise ValueError("No class of y contains more than one site, so sites cannot be compared.")

    # Quantities that do not change when site labels are permuted within strata.
    total = X.T @ X if metric == "mahalanobis" else np.sum(X**2, axis=0)
    shrinkage = 0.0
    if metric == "mahalanobis":
        residuals = X - _group_means(X, cells, n_strata * n_sites)[0][cells]
        shrinkage = ledoit_wolf_shrinkage(residuals * _inverse(residuals.std(axis=0)), assume_centered=True)

    def statistic(site_codes: npt.NDArray) -> tuple[float, float, npt.NDArray]:
        """Pseudo-F, R² and the inner-product weights for one site labelling."""
        means, counts = _group_means(X, strata * n_sites + site_codes, n_strata * n_sites)
        if metric == "mahalanobis":
            between = (means.T * counts) @ means
            # Shrink the within-site correlation (not covariance) so that the result does not depend on feature units.
            inv_sd = _inverse(np.sqrt(np.diag(total - between) / df_within))
            correlation = (total - between) / df_within * np.outer(inv_sd, inv_sd)
            weights = np.linalg.pinv(shrunk_covariance(correlation, shrinkage), hermitian=True) * np.outer(inv_sd, inv_sd)
        else:
            between = counts @ means**2  # diagonal of the between-site scatter
            within_var = (total - between) / df_within
            weights = np.ones_like(within_var) if metric == "euclidean" else _inverse(within_var)
        tr_between, tr_within = np.sum(weights * between), np.sum(weights * (total - between))
        f_stat = (tr_between / df_between) / (tr_within / df_within)
        return f_stat, tr_between / (tr_between + tr_within), weights

    f_stat, r2, weights = statistic(site_codes)

    null = _permutation_null(statistic, site_codes, strata, n_permutations, random_state)
    summary = _summarize_null(f_stat, null)

    pairwise = _pairwise_centroid_distance(X, site_codes, n_sites, weights)
    logger.info(f"Site centroid distance ({metric}): F = {f_stat:.3f}", **summary)

    return {
        "statistic": float(f_stat),
        "r2": float(r2),
        **summary,
        "null_distribution": null,
        "pairwise_distance": pd.DataFrame(pairwise, index=site_labels, columns=site_labels),
    }


def _inverse(values: npt.NDArray) -> npt.NDArray:
    """Element-wise 1 / values, with 0 where values is 0 (constant features get no weight)."""
    return np.divide(1.0, values, out=np.zeros_like(values), where=values > 0)


def _group_means(X: npt.NDArray, groups: npt.NDArray, n_groups: int) -> tuple[npt.NDArray, npt.NDArray]:
    """Return the mean of ``X`` per group and the group sizes (empty groups have mean 0)."""
    counts = np.bincount(groups, minlength=n_groups)
    sums = np.zeros((n_groups, X.shape[1]))
    np.add.at(sums, groups, X)
    return sums / np.maximum(counts, 1)[:, None], counts


def _pairwise_centroid_distance(X: npt.NDArray, site_codes: npt.NDArray, n_sites: int, weights: npt.NDArray) -> npt.NDArray:
    """Bias-corrected distances between site centroids under the inner product ``weights``.

    ``weights`` is a vector (diagonal metric) or a symmetric matrix.
    """

    def weigh(A: npt.NDArray) -> npt.NDArray:
        return A * weights if weights.ndim == 1 else A @ weights

    centroids, counts = _group_means(X, site_codes, n_sites)
    gram = weigh(centroids) @ centroids.T
    sq_dist = np.diag(gram)[:, None] + np.diag(gram)[None, :] - 2 * gram

    # Sampling noise of each centroid: tr(M S_k) / n_k, with S_k the site covariance.
    residuals = X - centroids[site_codes]
    sq_norms = np.sum(weigh(residuals) * residuals, axis=1)
    noise = np.bincount(site_codes, weights=sq_norms) / (counts - 1) / counts

    sq_dist -= noise[:, None] + noise[None, :]
    np.fill_diagonal(sq_dist, 0)
    return np.sqrt(np.clip(sq_dist, 0, None))


def _encode_labels(X: npt.NDArray, sites: npt.ArrayLike, y: npt.ArrayLike | None) -> tuple[npt.NDArray, ...]:
    """Validate ``sites`` and ``y`` and return the site labels, site codes and strata (codes of ``y``)."""
    sites = check_array(sites, dtype=None, ensure_2d=False)
    check_consistent_length(X, sites)
    validate_sites(sites)
    site_labels, site_codes, site_counts = np.unique(sites, return_inverse=True, return_counts=True)
    if site_counts.min() < 2:
        raise ValueError(f"Every site needs at least 2 samples, got {dict(zip(site_labels, site_counts, strict=True))}")

    if y is None:
        return site_labels, site_codes, np.zeros(X.shape[0], dtype=int)
    y = check_array(y, dtype=None, ensure_2d=False)
    check_consistent_length(X, y)
    return site_labels, site_codes, np.unique(y, return_inverse=True)[1]


def _permutation_null(
    statistic: Callable, site_codes: npt.NDArray, strata: npt.NDArray, n_permutations: int, random_state
) -> npt.NDArray:
    """Statistic for ``n_permutations`` permutations of the site labels within strata."""
    rng = check_random_state(random_state)
    strata_idx = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    null = np.empty(n_permutations)
    for i in range(n_permutations):
        permuted = site_codes.copy()
        for idx in strata_idx:
            permuted[idx] = rng.permutation(site_codes[idx])
        null[i] = statistic(permuted)[0]
    return null


def _summarize_null(f_stat: float, null: npt.NDArray) -> dict[str, float]:
    """Compare the observed statistic with its permutation null (NaN without permutations)."""
    if len(null) < 2:
        return dict.fromkeys(("null_mean", "null_std", "z_score", "p_value"), np.nan)
    null_mean, null_std = null.mean(), null.std(ddof=1)
    return {
        "null_mean": float(null_mean),
        "null_std": float(null_std),
        "z_score": float((f_stat - null_mean) / null_std),
        "p_value": float((1 + np.sum(null >= f_stat)) / (1 + len(null))),
    }
