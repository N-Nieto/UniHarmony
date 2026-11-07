(site_centroid_distance-utilities)=
# Site Centroid Distance

`uniharmony.metrics.site_centroid_distance` measures how far apart the site
centroids (per-site mean vectors) are, relative to what sampling noise alone
would produce. It is a data-driven measure of the location Effect of Site
(EoS) that remains in the data, before or after harmonization.

## Quick Start

```python
from uniharmony.datasets import make_multisite_classification
from uniharmony.metrics import site_centroid_distance

X, y, sites = make_multisite_classification(n_sites=4, n_samples=800)
result = site_centroid_distance(X, sites, y=y)

result["statistic"], result["null_mean"], result["p_value"]
result["pairwise_distance"]  # site x site DataFrame
```

## What is computed

With site centroids $\bar x_k$, grand mean $\bar x$ and $n_k$ samples per site,
the between-site and within-site scatter matrices are

$$
B = \sum_k n_k (\bar x_k - \bar x)(\bar x_k - \bar x)^\top, \qquad
W = \sum_k \sum_{i \in k} (x_i - \bar x_k)(x_i - \bar x_k)^\top .
$$

The statistic is the pseudo-F

$$
F = \frac{\operatorname{tr}(M B) / (K - 1)}{\operatorname{tr}(M W) / (n - K)},
$$

where $M$ sets the geometry (`metric`):

| `metric` | $M$ | $F$ equals |
|---|---|---|
| `"euclidean"` | identity | Calinski-Harabasz index of the site partition |
| `"standardized"` | inverse pooled within-site variances | mean univariate ANOVA F over features |
| `"mahalanobis"` (default) | inverse pooled within-site covariance (Ledoit-Wolf shrinkage of the correlation) | Lawley-Hotelling trace $/\,((K-1)\,p)$ (without shrinkage) |

Without EoS, $E[B] = (K-1)\Sigma$ and $E[W] = (n-K)\Sigma$, so $F \approx 1$
for the first two metrics. The exact null is obtained by permuting the site
labels, which gives `null_mean`, `z_score` and `p_value`. `r2` is the share of
the variance explained by site.

## Pairwise distances

The squared distance between two estimated centroids is biased by sampling
noise:

$$
E\,\lVert \bar x_a - \bar x_b \rVert_M^2 = \lVert \mu_a - \mu_b \rVert_M^2
+ \frac{\operatorname{tr}(M S_a)}{n_a} + \frac{\operatorname{tr}(M S_b)}{n_b} .
$$

Small sites therefore look far away even without EoS. `pairwise_distance`
subtracts this term (the unbiased linear-kernel MMD estimator) and clips
negative values to 0. With the Mahalanobis metric a distance $D$ is a
multivariate standardized mean difference: for Gaussian sites, the best
balanced accuracy of telling the two sites apart is $\Phi(D/2)$
(0.69 for $D = 1$, 0.84 for $D = 2$).

## Good practice

- **Compute it on held-out data.** ComBat-like methods equalize the site means
  of the data they were fitted on by construction.
- **Pass the biology as `y`** when its distribution differs between sites
  (e.g. diagnosis). Sites are then compared within each class and the labels
  are permuted within classes, so biological differences are not counted as
  EoS. `y` must be categorical.
- **Compute it in the feature space, not on t-SNE or UMAP embeddings**, which
  do not preserve distances between clusters. PCA projections can be used to
  *show* the centroids.
- **It only measures site means.** Differences in variance or covariance
  between sites are not captured; complement it with
  `uniharmony.metrics.evaluate_site_prediction` using a flexible model.
- **Cost.** The Mahalanobis metric inverts an `n_features x n_features` matrix
  per permutation; for many features use `"standardized"` or fewer
  permutations.
