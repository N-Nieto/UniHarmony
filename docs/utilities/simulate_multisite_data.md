(simulate-multisite-data)=
# Multisite Data Simulation

## Overview

``uniharmony.datasets.make_multisite_classification`` generates synthetic datasets that mimic multi-center studies,
with a known class signal, a known Effect of Site (EoS) and noise. It is used to test harmonization methods
and to teach how site effects bias machine learning.

## Usage

```python
from uniharmony.datasets import make_multisite_classification

# 3 sites, 500 samples in total
X, y, sites = make_multisite_classification(
    n_sites=3,
    n_samples=500,
    n_features=20,
    random_state=42,  # for reproducibility
)
```

## Behind the Scenes

Each sample combines three components:

Feature = Signal + Site Effect + Noise

1. A pool of base samples (signal only) is generated with a scikit-learn generator (`signal_type`: `"linear"`,
   `"blobs"`, `"gaussian_quantiles"`, `"moons"` or `"circles"`), with exactly the number of samples of each class
   that the sites need.
2. Each site draws its samples from the pool **without replacement**, following its class proportions
   (`balance_per_site`). No sample appears twice, whatever the imbalance or the site sizes.
3. Optional covariates (`covariates`, e.g. `"age"`, `"sex"`) are drawn per site, correlated with the base samples.
4. The site effect (`site_effect_type`, `site_effect_strength`) is applied; a strength of 0 means no effect.
5. Gaussian noise (`noise_strength`) is added.

With `return_base_data=True` the base samples are also returned, row-aligned with `X`, so that `X - X_base` is the
site effect plus noise of every sample.

## Site effects

| `site_effect_type` | Effect for a site with strength `s` |
|---|---|
| `"location"` | adds a per-feature offset drawn from U(0, s) |
| `"scale"` | multiplies the deviations from the global mean by a per-feature factor exp(U(-0.1 s, 0.1 s)) |
| `"location+scale"` | scale, then location |
| `"variance"` | adds Gaussian noise with a per-feature SD drawn from U(0, s) |
| `"nonlinear"` | adds `w * X**2`, with `w` drawn from U(0, 0.1 s) |
| `"dropout"` | sets values to 0 with probability min(0.1 s, 1) |

With `site_effect_homogeneous=False`, each sample's effect varies around its site's effect, by as much as the site
effects vary between sites.

The offsets are drawn per feature, so with the same strength, sites are easier to tell apart when there are more
features.

## Flexible Configuration

You can choose the number of sites, the total number of samples or the size of each site, the number of features
and classes, the class proportions of each site, and the strength of the site effect and of the noise per site.
