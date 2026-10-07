(intrasite-long)=
# Intra-Site Interpolation (ISI)

Intra-Site Interpolation (ISI) removes the association between acquisition site and target in a training set by
**balancing the classes within every site**. When one site recruits mostly patients and another mostly controls, a
machine-learning model can predict the diagnosis from the effect of site (EoS) instead of from biology. After ISI, every
class has the same number of samples in every site, so `P(y | site)` is uniform and the site carries no information about
the target.

Minority classes are over-sampled by **interpolating between real samples of the same class and the same site** (and,
optionally, the same covariate stratum, e.g. sex and age bin). The synthetic samples therefore keep the biological
variability of their class *and* the site effect of their site: the site effect becomes identical across classes, so it
can no longer be used as a shortcut. Interpolation cannot create information, so ISI limits how many synthetic samples a
class may receive and closes the remaining imbalance by **under-sampling** the larger classes: both meet in the middle.

Key features
------------
- Site-wise class balancing with any imblearn over-sampler (SMOTE and variants, ADASYN, random over-sampling).
- A data-driven cap on the number of synthetic samples (variance preservation), or a fixed / custom cap.
- Any count-controlled imblearn under-sampler for the remaining imbalance (random, NearMiss, ClusterCentroids, ...).
- Optional covariate strata that restrict who is interpolated with whom while preserving `P(covariates | class, site)`.
- Classification and regression (the target is binned; synthetic targets are interpolated from their parents).
- A full audit trail: `sample_indices_`, `is_synthetic_`, `summary()`.

---

## Algorithm

For every site `s` and class `c` with `n_sc` real samples:

1. **Cap.** The amplification of the cell, `r = n_synthetic / n_sc`, may not exceed `r*_sc` (see below).
2. **Target.** Every class of the site is brought to

   ```
   T_s = min( n_max ,  min_c floor(n_sc * (1 + r*_sc)) )
   ```

   with `n_max` the largest class of the site (`balance_strategy="per_site"`) or the largest class of any site
   (`balance_strategy="global_max"`).
3. **Resample.** Classes below `T_s` are over-sampled with the `interpolator`; classes above `T_s` are under-sampled with
   the `undersampler`. Kept real samples are returned unchanged.

Examples (`per_site`):

| Site | Class counts | Cap of the minority class | Result |
|---|---|---|---|
| A | 300 / 100 | none (`max_amplification=None`) | 300 / 300 (200 synthetic) |
| A | 300 / 100 | 1.0 | 200 / 200 (100 synthetic, 100 real removed) |
| A | 300 / 100 | 0 | 100 / 100 (pure under-sampling) |
| B | 10000 / 2 | auto (two samples cannot be interpolated into thousands) | 2 / 2, with a warning |

---

## How many samples can be created safely? (`max_amplification="auto"`)

SMOTE-like interpolation draws new samples on segments between a sample and one of its nearest neighbours. Such samples
are **less spread** than real ones: their variance is a fraction `rho` of the real variance of the class. `rho` is close
to 1 when the class is densely sampled (many samples per effective dimension, so neighbours are close) and drops to about
0.5 when a few samples have to span many dimensions.

The default `max_amplification="auto"` measures `rho` for each site-class cell, from a pilot batch of synthetic samples
(`rho = mean_j var_synthetic_j / var_real_j`). Mixing `r` synthetic samples per real sample changes the variance of the
class by `r / (1 + r) * |1 - rho|`; keeping this below `variance_tolerance = eps` (default 0.1) gives

```
r* = eps / (|1 - rho| - eps)    if |1 - rho| > eps,    otherwise no cap
```

The number of samples, the number of features and their collinearity therefore enter through the data themselves. In
simulations with SMOTE (k = 5), `rho` depends mainly on the number of real samples per effective dimension `n / d_eff`
(`d_eff` = participation ratio of the feature correlation matrix, reported as `effective_dim_`):

| `n / d_eff` | < 3 | 3 – 10 | 10 – 30 | 30 – 100 | 100 – 300 | > 300 |
|---|---|---|---|---|---|---|
| `rho` | ~0.53 | ~0.64 | ~0.73 | ~0.80 | ~0.84 | ~0.92 |
| `r*` (eps = 0.1) | ~0.25 | ~0.4 | ~0.55 | ~1 | ~1.5 | no cap |

The table holds for a low effective dimension (`d_eff` up to about 10, typical of correlated imaging features). With many
independent features (`d_eff` of 20 to 100), nearest neighbours stay far apart whatever the sample size: `rho` levels off at
about 0.7 to 0.75 and the cap at about 0.5 to 0.7 synthetic samples per real one.

For example, 30 samples of a class with 5 effective dimensions can receive about 13 synthetic samples; 3000 samples with
3 effective dimensions are not capped; 3000 samples of 100 independent features (no collinearity) can receive about 1400.

Other options:

- `max_amplification=<float>`: the same cap for every cell (`0` = under-sampling only).
- `max_amplification=<callable>`: `f(X_cell) -> float`, for custom rules.
- `max_amplification=None`: no cap, i.e. pure over-sampling (behaviour of earlier versions).

Random over-sampling (`interpolator="random"`) duplicates samples: it keeps the variance (`rho ~ 1`) and is not capped by
this rule, but it adds no new variability. It is kept as a baseline.

A cell with a single real sample cannot be interpolated (cap 0). When more than half of the real samples of a site must be
removed, ISI warns: such a site cannot be balanced without inventing or discarding most of its data.

---

## Basic usage

```python
from uniharmony.datasets import make_multisite_classification
from uniharmony.interpolation import IntraSiteInterpolation

X, y, sites = make_multisite_classification(balance_per_site=[[0.8, 0.2], [0.3, 0.7]])

isi = IntraSiteInterpolation(interpolator="smote", random_state=42)
X_balanced, y_balanced = isi.fit_resample(X, y, sites=sites)

isi.summary()               # one row per site and class: real / removed / created samples, cap, rho
isi.sites_resampled_        # site of each output sample
isi.is_synthetic_           # which output samples were created
isi.sample_indices_         # index of each real output sample in the input (-1 for synthetic samples)
```

### Use it on training data only

ISI must only see training data. In cross-validation, put it in an `imblearn.pipeline.Pipeline` and route `sites` with
scikit-learn metadata routing:

```python
import sklearn
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

sklearn.set_config(enable_metadata_routing=True)
isi = IntraSiteInterpolation(random_state=0).set_fit_resample_request(sites=True)
pipe = Pipeline([("isi", isi), ("clf", LogisticRegression())])
scores = cross_validate(pipe, X, y, params={"sites": sites}, scoring="roc_auc")
```

Applying ISI before splitting leaks information from the test folds into the synthetic training samples.

### Choosing the samplers

```python
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import NearMiss

isi = IntraSiteInterpolation(
    interpolator=BorderlineSMOTE(k_neighbors=3),   # or "smote", "svm-smote", "adasyn", "kmeans-smote", "random"
    undersampler=NearMiss(version=1),               # or "random", "nearmiss-2", "cluster-centroids", ...
    max_amplification=1.0,                          # at most one synthetic sample per real sample
)
```

Neighbour parameters are reduced automatically for small cells. Under-samplers must accept a target count per class;
cleaning methods (Tomek links, ENN, ...) cannot, and should be applied before ISI if needed. Prototype generators such as
`ClusterCentroids` create new samples, which are flagged in `is_synthetic_`. With `undersampler=None`, ISI raises an error
if the cap would require removing samples.

---

## Covariates

Covariates restrict **who is interpolated with whom**: synthetic samples are only interpolated between samples of the same
class, site and covariate stratum (unique combination of categorical values and continuous-covariate bins, computed within
each site). They are spread over the strata in proportion to the real samples of the class, and under-sampling is
stratified the same way, so the distribution of the covariates within each class and site is preserved (for example, if
80% of the patients of a site are women, 80% of its synthetic patients are interpolated between women).

```python
isi = IntraSiteInterpolation(n_bins_cont_cov=3, random_state=0)
X_bal, y_bal = isi.fit_resample(X, y, sites=sites, categorical_covariate=sex, continuous_covariate=age)
```

Strata with fewer samples than the interpolator needs are skipped. Covariates of the real output samples can be recovered
with `sample_indices_`.

---

## Regression

For continuous targets (e.g. brain age), the target is binned (`n_bins`, `binning_strategy`) and each bin is treated as a
class. The target of a synthetic sample is interpolated between its two parent samples (`y = y_a + lam * (y_b - y_a)`),
which recovers SMOTE's interpolation exactly. A bin that is missing from a site cannot be created by interpolation; ISI
warns and balances that site over the bins it has.

---

## Balance strategies

### `per_site` (default)
Every site is balanced to its own largest class (within the cap).

- Site A: 100 class-0, 20 class-1, no cap → 100 / 100
- Site B: 30 class-0, 70 class-1, no cap → 70 / 70

### `global_max`
Every site is balanced towards the largest class of any site, so sites also get similar sizes. Small sites need much more
amplification; the cap still applies, so sites that cannot be amplified that much stay smaller (but balanced).

- Site A: 100 class-0, 20 class-1; Site B: 30 class-0, 70 class-1; no cap → both sites 100 / 100

---

## Attributes

| Attribute | Content |
|---|---|
| `sites_resampled_` | Site of each output sample |
| `sample_indices_`, `is_synthetic_` | Origin of each output sample |
| `target_counts_` | Samples per class in each site after resampling |
| `samples_created_`, `samples_removed_` | `{site: {class: n}}` |
| `amplification_`, `amplification_cap_`, `variance_ratio_` | Realised amplification, its cap and the measured `rho` per site and class |
| `effective_dim_` | Participation ratio of the features (within site-class cells) |
| `summary()` | All of the above as a table |

---

## Reference implementation

Source code: https://github.com/N-Nieto/IntraSiteInterpolator.git

## Citation

Paper in preparation.
