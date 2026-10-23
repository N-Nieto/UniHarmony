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
can no longer be used as a shortcut. **ISI only adds samples: no real sample is ever removed.**

Key features
------------
- Site-wise class balancing with any imblearn over-sampler (SMOTE and variants, ADASYN, random over-sampling).
- A diagnostic of how much interpolation each class can take without losing its variance, with a clear warning when
  balancing needs more.
- `SiteBalancedResampler`, a separate wrapper that limits interpolation to a cap (for example the safe amount) and
  closes the remaining imbalance by under-sampling the larger classes (see below).
- Optional covariate strata that restrict who is interpolated with whom while preserving `P(covariates | class, site)`.
- Classification and regression (the target is binned; synthetic targets are interpolated from their parents).
- A full audit trail: `sample_indices_`, `is_synthetic_`, `summary()`.

---

## Algorithm

For every site `s` and class `c` with `n_sc` real samples:

1. **Target.** Every class of the site is brought to `T_s`: the largest class of the site
   (`balance_strategy="per_site"`), the largest class of any site (`balance_strategy="global_max"`) or a given number per
   site (`balance_strategy={site: T_s}`, at least the largest class of the site).
2. **Interpolation.** Classes below `T_s` are over-sampled with the `interpolator`. Real samples are returned unchanged.
3. **Diagnostic.** For every class that needs over-sampling, ISI measures how much of its variance interpolation keeps
   and derives the *safe amplification* `r*` (next section). If balancing needs more synthetic samples than that, ISI
   warns and points to `SiteBalancedResampler`.

Every class needs at least **two samples in every site**: a single sample cannot be interpolated (ISI raises an error).
For regression, target bins with fewer samples in a site are left as they are, with a warning.

Examples (`per_site`):

| Site | Class counts | Result |
|---|---|---|
| A | 300 / 100 | 300 / 300 (200 synthetic) |
| B | 1000 / 2 | 1000 / 1000, with a warning: 998 of the 1000 samples of the small class are synthetic |

---

## How much interpolation is safe? The variance rule

**Interpolation shrinks a class.** SMOTE creates a new sample on the segment between a real sample and one of its
nearest neighbours of the same class. The new sample is therefore pulled towards the inside of the class: synthetic
samples are less spread out than real ones. ISI measures this with a pilot batch of synthetic samples drawn from **all
real samples of the site** (nothing is removed for this):

```
rho = variance of the synthetic samples / variance of the real samples      (per feature, averaged)
```

- When many samples densely cover the class, neighbours are close, a new sample lands near a real one and `rho` is close
  to 1.
- When the samples are sparse, neighbours are far apart and a new sample lands in between, closer to the centre:
  mixing two independent points with a uniform weight `l` keeps `E[(1 - l)^2 + l^2] = 2/3` of the variance.
- With only **two samples**, all synthetic samples lie on one segment: `rho = 1/6`.

**How much variance the class keeps.** After over-sampling, a class has `n` real samples (variance `v`) and `r * n`
synthetic samples (variance `rho * v`). Its variance is the weighted average

```
v_after = v * (1 + r * rho) / (1 + r)      ->   loss = r * (1 - rho) / (1 + r)
```

**Safe amplification.** Keeping the loss below `variance_tolerance = eps` (default 0.2, i.e. the class keeps at least
80% of its variance) gives

```
r* = eps / ((1 - rho) - eps)     if 1 - rho > eps;   no limit otherwise
```

For example, `rho = 0.6` gives `r* = 0.2 / (0.4 - 0.2) = 1`: at most one synthetic sample per real one. A class of 100
real samples facing a class of 500 needs `r = 4` and would keep only `(1 + 4 * 0.6) / 5 = 68%` of its variance, so ISI
warns. Two samples (`rho = 1/6`) give `r* = 0.32`: not even one synthetic sample is safe.

**Where samples, features and collinearity come in.** `rho` is measured on the data, so it reflects the number of
samples, the number of features and their correlation together. In simulations with SMOTE (k = 5) it depends mainly on
the number of real samples per *effective* dimension `n / d_eff` (`d_eff` = participation ratio of the feature
correlation matrix, reported as `effective_dim_` and estimated from at most 2000 samples):

| `n / d_eff` | < 3 | 3 – 10 | 10 – 30 | 30 – 100 | > 100 |
|---|---|---|---|---|---|
| `rho` | ~0.53 | ~0.64 | ~0.73 | ~0.80 | 0.84 – 0.92 |
| `r*` (eps = 0.2) | ~0.65 | ~1.5 | ~2.5 | no limit | no limit |

The table holds for a low effective dimension (`d_eff` up to about 10, typical of correlated imaging features). With many
independent features (`d_eff` of 20 to 100), nearest neighbours stay far apart whatever the sample size: `rho` levels off
at about 0.65 to 0.75 and `r*` at about 1.5 to 3.5.

Random over-sampling (`interpolator="random"`) duplicates samples: it keeps the variance (`rho ~ 1`) and is not limited
by this rule, but it adds no new variability. It is kept as a baseline.

`estimate_safe_amplification(X, y, sites)` returns `rho` and `r*` of every site and class without resampling, to check a
data set before choosing a method.

---

## Basic usage

```python
from uniharmony.datasets import make_multisite_classification
from uniharmony.interpolation import IntraSiteInterpolation

X, y, sites = make_multisite_classification(balance_per_site=[[0.8, 0.2], [0.3, 0.7]])

isi = IntraSiteInterpolation(interpolator="smote", random_state=42)
X_balanced, y_balanced = isi.fit_resample(X, y, sites=sites)

isi.summary()               # per site and class: real / created samples, rho, safe amplification, ...
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

### Tune hyper-parameters with ISI inside the inner cross-validation

The same leak happens *inside* the training set when a model tunes itself on ISI output. Synthetic samples are
interpolated between their parents, so validating on a sample whose parents were used for training is optimistic.
Estimators with built-in tuning (`RidgeCV` / `RidgeClassifierCV` with generalised cross-validation,
`LogisticRegressionCV`, early stopping on a validation split) then select too little regularisation: in our simulations
`RidgeCV` fitted on ISI output chose penalties about 10 times smaller for sex classification and 30-10,000 times
smaller for age regression than with ISI nested in the tuning, which overfitted and worsened the performance on new
sites (by 0.015 AUC and 0.9 years of mean absolute error). Tune the whole pipeline instead:

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

with sklearn.config_context(enable_metadata_routing=True):
    isi = IntraSiteInterpolation(random_state=0).set_fit_resample_request(sites=True)
    pipe = Pipeline([("isi", isi), ("scale", StandardScaler()), ("ridge", Ridge())])
    search = GridSearchCV(pipe, {"ridge__alpha": np.logspace(-1, 5, 13)})
    search.fit(X, y, sites=sites)
```

### Combining interpolation with under-sampling: `SiteBalancedResampler`

ISI itself only interpolates. When it warns that a class needs more interpolation than is safe, wrap it in
`SiteBalancedResampler`, which limits the amplification of every class to a cap and **under-samples** the larger classes
of the site to meet the smaller ones:

```python
from uniharmony.interpolation import IntraSiteInterpolation, SiteBalancedResampler

sbr = SiteBalancedResampler(
    IntraSiteInterpolation("smote", random_state=0),  # any interpolator, also "random" (duplication)
    max_amplification="auto",            # the safe amplification of each class; or a number, f(X_cell) or None
    undersampler="cluster-centroids",    # default; or "nearmiss", "nearmiss-2", "nearmiss-3", "random", an imblearn instance
    random_state=0,
)
X_bal, y_bal = sbr.fit_resample(X, y, sites=sites)
sbr.summary()                            # real, removed, created and final samples per site and class
```

For every site, each class below the largest one gets a cap `r_sc` and the site target becomes
`T_s = min(n_max, min_c floor(n_sc * (1 + r_sc)))`. The classes above `T_s` are under-sampled (stratified by the
covariates), and ISI brings the classes below `T_s` up to it. With `max_amplification="auto"` the cap is the safe
amplification `r*`, measured on **all** real samples of the site before anything is removed. `max_amplification=0`
gives pure within-site under-sampling and `max_amplification=None` gives ISI alone.

| Site | Class counts | `max_amplification` | Result |
|---|---|---|---|
| A | 300 / 100 | `None` | 300 / 300 (200 synthetic; same as ISI) |
| A | 300 / 100 | `1.0` | 200 / 200 (100 synthetic, 100 real removed) |
| A | 300 / 100 | `0` | 100 / 100 (pure under-sampling) |
| B | 1000 / 2 | `"auto"` | 2 / 2, with a warning that 998 real samples were removed |

The default under-sampler, `"cluster-centroids"`, runs k-means on the class and keeps the real sample closest to each
centroid (`ClusterCentroids(voting="hard")`), so the kept samples cover the whole class. Any imblearn under-sampler that
selects real samples and accepts a target count per class can be used. Cleaning methods (Tomek links, ENN, ...) cannot
reach a count, and prototype generators (`ClusterCentroids(voting="soft")`) create new samples; both are rejected.
`SiteBalancedResampler` warns when more than half of the real samples of a site are removed: the imbalance is then beyond
what within-site balancing can repair without losing most of the data.

The cap trades the two failure modes of within-site balancing against each other: up-weighting (interpolation,
duplication, re-weighting) keeps all data but, with many features, leaves part of the site-target association when
classes are strongly imbalanced; under-sampling removes the association but discards data. Which one costs less depends
on the data, so check both with the probe below.

### When ISI is not enough

- **Strong imbalance with many features.** With more features than samples (voxel-wise images), models that are only
  moderately regularised can fit the few real samples of a heavily over-sampled class one by one, and the synthetic
  samples add little to that fit. ISI then removes the site-target association only partly, exactly like re-weighting or
  random over-sampling with the same amplification; under-sampling does not have this problem. If ISI warns that
  classes exceed their safe amplification, use `SiteBalancedResampler(IntraSiteInterpolation(), max_amplification="auto")`.
- **Local and kernel models.** Synthetic samples are denser and less dispersed than real ones. In our simulations a
  k-nearest-neighbour classifier recognised the synthetic class of each site by its density and learned the inverse
  shortcut, and an RBF support vector regression became worse than without correction. For such models prefer sample
  weights where the model accepts them, under-sampling (`SiteBalancedResampler(max_amplification=0)`) or random
  over-sampling (`interpolator="random"`), and check the result with the probe below.
- **Missing target ranges** (see Regression).

Check what is left of the shortcut by repeating the cross-validation after permuting the target *within* each site: the
biology is destroyed, the site-target association is kept, and any performance above chance comes from the site.

```python
rng = np.random.default_rng(0)
y_perm = y.copy()
for s in np.unique(sites):
    idx = np.flatnonzero(sites == s)
    y_perm[idx] = y[rng.permutation(idx)]
probe = cross_validate(pipe, X, y_perm, params={"sites": sites}, scoring="roc_auc")  # ~0.5 if no shortcut is left
```

### Choosing the interpolator

`"smote"` (default) works with any class of at least two samples. `"borderline-smote"`, `"svm-smote"`, `"adasyn"` and
`"kmeans-smote"` concentrate samples near the class boundary or in dense clusters and can fail on very small classes or
strata; the error then suggests SMOTE. Neighbour parameters are reduced automatically for small classes.

---

## Two further uses

### A sanity check: how much of my score comes from the site?

In real data you do not know how much of a cross-validated score comes from the effect of site. Run the same pipeline
twice, without and with ISI inside the training folds, and compare the scores. The test folds keep the site-target
association of the data, so a model that relied on the site loses that advantage once its training folds no longer
reward it:

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(5, shuffle=True, random_state=0)
base = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression())])
score = cross_val_score(base, X, y, cv=cv, scoring="roc_auc").mean()

with sklearn.config_context(enable_metadata_routing=True):
    isi = IntraSiteInterpolation(random_state=0).set_fit_resample_request(sites=True)
    balanced = Pipeline([("isi", isi), ("scale", StandardScaler()), ("clf", LogisticRegression())])
    score_isi = cross_val_score(balanced, X, y, cv=cv, scoring="roc_auc", params={"sites": sites}).mean()

drop = score - score_isi   # large: the score was driven by the site
```

Because ISI keeps every participant, it costs almost nothing when the site cannot help, so a drop can be attributed to
the site. Under-sampling also changes the sample size, so its drop mixes the loss of the shortcut with the loss of data
and should not be used for this check. The drop is a **lower bound**: whatever shortcut survives ISI (strong imbalance
with many features, target ranges that some sites do not cover) also survives in its score. A small drop is therefore
consistent with, but does not prove, a model that does not use the site; report it together with the within-site
permutation probe (above).

### Harmonisation without the target: ISI before ComBat

ComBat-type methods estimate a location and a scale per site and feature from the data of each site. When the target
is associated with the site, these estimates contain the biology of the target: without the target as a covariate,
ComBat removes biological signal together with the effect of site; with the target as a covariate, it needs the target
of every participant it harmonises, including test participants, which is leakage. After ISI every site has the same
target distribution, so the site estimates no longer contain the target. Fit the harmoniser on the balanced training data
and apply it to real participants without their targets:

```python
from uniharmony.combat import NeuroComBat

isi = IntraSiteInterpolation(random_state=0)
X_bal, y_bal = isi.fit_resample(X_train, y_train, sites=sites_train)
combat = NeuroComBat().fit(X_bal, isi.sites_resampled_)       # no covariates needed

X_bal_h = combat.transform(X_bal, isi.sites_resampled_)        # training data for the model
X_test_h = combat.transform(X_test, sites_test)                # new participants of the training sites, no labels
model.fit(X_bal_h, y_bal)
```

New participants of the training sites are harmonised with the training estimates. A new, unseen site needs its own
estimates, which come from its own data and therefore contain its own target distribution: without labels this is
unbiased only if that distribution matches the balanced training distribution.

---

## Covariates

Covariates restrict **who is interpolated with whom**: synthetic samples are only interpolated between samples of the same
class, site and covariate stratum (unique combination of categorical values and continuous-covariate bins, computed within
each site). They are spread over the strata in proportion to the real samples of the class (and `SiteBalancedResampler`
stratifies its under-sampling the same way), so the distribution of the covariates within each class and site is
preserved (for example, if 80% of the patients of a site are women, 80% of its synthetic patients are interpolated
between women).

```python
isi = IntraSiteInterpolation(n_bins_cont_cov=3, random_state=0)
X_bal, y_bal = isi.fit_resample(X, y, sites=sites, categorical_covariate=sex, continuous_covariate=age)
```

Strata with fewer than two samples of a class are skipped for interpolation. Covariates of the real output samples can be
recovered with `sample_indices_`.

---

## Regression

For continuous targets (e.g. brain age), the target is binned (`n_bins`, `binning_strategy`) and each bin is treated as a
class. The target of a synthetic sample is interpolated between its two parent samples (`y = y_a + lam * (y_b - y_a)`),
which recovers SMOTE's interpolation exactly. Bins that are missing from a site, or have a single sample there, cannot be
created by interpolation; ISI warns and leaves them as they are. The site then still predicts the target: if one site
recruited only young adults and another only older adults, no within-site method can make age independent of site.
Use few bins, and if possible restrict the training data to the target range that all sites share.

---

## Balance strategies

### `per_site` (default)
Every site is balanced to its own largest class.

- Site A: 100 class-0, 20 class-1 → 100 / 100
- Site B: 30 class-0, 70 class-1 → 70 / 70

### `global_max`
Every site is balanced towards the largest class of any site, so sites also get similar sizes. Small sites need much more
interpolation, which the variance diagnostic will often flag.

- Site A: 100 class-0, 20 class-1; Site B: 30 class-0, 70 class-1 → both sites 100 / 100

### `{site: n}`
Every site is balanced to the given number of samples per class, which must be at least its largest class (ISI only
adds samples). `SiteBalancedResampler` uses this to pass the capped targets.

---

## Attributes

| Attribute | Content |
|---|---|
| `sites_resampled_` | Site of each output sample |
| `sample_indices_`, `is_synthetic_` | Origin of each output sample |
| `class_counts_` | Real samples per site and class before resampling |
| `target_counts_` | Samples per class in each site after resampling |
| `samples_created_` | `{site: {class: n}}` |
| `amplification_` | Synthetic samples per real sample |
| `variance_ratio_`, `safe_amplification_` | `rho` and `r*` of each class that needed over-sampling |
| `effective_dim_` | Participation ratio of the features (within site-class cells) |
| `summary()` | All of the above as a table |

`SiteBalancedResampler` has the same attributes for the whole procedure (`variance_ratio_` and `safe_amplification_`
only with `max_amplification="auto"`), plus `samples_removed_`, `amplification_cap_` (`inf` where not limited) and
`oversampler_` (the fitted ISI, which saw the kept samples only).

---

## Reference implementation

Source code: https://github.com/N-Nieto/IntraSiteInterpolator.git

## Citation

Paper in preparation.
