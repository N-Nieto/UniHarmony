(bartharm-long)=
# BARTharm

**Paper**

Prevot E, et al., (2025). BARTharm: MRI Harmonization Using Image Quality Metrics and Bayesian Non-parametric. bioRxiv. Published online 2025. doi:10.1101/2025.06.04.657792 https://www.biorxiv.org/content/10.1101/2025.06.04.657792v1

**Source code**

- https://github.com/NeuroSML/BARTharm

---

## Overview

**BARTharm** is a statistical harmonization framework for MRI-derived phenotypes (IDPs) that removes scanner-induced variability while preserving biological signal.

Unlike traditional methods (e.g., ComBat), BARTharm:
- **does not rely on discrete scanner/site labels**
- leverages **Image Quality Metrics (IQMs)** as continuous proxies of acquisition variability
- uses **Bayesian Additive Regression Trees (BART)** to model complex, non-linear effects

This enables:
- modeling **continuous scanner variation**
- capturing **within-scanner heterogeneity**
- harmonizing **unseen or anonymized datasets**

The method jointly models:
- biological signal
- scanner-related variation

within a unified Bayesian framework.


BARTharm is a fully data-driven harmonization framework with clear advantages in scenarios such as model misspecification or when scanner-related variables are correlated with biological covariates, where standard methods can lead to inflated false positive rates (FPR). By flexibly modeling scanner effects as non-linear functions of IQMs, BARTharm reduces residual acquisition-related variability that would otherwise bias downstream analyses, resulting in better-calibrated inference and more reliable detection of true biological effects.



---

## Method Summary

BARTharm decomposes each IDP as:

$$
y = \mu(\text{IQMs}) + \tau(\text{biological covariates}) + \epsilon
$$

- **μ(·)** → scanner-related effects (learned from IQMs)
- **τ(·)** → biological signal
- **ε** → noise

Both components are modeled using **independent BART ensembles**, allowing:
- non-linear effects
- high-order interactions
- fully data-driven learning

Harmonized data is obtained by removing the estimated scanner component:

$$
\hat{y} = y - \hat{\mu}
$$

This avoids the restrictive **location-scale assumptions** of classical approaches like ComBat.

Above is the **homoskedastic version**, which captures scanner effects in the **mean structure** through flexible, non-linear functions of IQMs, without requiring scanner or site labels. This allows the model to account for complex, continuous acquisition variability and within-scanner heterogeneity.

There is also a **heteroskedastic version**, which extends the model to account for **scanner-specific differences in variance**. In this setting, the residual variance is allowed to vary across scanners (when available), introducing a multiplicative scaling term that captures differences in noise levels and reliability across acquisition settings. The resulting harmonization removes both **additive (mean)** and **multiplicative (variance)** scanner effects, while preserving the estimated biological signal.


---

## Key Advantages

- **No reliance on scanner IDs**
- Works with **missing or anonymized metadata**
- Handles **non-linear scanner effects**
- Captures **continuous acquisition variability**
- Naturally extends to **unseen datasets**
- Provides **uncertainty quantification** via Bayesian inference

Compared to standard harmonization:
- ComBat assumes **linear additive + multiplicative effects**
- BARTharm learns **flexible functions directly from data**

---



## Implementation

`uniharmony.iqm.BARTharm` is a Python translation of the BARTharm R code, including the soft BART
forests of the [SoftBart](https://github.com/theodds/SoftBART) R package (Linero & Yang, 2018) that it
builds on. Defaults follow the R code: 5000 Gibbs iterations, 500 burn-in, thinning 2, 200 trees for
the scanner forest and 50 for the biological forest, both with depth prior `0.95 * (1 + depth)^-2`.

```python
from uniharmony.iqm import BARTharm

harmonizer = BARTharm(random_state=0, n_jobs=-1)

# X: (n_samples, n_features) imaging-derived phenotypes
# iqms: (n_samples, n_iqms) image quality metrics, e.g. SNR, CNR
# bio: (n_samples, n_covariates) biological covariates, e.g. age, sex (numerically coded)
X_harmonized = harmonizer.fit_transform(X, iqms, biological_covariates=bio)

# New subjects only need their features and IQMs
X_test_harmonized = harmonizer.transform(X_test, iqms_test)
```

Heteroskedastic version (site-specific variance scaling), which needs site labels at fit and
transform time:

```python
harmonizer = BARTharm(var_scaling=True, random_state=0)
X_harmonized = harmonizer.fit_transform(X, iqms, biological_covariates=bio, sites=sites)
X_test_harmonized = harmonizer.transform(X_test, iqms_test, biological_covariates=bio_test, sites=sites_test)
```

### How it works

For every feature independently:

1. The feature is z-scored; IQMs and biological covariates are quantile normalized to [0, 1].
2. A Gibbs sampler alternates between updating the scanner forest μ(IQMs) given τ, the biological
   forest τ(covariates) given μ, the site variance scales (with `var_scaling=True`) and the error
   variance (inverse-gamma prior with shape and rate 0.01).
3. After burn-in and thinning, every kept draw gives a harmonized feature, `y - μ` or, with variance
   scaling, `(y - μ - τ) / δ_site + τ` with the site scales `δ` normalized to geometric mean 1. The
   draws are summarized by their mean (or median, `posterior_summary="median"`) and transformed back
   to the original scale.

### Things to know

- **Biological covariates** are optional but strongly recommended: without them, biological
  variation correlated with the IQMs can be attributed to the scanner and removed. Without
  `var_scaling`, they are only needed at fit time, so `transform` does not need them.
- **Location of the harmonized features.** The intercept is shared between μ and τ and is not
  identified, so harmonized features can be shifted by a constant with respect to the raw features.
  Differences between subjects, which is what downstream analyses use, are not affected.
- **New data.** `transform` uses `n_stored_draws` (default 200) evenly spaced posterior draws of
  the forests. New IQMs are normalized by interpolating the training distribution, so values outside
  the training range are clipped to it. `fit_transform` uses all kept draws, as the R code does.
- **Run time.** Every feature needs `n_iter` sweeps over `n_trees_mu + n_trees_tau` trees, which
  takes minutes per feature for typical sample sizes. Use `n_jobs` to fit features in parallel.
- **ML pipelines.** With `var_scaling=True` the biological covariates are required at transform
  time: never use the target of a downstream model as a biological covariate.

### Differences to the R code

- The R code appends the integer site codes, not normalized, as an extra IQM column when site labels
  are given. As the trees only split within [0, 1], that column carries almost no information, so
  it is not added.
- `transform` on new subjects is an addition; the R code only harmonizes the subjects it was fitted on.
- Missing values raise an error (the R code drops incomplete rows), and constant features are
  returned unchanged (the R code returns NaN).
- When `burn_in` is not a multiple of `thinning_interval`, the R code also drops the last draw.
- Random numbers come from NumPy, so results match the R code in distribution, not draw by draw.

**References**

- Linero, A. R., & Yang, Y. (2018). Bayesian regression tree ensembles that adapt to smoothness and
  sparsity. *Journal of the Royal Statistical Society: Series B*, 80(5), 1087-1110.
  doi:10.1111/rssb.12293
