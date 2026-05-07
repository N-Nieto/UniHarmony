"""IntraSiteInterpolation advance usage examples."""

# %%
### Global Maximum Balancing

# Balance all sites to the single largest class count found anywhere:

import numpy as np

from uniharmony.datasets import make_multisite_classification
from uniharmony.interpolation import IntraSiteInterpolation


X, y, sites = make_multisite_classification(balance_per_site=[0.3, 0.7])
isi = IntraSiteInterpolation(balance_strategy="global_max", interpolator="random", random_state=42)

X_balanced, y_balanced = isi.fit_resample(X, y, sites=sites)
print(f"Global target count: {isi.target_count_}")

# %%
## Covariates

### Stratified Interpolation with Covariates

# Preserve demographic distributions while balancing classes.
# Synthetic samples are interpolated only between participants matching on all covariates:
rng = np.random.default_rng(54)
n_samples = 1000
X, y, sites = make_multisite_classification(n_samples=n_samples, balance_per_site=[0.3, 0.7])
sex = rng.integers(0, 2, (n_samples, 1))
age = rng.standard_normal((n_samples, 1)) * 10 + 50
isi = IntraSiteInterpolation(balance_strategy="per_site", random_state=42)

X_balanced, y_balanced = isi.fit_resample(
    X, y, sites=sites, categorical_covariate=sex, continuous_covariate=age, n_bins_cont_cov=2
)  # Age binned with 5 bins

# %%
