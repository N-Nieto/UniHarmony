"""
Using Inter-Site Matched Interpolation (ISMI) with matching covariates
======================================================================

This notebook demonstrates the use of :class:`.InterSiteMatchedInterpolation` for harmonizing multi-site data using matching covariates.

"""
# %%
import numpy as np
from uniharmony.interpolation import InterSiteMatchedInterpolation
from uniharmony.datasets import make_multisite_classification

# Generate sample data with 3 sites
X, y, sites, covars = make_multisite_classification(n_samples=3000, n_sites=3, covariates=["age", "sex"])

# Define covariates for matching
categorical_covariate = covars["sex"].reshape(-1,1)
continuous_covariate = covars["age"].reshape(-1,1)
covariate_tolerance = np.array([5.0])  # ±5 years tolerance

# Create interpolator with pairwise mode and k=2 matches
ismi = InterSiteMatchedInterpolation(alpha=(0.2, 0.4),
                                     covariate_tolerance=covariate_tolerance,
                                     k=2,
                                     mode="pairwise",
                                     random_state=42)

# Generate harmonized dataset
X_res, y_res = ismi.fit_resample(X, y, sites,
                                 categorical_covariate=categorical_covariate,
                                 continuous_covariate=continuous_covariate)

# Check unmatched samples
print(ismi.unmatched_samples_)

# %%
