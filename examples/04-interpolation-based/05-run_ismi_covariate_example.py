"""
Using Inter-Site Matched Interpolation (ISMI) with matching covariates
======================================================================

This notebook demonstrates the use of :class:`.InterSiteMatchedInterpolation` for harmonizing multi-site data using matching covariates.

"""
# %%
import numpy as np
from uniharmony.interpolation import InterSiteMatchedInterpolation


# Generate sample data with 3 sites
rng = np.random.RandomState(42)
X = rng.randn(150, 10)
y = rng.randint(0, 2, 150)
sites = np.array(["A"] * 50 + ["B"] * 50 + ["C"] * 50)

# Define covariates for matching
categorical_covariate = np.array([["M"], ["F"]] * 75)  # Sex
continuous_covariate = rng.randint(20, 80, (150, 1))  # Age
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
