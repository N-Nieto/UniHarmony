"""
CovBat Use-Case Example: Harmonising Covariance Across Three Sites
====================================================================

This script demonstrates how CovBat removes batch effects from the
**covariance matrix** of multi-site data, something that standard ComBat
cannot do.

We simulate three sites that share the same biological effect (age-related
brain atrophy) but have different covariance structures.  After ComBat, a
classifier can still detect which site a scan came from by exploiting the
remaining covariance differences.  After CovBat, this is no longer possible.

Run with::

    python use_case_covbat.py

"""
# %%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold

# ---------------------------------------------------------------------------
# If UniHarmony is installed, import from the package.
# Otherwise fall back to a local copy of _covbat.py placed in the same dir.
# ---------------------------------------------------------------------------
from uniharmony import verbosity
verbosity("error")
from uniharmony.combat import CovBat
from uniharmony.combat import NeuroComBat
from uniharmony.datasets import make_multisite_classification

X, y, sites, covars = make_multisite_classification(n_samples=100, n_features=5, covariates=["age"], site_effect_strength=10)
age = covars["age"]
# =============================================================================
# 1. Simulate multi-site data with covariance batch effects
# =============================================================================


# ---------------------------------------------------------------------------
# 2. Helper: classify site from covariance features
# ---------------------------------------------------------------------------


lda = LinearDiscriminantAnalysis()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)



# ---------------------------------------------------------------------------
# 3. Harmonize with standard ComBat and with CovBat
# ---------------------------------------------------------------------------

# --- Standard ComBat (mean + variance only) ---
combat = NeuroComBat()
X_combat = combat.fit_transform(
    X, sites, continuous_covariates=age[:, None]
)

# --- CovBat (mean + variance + covariance) ---
covbat = CovBat(std_var=True, pct_var=0.99, score_eb=False)
X_covbat = covbat.fit_transform(
    X, sites, continuous_covariates=age[:, None]
)

# ---------------------------------------------------------------------------
# 4. Quantitative comparison: can we still detect site?
# ---------------------------------------------------------------------------

acc_raw =  cross_val_score(lda, X, sites, cv=cv, scoring="accuracy").mean()
acc_combat =  cross_val_score(lda, X_combat, sites, cv=cv, scoring="accuracy").mean()
acc_covbat =  cross_val_score(lda, X_covbat, sites, cv=cv, scoring="accuracy").mean()

print("=" * 60)
print("Site-prediction accuracy (LDA 5-fold CV)")
print("-" * 60)
print(f"  Raw data          : {acc_raw:.3f}")
print(f"  After ComBat      : {acc_combat:.3f}")
print(f"  After CovBat      : {acc_covbat:.3f}")
print(f"  Chance level (3 sites): {1/3:.3f}")
print("=" * 60)

# ---------------------------------------------------------------------------
# 5. Visualise covariance matrices
# ---------------------------------------------------------------------------

# Assuming X is your covariate matrix (DataFrame or numpy array)
# If X is a DataFrame:
import pandas as pd
X = pd.DataFrame(X, columns=[f'Var_{i}' for i in range(X.shape[1])])
corr_matrix = X.corr()


# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix,
            annot=True,  # Show correlation values
            cmap='coolwarm',  # Color scheme
            center=0,  # Center colormap at 0
            fmt='.2f',  # Format for annotations
            square=True,  # Make cells square
            linewidths=0.5)  # Add gridlines

plt.title('Covariate/Correlation Matrix', fontsize=16)
plt.tight_layout()
plt.show()

# Assuming X is your covariate matrix (DataFrame or numpy array)
# If X is a DataFrame:
X_covbat = pd.DataFrame(X_covbat, columns=[f'Var_{i}' for i in range(X.shape[1])])
corr_matrix = X_covbat.corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix,
            annot=True,  # Show correlation values
            cmap='coolwarm',  # Color scheme
            center=0,  # Center colormap at 0
            fmt='.2f',  # Format for annotations
            square=True,  # Make cells square
            linewidths=0.5)  # Add gridlines

plt.title('Covariate/Correlation Matrix', fontsize=16)
plt.tight_layout()
plt.show()


# %%
# Assuming X is your covariate matrix (DataFrame or numpy array)
# If X is a DataFrame:
X_combat = pd.DataFrame(X_combat, columns=[f'Var_{i}' for i in range(X.shape[1])])
corr_matrix = X_combat.corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix,
            annot=True,  # Show correlation values
            cmap='coolwarm',  # Color scheme
            center=0,  # Center colormap at 0
            fmt='.2f',  # Format for annotations
            square=True,  # Make cells square
            linewidths=0.5)  # Add gridlines

plt.title('Covariate/Correlation Matrix', fontsize=16)
plt.tight_layout()
plt.show()

# %%
