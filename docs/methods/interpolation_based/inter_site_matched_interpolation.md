(intersitematched-long)=
# Inter-Site Matched Interpolation (ISMI)

Inter-Site Matched Interpolation (ISMI) is a data harmonization technique that generates synthetic training samples by interpolating between participants matched across different data acquisition sites or scanners. Unlike traditional harmonization methods that remove site effects, ISMI explicitly models cross-site variation by creating intermediate samples between matched pairs, aiming to improve model generalization to unseen scanners and datasets.

This method is particularly effective when:
- Training data comes from multiple sites with different equipment/protocols
- You need to balance site representation without discarding real data
- Site-specific covariate distributions (age, sex) differ between locations
- Traditional ComBat harmonization removes signal you want to preserve

## Mathematical Formulation

Given two sites $S_i$ and $S_j$, for each sample $\mathbf{x}_k \in S_i$ with target $y_k$, ISMI finds matches in $S_j$ where $y_m \approx y_k$ (and optionally matching categorical/continuous covariates). Synthetic samples are generated via:

$$

\mathbf{x}_{\text{synth}} = \mathbf{x}_k + \alpha (\mathbf{x}_m - \mathbf{x}_k)

$$

$$

y_{\text{synth}} = y_k + \alpha (y_m - y_k) \quad \text{(regression)}

$$

$$

y_{\text{synth}} = y_k \quad \text{(classification)}

$$

Where $\alpha \in (0, 1)$ controls interpolation strength (default: 0.3). An alpha of aproximately 0 will generate a interpolated sample more similar to the *base* sample. On the other hand, a value closer to 1 will generate an interpolated sample more similar to the *target* sample.

For **regression**, targets are interpolated continuously. For **classification**, targets remain discrete while features are interpolated.

---

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | float or tuple | 0.3 | Interpolation weight. If float, constant. If tuple `(min, max)`, sampled uniformly per sample. Range [0, 1]. |
| `target_tolerance` | float or None | None | Tolerance for target matching. `None` = exact match (classification) or 10% of range (regression). |
| `covariate_tolerance` | array-like or None | None | Tolerance for continuous covariates (e.g., age tolerance in years). |
| `k` | int, "max", or "average" | 1 | Number of matches per sample: - `int`: Specific count - `"max"`: All available matches - `"average"`: Interpolate toward mean of all matches |
| `mode` | "pairwise" or "base_to_others" | "pairwise" | Site pairing strategy. `base_to_others` forces `k="average"` and interpolates each site against all others combined. |
| `concatenate` | bool | True | If True, returns original + synthetic data. If False, returns only synthetic samples. |
| `random_state` | int or None | None | Random seed for reproducibility. |

---

## Basic Usage

```python
from uniharmony.interpolation import InterSiteMatchedInterpolation
import numpy as np

# Your multi-site data
X = np.random.randn(300, 50)  # Features
y = np.random.randint(0, 2, 300)  # Binary targets
sites = np.array(["Site_A"]*100 + ["Site_B"]*100 + ["Site_C"]*100)

# Initialize and fit
ismi = InterSiteMatchedInterpolation(
    alpha=0.3,
    k=2,
    random_state=42
)

X_harmonized, y_harmonized = ismi.fit_resample(X, y, sites=sites)

print(f"Original: {len(X)} samples")
print(f"Harmonized: {len(X_harmonized)} samples")
print(f"Unmatched by direction: {ismi.unmatched_samples_}")
```

## Advanced Usage with Covariates
For strict matching on demographics (e.g., exact sex match, age within 5 years):

# Covariates

```python
sex = np.array([["M"], ["F"], ...])  # Categorical
age = np.array([[25], [34], ...])    # Continuous

ismi = InterSiteMatchedInterpolation(
    alpha=(0.2, 0.6),           # Random interpolation per sample
    k="average",                 # Interpolate toward mean of matches
    target_tolerance=None,       # Exact class match required
    covariate_tolerance=np.array([5.0]),  # Age within 5 years
    mode="pairwise",
    random_state=42
)

X_res, y_res = ismi.fit_resample(
    X, y, sites=sites,
    categorical_covariate=sex,
    continuous_covariate=age
)
```


## Understanding Unmatched Samples
After fitting, ismi.unmatched_samples_ contains a dictionary mapping (source_site, target_site) to counts of samples that couldn't be matched.
Important: Asymmetry is Expected
The unmatched matrix is directional and typically asymmetric:

```python
{
    ('Site_A', 'Site_B'): 45,   # 45 samples from A couldn't match in B
    ('Site_B', 'Site_A'): 40,   # 40 samples from B couldn't match in A
    ('Site_A', 'Site_C'): 12,
    ('Site_C', 'Site_A'): 8,
    ...
}
```

## Algorithm Modes
### Pairwise Mode (`mode="pairwise"`)
Creates synthetic samples for every site pair combination (A↔B, A↔C, B↔C). Best for:
Understanding specific site-to-site relationships
When you want maximum synthetic diversity

### Base-to-Others Mode (`mode="base_to_others"`)
Each site is interpolated against all other sites combined. Forces k="average". Best for:
Reducing computational cost with many sites
Creating "site vs. population" harmonization


## Implementation Notes
Classification: Targets are preserved exactly for synthetic samples (no interpolation of labels)
Regression: Targets are interpolated continuously between matched pairs
Memory: With k="max" and many sites, memory usage scales as O(n²). Use k=1 or k=5 for large datasets
Validation: Always check unmatched_samples_ after fitting. High unmatched counts (>50%) indicate poor site compatibility for the chosen matching criteria


## Reference Implementation
This implementation generalizes the method described in:

- Nieto, N., Asati, A., Jadhav, K., & Patil, K. R. (2026). Data harmonizing via interpolation applied to brain age prediction. Discover Data, 4(1), 3. https://doi.org/10.1007/s44248-026-00100-7

Key differences from the paper:
- The original implementation was specific to brain age prediction (regression)
- This version supports both classification and regression
- Added support for categorical covariates beyond binary sex
- Generalized to arbitrary site counts and pairing strategies

Source code for original study: https://github.com/Aditi-Asati/Interpolation-And-Brain-Age-Prediction

# Citation
If you use this method in your research, please cite:

```bibtex
@article{nieto2026data,
  title={Data harmonizing via interpolation applied to brain age prediction},
  author={Nieto, Nicol{\'a}s and Asati, Aditi and Jadhav, Kushaj and Patil, Kaustubh R},
  journal={Discover Data},
  volume={4},
  number={1},
  pages={3},
  year={2026},
  publisher={Springer},
  doi={10.1007/s44248-026-00100-7}
}
```
