(longcombat-long)=
# Longitudinal ComBat (LongComBat)

[LongComBat](https://doi.org/10.1016/j.neuroimage.2020.117129) extends ComBat to **longitudinal neuroimaging data** with multiple timepoints per subject [^1].

Standard ComBat assumes independent observations, violating the **within-subject correlation structure** of longitudinal studies. LongComBat preserves these correlations while removing site/scanner effects.

## Method

LongComBat uses a **mixed-effects model**:

$$

Y_{ijsf} = \alpha_f + X_{ijs}\beta_f + \gamma_{sf} + \delta_{sf}\epsilon_{ijs} + \omega_{if}

$$

where:

- $i$ = subject, $j$ = timepoint, $s$ = site, $f$ = feature
- $\omega_{if}$ = **random intercept** per subject (captures within-subject correlation)
- $\gamma_{sf}$, $\delta_{sf}$ = site-specific location and scale effects (empirical Bayes shrunk)

## Key points

| Aspect | Detail |
|--------|--------|
| **Advantage** | Preserves longitudinal within-subject correlations |
| **Requirement** | Subject IDs linking multiple timepoints |
| **Limitation** | Assumes linear trajectories (use LongComBat-GAM for non-linear) |
| **Use when** | Harmonizing multi-scanner longitudinal data (e.g., baseline + follow-up) |

## Reference

[^1]: Beer, J. C., Tustison, N. J., Cook, P. A., et al. (2020). Longitudinal ComBat: A method for harmonizing longitudinal multi-scanner imaging data. *NeuroImage*, 220, 117129. https://doi.org/10.1016/j.neuroimage.2020.117129
"""
