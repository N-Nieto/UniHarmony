(neurocombat-long)=
# ComBat and neuroComBat

[ComBat](https://doi.org/10.1093/biostatistics/kxj037) is one of the most widely used statistical harmonization methods [^1]. It was originally proposed for genomics to correct **batch effects** in microarray gene-expression data. In these datasets, samples are processed in batches (e.g., laboratory runs), and systematic technical differences between batches introduce unwanted variability that can obscure biological signals.

The method was later adapted to **multi-site neuroimaging datasets**, with the name neuroComBat, where similar sources of variability arise from differences in scanner hardware, acquisition protocols, and preprocessing pipelines [^2].

ComBat models each feature as the combination of:

- biological effects,
- site-specific effects,
- residual noise.

For each feature and each site, ComBat estimates:

- a **site-specific location parameter** (mean shift), and
- a **site-specific scale parameter** (variance scaling).

These parameters are estimated using an **empirical Bayes framework**, which stabilizes the estimates by borrowing information across features [^1]. This procedure is particularly useful when dealing with small samples per site.

After estimating its parameters, ComBat removes site-related variability by transforming the data so that feature distributions across sites align with the **pooled mean and variance**, while preserving variability explained by biological covariates.

Because of its robustness and simplicity, ComBat has become a standard harmonization approach in neuroimaging and has inspired a large family of extensions and variants.

A higly recomended simple and easy-to-follow explanation of ComBat, its advantajes, disandantages, limitations and mathematical fundations can be found at [^3].

## ComBat statistical model

Flowing the nomenclature used in [^3], ComBat assumes the following model for each feature $Y_{isf}$:

$$

Y_{isf} = \alpha_f + X\beta_f + \gamma_{sf} + \delta_{sf} \epsilon_{isf}

$$

where:

- $i$ is the individual (subject) index, $f$ the feature index and $s$ denotes the site/batch.
- $X$ represents the biological covariates matrix.

Non-site related effects:
- $\alpha_f$ overall mean per feature (across all sites).
- $\beta_f$ corresponds to the coefficients of the covariate matrix $X$. This is also calculated as the same for all features.
- $\epsilon_{isf}$ is the noise term, with mean 0 and site specific variance $\sigma_{f}^{2}$.

Site related effects:
- $\gamma_{fs}$ is the additive **site-specific location effect**. Represents the estimated offset per site per feature.
- $\delta_{fs}$ is the multiplicative **site-specific scale effect**. Represents the scale value that controls the dispersion around the mean.


```{note}
ComBat nomenclature does not follow typical ML nomenclature, where $X$ denotes features and $y$ the target. In the case of ComBat, $y$ represents the features and $X$ (covariate matrix) represents the variables which variance we want to preserve, typically our target.
```

The empirical Bayes estimation shrinks these parameters toward shared distributions across features, improving robustness when sample sizes per site are small.

---

# Main assumptions and limitations of ComBat

Although highly effective, ComBat relies on several assumptions that introduce practical limitations.

## Whole-dataset estimation

**Assumption**

Original ComBat and neuroComBat implementations require access to **the full dataset** during parameter estimation.

**Implication**

This introduces **data leakage** in machine learning pipelines because information from the test set influences the training transformation [^6].

Newer implementations of neuroComBat (including the one in UniHarmony), allow the models to find its parameters in a training set and to be applied in a test set, allowing its integration into ML pipelines.

## Linear biological covariate effects

**Assumption**

Biological covariates influence features **linearly**.

**Implication**

Non-linear biological relationships (for example age-related brain trajectories) may not be preserved accurately.

Please see [ComBat-GAM](./combat_gam.md) as alternative.

## Additive and multiplicative site effects

**Assumption**

Site effects can be fully modeled using **feature-wise mean shifts and variance scaling**.

**Implication**

More complex scanner differences, such as differences in **feature covariance structure**, are not addressed.

## Balanced site representation

**Assumption**

Sites have roughly similar numbers of samples.

**Implication**

If one site dominates the dataset, it can strongly influence the pooled distribution used in harmonization, potentially leading to **overcorrection of smaller sites**.

Empirical Bayes shrinkage partially mitigates this issue, but severe imbalance may still affect results.

## Closed-set site assumption

**Assumption**

All scanner sites are known during training.

**Implication**

Standard ComBat cannot harmonize data from **previously unseen sites**.

---

# Cross-site class imbalance

A common challenge in multi-site datasets occurs when **class distributions differ across sites**. For example:

- Site A contains mostly control subjects
- Site B contains mostly patients

In such cases, **biological signal and site effects become confounded**.

If harmonization removes site effects without accounting for the biological variable, the biological signal may also be partially removed.

To address this, ComBat allows users to include biological variables as **covariates to preserve** during harmonization.

However, this introduces an issue in **machine-learning pipelines**.

If the covariate being preserved is also the **target variable of the ML model**, then the target must be known during harmonization. In real-world prediction scenarios this information is unavailable, creating a new form of **data leakage** [^4].

Even if harmonization parameters are estimated using only the training set, the transformation of test data still requires the target variable.

A suitable alternative in these scenarios is [**PrettYharmonize**](./prettyharmony.md), which allows ComBat-based harmonization to be integrated into ML pipelines **without requiring the target variable during inference**.

---

# Further reading

- Pomponio, R., Erus, G., Habes, M., et al. (2020). Harmonization of large MRI datasets for the analysis of brain imaging patterns throughout the lifespan. *NeuroImage*, 208, 116450. https://doi.org/10.1016/j.neuroimage.2019.116450

- Chen, A. A., Beer, J. C., Tustison, N. J., Cook, P. A., Shinohara, R. T., & Shou, H. (2022). Mitigating site effects in covariance for machine learning in neuroimaging data. *Human Brain Mapping*, 43(4), 1179-1195. https://doi.org/10.1002/hbm.25688

- Beer, J. C., Tustison, N. J., Cook, P. A., et al. (2020). Longitudinal ComBat: A method for harmonizing longitudinal multi-scanner imaging data. *NeuroImage*, 220, 117129. https://doi.org/10.1016/j.neuroimage.2020.117129

- Garcia-Dias, R., Scarpazza, C., Baecker, L., et al. (2020). Neuroharmony: A new tool for harmonizing volumetric MRI data from unseen scanners. *NeuroImage*, 220, 117127. https://doi.org/10.1016/j.neuroimage.2020.117127

- Zhang, Y., Parmigiani, G., & Johnson, W. E. (2020). ComBat-seq: batch effect adjustment for RNA-seq count data. *NAR Genomics and Bioinformatics*, 2(3). https://doi.org/10.1093/nargab/lqaa078

- Dinsdale, N. K., Jenkinson, M., & Namburete, A. I. (2021). Deep learning-based unlearning of dataset bias for MRI harmonisation and confound removal. *NeuroImage*, 228, 117689. https://doi.org/10.1016/j.neuroimage.2020.117689

[^1]: Johnson, W. E., Li, C., & Rabinovic, A. (2007). Adjusting batch effects in microarray expression data using empirical Bayes methods. *Biostatistics*, 8(1), 118-127. https://doi.org/10.1093/biostatistics/kxj037

[^2]: Fortin, J. P., Cullen, N., Sheline, Y. I., et al. (2018). Harmonization of cortical thickness measurements across scanners and sites. *NeuroImage*, 167, 104-120. https://doi.org/10.1016/j.neuroimage.2017.11.024

[^3]: Bayer, J. M., Thompson, P. M., Ching, C. R., Liu, M., Chen, A., Panzenhagen, A. C., et al. (2022). Site effects how-to and when: An overview of retrospective techniques to accommodate site effects in multi-site neuroimaging analyses. *Frontiers in Neurology*, 13, 923988. https://doi.org/10.3389/fneur.2022.923988

[^4]: Sasse, L., Nicolaisen-Sobesky, E., Dukart, J., et al. (2025). Overview of leakage scenarios in supervised machine learning. *Journal of Big Data*, 12(1), 135. https://doi.org/10.1186/s40537-025-01193-8
