# 🤺 ComBat-based Harmonization

(neurocombat-short)=
## [ComBat - neuroComBat](#neurocombat-long)

ComBat is one of the most widely used statistical harmonization methods [^combat]. It was originally proposed for genomics to correct **batch effects** in microarray gene-expression data. In these datasets, samples are processed in batches (e.g., laboratory runs), and systematic technical differences between batches introduce unwanted variability that can obscure biological signals.

The method was later adapted to **multi-site neuroimaging datasets**, as neuroComBat, where similar sources of variability arise from differences in scanner hardware, acquisition protocols, and preprocessing pipelines [^neurocombat].

For each feature and each site, ComBat estimates:

- a **site-specific location parameter** (mean shift), and
- a **site-specific scale parameter** (variance scaling).

Additionally, ComBat allows biological relevant variance to be linearly preserved when passed as covariate.

Because of its robustness and simplicity, ComBat has become a standard harmonization approach in neuroimaging and has inspired a large family of extensions and variants.

[^combat]: Johnson, W. E., Li, C., & Rabinovic, A. (2007). Adjusting batch effects in microarray expression data using empirical Bayes methods. *Biostatistics*, 8(1), 118-127. https://doi.org/10.1093/biostatistics/kxj037

[^neurocombat]: Fortin, J. P., Cullen, N., Sheline, Y. I., et al. (2018). Harmonization of cortical thickness measurements across scanners and sites. *NeuroImage*, 167, 104-120. https://doi.org/10.1016/j.neuroimage.2017.11.024

**Source code**

- https://github.com/Jfortin1/neuroCombat/blob/master/neuroCombat/

- https://github.com/Warvito/neurocombat_sklearn/blob/master/examples/

---

(combatgam-short)=
## [ComBat-GAM](#combatgam-long): Preservation of non-Linear biological covariate effects.

While neuroComBat can preserve biological covariates influence features **linearly**, non-linear biological relationships (for example age-related brain trajectories) may not be preserved accurately.

ComBat-GAM extends the original ComBat by allowing **non-linear covariate effects** using generalized additive models (GAMs) [^combatgam].

This is particularly important for neuroimaging variables such as **age**, which often show strong non-linear effects on brain features.

Key contributions:

- modeling of **non-linear biological covariates**
- improved preservation of biological signals
- compatibility with ML pipelines

[^combatgam]: Pomponio, R., Erus, G., Habes, M., et al. (2020). Harmonization of large MRI datasets for the analysis of brain imaging patterns throughout the lifespan. *NeuroImage*, 208, 116450. https://doi.org/10.1016/j.neuroimage.2019.116450

**Source code**

- https://github.com/rpomponio/neuroHarmonize

---

(covbat-short)=
## [CovBat](): Harmonization of covariate matrix

ComBat assumes that site effects can be fully modeled using **feature-wise mean shifts and variance scaling**. However, more complex scanner differences, such as differences in **feature covariance structure**, are not addressed. While standard ComBat aligns means and variances, differences in **feature correlations** may still remain and affect multivariate analyses.

**CovBat** extends ComBat by harmonizing the **covariance structure** of features across sites. CovBat corrects these differences and has shown improved performance in **machine learning applications using neuroimaging features** [^covbat].

[^covbat]: Chen, A. A., Beer, J. C., Tustison, N. J., Cook, P. A., Shinohara, R. T., & Shou, H. (2022). Mitigating site effects in covariance for machine learning in neuroimaging data. *Human Brain Mapping*, 43(4), 1179-1195. https://doi.org/10.1002/hbm.25688

**Source code**

- https://github.com/andy1764/CovBat_Harmonization

---

(neuroharmony-short)=
## [Neuroharmony](#neuroharmony-long): Harmonization based on IQMs

ComBat can not be applied to unknown sites, as the location/scale parameters are learnt by site.

Neuroharmony relies on **scanner-independent image metrics**, instead of site tags.

**Neuroharmony** is a harmonization approach based on **image quality metrics (IQMs)** rather than explicit site labels [^neuroharmony].

Because it relies on scanner-related image characteristics instead of site identifiers, it can generalize to **previously unseen scanners or sites**.

[^neuroharmony]: Garcia-Dias, R., Scarpazza, C., Baecker, L., et al. (2020). Neuroharmony: A new tool for harmonizing volumetric MRI data from unseen scanners. *NeuroImage*, 220, 117127. https://doi.org/10.1016/j.neuroimage.2020.117127

**Source code**

- https://github.com/garciadias/Neuroharmony

---

(bartharm-short)=
## [BARTharm](#bartharm-long): Harmonization based on IQMs using Bayesian Additive Regression Trees.

**Paper**

Prevot E, et al., (2025). BARTharm: MRI Harmonization Using Image Quality Metrics and Bayesian Non-parametric. bioRxiv. Published online 2025. doi:10.1101/2025.06.04.657792 https://www.biorxiv.org/content/10.1101/2025.06.04.657792v1

**Source code**

- https://github.com/NeuroSML/BARTharm

---

(longcombat-short)=
## [Longitudinal ComBat (LongComBat)](#longcombat-long): Adapted ComBat for repeated scaners.

**Longitudinal ComBat** adapts the ComBat framework for **longitudinal studies**, where subjects are scanned repeatedly over time.

Standard ComBat assumes independent observations, which is violated in repeated-measure designs.

Longitudinal ComBat introduces **subject-specific random effects** to model within-subject correlations [^longcombat].

[^longcombat]: Beer, J. C., Tustison, N. J., Cook, P. A., et al. (2020). Longitudinal ComBat: A method for harmonizing longitudinal multi-scanner imaging data. *NeuroImage*, 220, 117129. https://doi.org/10.1016/j.neuroimage.2020.117129

**Source code**

- https://github.com/jcbeer/longCombat

---

## [DeepComBat](#deepcombat-long): Hybrid approach of deep learning and ComBat.

**DeepComBat** integrates ComBat with **deep learning-based feature modeling**.

The method uses neural networks to capture complex non-linear structure in the data while preserving the statistical harmonization principles of ComBat.

This hybrid approach allows the modeling of **complex scanner effects and feature interactions** that cannot be captured by linear models [^deepcombat].

[^deepcombat]: Hu, F., Lucas, A., Chen, A. A., Coleman, K., Horng, H., Ng, R. W., ... & Alzheimer's Disease Neuroimaging Initiative. (2024). Deepcombat: A statistically motivated, hyperparameter‐robust, deep learning approach to harmonization of neuroimaging data. Human brain mapping, 45(11), e26708. https://doi.org/10.1016/j.neuroimage.2020.117689

**Source code**

- https://github.com/hufengling/DeepComBat

---

(prettyharmonize-short)=
## [PrettYharmony](#prettyharmonize-long): A framework to integrate ComBat-based harmonization methods into Machine learning pipelines.

A common challenge in multi-site datasets occurs when **class distributions differ across sites**. For example:

- Site A contains mostly control subjects
- Site B contains mostly patients

In such cases, **biological signal and site effects become confounded**.

If harmonization removes site effects without accounting for the biological variable, the biological signal may also be partially removed.

To address this, ComBat allows users to include biological variables as **covariates to preserve** during harmonization.

However, this introduces an issue in **machine-learning pipelines**.

If the covariate being preserved is also the **target variable of the ML model**, then the target must be known during harmonization. In real-world prediction scenarios this information is unavailable, creating a new form of **data leakage** ${^6}$.

Even if harmonization parameters are estimated using only the training set, the transformation of test data still requires the target variable.

A suitable alternative in these scenarios is **PrettYharmonize**, which allows ComBat-based harmonization to be integrated into ML pipelines **without requiring the target variable during inference** [^prettyharmony].


[^prettyharmony]: Nieto, N., Eickhoff, S. B., Jung, C., Reuter, M., Diers, K., Kelm, M., ... & Patil, K. R. (2026). Impact of leakage on data harmonization in machine learning pipelines in class imbalance across sites. Neurocomputing, 133146. https://doi.org/10.1016/j.neucom.2026.133146

**Source Code**

- https://github.com/juaml/PrettYharmonize
- https://github.com/juaml/harmonize_project

---

## Methods

```{toctree}
:maxdepth: 2
combat
combat_gam
covbat
neuroharmony
bartharm
longcombat
deepcombat
prettyharmony
```
