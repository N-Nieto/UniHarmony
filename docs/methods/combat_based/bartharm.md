# BARTharm

**MRI Harmonization using Image Quality Metrics and Bayesian Non-parametrics**

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

\[
y = \mu(\text{IQMs}) + \tau(\text{biological covariates}) + \epsilon
\]

- **μ(·)** → scanner-related effects (learned from IQMs)  
- **τ(·)** → biological signal  
- **ε** → noise  

Both components are modeled using **independent BART ensembles**, allowing:
- non-linear effects  
- high-order interactions  
- fully data-driven learning  

Harmonized data is obtained by removing the estimated scanner component:

\[
\hat{y} = y - \hat{\mu}
\]

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

**Paper**

Prevot E, et al., (2025). BARTharm: MRI Harmonization Using Image Quality Metrics and Bayesian Non-parametric. bioRxiv. Published online 2025. doi:10.1101/2025.06.04.657792 https://www.biorxiv.org/content/10.1101/2025.06.04.657792v1

**Source code**

- https://github.com/NeuroSML/BARTharm

# Implementation
- To be implemented.
