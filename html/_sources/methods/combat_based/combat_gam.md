(combatgam-long)=
# ComBat-GAM

ComBat-GAM is an extension of the NeuroComBat harmonization framework that addresses a key limitation of the original method: the assumption of linear relationships between biological covariates and imaging features [^1].

While standard ComBat and NeuroComBat assume that covariates such as age influence brain measures linearly, neurodevelopmental and aging studies often exhibit non-linear trajectories across the lifespan. For example, cortical thickness follows an inverted U-shaped pattern across childhood and adolescence, while white matter volume shows complex non-linear changes in aging populations.

ComBat-GAM overcomes this limitation by incorporating Generalized Additive Models (GAMs) into the harmonization framework. This allows the method to preserve non-linear biological effects while removing site-related variability, making it particularly suitable for:

Longitudinal studies spanning wide age ranges

Lifespan studies from childhood to older adulthood

Developmental studies with rapid non-linear changes

Any dataset where covariate effects are expected to be non-linear

The method was originally developed by Pomponio et al. [^1] and has been widely adopted for harmonizing large-scale multi-site neuroimaging datasets

[^1]: Pomponio, R., Erus, G., Habes, M., et al. (2020). Harmonization of large MRI datasets for the analysis of brain imaging patterns throughout the lifespan. *NeuroImage*, 208, 116450. https://doi.org/10.1016/j.neuroimage.2019.116450
