(covbat-long)=
# CovBat

CovBat extends NeuroComBat by harmonizing not only means and variances but also feature covariances across sites [^1].

Standard ComBat assumes that site effects are limited to location (mean) and scale (variance) shifts. However, multi-site neuroimaging studies often exhibit more complex batch effects where the correlation structure among features differs across scanners—for example, the relationship between hippocampal volume and cortical thickness may vary systematically by scanner manufacturer.

CovBat addresses this limitation by harmonizing the full covariance structure of the data.


## Method
CovBat operates in three steps:

- Apply ComBat to remove mean and variance site effects

- Decompose residuals via PCA within each site to obtain site-specific PC scores

- Harmonize PC score covariances to a reference site, then reconstruct features


## References
[^1]: Chen, A. A., Beer, J. C., Tustison, N. J., Cook, P. A., Shinohara, R. T., & Shou, H. (2022). Mitigating site effects in covariance for machine learning in neuroimaging data. *Human Brain Mapping*, 43(4), 1179-1195. https://doi.org/10.1002/hbm.25688
