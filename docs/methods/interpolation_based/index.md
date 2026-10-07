# 📈 Interpolation-based Harmonization

Interpolation based methods are designed to remove or hide the Effect of Site from the ML models. If you use the interpolation methods for
data harmonization, you should use the imblearn.Pipeline instead of the sklearn.Pipiline, as the models don't have a `fit` and `transform`, but
rather a `fit_resample` method, which is consistent with imblearn.Pipeline.

(intersitematched-short)=
## [Inter Site Matched Interpolation](#intersitematched-long)

Matched interpolation buils on the core assumtion that if you interpolate between two samples from different scanners with the same covariates,
for example age and gender, the resulted sample will preserve the matched characteristics (or interpolate between the posibilities), but will
remove effect of site.

---

(intrasite-short)=
## [Intra Site Interpolation](#intrasite-long)

IntraSiteInterpolation (ISI) balances the classes within each site. Minority classes are over-sampled by interpolating between samples of the
same class and site, up to a data-driven limit, and the remaining imbalance is closed by under-sampling the larger classes. At the end, all sites
have the same proportions (balanced) of samples for all classes. This breaks the correlation between site and target, so ML models cannot pick
up that signal and give a prediction fraudulently based on EoS instead of the true biological signal. Classification and regression are supported.

---

## Methods

```{toctree}
:maxdepth: 2
inter_site_matched_interpolation
intra_site_interpolation
```
