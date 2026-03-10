# Interpolation

Interpolation based methods are designed to remove or hide the Effect of Site from the ML models. If you use the interpolation methods for
data harmonization, you should use the imblearn.Pipeline instead of the sklearn.Pipiline, as the models don't have a `fit` and `transform`, but
rather a `fit_resample` method, which is consistent with imblearn.Pipeline.

## Matched Interpolator

Matched interpolation buils on the core assumtion that if you interpolate between two samples from different scanners with the same covariates,
for example age and gender, the resulted sample will preserve the matched characteristics (or interpolate between the posibilities), but will
remove effect of site.

## IntraSiteInterpolation

IntraSiteInterpolation (ISI) balances the samples for all the presented classes in each site. At the end of the interpolation, all sites will have the
same proportions (balanced) of samples for all classes. This will break any correlation between site and target, make it invisible for the ML
models to pick up that signal and give you a prediction fraudulently based on EoS and not on true biological signal.

At for now, the method only supports classification problems and not regression ones.
