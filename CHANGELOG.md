## [0.0.5](https://github.com/N-Nieto/UniHarmony/tree/0.0.5) - 2026-11-03

### Removed

- Remove `uniharmony.datasets.load_onharmony_structural_features`, features not longer available
- Remove the `uniharmony.prettyharmonize.prettyharmonize` placeholder function, superseded by the PrettYharmonize estimators.

### Deprecated

- Passing `n_bins_cont_cov` / `binning_strategy_cont_cov` to `IntraSiteInterpolation.fit_resample` is deprecated; set them in the constructor.

### Added

- Add `SiteBalancedResampler`, which combines within-site over-sampling with `IntraSiteInterpolation` (any interpolator, including random duplication) and within-site under-sampling: the amplification of every class is limited to `max_amplification` (`"auto"`, the safe amplification of ISI's variance rule, by default; a number, a callable or `None`) and the larger classes of the site are under-sampled to meet the smaller ones with any count-controlled imblearn under-sampler that selects real samples (default `ClusterCentroids(voting="hard")`). `max_amplification=0` gives pure within-site under-sampling and `None` ISI alone. It reports `samples_removed_`, `samples_created_`, `amplification_cap_` and `summary()`, and warns when most samples of a site are removed.
- Add `random_state` to `ComBatGAM` to make the k-fold selection of GAM penalty weights reproducible.
- Add `uniharmony.dl.DeepHarmony` (Dewey et al., 2019), a U-Net based contrast harmonization method for multi-contrast MR images trained on an overlap cohort, with 2.5D prediction and optional harmonization of target-protocol images
- Add `uniharmony.iqm.BARTharm` (Prevot et al., 2025), which harmonizes scanner effects using image quality metrics instead of site labels, with Bayesian additive regression trees. Translated from the BARTharm R code, including the SoftBART forests (Linero & Yang, 2018), with optional site-specific variance scaling and harmonization of new subjects.
- Add `uniharmony.metrics.site_centroid_distance`, which quantifies the Effect of Site as the spread of the site centroids relative to the within-site spread (a pseudo-F with Euclidean, standardized or Mahalanobis geometry), with a permutation test, the share of variance explained by site, bias-corrected pairwise distances between site centroids, and optional conditioning on a categorical biological variable.
- Add `uniharmony.prettyharmonize` with `PrettYharmonizeClassifier` and `PrettYharmonizeRegressor` (Nieto et al., 2026). They wrap a ComBat-based model (`harm_model`: e.g. `NeuroComBat`), a `base_model` and a `stack_model` (ML models): the target is preserved when fitting the harmonization model and pretended for the data to predict, and the stack model, makes the prediction. Other covariates are passed as `additional_covariates`. Also add a registry of the ComBat-based models (`uniharmony.combat._available_models`) listing their covariate arguments.
- Add a variance diagnostic to `IntraSiteInterpolation`: for every class it over-samples, it measures how much of the class variance the interpolated samples keep (`variance_ratio_`) and derives the safe amplification (`safe_amplification_`, controlled by `variance_tolerance`, default 0.2), warning clearly when balancing needs more synthetic samples than that; `estimate_safe_amplification()` reports the same without resampling. `balance_strategy` also accepts a target count per site, and `n_bins` explicit bin edges. Add `summary()` and the attributes `sample_indices_`, `is_synthetic_`, `class_counts_`, `target_counts_`, `amplification_` and `effective_dim_`. ISI only interpolates and never removes a sample.
- Add optional `dl` dependency group (`pip install uniharmony[dl]`) providing PyTorch for the deep learning methods in `uniharmony.dl`
- New "Implementations conformance" gallery. Its first example compares `NeuroComBat` with `neuroCombat` 0.2.12 (and, offline, R `sva` 3.50.0): the ComBat model, where the implementations agree, the equations and lines of code where they differ, simulations with a known ground truth, harmonization of new subjects, and runtime and memory. Added `tests/combat/test_reference_conformance.py` with a frozen `neuroCombat` 0.2.12 reference.
- Stronger conformance tests for `NeuroComBat`: a frozen neuroCombat 0.2.12 reference (identical to R `sva::ComBat` 3.50.0 to 1e-13) over six designs (several covariates, no covariates, imbalanced sites, large-scale covariates, a covariate confounded with site) and all four modes, with float32-exact inputs so the tolerance is 1e-8 of the data scale instead of 1e-6. The reference is rebuilt with `tests/combat/make_neurocombat_reference.py`.

### Changed

- Documented a third difference with R `sva::ComBat` and `neuroCombat`: their parametric empirical Bayes solver measures the relative change against the signed previous estimate, so features with a negative location effect never block the stop and the output is not converged to the stated tolerance (up to about 1e-5 of the data scale). UniHarmony uses the absolute value; run to convergence, both give the same solution. The "Implementations conformance" example explains it, and its pooled-variance figure now separates the effect of the denominator (exactly `(n - S) / n`) from the comparison with the truth, with confidence intervals and with and without empirical Bayes.
- Install the `dl` extra with the CPU build of PyTorch in the tox test environments, so the deep learning methods are tested in CI
- Replace the regex-based `python-no-eval` pre-commit hook with ruff rule `S307`, which flags only the builtin `eval()` and allows method calls such as PyTorch's `model.eval()`
- The type alias of the registered ComBat-based models in `uniharmony.combat._available_models` is renamed from `ComBatModel` to `ComBatModels`.
- `IntraSiteInterpolation` covariates now only restrict which samples are interpolated together and preserve the covariate distribution of each class and site, instead of balancing classes inside every stratum; `global_max` with covariates no longer fills every stratum up to the global maximum. Every class that needs over-sampling must have at least two samples in its site (one for random over-sampling), otherwise an error is raised; regression bins with too few samples in a site are left as they are, with a warning. Each site and stratum gets its own random stream, and the constructor parameters are no longer modified by `fit_resample`.
- `IntraSiteInterpolation`: the parent search that assigns interpolated targets in regression is vectorised (one Gram matrix per cell instead of a nearest-neighbour search per synthetic sample); `effective_dimension` uses the trace and Frobenius norm of the Gram matrix instead of an eigendecomposition and estimates the participation ratio from at most 2000 samples (`max_samples`), which keeps it cheap for voxel-wise data. The warnings for unsafe amplification and for regression bins that cannot be balanced now explain that part of the site-target association remains, and the documentation describes how to tune hyper-parameters with ISI nested in the inner cross-validation, the limits of ISI (strong imbalance with many features, local models, missing target ranges) and the within-site permutation probe. The documentation also describes two further uses of ISI: as a sanity check of how much of a cross-validated score comes from the site (score without minus score with ISI), and as a pre-processing step that lets ComBat-type harmonisers estimate site effects without the target as a covariate.

### Fixed

- Fix ComBat-based methods (`NeuroComBat`, `CovBat`, `ComBatGAM`) when more than one categorical, continuous or smooth covariate column is given; columns were previously flattened into a single column. `CovBat.transform` now also accepts 1D covariates.
- Fix `IntraSiteInterpolation` crashing on small covariate strata or small classes (neighbour counts are adapted and strata that are too small are skipped), interpolators that produce approximately the requested number of samples (SVM-SMOTE, ADASYN), and upper-case interpolator names in `create_interpolator`. When an interpolator fails, the error now suggests SMOTE.
- Fix `IntraSiteInterpolation` overwriting targets: original samples now keep their original `y` (it was replaced for every row, also for classification), classification labels keep their dtype (they were returned as floats; string labels crashed), synthetic regression targets are interpolated between the targets of the samples they were generated from (they were random values from their bin), `samples_created_` uses the fitted bins for regression, and `random_state` is no longer overwritten in `fit_resample` and also seeds the bootstrap fill-up. Interpolators must return the original samples first, as imblearn over-samplers do.
- Fix `NeuroComBat`, `CovBat` and `ComBatGAM` with more than one categorical or continuous covariate: a 2-D covariate array was flattened into a single column, so fitting failed. A 1-D array is still treated as a single covariate.
- Fix `OptimalTransportDomainAdaptation.transform` and `inverse_transform` on new data: any subset of the fitted sites is accepted (a single site, or a batch without the reference site), rows are selected by site instead of by value (duplicate rows no longer crash), DataFrames are supported, unseen sites raise a clear error, and `inverse_transform` maps the non-reference rows back as the inverse of `transform` (it crashed on the output of `transform`).
- Fix `fit_transform` of `NeuroComBat` and `ComBatGAM` forwarding fit-only options (`var_epsilon`, `delta_epsilon`, `tau_2_epsilon`, `max_iter`, `df`, `degree`, `smooth_covariates_bounds`) to `transform`, which raised a `TypeError`. Every ComBat variant now declares `fit_transform` with the same typed, documented signature as its `fit` and delegates to a shared helper that passes to `transform` only the arguments it accepts.
- Fix `uniharmony.verbosity` having no effect on the ComBat-based harmonizers when it was called after importing them: their loggers were bound at import time. The `src` field is now also added to the logs of `NeuroComBat`, `CovBat` and `ComBatGAM`.
- Fix the return type annotation of `uniharmony._verbose.verbosity_context`, so that type checkers accept it in `with` statements.
- Small numerical and code fixes: the ComBat regression is solved with an SVD-based least-squares solver instead of an explicit matrix inverse (stable for nearly collinear covariates, minimum-norm solution with a warning for a rank-deficient design); the inverse-gamma priors are kept in float64 (they were cast to float32); `MetricType` of `OptimalTransportDomainAdaptation` no longer lists metrics removed from SciPy (`kulsinski`, `wminkowski`, `sokalmichener`) and lists `sqeuclidean`; a duplicated line was removed from `minimum_samples_warning`.
- The scikit-learn tags of the ComBat variants and `OptimalTransportDomainAdaptation` now describe them correctly: missing values are not allowed (`allow_nan=False`, they are rejected at fit), and `sites`, which takes the place of `y`, is a required 1D label of any value (`two_d_labels` and `positive_only` are no longer set). `ComBatGAM` is tagged non-deterministic unless `random_state` is fixed.
- UniHarmony prints only warnings and errors by default (it printed every debug line before). `uniharmony.verbosity` and `uniharmony.verbosity_context` now set UniHarmony's own level without changing structlog's global configuration, so applications that configure structlog keep their settings, and the default argument of both functions is `"warning"`. Transforming samples from only some of the fitted sites no longer logs a warning for each missing site.
- `ComBatGAM.transform` no longer fails when smooth covariates fall outside the fitted range: values are clipped to the range boundary (constant extrapolation) and a warning is logged.
- `NeuroComBat`, `ComBatGAM` and `CovBat` divide the pooled residual sum of squares by `n - S` (`S` sites) for every sample size, so that the harmonized variance is unbiased. The former switch to `n - 1` below 30 samples, which had no statistical basis and made the output scale jump at `n = 30`, was removed. R `sva::ComBat` and `neuroCombat` divide by `n`, which shrinks the harmonized variance by `(n - S) / n`; outside `mean_only`, UniHarmony's output equals theirs with the deviations from the fitted biology scaled by `sqrt(n / (n - S))`.
- `NeuroComBat`, `CovBat`, `ComBatGAM` and `OptimalTransportDomainAdaptation` store `n_features_in_` (and `feature_names_in_` for DataFrames) at fit, and `transform` raises a clear error when the number or names of the features differ, instead of a NumPy broadcasting error.
- `mean_only=True` in `NeuroComBat` and `ComBatGAM` is documented and flagged (an info log at fit) as a deliberate correction of R `sva::ComBat` and `neuroCombat`: the empirical Bayes posterior of the site effect uses the site sample size, while the references use `n = 1`, which leaves most of the feature-specific site effect in the data.
- `transform` of the ComBat variants raises a clear `ValueError` when a covariate given to `fit` is missing, or a covariate not given to `fit` is passed (it previously failed with a `TypeError`, an `AttributeError`, or "Input contains NaN", or ignored the covariate). `ComBatGAM` reports missing `smooth_covariates` explicitly in `fit` and `transform`.
- `uniharmony.ot` is exported by the top-level package, like the other modules.


## [0.0.4](https://github.com/N-Nieto/UniHarmony/tree/0.0.4) - 2026-06-10

### Added

- Expose `__version__` as package-level attribute ([#73](https://github.com/N-Nieto/UniHarmony/issues/73))

### Changed

- Remove `copy` parameter from ComBat-based methods and change its default from True to False ([#65](https://github.com/N-Nieto/UniHarmony/issues/65))


## [0.0.3](https://github.com/N-Nieto/UniHarmony/tree/0.0.3) - 2026-05-08

### Added

- Add ComBatGAM harmonisation method ([#15](https://github.com/N-Nieto/UniHarmony/issues/15))
- Add support for fetching BIDS-compatible datasets using DataLad and use ON-Harmony dataset as an example ([#58](https://github.com/N-Nieto/UniHarmony/issues/58))
- Export `uniharmony.metrics` and allow for multiple metrics in `uniharmony.metrics.report_metrics_by_site` ([#59](https://github.com/N-Nieto/UniHarmony/issues/59))
- Add global and per site functionality to IntraSiteInterpolation class.
  Add covariate match interpolation to IntraSiteInterpolation class for better preserve biological plausibility. ([#61](https://github.com/N-Nieto/UniHarmony/issues/61))
- Add `BaseComBat`, `DesignMatrixMixin`, `StandardizationMixin` and `LocationAndScaleMixin` for ComBat-based methods ([#64](https://github.com/N-Nieto/UniHarmony/issues/64))

### Changed

- Refactor `make_multisite_classification` to use `sklearn` functions to generate base samples ([#54](https://github.com/N-Nieto/UniHarmony/issues/54))
- Expose ``uniharmony.datasets`` and hide previously exposed functions from it ([#58](https://github.com/N-Nieto/UniHarmony/issues/58))
- Fix `uniharmony.metrics.report_metrics_by_site` when requested metric requires `y_score` instead of `y_pred` ([#59](https://github.com/N-Nieto/UniHarmony/issues/59))
- Change basic to IntraSiteInterpolation class structure to allow for classification and regression problems. ([#61](https://github.com/N-Nieto/UniHarmony/issues/61))
- Refactor ComBat-based method implementations to use mixin classes ([#64](https://github.com/N-Nieto/UniHarmony/issues/64))


## [0.0.2](https://github.com/N-Nieto/UniHarmony/tree/0.0.2) - 2026-04-16

### Added

- Introduce `uniharmony.verbosity` and `uniharmony.verbosity_context` to manage logging levels ([#10](https://github.com/N-Nieto/UniHarmony/issues/10))
- Add InterSiteMatchedInterpolation harmonisation method ([#16](https://github.com/N-Nieto/UniHarmony/issues/16))
- Add `plot` sub-package for providing plotting helpers ([#26](https://github.com/N-Nieto/UniHarmony/issues/26))

### Changed

- Improve `sites` validation, handling and storage across transformers ([#17](https://github.com/N-Nieto/UniHarmony/issues/17))
- Split up API reference in docs ([#19](https://github.com/N-Nieto/UniHarmony/issues/19))
- Fix IntraSiteInterpolation logic and improve test coverage ([#25](https://github.com/N-Nieto/UniHarmony/issues/25))
- Improve multi-site data generation and move it to `datasets` ([#27](https://github.com/N-Nieto/UniHarmony/issues/27))
- Improve NeuroComBat tests ([#29](https://github.com/N-Nieto/UniHarmony/issues/29))
- Add Optimal Transport for Domain Adaptation wrapper that allows for harmonizing data to a reference site(s) ([#30](https://github.com/N-Nieto/UniHarmony/issues/30))
- Replace `zensical` with `sphinx` for documentation ([#35](https://github.com/N-Nieto/UniHarmony/issues/35))


## [0.0.1](https://github.com/N-Nieto/UniHarmony/tree/0.0.1) - 2026-03-26

### Added

- Setup documentation and add initial info ([#2](https://github.com/N-Nieto/UniHarmony/issues/2))
- Add IntraSiteInterpolation harmonisation method, MAREoS dataset loading and multisite data generation functions ([#4](https://github.com/N-Nieto/UniHarmony/issues/4))
- Add NeuroComBat harmonisation method ([#5](https://github.com/N-Nieto/UniHarmony/issues/5))


## [0.0.0](https://github.com/N-Nieto/UniHarmony/tree/0.0.0) - 2026-01-15

### Added

- Clean up repository and publish package ([#1](https://github.com/N-Nieto/UniHarmony/issues/1))
