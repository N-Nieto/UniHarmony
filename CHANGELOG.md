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
