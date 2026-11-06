(simulate-multisite-data)=
# Multisite Data Simulation

## Overview

Three functions of `uniharmony.datasets` generate synthetic datasets that mimic multi-center studies, with a known
signal, a known Effect of Site (EoS) and noise. They are used to test harmonization methods and to teach how site
effects bias machine learning:

| Function | Design |
|---|---|
| `make_multisite_classification` | Different subjects in every site, with class labels. |
| `make_multisite_regression` | Different subjects in every site, with a continuous target (e.g. brain age); each site can cover its own range of the target. |
| `make_traveling_subjects` | The same subjects measured in every site, the gold-standard design to check a harmonization. |

The three share the covariates, the noise and the site effects described below; the sections at the end describe what
is specific to regression and traveling subjects.

## Usage

```python
from uniharmony.datasets import make_multisite_classification

# 3 sites, 500 samples in total
X, y, sites = make_multisite_classification(
    n_sites=3,
    n_samples=500,
    n_features=20,
    signal={"linear": 1.0},        # generator and class separation (the default)
    site_effect={"location": 3.0},  # type and strength of the site effect (the default)
    random_state=42,               # for reproducibility
)
```

## Behind the Scenes

Each sample combines four components:

Feature = SiteEffect(Signal + Covariate effects + Noise)

1. A pool of base samples (the signal) is generated with a scikit-learn generator, given as `signal={type: strength}`
   with type `"linear"`, `"blobs"`, `"gaussian_quantiles"`, `"moons"` or `"circles"` (the strength sets the class
   separation; moons and circles have none), with exactly the number of samples of each class
   that the sites need.
2. Each site draws its samples from the pool **without replacement**, following its class proportions
   (`balance_per_site`). No sample appears twice, whatever the imbalance or the site sizes. A proportion of 0 leaves
   a class out of a site; a non-zero proportion always gives the class at least one sample there.
3. Optional covariates (`covariates`, e.g. `"age"`, `"sex"`, `"tiv"`) are drawn per site, and their effects on the features
   are added to the signal. The effects are the **same in every site**, so when a covariate's distribution differs
   between sites (an older site, say), the sites also differ through it: the covariate is a confounder of the site, as
   in real multi-site data.
4. Gaussian noise (`noise_strength`) is added. Signal, covariate effects and noise form the data **without site
   effect**.
5. The site effect is applied (`site_effect={type: strength, ...}`).

### Units

All strengths are in units of a **reference SD** per feature: the pooled within-class standard deviation of the
feature in the signal (about 1 for `"linear"` and `"blobs"`, 0.3–0.7 for `"moons"` and `"circles"`). A location
offset of `s` is therefore a standardized mean difference (Cohen's d) of `s`, whatever the generator and the class
separation, and the same strength means the same thing for every signal. For regression and traveling subjects, the
reference SD is the SD of the feature in the population of the signal (see their sections).

## Site effects

| Type | Effect for a site with strength `s` (`sd`: reference SD, `c`: expected value without site effect) |
|---|---|
| `"location"` | adds a per-feature offset `o * sd`, with `o` drawn from U(-s/2, s/2) |
| `"scale"` | multiplies the deviations from `c` by a per-feature factor exp(U(-0.1 s, 0.1 s)) |
| `"variance"` | adds Gaussian noise with a per-feature SD drawn from U(0, s) reference SDs |
| `"nonlinear"` | adds `w * (X - c)**2 / sd`, with `w` drawn from U(0, 0.1 s) |
| `"dropout"` | replaces values: takes `{"value": v, "prob": p}` (with `p` a probability, a float or one per site); in each site, every affected feature is replaced by `v` with probability `p`, e.g. `np.nan` for missing values. With `"per": "value"`, single values are replaced instead of whole features. |
| `"combat"` | ComBat's generative model: offsets `gamma ~ N(g, (s/4)**2)` with a site mean `g` from U(-s/2, s/2), and factors `delta = sqrt(d)` with `d` from an inverse gamma distribution (shape 6, mean exp(U(-0.2 s, 0.2 s))): the priors that ComBat's empirical Bayes step assumes |
| `"class_interaction"` | every class of every site gets an extra offset from U(-s/2, s/2) reference SDs: the site effect differs between classes, so the class signal differs between sites |

`c` is the global mean plus the sample's covariate effects: the scale effect multiplies the signal and the noise of a
sample, but not its covariate effects, as in ComBat's model `y = alpha + X beta + gamma + delta * epsilon`. The
location offsets are centred: the pooled mean does not drift, and two sites differ by up to `s` reference SDs on each
feature.

### Combining types and choosing strengths

`site_effect` is a dict with the strength of each type, a float (every site) or a list (one value per site). The
types are applied in a fixed order (scale and location, then variance, nonlinear and dropout), whatever the order
of the dict:

```python
X, y, sites = make_multisite_classification(
    site_effect={"location": 4, "scale": [1, 0.5, 3]},  # per site for the scale
    n_sites=3,
)
```

A strength of 0 means no site effect of that type in that site, and `site_effect={}` gives data without site
effect.

### Which features, which samples, which classes

| Parameter | Effect |
|---|---|
| `site_effect_features` | Only these features have a site effect: the first `k`, a fraction, or a list of indices (default: all). |
| `site_effect_heterogeneity` | Each sample's site effect is multiplied by its own intensity exp(h z - h²/2) (mean 1), `z` drawn per sample and shared by its features, e.g. a scanner effect that depends on head size or motion. Samples move further or less far along their site's offset, which adds a site-specific covariance that a per-feature mean and variance correction does not remove (the case CovBat addresses). |

## Covariates

A {class}`~uniharmony.datasets.Covariate` describes the distribution of a covariate in every site and its effect on
the features:

```python
from uniharmony.datasets import Covariate, CovariateSiteDistribution, make_multisite_classification

age = Covariate(
    name="age",
    site_distributions=[  # a young and an old site
        CovariateSiteDistribution(loc=30, scale=5, clip=(18, 90)),
        CovariateSiteDistribution(loc=65, scale=5, clip=(18, 90)),
    ],
    effect=-0.5,               # -0.5 reference SD per SD of age (on average over the features)
    effect_shape="quadratic",  # or "linear", "sigmoid", or a function of the raw values
    effect_features=5,         # only the first 5 features
)
X, y, sites, truth = make_multisite_classification(n_sites=2, covariates=[age, "sex"], return_ground_truth=True)
age_values, sex_values = truth["covariates"]["age"], truth["covariates"]["sex"]
```

| Parameter | Meaning |
|---|---|
| `effect` | Continuous covariate: SD of its contribution to a feature (with `"linear"`, the change per SD of the covariate). Categorical: mean difference between each category and the first one. A scalar is multiplied by a per-feature weight from U(0.5, 1.5); an array gives the effect on every feature exactly. |
| `effect_shape` | `"linear"`, `"quadratic"`, `"sigmoid"` or a callable, applied to the standardized covariate and standardized again, so that `effect` keeps its meaning. |
| `effect_features` | All features (`None`), the first `k`, a fraction, or a list of indices. |
| `depends_on` | Makes a continuous covariate depend on a categorical one: `{"sex": [0, -160]}` adds the shift of each sample's category to its value. |

The presets `"age"` (linear effect 0.5 on every feature), `"sex"` (category 1 is 0.3 SD higher) and `"tiv"` (total
intracranial volume, linear effect 0.5) have effects; `"quality"` has none. `"tiv"` depends on sex: it is drawn
from N(1520, 120²) cm³ for sex 0 and N(1360, 120²) for sex 1 (approximate adult values), so it correlates about
-0.55 with sex when the sexes are balanced, and `"sex"` must also be in the covariates:

```python
X, y, sites, truth = make_multisite_classification(covariates=["sex", "tiv"], return_ground_truth=True)
```

## Outputs and ground truth

`make_multisite_classification` returns `X, y, sites`. With `site_names=[...]` the sites get these labels; with
`as_frame=True`, `X` is a DataFrame (`feature_0`, ...), `y` and `sites` are Series, and in the ground truth the
covariates are a DataFrame and `site_location` and `site_scale` DataFrames indexed by the site names.

With `return_ground_truth=True`, a dict with every component of the data is returned as a fourth output, row-aligned
with `X`:

| Key | Content |
|---|---|
| `"covariates"` | The covariate values (a DataFrame with `as_frame=True`). |
| `"signal"`, `"covariate_effect"`, `"noise"` | The three components of the data without site effect. |
| `"site_free"` | `signal + covariate_effect + noise`: the data without site effect, the target of a perfect harmonization. |
| `"site_effect"` | `X - site_free`. |
| `"site_location"`, `"site_scale"` | Offset (data units) and factor of every site and feature. |
| `"site_class_location"` | Offsets of the site-by-class interaction, per site, class and feature (classification only). |
| `"intensity"` | The per-sample intensity of the site effect (all 1 without heterogeneity). |
| `"dropout_mask"` | True where `"dropout"` replaced a value. |
| `"covariate_coefficients"`, `"feature_scale"`, `"site_effect_features"` | Covariate effects per feature (reference-SD units), the reference SD of every feature, the affected features. |

This allows harmonization methods to be scored exactly, e.g. by the error between harmonized data and `site_free`,
or by how much of the covariate effect they keep.

## Regression

`make_multisite_regression` uses the same model for the features, and a target

y = s · f(Signal) + e, e ~ N(0, 1),

where `f` is standardized in the population, so the strength `s` of `signal={type: s}` sets the explained variance:
R² = s² / (1 + s²) (0.8 for the default `s = 2`). `"linear"` draws the features from N(0, 1) and combines
`n_informative` of them; `"friedman1"` draws them from U(0, 1) and uses the nonlinear Friedman #1 function of the first
five. The site effect changes the features, never the target.

`target_range_per_site` gives each site a range of the population target, as quantiles: with
`[(0, 0.5), (0.5, 1)]`, one site has the lower half of the targets and the other the upper half (a young and an old
cohort, say). The signal follows the target, so the sites differ even without site effect: a harmonization that
removes every difference between sites also removes part of the target signal.

```python
from uniharmony.datasets import make_multisite_regression

X, y, sites, truth = make_multisite_regression(
    n_sites=2,
    target_range_per_site=[(0, 0.6), (0.4, 1)],  # overlapping ranges
    signal={"linear": 2.0},
    site_effect={"location": 2.0},
    return_ground_truth=True,
)
```

The ground truth adds `"target_signal"` (`s · f`, the explained part of `y`) and, for `"linear"`,
`"target_coefficients"` (`target_signal = signal @ target_coefficients`).

## Traveling subjects

`make_traveling_subjects` measures the same `n_subjects` in every site, `n_repeats` times. A subject has a trait
(N(0, 1) per feature) and covariates drawn from one population, the same in every site; each measurement adds its own
noise (`noise_strength`, in units of the between-subject SD, so the reliability within a site is 1 / (1 + r²)) and
the site effect. The differences between the measurements of a subject are therefore noise and site effects only: after
a perfect harmonization, the measurements of a subject agree across sites as well as within a site.

```python
from uniharmony.datasets import make_traveling_subjects

X, sites, subjects, truth = make_traveling_subjects(
    n_subjects=20, n_sites=3, n_repeats=2, site_effect={"location": 2.0, "scale": 2.0}, return_ground_truth=True
)
```

With `site_effect_heterogeneity`, the intensity of the site effect is a property of the subject (e.g. head size): the
same in every site.

## Not simulated yet

Planned improvements of the simulators:

- **Site effects correlated across features.** The offsets and factors are drawn independently for each feature,
  while real scanner effects are spatially smooth (correlated between neighbouring regions). Low-rank or smooth site
  effects would test methods that use the covariance (e.g. CovBat).
- **Site × covariate interactions.** The covariate effects are the same in every site; a scanner effect that changes
  with age, for example, would violate ComBat's assumption of a site effect independent of the covariates.
- **Incomplete traveling-subjects designs**, where not every subject visits every site.
