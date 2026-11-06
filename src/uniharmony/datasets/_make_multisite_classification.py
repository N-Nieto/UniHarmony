"""Data simulation module for multi-site data generation."""

import numbers
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, get_args, overload

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.datasets import make_blobs, make_circles, make_classification, make_gaussian_quantiles, make_moons
from sklearn.utils import check_random_state

from uniharmony._verbose import get_logger

from ._make_covariates import (
    PRESET_COVARIATES,
    Covariate,
    _covariate_effects,
    _make_covariate,
    _resolve_covariates,
    _resolve_feature_subset,
)


__all__ = [
    "make_multisite_classification",
]

# Generators of the base problem
SIGNAL_TYPES = Literal["linear", "circles", "moons", "blobs", "gaussian_quantiles"]
# Types of Effect of Site (EoS); several can be combined
SITE_EFFECT_TYPES = Literal["location", "scale", "variance", "nonlinear", "dropout", "combat", "class_interaction"]
_DEFAULT_SIGNAL = {"linear": 1.0}
_DEFAULT_SITE_EFFECT = {"location": 3.0}
# Samplers such as SMOTE need k_neighbors + 1 samples of a class in a site (6 by default). Only used for warning.
_FEW_SAMPLES = 6

logger = get_logger(src="make_multisite_classification")

SiteEffect = Mapping[str, float | Sequence[float]]


@overload
def make_multisite_classification(
    n_sites: int = 2,
    n_samples: int | Sequence[int] = 1000,
    n_features: int = 10,
    n_classes: int = 2,
    balance_per_site: Sequence[float] | Sequence[Sequence[float]] | None = None,
    signal: Mapping[str, float] | None = None,
    noise_strength: float | Sequence[float] = 0.1,
    site_effect: SiteEffect | None = None,
    site_effect_features: int | float | Sequence[int] | None = None,
    site_effect_heterogeneity: float = 0.0,
    covariates: Sequence[Covariate | str | PRESET_COVARIATES] | None = None,
    site_names: Sequence | None = None,
    as_frame: bool = False,
    return_ground_truth: Literal[False] = False,
    random_state: int | np.random.RandomState | np.random.Generator | None = 42,
    **kwargs,
) -> tuple[Any, Any, Any]: ...


@overload
def make_multisite_classification(
    n_sites: int = 2,
    n_samples: int | Sequence[int] = 1000,
    n_features: int = 10,
    n_classes: int = 2,
    balance_per_site: Sequence[float] | Sequence[Sequence[float]] | None = None,
    signal: Mapping[str, float] | None = None,
    noise_strength: float | Sequence[float] = 0.1,
    site_effect: SiteEffect | None = None,
    site_effect_features: int | float | Sequence[int] | None = None,
    site_effect_heterogeneity: float = 0.0,
    covariates: Sequence[Covariate | str | PRESET_COVARIATES] | None = None,
    site_names: Sequence | None = None,
    as_frame: bool = False,
    *,
    return_ground_truth: Literal[True],
    random_state: int | np.random.RandomState | np.random.Generator | None = 42,
    **kwargs,
) -> tuple[Any, Any, Any, dict[str, Any]]: ...


def make_multisite_classification(
    n_sites: int = 2,
    n_samples: int | Sequence[int] = 1000,
    n_features: int = 10,
    n_classes: int = 2,
    balance_per_site: Sequence[float] | Sequence[Sequence[float]] | None = None,
    signal: Mapping[str, float] | None = None,
    noise_strength: float | Sequence[float] = 0.1,
    site_effect: SiteEffect | None = None,
    site_effect_features: int | float | Sequence[int] | None = None,
    site_effect_heterogeneity: float = 0.0,
    covariates: Sequence[Covariate | str | PRESET_COVARIATES] | None = None,
    site_names: Sequence | None = None,
    as_frame: bool = False,
    return_ground_truth: bool = False,
    random_state: int | np.random.RandomState | np.random.Generator | None = 42,
    **kwargs,
) -> tuple[Any, ...]:
    """Simulate multi-site classification data with signal, covariates, noise and site effects.

    The data follow the model::

        X = SiteEffect(Signal(y) + CovariateEffects + Noise; site)

    1. A pool of base samples (the signal) is generated with the
       scikit-learn generator selected by ``signal``.
    2. Each site draws its samples from the pool without replacement,
       following its class balance. No sample is used twice.
    3. Optional covariates are drawn per site (their distributions may
       differ between sites) and their effects, the same in every site, are
       added to the signal.
    4. Independent Gaussian noise is added. Signal, covariate effects and
       noise form the data without site effect is obtained (``site_free``).
    5. The Effect of Site (EoS) of ``site_effect`` is applied.

    **Units.** All strengths (site effects, noise, covariate effects) are in
    units of a reference scale per feature: the pooled within-class standard
    deviation of the feature in the signal (about 1 for ``"linear"`` and
    ``"blobs"``, 0.3-0.7 for ``"moons"`` and ``"circles"``). A location
    offset of ``s`` is therefore a standardized mean difference (Cohen's d)
    of ``s``, whatever the generator and the class separation.

    Parameters
    ----------
    n_sites : int, optional (default 2)
        Number of sites to simulate.

    n_samples : int or list of int, optional (default 1000)
        If an int, the total number of samples, split as evenly as possible
        across sites. If a list (or array), the number of samples of each
        site; every site needs at least 1 sample.

    n_features : int, optional (default 10)
        Number of features per sample.

    n_classes : int, optional (default 2)
        Number of classes to simulate (2 for binary, >2 for multi-class).

    balance_per_site : list of float, list of list of float or None, optional (default None)
        Class proportions. A flat list of ``n_classes`` proportions applies
        to every site; a list of ``n_sites`` such lists gives one per site.
        Each must sum to 1. ``None`` gives balanced classes. Counts are
        rounded with the largest-remainder method. A proportion of 0 leaves
        the class out of that site (e.g. to simulate a site that only
        recruits patients); a class with a non-zero proportion gets at least
        one sample in that site, and every class must be present in some
        site.

    signal : dict or None, optional (default None, i.e. ``{"linear": 1.0}``)
        Generator of the base problem and its strength, as ``{type:
        strength}`` with one type:

        - ``"linear"``: ``sklearn.datasets.make_classification``; the
          strength is ``class_sep``. Needs ``n_classes <= 2**n_features``.
        - ``"blobs"``: ``make_blobs``; the strength is the half-width of the
          ``center_box`` of the class centres.
        - ``"gaussian_quantiles"``: ``make_gaussian_quantiles``; the strength
          is the covariance of the Gaussian, which scales the data but does
          not change how separable the classes are.
        - ``"moons"``, ``"circles"``: ``make_moons``, ``make_circles``;
          always 2 features and 2 classes. They have no strength parameter
          (a value other than 1 warns); pass ``noise=...`` to control their
          class overlap.

    noise_strength : float or list of float, optional (default 0.1)
        Standard deviation of the Gaussian noise, in reference-SD units, for
        all sites or per site.

    site_effect : dict or None, optional (default None, i.e. ``{"location": 3.0}``)
        Types of EoS and their strength ``s``, as ``{type: strength}``. A
        strength is a non-negative float (reference-SD units) for every site,
        or a list with one value per site; a strength of 0 means no EoS of
        that type in that site, and ``{}`` means no site effect. Several
        types can be combined:

        - ``"location"``: adds a per-feature offset ``o * sd`` with ``o``
          drawn from U(-s/2, s/2). The offsets are centred: the pooled mean
          does not drift, and two sites differ by up to ``s`` reference SDs.
        - ``"scale"``: multiplies the deviations from ``c`` by a per-feature
          factor exp(U(-0.1 s, 0.1 s)) (for ``s = 3``, factors in
          [0.74, 1.35]). Covariate effects are not scaled.
        - ``"variance"``: adds Gaussian noise with a per-feature standard
          deviation drawn from U(0, s) reference SDs.
        - ``"nonlinear"``: adds ``w * (X - c)**2 / sd`` with ``w`` drawn
          from U(0, 0.1 s).
        - ``"dropout"``: sets values to 0 with probability ``min(0.1 s, 1)``
          (whole features of a site).
        - ``"combat"``: the generative model of ComBat, location and scale
          with the priors of its empirical Bayes step: offsets
          ``gamma ~ N(g, (s/4)**2)`` with a site mean ``g`` drawn from
          U(-s/2, s/2), and factors ``delta = sqrt(d)`` with ``d`` drawn
          from an inverse gamma distribution of shape 6 (coefficient of
          variation 0.5) and mean exp(U(-0.2 s, 0.2 s)).
        - ``"class_interaction"``: every class of every site gets an extra
          per-feature offset drawn from U(-s/2, s/2): the site effect
          differs between classes, so the class signal differs between
          sites.

        They are applied in the order factors, offsets, variance, nonlinear,
        dropout.

    site_effect_features : int, float, sequence of int or None, optional (default None)
        Features affected by the EoS: ``None`` for all, an int ``k`` for the
        first ``k``, a float in (0, 1] for that fraction of the first
        features, or a sequence of indices. The other features have no site
        effect.

    site_effect_heterogeneity : float, optional (default 0.0)
        Variation of the EoS between the samples of a site. Each sample's
        site effect (offsets, log-factors, variance and nonlinear terms) is
        multiplied by an intensity ``exp(h * z - h**2 / 2)`` (mean 1), with
        ``z ~ N(0, 1)`` drawn per sample and shared by all its features.
        This adds a site-specific covariance along the site effect, which a per-feature
        mean and variance correction does not remove.
        0 gives the same effect to every sample of a site.

    covariates : list of Covariate or str, or None, optional (default None)
        Covariate specifications. Each entry is a :class:`Covariate` or a
        preset name (``"age"``, ``"sex"``, ``"tiv"``, ``"quality"``). A
        covariate with an ``effect`` changes the features in the same way in
        every site; when its distribution differs between sites, the sites
        also differ through it (confounding). The values are in the ground
        truth (``ground_truth["covariates"]``).

    site_names : sequence or None, optional (default None)
        Labels of the sites, one per site (e.g. scanner names), used in
        ``sites`` and, with ``as_frame``, as the index of the per-site ground
        truth. ``None`` gives ``0`` to ``n_sites - 1``.

    as_frame : bool, optional (default False)
        If True, ``X`` is a DataFrame with columns ``feature_0`` ...,
        ``y`` a Series named ``"target"``, ``sites`` a Series named
        ``"site"``; in the ground truth, the covariates are a DataFrame and
        ``site_location`` and ``site_scale`` DataFrames indexed by the site
        names.

    return_ground_truth : bool, optional (default False)
        If True, also return a dict with the components of ``X``, all
        row-aligned with ``X``:

        - ``"signal"``: the base samples (class signal only);
        - ``"covariates"``: the covariate values (empty without covariates);
        - ``"covariate_effect"``: the sum of the covariate effects;
        - ``"covariate_coefficients"``: per covariate, its effect on each
          feature in reference-SD units (see :class:`Covariate`);
        - ``"noise"``: the Gaussian noise;
        - ``"site_free"``: ``signal + covariate_effect + noise``, the data
          without site effect (the target of a perfect harmonization);
        - ``"site_effect"``: ``X - site_free``;
        - ``"site_location"``, ``"site_scale"``: shape (n_sites, n_features),
          rows in the order of the sites: the offset (data units) and the
          factor of each site, from the ``"location"``, ``"scale"`` and
          ``"combat"`` types (0 and 1 otherwise);
        - ``"site_class_location"``: shape (n_sites, n_classes, n_features),
          the offsets of ``"class_interaction"``;
        - ``"intensity"``: the per-sample intensity of the site effect
          (``site_effect_heterogeneity``; all 1 when it is 0);
        - ``"site_effect_features"``: indices of the affected features;
        - ``"feature_scale"``: the reference SD of each feature.

    random_state : int, RandomState, Generator or None, optional (default 42)
        The seed of the pseudo random number generator, or a RandomState or
        Generator, for reproducibility. A Generator is used to seed a
        RandomState (and advances).

    **kwargs
        Additional keyword arguments passed to the scikit-learn generator.

    Returns
    -------
    X : np.ndarray or DataFrame of shape (n_samples, n_features)
        Simulated features.
    y : np.ndarray or Series of shape (n_samples,)
        Class labels (0 to ``n_classes`` - 1).
    sites : np.ndarray or Series of shape (n_samples,)
        Site labels (``site_names``, or 0 to ``n_sites`` - 1).
    ground_truth : dict
        Only returned when ``return_ground_truth`` is True.

    Raises
    ------
    ValueError
        If a parameter has an invalid value.
    TypeError
        If a parameter has an invalid type.

    Examples
    --------
    >>> X, y, sites = make_multisite_classification(n_sites=3, n_samples=300, n_features=20, n_classes=3)
    >>> X.shape, y.shape, sites.shape
    ((300, 20), (300,), (300,))

    A strong location effect and a mild scale effect on half of the
    features, in sites that differ in age (which affects every feature):

    >>> from uniharmony.datasets import Covariate, CovariateSiteDistribution
    >>> age = Covariate(
    ...     name="age",
    ...     site_distributions=[CovariateSiteDistribution(loc=30, scale=5), CovariateSiteDistribution(loc=60, scale=5)],
    ...     effect=0.5,
    ... )
    >>> X, y, sites, truth = make_multisite_classification(
    ...     site_effect={"location": 4, "scale": 1},
    ...     site_effect_features=0.5,
    ...     covariates=[age],
    ...     return_ground_truth=True,
    ... )
    >>> truth["covariates"]["age"].shape
    (1000,)

    """
    s_params = _validate_inputs(
        n_sites=n_sites,
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        balance_per_site=balance_per_site,
        signal=signal,
        noise_strength=noise_strength,
        site_effect=site_effect,
        site_effect_features=site_effect_features,
        site_effect_heterogeneity=site_effect_heterogeneity,
        covariates=covariates,
        site_names=site_names,
        random_state=random_state,
        kwargs=kwargs,
    )
    rng = s_params.random_state

    # Steps 1-2: a pool with exactly the class counts the sites need, split without replacement.
    X_pool, y_pool = _generate_base_samples(
        s_params.class_counts.sum(axis=0),
        s_params.n_features,
        s_params.n_classes,
        s_params.signal_type,
        s_params.signal_strength,
        rng,
        **kwargs,
    )
    site_indices = _split_pool_by_site(y_pool, s_params.class_counts, rng)
    signal_samples = np.vstack([X_pool[idx] for idx in site_indices])
    y = np.concatenate([y_pool[idx] for idx in site_indices])
    site_codes = np.repeat(np.arange(s_params.n_sites), s_params.class_counts.sum(axis=1))
    site_rows = [np.flatnonzero(site_codes == site) for site in range(s_params.n_sites)]
    feature_scale = _within_class_sd(signal_samples, y)  # the unit of all strengths

    # Step 3: covariates, drawn per site, and their effects, the same in every site
    covariate_values, covariate_effect, covariate_coefficients = _simulate_covariates(
        s_params.covariates, site_rows, s_params.n_features, feature_scale, rng
    )

    # Step 4: noise; signal + covariate effects + noise is the data without site effect
    noise = np.zeros_like(signal_samples)
    for site, rows in enumerate(site_rows):
        noise[rows] = rng.normal(size=(len(rows), s_params.n_features)) * s_params.noise_strength[site] * feature_scale
    site_free = signal_samples + covariate_effect + noise
    center = (signal_samples + noise).mean(axis=0) + covariate_effect  # expected value of each sample without site effect

    # Step 5: site effects
    X, site_location, site_scale, site_class_location, intensity = _apply_site_effects(
        site_free, center, y, site_rows, feature_scale, s_params, rng
    )

    logger.info(f"Generated {len(X)} samples across {s_params.n_sites} sites")
    logger.debug(f"Class counts per site: {s_params.class_counts.tolist()}")

    truth = None
    if return_ground_truth:
        truth = {
            "signal": signal_samples,
            "covariates": covariate_values,
            "covariate_effect": covariate_effect,
            "covariate_coefficients": covariate_coefficients,
            "noise": noise,
            "site_free": site_free,
            "site_effect": X - site_free,
            "site_location": site_location,
            "site_scale": site_scale,
            "site_class_location": site_class_location,
            "intensity": intensity,
            "site_effect_features": s_params.affected_features,
            "feature_scale": feature_scale,
        }
    return _format_outputs(X, y, site_codes, s_params.site_names, as_frame, truth)


def _format_outputs(
    X: npt.NDArray,
    y: npt.NDArray,
    site_codes: npt.NDArray,
    site_names: npt.NDArray,
    as_frame: bool,
    truth: dict[str, Any] | None,
) -> tuple:
    """Label the sites with ``site_names``; return arrays, or DataFrames and Series when ``as_frame``."""
    sites = site_names[site_codes]
    if as_frame:
        features = [f"feature_{j}" for j in range(X.shape[1])]
        index = pd.Index(site_names, name="site")
        X, y, sites = pd.DataFrame(X, columns=features), pd.Series(y, name="target"), pd.Series(sites, name="site")
        if truth is not None:
            truth["covariates"] = pd.DataFrame(truth["covariates"])
            truth["site_location"] = pd.DataFrame(truth["site_location"], index=index, columns=features)
            truth["site_scale"] = pd.DataFrame(truth["site_scale"], index=index, columns=features)
    return (X, y, sites) if truth is None else (X, y, sites, truth)


##########################################################################################################
############################################### VALIDATION ###############################################
##########################################################################################################
@dataclass
class _SimParams:
    """Validated and resolved simulation parameters."""

    random_state: np.random.RandomState
    n_sites: int
    n_features: int
    n_classes: int
    class_counts: npt.NDArray  # (n_sites, n_classes)
    signal_type: str
    signal_strength: float
    noise_strength: npt.NDArray  # (n_sites,)
    site_effect: dict[str, npt.NDArray]  # type -> strength per site, (n_sites,)
    affected_features: npt.NDArray
    heterogeneity: float
    covariates: list[Covariate] | None
    site_names: npt.NDArray


def _validate_inputs(**params: Any) -> _SimParams:
    """Validate every parameter of ``make_multisite_classification`` and resolve defaults and broadcasts."""
    n_sites, n_features, n_classes = _check_integers(
        n_sites=params["n_sites"], n_features=params["n_features"], n_classes=params["n_classes"]
    )
    samples_per_site = _validate_sizes(n_sites, params["n_samples"], n_features, n_classes)
    signal_type, signal_strength, n_classes, n_features = _validate_signal(
        params["signal"], n_classes, n_features, params["kwargs"]
    )
    balance = _validate_balance_per_site(params["balance_per_site"], n_sites, n_classes)
    class_counts = np.array([_allocate(n, proportions) for n, proportions in zip(samples_per_site, balance, strict=True)])
    _check_class_counts(class_counts)
    return _SimParams(
        random_state=_check_random_state(params["random_state"]),
        n_sites=n_sites,
        n_features=n_features,
        n_classes=n_classes,
        class_counts=class_counts,
        signal_type=signal_type,
        signal_strength=signal_strength,
        noise_strength=_per_site(params["noise_strength"], n_sites, "noise_strength"),
        site_effect=_validate_site_effect(params["site_effect"], n_sites),
        affected_features=_resolve_feature_subset(params["site_effect_features"], n_features, "site_effect_features"),
        heterogeneity=_check_strength(params["site_effect_heterogeneity"], "site_effect_heterogeneity"),
        covariates=_resolve_covariates(params["covariates"], n_sites),
        site_names=_validate_site_names(params["site_names"], n_sites),
    )


def _check_random_state(seed: int | np.random.RandomState | np.random.Generator | None) -> np.random.RandomState:
    """Return a RandomState; a NumPy Generator seeds a new RandomState."""
    if isinstance(seed, np.random.Generator):
        return np.random.RandomState(int(seed.integers(0, 2**32 - 1)))
    return check_random_state(seed)


def _check_integers(**values: object) -> tuple[int, ...]:
    """Check that the size parameters are integers (not booleans, not floats)."""
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise TypeError(f"{name} must be an int, got {value!r} ({type(value).__name__}).")
    return tuple(int(value) for value in values.values())  # type: ignore[call-overload]


def _check_strength(value: object, name: str) -> float:
    """Check a single non-negative, finite number."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"Invalid type for {name}: must be a float, got {value!r} ({type(value).__name__}).")
    if not (np.isfinite(value) and value >= 0):
        raise ValueError(f"{name} must be non-negative and finite, got {value}")
    return float(value)


def _per_site(value: object, n_sites: int, name: str) -> npt.NDArray:
    """Broadcast a non-negative strength to all sites, or validate one strength per site."""
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return np.full(n_sites, _check_strength(value, name))
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise TypeError(f"Invalid type for {name}: must be a float or list[float], got {type(value)}")
    if len(value) != n_sites:
        raise ValueError(f"{name} must have length n_sites ({n_sites}), got {len(value)}")
    return np.array([_check_strength(v, f"{name}[{i}]") for i, v in enumerate(value)])


def _validate_sizes(n_sites: int, n_samples: int | Sequence[int], n_features: int, n_classes: int) -> npt.NDArray:
    """Validate the problem size and return the number of samples of each site."""
    if n_sites < 1:
        raise ValueError(f"n_sites must be at least 1, got {n_sites}")
    if n_sites == 1:
        logger.warning("n_sites is 1, which may not be suitable for testing multi-site methods.")
    if n_features <= 0:
        raise ValueError(f"n_features must be positive, got {n_features}")
    if n_classes < 2:
        raise ValueError(f"n_classes must be at least 2, got {n_classes}")

    if isinstance(n_samples, numbers.Integral) and not isinstance(n_samples, bool):
        if n_samples < n_sites:
            raise ValueError(f"n_samples ({n_samples}) is less than n_sites ({n_sites}). Some sites will have 0 samples.")
        samples_per_site = np.full(n_sites, n_samples // n_sites, dtype=int)
        samples_per_site[: n_samples % n_sites] += 1
        return samples_per_site

    samples_per_site = np.asarray(n_samples)
    if samples_per_site.ndim != 1 or not np.issubdtype(samples_per_site.dtype, np.integer):
        raise TypeError(f"n_samples must be int or list[int], got {type(n_samples)}.")
    if len(samples_per_site) != n_sites:
        raise ValueError(f"n_samples has len {len(samples_per_site)} and does not match with n_sites={n_sites}")
    if samples_per_site.min() < 1:
        raise ValueError(f"Every site needs at least 1 sample, got n_samples={samples_per_site.tolist()}")
    return samples_per_site.astype(int)


def _validate_signal(
    signal: Mapping[str, float] | None, n_classes: int, n_features: int, kwargs: dict
) -> tuple[str, float, int, int]:
    """Return the signal type and strength, and the number of classes and features the generator produces."""
    signal = _DEFAULT_SIGNAL if signal is None else signal
    if not isinstance(signal, Mapping) or len(signal) != 1:
        error = ValueError if isinstance(signal, Mapping) else TypeError
        raise error(f"signal must be a dict with one type and its strength, e.g. {{'linear': 1.0}}, got {signal!r}.")
    ((signal_type, strength),) = signal.items()
    if signal_type not in get_args(SIGNAL_TYPES):
        raise ValueError(f"Unsupported signal type: {signal_type!r}. Choose from {get_args(SIGNAL_TYPES)}.")
    strength = _check_strength(strength, f"signal['{signal_type}']")

    if signal_type in {"moons", "circles"}:
        if n_features != 2:
            logger.warning(f"signal '{signal_type}' always produces 2 features; n_features={n_features} is set to 2.")
        if n_classes != 2:
            logger.warning(f"signal '{signal_type}' always produces 2 classes; n_classes={n_classes} is set to 2.")
        if strength != 1.0:
            warnings.warn(
                f"signal '{signal_type}' has no strength parameter; pass noise=... to control the class overlap.",
                UserWarning,
                stacklevel=4,
            )
        return signal_type, strength, 2, 2
    if signal_type == "linear":
        n_informative = kwargs.get("n_informative", min(n_features, 2 * n_classes))
        n_clusters = kwargs.get("n_clusters_per_class", 1)
        if n_classes * n_clusters > 2**n_informative:
            raise ValueError(
                f"signal 'linear' places each class on a vertex of a hypercube of n_informative={n_informative} "
                f"dimensions, which has {2**n_informative} vertices: n_classes * n_clusters_per_class = "
                f"{n_classes * n_clusters} is too many. Increase n_features (and n_informative), or reduce n_classes."
            )
    if strength == 0.0:
        logger.warning("The signal strength is 0. Adding a delta (1e-6) to it to avoid degenerate data.")
        strength = 1e-6
    return signal_type, strength, n_classes, n_features


def _validate_site_effect(site_effect: SiteEffect | None, n_sites: int) -> dict[str, npt.NDArray]:
    """Return the strength of every site effect type in every site, shape (n_sites,) per type."""
    site_effect = _DEFAULT_SITE_EFFECT if site_effect is None else site_effect
    if not isinstance(site_effect, Mapping):
        raise TypeError(f"site_effect must be a dict {{type: strength}}, e.g. {{'location': 3.0}}, got {site_effect!r}.")
    strengths = {}
    for name, strength in site_effect.items():
        if name not in get_args(SITE_EFFECT_TYPES):
            raise ValueError(f"Unsupported site effect type: {name!r}. Choose from {get_args(SITE_EFFECT_TYPES)}.")
        strengths[name] = _per_site(strength, n_sites, f"site_effect['{name}']")
    return strengths


def _validate_site_names(site_names: Sequence | None, n_sites: int) -> npt.NDArray:
    """Return the site labels (default 0 to n_sites - 1), checking their number and uniqueness."""
    if site_names is None:
        return np.arange(n_sites)
    labels = np.asarray(site_names)
    if labels.ndim != 1 or len(labels) != n_sites:
        raise ValueError(f"site_names must have one label per site ({n_sites}), got {list(site_names)}.")
    if len(set(labels.tolist())) != n_sites:
        raise ValueError(f"site_names must be distinct, got {labels.tolist()}.")
    return labels


##########################################################################################################
################################################# BALANCE ################################################
##########################################################################################################
def _validate_balance_per_site(
    balance_per_site: Sequence[float] | Sequence[Sequence[float]] | None,
    n_sites: int,
    n_classes: int,
) -> npt.NDArray:
    """Validate the class proportions and return them as an array of shape (n_sites, n_classes)."""
    if balance_per_site is None:
        return np.full((n_sites, n_classes), 1.0 / n_classes)
    if not isinstance(balance_per_site, (list, tuple, np.ndarray)):
        raise TypeError(f"balance_per_site must be a list, got {type(balance_per_site)}")

    nested = len(balance_per_site) > 0 and isinstance(balance_per_site[0], (list, tuple, np.ndarray))
    rows = balance_per_site if nested else [balance_per_site] * n_sites
    if len(rows) != n_sites:
        raise ValueError(f"balance_per_site must have length n_sites ({n_sites}), got {len(rows)}")
    for i, row in enumerate(rows):
        if not all(isinstance(p, numbers.Real) for p in row):
            raise TypeError(f"balance_per_site[{i}] must contain class proportions (floats), got {row}")
        if len(row) != n_classes:
            raise ValueError(f"balance_per_site[{i}] must have length n_classes ({n_classes}), got {len(row)}")

    balance = np.asarray(rows, dtype=float)
    for i, row in enumerate(balance):
        if not ((row >= 0) & (row <= 1)).all():
            raise ValueError(f"balance_per_site[{i}] must be between 0 and 1, got {row.tolist()}")
        if not np.isclose(row.sum(), 1.0):
            raise ValueError(f"balance_per_site[{i}] must sum to 1.0, got {row.sum():.6f}")
    return balance


def _allocate(n: int, proportions: npt.ArrayLike) -> npt.NDArray:
    """Split ``n`` into counts proportional to ``proportions``; every non-zero proportion gets at least 1.

    The largest-remainder method rounds the counts. A class with a non-zero
    proportion that would be rounded to 0 takes one sample from the largest
    class, so that a requested class is never silently left out.
    """
    proportions = np.asarray(proportions, dtype=float)
    raw = n * proportions
    counts = np.floor(raw).astype(int)
    remainder_order = np.argsort(-(raw - counts), kind="stable")
    counts[remainder_order[: n - counts.sum()]] += 1
    present = proportions > 0
    if present.sum() > n:
        raise ValueError(f"A site with {n} samples cannot hold the {present.sum()} classes of its balance_per_site.")
    for c in np.flatnonzero(present & (counts == 0)):
        counts[np.argmax(counts)] -= 1
        counts[c] = 1
    return counts


def _check_class_counts(class_counts: npt.NDArray) -> None:
    """Raise if a class is absent from every site; log a warning for classes with very few samples in a site."""
    empty = np.flatnonzero(class_counts.sum(axis=0) == 0)
    if len(empty) > 0:
        raise ValueError(
            f"Class(es) {empty.tolist()} have proportion 0 in every site of balance_per_site; "
            "every class must be present in at least one site."
        )
    few = np.argwhere((class_counts > 0) & (class_counts < _FEW_SAMPLES))
    if len(few) > 0:
        cells = ", ".join(f"site {s} class {c}: {class_counts[s, c]}" for s, c in few[:5])
        logger.warning(
            f"Some classes have fewer than {_FEW_SAMPLES} samples in a site ({cells}); many methods need more "
            "(e.g. SMOTE-based samplers need k_neighbors + 1)."
        )


##########################################################################################################
############################################### BASE SAMPLES #############################################
##########################################################################################################
def _generate_base_samples(
    class_counts: npt.NDArray,
    n_features: int,
    n_classes: int,
    signal_type: str,
    signal_strength: float,
    random_state: np.random.RandomState,
    **kwargs,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Generate a pool of base samples with at least ``class_counts[c]`` samples of each class ``c``.

    Parameters
    ----------
    class_counts : np.ndarray of shape (n_classes,)
        Number of samples needed of each class.
    n_features : int
        Number of features per sample.
    n_classes : int
        Number of classes.
    signal_type : str
        One of ``SIGNAL_TYPES``.
    signal_strength : float
        Strength of the signal separating the classes.
    random_state : RandomState instance
        The RandomState for reproducibility.
    **kwargs
        Additional keyword arguments passed to the scikit-learn generator.

    Returns
    -------
    X : np.ndarray of shape (n_pool, n_features)
        Base samples.
    y : np.ndarray of shape (n_pool,)
        Class labels.

    """
    n_needed = int(class_counts.sum())
    if signal_type == "linear":
        make_classification_kwargs = {
            "n_redundant": 0,
            "flip_y": 0.0,
            "n_clusters_per_class": 1,
            "n_informative": min(n_features, n_classes * 2),
            **kwargs,
        }
        # Class sizes follow `weights` up to rounding (and `flip_y`): grow the pool until every class is large enough.
        n_pool = n_needed
        while True:
            X, y = make_classification(
                n_samples=n_pool,
                n_features=n_features,
                n_classes=n_classes,
                weights=list(class_counts / n_needed),
                class_sep=signal_strength,
                random_state=random_state,
                **make_classification_kwargs,
            )
            if (np.bincount(y, minlength=n_classes) >= class_counts).all():
                break
            n_pool = int(1.2 * n_pool) + n_classes
    elif signal_type == "circles":
        X, y = make_circles(n_samples=tuple(class_counts), random_state=random_state, **kwargs)
    elif signal_type == "moons":
        X, y = make_moons(n_samples=tuple(class_counts), random_state=random_state, **kwargs)
    elif signal_type == "blobs":
        X, y = make_blobs(
            n_samples=list(class_counts),
            n_features=n_features,
            center_box=(-signal_strength, signal_strength),
            random_state=random_state,
            **kwargs,
        )
    else:  # gaussian_quantiles: classes are equal-sized quantile shells
        X, y = make_gaussian_quantiles(
            cov=signal_strength,
            n_features=n_features,
            n_samples=n_classes * int(class_counts.max()),
            n_classes=n_classes,
            random_state=random_state,
            **kwargs,
        )

    return X.astype(float), y


def _split_pool_by_site(y: npt.NDArray, class_counts: npt.NDArray, random_state: np.random.RandomState) -> list[npt.NDArray]:
    """Draw, without replacement, ``class_counts[k, c]`` pool indices of class ``c`` for each site ``k``."""
    shuffled_by_class = [random_state.permutation(np.flatnonzero(y == c)) for c in range(class_counts.shape[1])]
    boundaries = np.vstack([np.zeros(class_counts.shape[1], dtype=int), np.cumsum(class_counts, axis=0)])
    return [
        random_state.permutation(
            np.concatenate([shuffled_by_class[c][boundaries[k, c] : boundaries[k + 1, c]] for c in range(class_counts.shape[1])])
        )
        for k in range(class_counts.shape[0])
    ]


def _within_class_sd(X: npt.NDArray, y: npt.NDArray) -> npt.NDArray:
    """Pooled within-class standard deviation of each feature (the reference scale; 1 if undefined or 0)."""
    classes = np.unique(y)
    dof = len(X) - len(classes)
    if dof <= 0:
        return np.ones(X.shape[1])
    residuals = X.copy()
    for c in classes:
        residuals[y == c] -= X[y == c].mean(axis=0)
    sd = np.sqrt((residuals**2).sum(axis=0) / dof)
    sd[~(sd > 0)] = 1.0
    return sd


def _simulate_covariates(
    specs: list[Covariate] | None,
    site_rows: list[npt.NDArray],
    n_features: int,
    feature_scale: npt.NDArray,
    random_state: np.random.RandomState,
) -> tuple[dict[str, npt.NDArray], npt.NDArray, dict[str, npt.NDArray]]:
    """Draw the covariates of every site and compute their effects on the features (the same in every site)."""
    n_total = sum(len(rows) for rows in site_rows)
    if specs is None:
        return {}, np.zeros((n_total, n_features)), {}
    parts: dict[str, list[npt.NDArray]] = {spec.name: [] for spec in specs}
    for site, rows in enumerate(site_rows):
        for name, values in _make_covariate(specs, site, len(rows), random_state).items():
            parts[name].append(values)
    values = {name: np.concatenate(chunks) for name, chunks in parts.items()}
    effect, coefficients = _covariate_effects(specs, values, n_features, feature_scale, random_state)
    return values, effect, coefficients


##########################################################################################################
############################################# EoS ########################################################
##########################################################################################################
def _apply_site_effects(
    site_free: npt.NDArray,
    center: npt.NDArray,
    y: npt.NDArray,
    site_rows: list[npt.NDArray],
    feature_scale: npt.NDArray,
    s_params: _SimParams,
    random_state: np.random.RandomState,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Apply the site effects of every site.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
        Data with site effects.
    site_location, site_scale : np.ndarray of shape (n_sites, n_features)
        Offset (data units) and factor of each site.
    site_class_location : np.ndarray of shape (n_sites, n_classes, n_features)
        Offsets of the site-by-class interaction (data units).
    intensity : np.ndarray of shape (n_samples,)
        Per-sample intensity of the site effect.

    """
    n_features = site_free.shape[1]
    affected = np.zeros(n_features, dtype=bool)
    affected[s_params.affected_features] = True
    h = s_params.heterogeneity
    # Always drawn, so that changing the heterogeneity keeps the site parameters (same random sequence)
    intensity = np.exp(h * random_state.normal(size=len(site_free)) - h**2 / 2)

    X = site_free.copy()
    site_location = np.zeros((s_params.n_sites, n_features))
    site_scale = np.ones((s_params.n_sites, n_features))
    site_class_location = np.zeros((s_params.n_sites, s_params.n_classes, n_features))
    for site, rows in enumerate(site_rows):
        strength = {name: s[site] for name, s in s_params.site_effect.items()}
        log_factor, offset = _site_parameters(strength, n_features, random_state)
        k = strength.get("class_interaction", 0.0)
        class_offset = random_state.uniform(-k / 2, k / 2, size=(s_params.n_classes, n_features))
        log_factor[~affected], offset[~affected], class_offset[:, ~affected] = 0.0, 0.0, 0.0
        site_scale[site] = np.exp(log_factor)
        site_location[site] = offset * feature_scale
        site_class_location[site] = class_offset * feature_scale

        a = intensity[rows, None]
        deviation = site_free[rows] - center[rows]
        # Written as increments of site_free, so that a site without effect is left exactly unchanged
        Xs = site_free[rows] + deviation * np.expm1(a * log_factor) + a * (offset + class_offset[y[rows]]) * feature_scale
        Xs = _apply_other_effects(Xs, deviation, a, strength, affected, feature_scale, random_state)
        X[rows] = Xs
        logger.debug(f"Site {site}: EoS strengths {strength}")
    return X, site_location, site_scale, site_class_location, intensity


def _site_parameters(
    strength: dict[str, float], n_features: int, random_state: np.random.RandomState
) -> tuple[npt.NDArray, npt.NDArray]:
    """Per-feature log-factor and offset (reference-SD units) of one site, from the scale, location and combat types."""
    log_factor = np.zeros(n_features)
    offset = np.zeros(n_features)
    if "combat" in strength:
        s = strength["combat"]
        # Inverse gamma of shape 6 (CV 0.5) with mean m = exp(U(-0.2 s, 0.2 s)): 1 / Gamma(6, scale=1 / (5 m))
        mean_variance = np.exp(random_state.uniform(-0.2 * s, 0.2 * s))
        variance_factor = 1.0 / random_state.gamma(6.0, 1.0 / (5.0 * mean_variance), size=n_features)
        log_factor += 0.5 * np.log(variance_factor) if s > 0 else 0.0
        offset += random_state.uniform(-s / 2, s / 2) + random_state.normal(0.0, s / 4, size=n_features)
    if "scale" in strength:
        s = strength["scale"]
        log_factor += random_state.uniform(-0.1 * s, 0.1 * s, size=n_features)
    if "location" in strength:
        s = strength["location"]
        offset += random_state.uniform(-s / 2, s / 2, size=n_features)
    return log_factor, offset


def _apply_other_effects(
    X: npt.NDArray,
    deviation: npt.NDArray,
    intensity: npt.NDArray,
    strength: dict[str, float],
    affected: npt.NDArray,
    feature_scale: npt.NDArray,
    random_state: np.random.RandomState,
) -> npt.NDArray:
    """Apply the variance, nonlinear and dropout site effects of one site (to the affected features)."""
    n_features = X.shape[1]
    if "variance" in strength:
        sd = random_state.uniform(0.0, strength["variance"], size=n_features) * affected
        X = X + random_state.normal(size=X.shape) * sd * intensity * feature_scale
    if "nonlinear" in strength:
        weight = random_state.uniform(0.0, 0.1 * strength["nonlinear"], size=n_features) * affected
        X = X + intensity * weight * np.square(deviation) / feature_scale
    if "dropout" in strength:
        keep_probability = float(np.clip(1.0 - 0.1 * strength["dropout"], 0.0, 1.0))
        keep = random_state.binomial(1, keep_probability, size=n_features).astype(bool) | ~affected
        X = X * keep
    return X
