"""Simulation of multi-site regression data."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, get_args, overload

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from uniharmony._verbose import get_logger

from ._make_covariates import Covariate
from ._simulation_constants import (
    DEFAULT_REGRESSION_SIGNAL,
    MIN_TARGET_RANGE,
    PRESET_COVARIATES,
    REGRESSION_SIGNAL_TYPES,
    SiteEffect,
)
from ._simulation_core import (
    RandomStateLike,
    _apply_site_effects,
    _check_integers,
    _draw_noise,
    _format_outputs,
    _ground_truth,
    _samples_per_site,
    _SharedParams,
    _signal_type_and_strength,
    _simulate_covariates,
    _validate_shared,
)


__all__ = [
    "make_multisite_regression",
]

logger = get_logger(src="make_multisite_regression")

# Friedman #1, f = 10 sin(pi x0 x1) + 20 (x2 - 0.5)**2 + 10 x3 + 5 x4 with x ~ U(0, 1): exact mean and SD
# (numerical integration of the sine term; the other terms are polynomials of uniform variables)
_FRIEDMAN1_MEAN = 14.413297342419858
_FRIEDMAN1_SD = 4.881235885652423
# Samples used to find the target quantiles of non-Gaussian signals (drawn with a fixed seed)
_N_REFERENCE = 200_000

TargetRange = tuple[float, float]


@overload
def make_multisite_regression(
    n_sites: int = 2,
    n_samples: int | Sequence[int] = 1000,
    n_features: int = 10,
    target_range_per_site: TargetRange | Sequence[TargetRange] | None = None,
    signal: Mapping[str, float] | None = None,
    n_informative: int | None = None,
    noise_strength: float | Sequence[float] = 0.1,
    site_effect: SiteEffect | None = None,
    site_effect_features: int | float | Sequence[int] | None = None,
    site_effect_heterogeneity: float = 0.0,
    covariates: Sequence[Covariate | str | PRESET_COVARIATES] | None = None,
    site_names: Sequence | None = None,
    as_frame: bool = False,
    return_ground_truth: Literal[False] = False,
    random_state: RandomStateLike = 42,
) -> tuple[Any, Any, Any]: ...


@overload
def make_multisite_regression(
    n_sites: int = 2,
    n_samples: int | Sequence[int] = 1000,
    n_features: int = 10,
    target_range_per_site: TargetRange | Sequence[TargetRange] | None = None,
    signal: Mapping[str, float] | None = None,
    n_informative: int | None = None,
    noise_strength: float | Sequence[float] = 0.1,
    site_effect: SiteEffect | None = None,
    site_effect_features: int | float | Sequence[int] | None = None,
    site_effect_heterogeneity: float = 0.0,
    covariates: Sequence[Covariate | str | PRESET_COVARIATES] | None = None,
    site_names: Sequence | None = None,
    as_frame: bool = False,
    *,
    return_ground_truth: Literal[True],
    random_state: RandomStateLike = 42,
) -> tuple[Any, Any, Any, dict[str, Any]]: ...


def make_multisite_regression(
    n_sites: int = 2,
    n_samples: int | Sequence[int] = 1000,
    n_features: int = 10,
    target_range_per_site: TargetRange | Sequence[TargetRange] | None = None,
    signal: Mapping[str, float] | None = None,
    n_informative: int | None = None,
    noise_strength: float | Sequence[float] = 0.1,
    site_effect: SiteEffect | None = None,
    site_effect_features: int | float | Sequence[int] | None = None,
    site_effect_heterogeneity: float = 0.0,
    covariates: Sequence[Covariate | str | PRESET_COVARIATES] | None = None,
    site_names: Sequence | None = None,
    as_frame: bool = False,
    return_ground_truth: bool = False,
    random_state: RandomStateLike = 42,
) -> tuple[Any, ...]:
    """Simulate multi-site regression data with signal, covariates, noise and site effects.

    The data follow the model::

        y = s * f(Signal) + e,  e ~ N(0, 1)
        X = SiteEffect(Signal + CovariateEffects + Noise; site)

    1. Base samples (the signal) are drawn from the population of the
       generator selected by ``signal``, and their target ``y`` from them.
    2. Each site keeps the samples whose target falls in its range of the
       population (``target_range_per_site``), e.g. a site that recruits
       children and one that recruits adults: the sites then differ in
       their target, and therefore in their signal. No sample is used twice.
    3. Optional covariates are drawn per site and their effects, the same in
       every site, are added to the signal (not to the target).
    4. Independent Gaussian noise is added. Signal, covariate effects and
       noise form the data without site effect (``site_free``).
    5. The Effect of Site (EoS) of ``site_effect`` is applied. It changes
       the features only; the target is never affected.

    Except for the generator and the target ranges, the parameters and
    their meaning are those of :func:`make_multisite_classification`
    (without the ``"class_interaction"`` site effect, which needs classes).

    **Units.** All strengths (site effects, noise, covariate effects) are in
    units of the SD of each feature in the population of the signal (1 for
    ``"linear"``, ``1 / sqrt(12)`` for ``"friedman1"``).

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

    target_range_per_site : tuple, list of tuple or None, optional (default None)
        Range of the target in each site, as ``(low, high)`` quantiles of the
        population distribution of ``y`` (``0 <= low < high <= 1``, with
        ``high - low >= 0.01``). One pair applies to every site; a list of
        ``n_sites`` pairs gives one per site, e.g. ``[(0, 0.5), (0.5, 1)]``
        for a site with the lower half of the targets and one with the upper
        half. Within its range, a site has the population distribution.
        ``None`` gives every site the whole population.

    signal : dict or None, optional (default None, i.e. ``{"linear": 2.0}``)
        Generator of the base problem and its strength ``s``, as ``{type:
        strength}`` with one type. ``f`` is standardized to mean 0 and SD 1
        in the population, so that ``s`` is the SD of the explained part of
        ``y`` and the population ``R**2 = s**2 / (1 + s**2)`` (0.8 for
        ``s = 2``; ``s = 0`` gives a target unrelated to the features):

        - ``"linear"``: features ``~ N(0, 1)``, ``f`` a linear combination
          of ``n_informative`` of them with random weights.
        - ``"friedman1"``: features ``~ U(0, 1)``, ``f`` the Friedman #1
          function ``10 sin(pi x0 x1) + 20 (x2 - 0.5)**2 + 10 x3 + 5 x4``
          of the first five features (needs ``n_features >= 5``), a
          nonlinear problem with interactions.

    n_informative : int or None, optional (default None)
        For ``"linear"``, the number of features that carry the signal (the
        first ``n_informative``); ``None`` for all.

    noise_strength : float or list of float, optional (default 0.1)
        Standard deviation of the Gaussian noise added to the features, in
        reference-SD units, for all sites or per site.

    site_effect : dict or None, optional (default None, i.e. ``{"location": 3.0}``)
        Types of EoS and their strength, as in
        :func:`make_multisite_classification` (``"location"``, ``"scale"``,
        ``"variance"``, ``"nonlinear"``, ``"dropout"``, ``"combat"``).

    site_effect_features : int, float, sequence of int or None, optional (default None)
        Features affected by the EoS: ``None`` for all, an int ``k`` for the
        first ``k``, a float in (0, 1] for that fraction of the first
        features, or a sequence of indices.

    site_effect_heterogeneity : float, optional (default 0.0)
        Variation of the EoS between the samples of a site: each sample's
        site effect is multiplied by ``exp(h * z - h**2 / 2)``, ``z ~ N(0,
        1)`` per sample.

    covariates : list of Covariate or str, or None, optional (default None)
        Covariate specifications (a :class:`Covariate` or a preset name);
        their effects change the features, not the target.

    site_names : sequence or None, optional (default None)
        Labels of the sites; ``None`` gives ``0`` to ``n_sites - 1``.

    as_frame : bool, optional (default False)
        If True, ``X`` is a DataFrame, ``y`` a Series named ``"target"`` and
        ``sites`` a Series named ``"site"``.

    return_ground_truth : bool, optional (default False)
        If True, also return a dict with the components of the data, all
        row-aligned with ``X``. It has the keys of
        :func:`make_multisite_classification` (but ``"site_class_location"``),
        and:

        - ``"target_signal"``: ``s * f(signal)``, the part of ``y``
          explained by the signal (``y - target_signal`` is ``e``);
        - ``"target_coefficients"``: for ``"linear"``, the coefficients
          ``beta`` with ``target_signal = signal @ beta``; None otherwise.

    random_state : int, RandomState, Generator or None, optional (default 42)
        The seed of the pseudo random number generator, or a RandomState or
        Generator, for reproducibility.

    Returns
    -------
    X : np.ndarray or DataFrame of shape (n_samples, n_features)
        Simulated features.
    y : np.ndarray or Series of shape (n_samples,)
        Target.
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

    See Also
    --------
    make_multisite_classification : The same model with class labels.
    make_traveling_subjects : The same subjects measured in every site.

    Examples
    --------
    A site with the lower and one with the upper half of the targets (e.g.
    a young and an old cohort for brain-age prediction):

    >>> X, y, sites = make_multisite_regression(target_range_per_site=[(0, 0.5), (0.5, 1)])
    >>> X.shape, bool(y[sites == 0].max() <= y[sites == 1].min())
    ((1000, 10), True)

    """
    s_params = _validate_inputs(
        n_sites=n_sites,
        n_samples=n_samples,
        n_features=n_features,
        target_range_per_site=target_range_per_site,
        signal=signal,
        n_informative=n_informative,
        noise_strength=noise_strength,
        site_effect=site_effect,
        site_effect_features=site_effect_features,
        site_effect_heterogeneity=site_effect_heterogeneity,
        covariates=covariates,
        site_names=site_names,
        random_state=random_state,
    )
    shared = s_params.shared
    rng = shared.random_state

    # Steps 1-2: the signal of each site, with its target in the site's range
    function, coefficients = _signal_function(s_params, rng)
    thresholds = _target_thresholds(s_params.signal_type, s_params.signal_strength, s_params.target_ranges)
    parts = [
        _draw_site(int(n), low, high, width, function, s_params, rng)
        for n, (low, high), width in zip(
            s_params.samples_per_site, thresholds, np.diff(s_params.target_ranges, axis=1)[:, 0], strict=True
        )
    ]
    signal_samples = np.vstack([part[0] for part in parts])
    target_signal = np.concatenate([part[1] for part in parts])
    y = np.concatenate([part[2] for part in parts])
    site_codes = np.repeat(np.arange(s_params.n_sites), s_params.samples_per_site)
    site_rows = [np.flatnonzero(site_codes == site) for site in range(s_params.n_sites)]
    feature_scale = np.full(s_params.n_features, _FEATURE_SD[s_params.signal_type])  # the unit of all strengths

    # Step 3: covariates, drawn per site, and their effects, the same in every site
    covariate_values, covariate_effect, covariate_coefficients = _simulate_covariates(
        shared.covariates, site_rows, s_params.n_features, feature_scale, rng
    )

    # Step 4: noise; signal + covariate effects + noise is the data without site effect
    noise = _draw_noise(site_rows, s_params.n_features, shared.noise_strength, feature_scale, rng)
    site_free = signal_samples + covariate_effect + noise
    center = (signal_samples + noise).mean(axis=0) + covariate_effect

    # Step 5: site effects (on the features only)
    effects = _apply_site_effects(site_free, center, site_rows, feature_scale, shared.site_effect, rng)
    logger.info(f"Generated {len(y)} samples across {s_params.n_sites} sites")

    truth = None
    if return_ground_truth:
        truth = _ground_truth(
            signal=signal_samples,
            target_signal=target_signal,
            target_coefficients=coefficients,
            site_free=site_free,
            effects=effects,
            covariate_values=covariate_values,
            covariate_effect=covariate_effect,
            covariate_coefficients=covariate_coefficients,
            noise=noise,
            params=shared.site_effect,
            feature_scale=feature_scale,
        )
    labels = {"target": y, "site": shared.site_names[site_codes]}
    return _format_outputs(effects.X, labels, shared.site_names, as_frame, truth)


##########################################################################################################
############################################### VALIDATION ###############################################
##########################################################################################################
@dataclass
class _SimParams:
    """Validated and resolved parameters of ``make_multisite_regression``."""

    n_sites: int
    n_features: int
    samples_per_site: npt.NDArray  # (n_sites,)
    target_ranges: npt.NDArray  # (n_sites, 2) quantiles of the population target
    signal_type: str
    signal_strength: float
    n_informative: int
    shared: _SharedParams


def _validate_inputs(**params: Any) -> _SimParams:
    """Validate every parameter of ``make_multisite_regression`` and resolve defaults and broadcasts."""
    n_sites, n_features = _check_integers(n_sites=params["n_sites"], n_features=params["n_features"])
    if n_features <= 0:
        raise ValueError(f"n_features must be positive, got {n_features}")
    samples_per_site = _samples_per_site(params["n_samples"], n_sites)
    signal_type, strength = _signal_type_and_strength(
        params["signal"], DEFAULT_REGRESSION_SIGNAL, get_args(REGRESSION_SIGNAL_TYPES)
    )
    n_informative = _validate_n_informative(params["n_informative"], signal_type, n_features)
    shared = _validate_shared(
        n_sites=n_sites,
        n_features=n_features,
        noise_strength=params["noise_strength"],
        site_effect=params["site_effect"],
        site_effect_features=params["site_effect_features"],
        site_effect_heterogeneity=params["site_effect_heterogeneity"],
        covariates=params["covariates"],
        site_names=params["site_names"],
        random_state=params["random_state"],
        with_classes=False,
    )
    return _SimParams(
        n_sites=n_sites,
        n_features=n_features,
        samples_per_site=samples_per_site,
        target_ranges=_validate_target_ranges(params["target_range_per_site"], n_sites),
        signal_type=signal_type,
        signal_strength=strength,
        n_informative=n_informative,
        shared=shared,
    )


def _validate_n_informative(n_informative: object, signal_type: str, n_features: int) -> int:
    """Return the number of features carrying the signal: all by default for "linear", 5 for "friedman1"."""
    if signal_type == "friedman1":
        if n_informative is not None:
            raise ValueError("n_informative only applies to the 'linear' signal; 'friedman1' uses the first 5 features.")
        if n_features < 5:
            raise ValueError(f"signal 'friedman1' needs n_features >= 5, got {n_features}.")
        return 5
    if n_informative is None:
        return n_features
    (n_informative,) = _check_integers(n_informative=n_informative)
    if not 1 <= n_informative <= n_features:
        raise ValueError(f"n_informative must be between 1 and n_features={n_features}, got {n_informative}.")
    return n_informative


def _validate_target_ranges(ranges: object, n_sites: int) -> npt.NDArray:
    """Return the (low, high) quantile range of every site, shape (n_sites, 2)."""
    if ranges is None:
        return np.tile([0.0, 1.0], (n_sites, 1))
    message = (
        "target_range_per_site must be a (low, high) pair of quantiles or a list of one pair per site, "
        f"e.g. [(0, 0.5), (0.5, 1)], got {ranges!r}."
    )
    try:
        array = np.asarray(ranges, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError(message) from error
    if array.shape == (2,):
        array = np.tile(array, (n_sites, 1))
    if array.ndim != 2 or array.shape[1] != 2:
        raise TypeError(message)
    if len(array) != n_sites:
        raise ValueError(f"target_range_per_site must have one pair per site ({n_sites}), got {len(array)}.")
    low, high = array[:, 0], array[:, 1]
    if not (np.isfinite(array).all() and (low >= 0).all() and (high <= 1).all() and (low < high).all()):
        raise ValueError(f"target_range_per_site needs 0 <= low < high <= 1 in every site, got {array.tolist()}.")
    if (high - low < MIN_TARGET_RANGE).any():
        raise ValueError(
            f"Each target range must cover at least {MIN_TARGET_RANGE} of the population (high - low), got {array.tolist()}."
        )
    return array


##########################################################################################################
################################################# SIGNAL #################################################
##########################################################################################################
_FEATURE_SD = {"linear": 1.0, "friedman1": 1.0 / np.sqrt(12.0)}
SignalFunction = Callable[[int, np.random.RandomState], tuple[npt.NDArray, npt.NDArray]]


def _signal_function(s_params: _SimParams, random_state: np.random.RandomState) -> tuple[SignalFunction, npt.NDArray | None]:
    """Return a function drawing ``n`` base samples and their standardized ``f``, and the linear coefficients."""
    n_features, strength = s_params.n_features, s_params.signal_strength
    if s_params.signal_type == "linear":
        weights = np.zeros(n_features)
        weights[: s_params.n_informative] = random_state.normal(size=s_params.n_informative)
        weights /= np.linalg.norm(weights)  # f = X @ weights has SD 1 in the population

        def linear(n: int, rng: np.random.RandomState) -> tuple[npt.NDArray, npt.NDArray]:
            X = rng.normal(size=(n, n_features))
            return X, X @ weights

        return linear, strength * weights
    return _friedman1(n_features), None


def _friedman1(n_features: int) -> SignalFunction:
    """Friedman #1 on the first five of ``n_features`` uniform features, standardized."""

    def friedman1(n: int, rng: np.random.RandomState) -> tuple[npt.NDArray, npt.NDArray]:
        X = rng.uniform(size=(n, n_features))
        f = 10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4]
        return X, (f - _FRIEDMAN1_MEAN) / _FRIEDMAN1_SD

    return friedman1


def _target_thresholds(signal_type: str, strength: float, ranges: npt.NDArray) -> npt.NDArray:
    """Target values at the quantiles ``ranges`` of the population distribution of ``y = s f + e``."""
    if signal_type == "linear":  # y ~ N(0, 1 + s**2) exactly
        return norm.ppf(ranges) * np.sqrt(1.0 + strength**2)
    reference = _reference_targets(signal_type, strength)
    thresholds = np.quantile(reference, np.clip(ranges, 0, 1))
    thresholds[ranges == 0], thresholds[ranges == 1] = -np.inf, np.inf
    return thresholds


@lru_cache(maxsize=8)
def _reference_targets(signal_type: str, strength: float) -> npt.NDArray:
    """Draw a large sample of the population target with a fixed seed (independent of ``random_state``)."""
    rng = np.random.RandomState(0)
    _, f = _friedman1(5)(_N_REFERENCE, rng)
    return np.sort(strength * f + rng.normal(size=_N_REFERENCE))


def _draw_site(
    n: int,
    low: float,
    high: float,
    width: float,
    function: SignalFunction,
    s_params: _SimParams,
    random_state: np.random.RandomState,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Draw ``n`` samples whose target is in ``[low, high]`` (rejection sampling from the population).

    ``width`` is the fraction of the population in the range, used to size the batches.
    """
    X_parts, f_parts, y_parts = [], [], []
    missing = n
    while missing > 0:
        batch = int(np.ceil(1.2 * missing / width)) + 10
        X, f = function(batch, random_state)
        target_signal = s_params.signal_strength * f
        y = target_signal + random_state.normal(size=batch)
        keep = np.flatnonzero((y >= low) & (y <= high))[:missing]
        X_parts.append(X[keep])
        f_parts.append(target_signal[keep])
        y_parts.append(y[keep])
        missing -= len(keep)
    return np.vstack(X_parts), np.concatenate(f_parts), np.concatenate(y_parts)
