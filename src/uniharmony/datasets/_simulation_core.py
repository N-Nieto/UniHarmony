"""Machinery shared by the multi-site simulators.

Validation of the shared parameters, covariates, noise, effects of site and
output formatting. The simulators (``make_multisite_classification``,
``make_multisite_regression``, ``make_traveling_subjects``) differ only in how
they generate the signal and assign the samples to sites; everything that
happens to the data afterwards is done here, so the model is the same for all:

    X = SiteEffect(Signal + CovariateEffects + Noise; site)
"""

import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, get_args

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.utils import check_random_state

from uniharmony._verbose import get_logger

from ._make_covariates import Covariate, _covariate_effects, _make_covariate, _resolve_covariates, _resolve_feature_subset
from ._simulation_constants import (
    CLASS_SITE_EFFECT_TYPES,
    DEFAULT_DROPOUT_UNIT,
    DEFAULT_SITE_EFFECT,
    DROPOUT_UNITS,
    SITE_EFFECT_TYPES,
    SiteEffect,
)


logger = get_logger(src="multisite_simulation")

RandomStateLike = int | np.random.RandomState | np.random.Generator | None


##########################################################################################################
############################################ SHARED PARAMETERS ###########################################
##########################################################################################################
@dataclass
class _Dropout:
    """Validated ``site_effect["dropout"]``."""

    value: float  # replacement value (NaN for missing values)
    prob: npt.NDArray  # (n_sites,) probability of replacing a feature (or a value) in each site
    per: str  # "feature": whole features of a site; "value": single values


@dataclass
class _SiteEffectParams:
    """Validated effect of site: strengths per type and site, and where it applies."""

    n_sites: int
    strengths: dict[str, npt.NDArray]  # type -> (n_sites,), every type but "dropout"
    dropout: _Dropout | None
    affected_features: npt.NDArray  # indices of the features with a site effect
    heterogeneity: float


@dataclass
class _SharedParams:
    """Validated parameters that every simulator has."""

    random_state: np.random.RandomState
    noise_strength: npt.NDArray  # (n_sites,)
    site_effect: _SiteEffectParams
    covariates: list[Covariate] | None
    site_names: npt.NDArray


def _validate_shared(
    *,
    n_sites: int,
    n_features: int,
    noise_strength: object,
    site_effect: SiteEffect | None,
    site_effect_features: int | float | Sequence[int] | None,
    site_effect_heterogeneity: object,
    covariates: Sequence[Covariate | str] | None,
    site_names: Sequence | None,
    random_state: RandomStateLike,
    with_classes: bool,
    covariate_sites: int | None = None,
) -> _SharedParams:
    """Validate the parameters shared by every simulator.

    ``covariate_sites`` is the number of covariate distributions expected
    (default ``n_sites``; 1 when the covariates describe one population).
    """
    strengths, dropout = _validate_site_effect(site_effect, n_sites, with_classes)
    return _SharedParams(
        random_state=_check_random_state(random_state),
        noise_strength=_per_site(noise_strength, n_sites, "noise_strength"),
        site_effect=_SiteEffectParams(
            n_sites=n_sites,
            strengths=strengths,
            dropout=dropout,
            affected_features=_resolve_feature_subset(site_effect_features, n_features, "site_effect_features"),
            heterogeneity=_check_strength(site_effect_heterogeneity, "site_effect_heterogeneity"),
        ),
        covariates=_resolve_covariates(covariates, n_sites if covariate_sites is None else covariate_sites),
        site_names=_validate_site_names(site_names, n_sites),
    )


def _check_random_state(seed: RandomStateLike) -> np.random.RandomState:
    """Return a RandomState; a NumPy Generator seeds a new RandomState."""
    if isinstance(seed, np.random.Generator):
        return np.random.RandomState(int(seed.integers(0, 2**32 - 1)))
    return check_random_state(seed)


def _check_integers(**values: object) -> tuple[int, ...]:
    """Check that the size parameters are integers (not booleans, not floats)."""
    checked = []
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an int, got {value!r} ({type(value).__name__}).")
        checked.append(int(value))
    return tuple(checked)


def _check_strength(value: object, name: str) -> float:
    """Check a single non-negative, finite number."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"Invalid type for {name}: must be a float, got {value!r} ({type(value).__name__}).")
    number = float(value)
    if not (np.isfinite(number) and number >= 0):
        raise ValueError(f"{name} must be non-negative and finite, got {number}")
    return number


def _per_site(value: object, n_sites: int, name: str) -> npt.NDArray:
    """Broadcast a non-negative strength to all sites, or validate one strength per site."""
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return np.full(n_sites, _check_strength(value, name))
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise TypeError(f"Invalid type for {name}: must be a float or list[float], got {type(value)}")
    if len(value) != n_sites:
        raise ValueError(f"{name} must have length n_sites ({n_sites}), got {len(value)}")
    return np.array([_check_strength(v, f"{name}[{i}]") for i, v in enumerate(value)])


def _samples_per_site(n_samples: object, n_sites: int) -> npt.NDArray:
    """Return the number of samples of each site: an int split evenly, or one count per site."""
    if n_sites < 1:
        raise ValueError(f"n_sites must be at least 1, got {n_sites}")
    if n_sites == 1:
        logger.warning("n_sites is 1, which may not be suitable for testing multi-site methods.")
    if isinstance(n_samples, (int, np.integer)) and not isinstance(n_samples, bool):
        total = int(n_samples)
        if total < n_sites:
            raise ValueError(f"n_samples ({total}) is less than n_sites ({n_sites}). Some sites will have 0 samples.")
        counts = np.full(n_sites, total // n_sites, dtype=int)
        counts[: total % n_sites] += 1
        return counts

    counts = np.asarray(n_samples)
    if counts.ndim != 1 or not np.issubdtype(counts.dtype, np.integer):
        raise TypeError(f"n_samples must be int or list[int], got {type(n_samples)}.")
    if len(counts) != n_sites:
        raise ValueError(f"n_samples has len {len(counts)} and does not match with n_sites={n_sites}")
    if counts.min() < 1:
        raise ValueError(f"Every site needs at least 1 sample, got n_samples={counts.tolist()}")
    return counts.astype(int)


def _signal_type_and_strength(signal: object, default: Mapping[str, float], allowed: tuple[str, ...]) -> tuple[str, float]:
    """Return the type and strength of a ``{type: strength}`` signal with one supported type."""
    signal = default if signal is None else signal
    if not isinstance(signal, Mapping) or len(signal) != 1:
        error = ValueError if isinstance(signal, Mapping) else TypeError
        example = f"{{{next(iter(default))!r}: {next(iter(default.values()))}}}"
        raise error(f"signal must be a dict with one type and its strength, e.g. {example}, got {signal!r}.")
    ((signal_type, strength),) = signal.items()
    if signal_type not in allowed:
        raise ValueError(f"Unsupported signal type: {signal_type!r}. Choose from {allowed}.")
    return signal_type, _check_strength(strength, f"signal['{signal_type}']")


def _validate_site_effect(
    site_effect: SiteEffect | None, n_sites: int, with_classes: bool
) -> tuple[dict[str, npt.NDArray], _Dropout | None]:
    """Return the strength of every site effect type in every site, shape (n_sites,) per type, and the dropout."""
    site_effect = DEFAULT_SITE_EFFECT if site_effect is None else site_effect
    if not isinstance(site_effect, Mapping):
        raise TypeError(f"site_effect must be a dict {{type: strength}}, e.g. {{'location': 3.0}}, got {site_effect!r}.")
    strengths: dict[str, npt.NDArray] = {}
    dropout = None
    for name, strength in site_effect.items():
        if name not in get_args(SITE_EFFECT_TYPES):
            raise ValueError(f"Unsupported site effect type: {name!r}. Choose from {get_args(SITE_EFFECT_TYPES)}.")
        if name in CLASS_SITE_EFFECT_TYPES and not with_classes:
            raise ValueError(f"Site effect type {name!r} needs class labels; it is only available for classification.")
        if name == "dropout":
            dropout = _validate_dropout(strength, n_sites)
        else:
            strengths[name] = _per_site(strength, n_sites, f"site_effect['{name}']")
    return strengths, dropout


def _validate_dropout(spec: object, n_sites: int) -> _Dropout:
    """Validate ``{"value": float, "prob": float or one per site, "per": "feature" or "value"}``."""
    name = "site_effect['dropout']"
    if not isinstance(spec, Mapping):
        raise TypeError(
            f"{name} must be a dict {{'value': float, 'prob': float or list per site}}, "
            f"e.g. {{'value': np.nan, 'prob': 0.2}}, got {spec!r}."
        )
    keys = set(spec)
    if not {"value", "prob"} <= keys or not keys <= {"value", "prob", "per"}:
        raise ValueError(f"{name} needs the keys 'value' and 'prob' (and optionally 'per'), got {sorted(keys)}.")
    value = spec["value"]
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name}['value'] must be a float (e.g. np.nan or 0.0), got {value!r}.")
    prob = _per_site(spec["prob"], n_sites, f"{name}['prob']")
    if (prob > 1).any():
        raise ValueError(f"{name}['prob'] must be between 0 and 1, got {prob.tolist()}.")
    per = spec.get("per", DEFAULT_DROPOUT_UNIT)
    if per not in get_args(DROPOUT_UNITS):
        raise ValueError(f"{name}['per'] must be one of {get_args(DROPOUT_UNITS)}, got {per!r}.")
    return _Dropout(value=float(value), prob=prob, per=per)


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
######################################## COVARIATES AND NOISE ############################################
##########################################################################################################
def _simulate_covariates(
    specs: list[Covariate] | None,
    site_rows: list[npt.NDArray],
    n_features: int,
    feature_scale: npt.NDArray,
    random_state: np.random.RandomState,
) -> tuple[dict[str, npt.NDArray], npt.NDArray, dict[str, npt.NDArray]]:
    """Draw the covariates of every site and compute their effects on the features (the same in every site).

    ``site_rows[k]`` are the rows drawn from the covariate distributions of
    site ``k``; the returned arrays follow the order of the rows.
    """
    n_total = sum(len(rows) for rows in site_rows)
    if specs is None:
        return {}, np.zeros((n_total, n_features)), {}
    values = {spec.name: np.empty(n_total, dtype=object) for spec in specs}
    for site, rows in enumerate(site_rows):
        for name, site_values in _make_covariate(specs, site, len(rows), random_state).items():
            values[name][rows] = site_values
    values = {name: np.array(column.tolist()) for name, column in values.items()}  # back to a numeric dtype
    effect, coefficients = _covariate_effects(specs, values, n_features, feature_scale, random_state)
    return values, effect, coefficients


def _draw_noise(
    site_rows: list[npt.NDArray],
    n_features: int,
    noise_strength: npt.NDArray,
    feature_scale: npt.NDArray,
    random_state: np.random.RandomState,
) -> npt.NDArray:
    """Gaussian noise with SD ``noise_strength[site]`` reference SDs, site by site."""
    noise = np.zeros((sum(len(rows) for rows in site_rows), n_features))
    for site, rows in enumerate(site_rows):
        noise[rows] = random_state.normal(size=(len(rows), n_features)) * noise_strength[site] * feature_scale
    return noise


##########################################################################################################
############################################# EFFECTS OF SITE ############################################
##########################################################################################################
@dataclass
class _SiteEffectResult:
    """Data with site effects and the parameters that produced them."""

    X: npt.NDArray  # (n_samples, n_features)
    site_location: npt.NDArray  # (n_sites, n_features), data units
    site_scale: npt.NDArray  # (n_sites, n_features)
    site_class_location: npt.NDArray | None  # (n_sites, n_classes, n_features), data units
    intensity: npt.NDArray  # (n_samples,)
    dropout_mask: npt.NDArray  # (n_samples, n_features), True where a value was replaced


def _apply_site_effects(
    site_free: npt.NDArray,
    center: npt.NDArray,
    site_rows: list[npt.NDArray],
    feature_scale: npt.NDArray,
    params: _SiteEffectParams,
    random_state: np.random.RandomState,
    classes: npt.NDArray | None = None,
    n_classes: int = 0,
    z: npt.NDArray | None = None,
) -> _SiteEffectResult:
    """Apply the site effects of every site.

    Parameters
    ----------
    site_free : np.ndarray of shape (n_samples, n_features)
        Data without site effect.
    center : np.ndarray of shape (n_samples, n_features)
        Expected value of each sample without site effect; the scale and
        nonlinear effects act on the deviations from it.
    site_rows : list of np.ndarray
        Rows of each site.
    feature_scale : np.ndarray of shape (n_features,)
        Reference SD of each feature (the unit of the strengths).
    params : _SiteEffectParams
        Validated site effect.
    random_state : RandomState
        Random state.
    classes : np.ndarray of shape (n_samples,) or None
        Class labels, for the ``"class_interaction"`` type (classification).
    n_classes : int
        Number of classes.
    z : np.ndarray of shape (n_samples,) or None
        Standard normal draws of the per-sample intensity; drawn here if None.

    """
    n_samples, n_features = site_free.shape
    affected = np.zeros(n_features, dtype=bool)
    affected[params.affected_features] = True
    h = params.heterogeneity
    # Always drawn, so that changing the heterogeneity keeps the site parameters (same random sequence)
    draws = random_state.normal(size=n_samples) if z is None else z
    intensity = np.exp(h * draws - h**2 / 2)

    X = site_free.copy()
    site_location = np.zeros((params.n_sites, n_features))
    site_scale = np.ones((params.n_sites, n_features))
    site_class_location = None if classes is None else np.zeros((params.n_sites, n_classes, n_features))
    dropout_mask = np.zeros((n_samples, n_features), dtype=bool)
    for site, rows in enumerate(site_rows):
        strength = {name: s[site] for name, s in params.strengths.items()}
        log_factor, offset = _site_parameters(strength, n_features, random_state)
        log_factor[~affected], offset[~affected] = 0.0, 0.0
        site_scale[site] = np.exp(log_factor)
        site_location[site] = offset * feature_scale
        sample_offset = np.broadcast_to(offset, (len(rows), n_features))
        if classes is not None and site_class_location is not None:
            k = strength.get("class_interaction", 0.0)
            class_offset = random_state.uniform(-k / 2, k / 2, size=(n_classes, n_features))
            class_offset[:, ~affected] = 0.0
            site_class_location[site] = class_offset * feature_scale
            sample_offset = sample_offset + class_offset[classes[rows]]

        a = intensity[rows, None]
        deviation = site_free[rows] - center[rows]
        # Written as increments of site_free, so that a site without effect is left exactly unchanged
        Xs = site_free[rows] + deviation * np.expm1(a * log_factor) + a * sample_offset * feature_scale
        Xs = _apply_other_effects(Xs, deviation, a, strength, affected, feature_scale, random_state)
        if params.dropout is not None:
            dropout_mask[rows] = _dropout_mask(params.dropout, site, len(rows), affected, random_state)
            Xs = np.where(dropout_mask[rows], params.dropout.value, Xs)
        X[rows] = Xs
        logger.debug(f"Site {site}: EoS strengths {strength}")
    return _SiteEffectResult(X, site_location, site_scale, site_class_location, intensity, dropout_mask)


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
    """Apply the variance and nonlinear site effects of one site (to the affected features)."""
    n_features = X.shape[1]
    if "variance" in strength:
        sd = random_state.uniform(0.0, strength["variance"], size=n_features) * affected
        X = X + random_state.normal(size=X.shape) * sd * intensity * feature_scale
    if "nonlinear" in strength:
        weight = random_state.uniform(0.0, 0.1 * strength["nonlinear"], size=n_features) * affected
        X = X + intensity * weight * np.square(deviation) / feature_scale
    return X


def _dropout_mask(
    dropout: _Dropout, site: int, n_rows: int, affected: npt.NDArray, random_state: np.random.RandomState
) -> npt.NDArray:
    """Values of one site to replace: whole features (``per="feature"``) or single values (``per="value"``)."""
    n_features = len(affected)
    shape = (1, n_features) if dropout.per == "feature" else (n_rows, n_features)
    mask = (random_state.uniform(size=shape) < dropout.prob[site]) & affected
    return np.broadcast_to(mask, (n_rows, n_features))


##########################################################################################################
################################################# OUTPUTS ################################################
##########################################################################################################
def _ground_truth(
    *,
    site_free: npt.NDArray,
    effects: _SiteEffectResult,
    covariate_values: dict[str, npt.NDArray],
    covariate_effect: npt.NDArray,
    covariate_coefficients: dict[str, npt.NDArray],
    noise: npt.NDArray,
    params: _SiteEffectParams,
    feature_scale: npt.NDArray,
    **extra: Any,
) -> dict[str, Any]:
    """Assemble the ground truth shared by every simulator; ``extra`` holds simulator-specific keys first."""
    truth: dict[str, Any] = {
        **extra,
        "covariates": covariate_values,
        "covariate_effect": covariate_effect,
        "covariate_coefficients": covariate_coefficients,
        "noise": noise,
        "site_free": site_free,
        "site_effect": effects.X - site_free,
        "site_location": effects.site_location,
        "site_scale": effects.site_scale,
    }
    if effects.site_class_location is not None:
        truth["site_class_location"] = effects.site_class_location
    truth.update(
        {
            "intensity": effects.intensity,
            "dropout_mask": effects.dropout_mask,
            "site_effect_features": params.affected_features,
            "feature_scale": feature_scale,
        }
    )
    return truth


def _format_outputs(
    X: npt.NDArray,
    labels: dict[str, npt.NDArray],
    site_names: npt.NDArray,
    as_frame: bool,
    truth: dict[str, Any] | None,
) -> tuple[Any, ...]:
    """Return ``X``, the label arrays and the ground truth, or DataFrames and Series when ``as_frame``.

    ``labels`` maps the name of each Series (e.g. ``"target"``, ``"site"``) to
    its values, in the order of the outputs.
    """
    outputs: list[Any] = [X, *labels.values()]
    if as_frame:
        features = [f"feature_{j}" for j in range(X.shape[1])]
        index = pd.Index(site_names, name="site")
        outputs = [pd.DataFrame(X, columns=features), *(pd.Series(v, name=name) for name, v in labels.items())]
        if truth is not None:
            truth["covariates"] = pd.DataFrame(truth["covariates"])
            truth["site_location"] = pd.DataFrame(truth["site_location"], index=index, columns=features)
            truth["site_scale"] = pd.DataFrame(truth["site_scale"], index=index, columns=features)
    return tuple(outputs) if truth is None else (*outputs, truth)
