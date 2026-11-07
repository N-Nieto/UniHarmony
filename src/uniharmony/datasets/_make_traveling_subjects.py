"""Simulation of a traveling-subjects study: the same subjects measured in every site."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, overload

import numpy as np

from uniharmony._verbose import get_logger

from ._make_covariates import Covariate
from ._simulation_constants import PRESET_COVARIATES, SiteEffect
from ._simulation_core import (
    RandomStateLike,
    _apply_site_effects,
    _check_integers,
    _draw_noise,
    _format_outputs,
    _ground_truth,
    _SharedParams,
    _simulate_covariates,
    _validate_shared,
)


__all__ = [
    "make_traveling_subjects",
]

logger = get_logger(src="make_traveling_subjects")


@overload
def make_traveling_subjects(
    n_subjects: int = 20,
    n_sites: int = 3,
    n_features: int = 10,
    n_repeats: int = 1,
    noise_strength: float | Sequence[float] = 0.3,
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
def make_traveling_subjects(
    n_subjects: int = 20,
    n_sites: int = 3,
    n_features: int = 10,
    n_repeats: int = 1,
    noise_strength: float | Sequence[float] = 0.3,
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


def make_traveling_subjects(
    n_subjects: int = 20,
    n_sites: int = 3,
    n_features: int = 10,
    n_repeats: int = 1,
    noise_strength: float | Sequence[float] = 0.3,
    site_effect: SiteEffect | None = None,
    site_effect_features: int | float | Sequence[int] | None = None,
    site_effect_heterogeneity: float = 0.0,
    covariates: Sequence[Covariate | str | PRESET_COVARIATES] | None = None,
    site_names: Sequence | None = None,
    as_frame: bool = False,
    return_ground_truth: bool = False,
    random_state: RandomStateLike = 42,
) -> tuple[Any, ...]:
    """Simulate a traveling-subjects study: the same subjects measured in every site.

    Every subject is measured ``n_repeats`` times in every site. Their true
    values are the same everywhere, so the differences between the
    measurements of a subject are measurement noise and site effects only.
    This design separates the site effect from the differences between
    populations, and gives a direct check of a harmonization: after it, the
    measurements of a subject should agree across sites as well as within a
    site.

    The data follow the model::

        X[subject, site, repeat] = SiteEffect(Trait[subject] + CovariateEffects[subject] + Noise; site)

    1. Each subject gets a trait, ``~ N(0, 1)`` per feature (independent
       features), and covariates drawn from one population distribution;
       their effects, the same in every site, are added to the trait.
    2. Every measurement adds its own Gaussian noise (scan-rescan noise).
    3. The Effect of Site (EoS) of ``site_effect`` is applied, as in
       :func:`make_multisite_classification`.

    **Units.** All strengths are in units of the SD of the trait between
    subjects (1 for every feature). With ``noise_strength = r``, the
    reliability of a measurement within a site (intraclass correlation,
    without covariates) is ``1 / (1 + r**2)``: 0.92 for the default 0.3.

    Parameters
    ----------
    n_subjects : int, optional (default 20)
        Number of subjects, each measured in every site.

    n_sites : int, optional (default 3)
        Number of sites.

    n_features : int, optional (default 10)
        Number of features per measurement.

    n_repeats : int, optional (default 1)
        Number of measurements of each subject in each site (more than 1
        gives the within-site reliability).

    noise_strength : float or list of float, optional (default 0.3)
        Standard deviation of the noise of a measurement, in units of the
        between-subject SD, for all sites or per site (e.g. a noisier
        scanner).

    site_effect : dict or None, optional (default None, i.e. ``{"location": 3.0}``)
        Types of EoS and their strength, as in
        :func:`make_multisite_classification` (``"location"``, ``"scale"``,
        ``"variance"``, ``"nonlinear"``, ``"dropout"``, ``"combat"``).

    site_effect_features : int, float, sequence of int or None, optional (default None)
        Features affected by the EoS: ``None`` for all, an int ``k`` for the
        first ``k``, a float in (0, 1] for that fraction of the first
        features, or a sequence of indices.

    site_effect_heterogeneity : float, optional (default 0.0)
        Variation of the EoS between subjects: the site effect of a subject
        is multiplied by ``exp(h * z - h**2 / 2)``, with ``z ~ N(0, 1)``
        drawn per subject and the same in every site (e.g. a scanner effect
        that depends on head size).

    covariates : list of Covariate or str, or None, optional (default None)
        Covariates of the subjects (a :class:`Covariate` with a single
        distribution, the population, or a preset name), constant across
        sites and repeats. A categorical covariate with an ``effect`` can
        play the role of a diagnosis.

    site_names : sequence or None, optional (default None)
        Labels of the sites; ``None`` gives ``0`` to ``n_sites - 1``.

    as_frame : bool, optional (default False)
        If True, ``X`` is a DataFrame and ``sites`` and ``subjects`` are
        Series named ``"site"`` and ``"subject"``.

    return_ground_truth : bool, optional (default False)
        If True, also return a dict with the components of the data, all
        row-aligned with ``X``: the keys of
        :func:`make_multisite_classification` (but
        ``"site_class_location"``), where ``"signal"`` is the trait of the
        subject of each row, and ``"repeat"``, the index of the repeat.

    random_state : int, RandomState, Generator or None, optional (default 42)
        The seed of the pseudo random number generator, or a RandomState or
        Generator, for reproducibility.

    Returns
    -------
    X : np.ndarray or DataFrame of shape (n_sites * n_repeats * n_subjects, n_features)
        Simulated measurements, ordered by site, then repeat, then subject.
    sites : np.ndarray or Series of shape (n_measurements,)
        Site of each measurement (``site_names``, or 0 to ``n_sites`` - 1).
    subjects : np.ndarray or Series of shape (n_measurements,)
        Subject of each measurement (0 to ``n_subjects`` - 1).
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
    make_multisite_classification : Different subjects in every site, with class labels.
    make_multisite_regression : Different subjects in every site, with a continuous target.

    Examples
    --------
    >>> X, sites, subjects = make_traveling_subjects(n_subjects=10, n_sites=3)
    >>> X.shape
    (30, 10)
    >>> int((subjects == 0).sum())  # subject 0 is measured once in each site
    3

    """
    s_params = _validate_inputs(
        n_subjects=n_subjects,
        n_sites=n_sites,
        n_features=n_features,
        n_repeats=n_repeats,
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
    n_subjects, n_features = s_params.n_subjects, s_params.n_features
    feature_scale = np.ones(n_features)  # the between-subject SD of the trait

    # Rows: site, then repeat, then subject
    per_site = s_params.n_repeats * n_subjects
    site_codes = np.repeat(np.arange(s_params.n_sites), per_site)
    subjects = np.tile(np.arange(n_subjects), s_params.n_sites * s_params.n_repeats)
    repeats = np.tile(np.repeat(np.arange(s_params.n_repeats), n_subjects), s_params.n_sites)
    site_rows = [np.flatnonzero(site_codes == site) for site in range(s_params.n_sites)]

    # Step 1: subjects, their covariates (one population) and the effects of the covariates
    trait = rng.normal(size=(n_subjects, n_features))
    subject_covariates, subject_covariate_effect, covariate_coefficients = _simulate_covariates(
        shared.covariates, [np.arange(n_subjects)], n_features, feature_scale, rng
    )
    z = rng.normal(size=n_subjects)  # intensity of the site effect of each subject (heterogeneity)

    # Step 2: one noise per measurement
    signal = trait[subjects]
    covariate_effect = subject_covariate_effect[subjects]
    noise = _draw_noise(site_rows, n_features, shared.noise_strength, feature_scale, rng)
    site_free = signal + covariate_effect + noise
    center = (signal + noise).mean(axis=0) + covariate_effect

    # Step 3: site effects
    effects = _apply_site_effects(site_free, center, site_rows, feature_scale, shared.site_effect, rng, z=z[subjects])
    logger.info(f"Generated {len(site_free)} measurements of {n_subjects} subjects in {s_params.n_sites} sites")

    truth = None
    if return_ground_truth:
        truth = _ground_truth(
            signal=signal,
            site_free=site_free,
            effects=effects,
            covariate_values={name: values[subjects] for name, values in subject_covariates.items()},
            covariate_effect=covariate_effect,
            covariate_coefficients=covariate_coefficients,
            noise=noise,
            params=shared.site_effect,
            feature_scale=feature_scale,
        )
        truth["repeat"] = repeats
    labels = {"site": shared.site_names[site_codes], "subject": subjects}
    return _format_outputs(effects.X, labels, shared.site_names, as_frame, truth)


@dataclass
class _SimParams:
    """Validated and resolved parameters of ``make_traveling_subjects``."""

    n_subjects: int
    n_sites: int
    n_features: int
    n_repeats: int
    shared: _SharedParams


def _validate_inputs(**params: Any) -> _SimParams:
    """Validate every parameter of ``make_traveling_subjects`` and resolve defaults and broadcasts."""
    n_subjects, n_sites, n_features, n_repeats = _check_integers(
        n_subjects=params["n_subjects"],
        n_sites=params["n_sites"],
        n_features=params["n_features"],
        n_repeats=params["n_repeats"],
    )
    for name, value, minimum in [("n_subjects", n_subjects, 2), ("n_sites", n_sites, 1), ("n_features", n_features, 1)]:
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}, got {value}")
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be at least 1, got {n_repeats}")
    if n_sites == 1:
        logger.warning("n_sites is 1: there is no site effect to measure between sites.")
    for covariate in params["covariates"] or []:
        if isinstance(covariate, Covariate) and len(covariate.site_distributions) != 1:
            raise ValueError(
                f"Covariate '{covariate.name}' has {len(covariate.site_distributions)} site_distributions: the same "
                "subjects visit every site, so a covariate needs a single distribution (the population)."
            )
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
        covariate_sites=1,
    )
    return _SimParams(n_subjects=n_subjects, n_sites=n_sites, n_features=n_features, n_repeats=n_repeats, shared=shared)
