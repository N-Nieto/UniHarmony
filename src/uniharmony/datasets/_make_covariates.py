"""Covariates of the multi-site simulators: distributions per site, presets and effects on the features."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import get_args

import numpy as np
import numpy.typing as npt

from uniharmony._verbose import get_logger

from ._simulation_constants import EFFECT_SHAPES, PRESET_COVARIATES


logger = get_logger(src="make_covariates")


__all__ = [
    "Covariate",
    "CovariateSiteDistribution",
    "_covariate_effects",
    "_make_covariate",
    "_resolve_covariates",
    "_resolve_feature_subset",
]


@dataclass
class CovariateSiteDistribution:
    """Distribution parameters for one covariate at one site.

    Pass ``loc`` / ``scale`` / ``clip`` for a **continuous** covariate, or
    ``probs`` for a **categorical** one.  The ``kind`` of the parent
    :class:`Covariate` is inferred automatically from which fields are set,
    so you never need to declare it explicitly.

    Parameters
    ----------
    loc : float | None, default None
        Mean of the Gaussian draw. Setting this marks the distribution as
        *continuous*.  Mutually exclusive with ``probs``.

    scale : float, default 1.0
        Standard deviation of the Gaussian draw. Only used when ``loc`` is
        set (continuous covariate).

    clip : tuple[float, float] | None, default None
        Optional ``(min, max)`` clipping applied after drawing continuous
        samples (e.g. ``(0.0, 100.0)`` for age). Only used for continuous
        covariates.

    probs : list[float] | None, default None
        Probability vector over the parent :attr:`Covariate.categories`.
        Setting this marks the distribution as *categorical*.  Values are
        normalised to sum to 1 automatically.  Mutually exclusive with
        ``loc``.

    Raises
    ------
    ValueError
        If both ``loc`` and ``probs`` are supplied, or if neither is
        supplied.

    Examples
    --------
    Continuous — older participants, mean 60, sd 8, clipped to [18, 90]:

    >>> older = CovariateSiteDistribution(loc=60.0, scale=8.0, clip=(18.0, 90.0))

    Categorical — 70 % female site:

    >>> female = CovariateSiteDistribution(probs=[0.3, 0.7])

    """

    loc: float | None = None
    scale: float = 1.0
    clip: tuple[float, float] | None = None
    probs: list[float] | None = None

    def __post_init__(self) -> None:
        has_loc = self.loc is not None
        has_probs = self.probs is not None
        if has_loc and has_probs:
            raise ValueError("CovariateSiteDistribution: supply either 'loc' (continuous) or 'probs' (categorical), not both.")
        if not has_loc and not has_probs:
            raise ValueError(
                "CovariateSiteDistribution: supply 'loc' for a continuous covariate or 'probs' for a categorical one."
            )

    @property
    def kind(self) -> str:
        """Inferred kind: ``'continuous'`` or ``'categorical'``."""
        return "continuous" if self.loc is not None else "categorical"


@dataclass
class Covariate:
    """Full specification for a single synthetic covariate across all sites.

    Each site gets its own :class:`CovariateSiteDistribution`, which allows
    arbitrarily different demographic profiles per site (e.g. site A young and
    male-heavy, site B older and female-heavy).

    The ``kind`` (``"continuous"`` or ``"categorical"``) is inferred
    automatically from the first entry in ``site_distributions`` — you never
    need to set it explicitly.

    If only **one** :class:`CovariateSiteDistribution` is supplied, it is
    broadcast to every site automatically.

    Parameters
    ----------
    name : str
        Column name used as the key in the returned covariates dictionary.

    site_distributions : list[CovariateSiteDistribution]
        One distribution per site (in site order), **or** a single entry that
        is broadcast to all sites. The ``kind`` of each entry must be
        consistent (all continuous or all categorical).

    categories : list, default [0, 1]
        Unique category labels for *categorical* covariates. Must match the
        length of every ``CovariateSiteDistribution.probs`` vector. Ignored
        for continuous covariates.

    effect : float or array-like, default 0.0
        Effect of the covariate on the features, the same in every site. It
        is added to the signal before the site effect, so a covariate whose
        distribution differs between sites makes the sites differ through it
        (confounding), as in real multi-site data. Unit: the reference SD of
        each feature (its pooled within-class SD in the signal, the unit of
        all strengths of ``make_multisite_classification``).

        - Continuous covariate: ``effect`` is the standard deviation, over
          all samples, of the covariate's contribution to an affected
          feature. With ``effect_shape="linear"`` it is the change of the
          feature per pooled SD of the covariate (a standardized slope).
        - Categorical covariate: ``effect`` is the mean difference between
          each category and the first category of ``categories``.

        A scalar is multiplied, for each affected feature, by a weight drawn
        from U(0.5, 1.5) (for a categorical covariate, independently for
        each non-reference category), so the features are affected by
        different amounts with mean ``effect``. An array of shape
        ``(n_features,)`` gives the effect on every feature exactly (for a
        categorical covariate, the same for every non-reference category; or
        an array of shape ``(n_categories - 1, n_features)``). Negative values
        decrease the features. The effects used are returned by
        ``make_multisite_classification(..., return_ground_truth=True)``.

    effect_shape : {"linear", "quadratic", "sigmoid"} or callable, default "linear"
        Shape of the effect of a continuous covariate, applied to the
        covariate ``z`` standardized over all samples: ``z``, ``z**2`` or
        ``tanh(2 z)``. A callable receives the raw covariate values (an array
        of shape ``(n_samples,)``) and returns an array of the same shape.
        The result is standardized to mean 0 and SD 1 over all samples, so
        ``effect`` keeps its meaning for every shape. Only ``"linear"`` is
        allowed for categorical covariates.

    effect_features : int, float, sequence of int or None, default None
        Features affected by the covariate: ``None`` for all features, an
        int ``k`` for the first ``k`` features, a float in (0, 1] for that
        fraction of the first features, or a sequence of feature indices.
        Ignored when ``effect`` is an array.

    depends_on : dict or None, default None
        Makes a continuous covariate depend on a categorical one, e.g. total
        intracranial volume on sex: ``{"sex": [0.0, -160.0]}`` adds, to the
        mean of each sample, the shift of its category of ``sex`` (one value
        per category, in the units of this covariate). The categorical
        covariate must be in the same ``covariates`` list.

    Raises
    ------
    ValueError
        If ``site_distributions`` is empty, entries mix continuous and
        categorical distributions, or an effect parameter is invalid.

    Examples
    --------
    Same distribution at every site (broadcast from one entry):

    >>> cov = Covariate(
    ...     name="age",
    ...     site_distributions=[
    ...         CovariateSiteDistribution(loc=45.0, scale=10.0, clip=(18.0, 90.0)),
    ...     ],
    ...     effect=0.5,
    ... )

    Different distribution per site (two sites, explicit):

    >>> cov = Covariate(
    ...     name="age",
    ...     site_distributions=[
    ...         CovariateSiteDistribution(loc=30.0, scale=5.0, clip=(18.0, 90.0)),
    ...         CovariateSiteDistribution(loc=60.0, scale=8.0, clip=(18.0, 90.0)),
    ...     ],
    ...     effect=-0.5,
    ...     effect_shape="quadratic",
    ... )

    Categorical covariate — kind inferred from ``probs``; category 1 is 0.3
    reference SD above category 0 on the first 5 features:

    >>> cov = Covariate(
    ...     name="sex",
    ...     site_distributions=[
    ...         CovariateSiteDistribution(probs=[0.4, 0.6]),
    ...     ],
    ...     effect=0.3,
    ...     effect_features=5,
    ... )

    """

    name: str
    site_distributions: list[CovariateSiteDistribution] = field(default_factory=list)
    categories: list = field(default_factory=lambda: [0, 1])
    effect: float | npt.ArrayLike = 0.0
    effect_shape: EFFECT_SHAPES | Callable[[np.ndarray], np.ndarray] = "linear"
    effect_features: int | float | Sequence[int] | None = None
    depends_on: dict[str, Sequence[float]] | None = None

    def __post_init__(self) -> None:
        if not self.site_distributions:
            raise ValueError(f"Covariate '{self.name}': 'site_distributions' must not be empty.")
        kinds = {sd.kind for sd in self.site_distributions}
        if len(kinds) > 1:
            raise ValueError(
                f"Covariate '{self.name}': all CovariateSiteDistribution entries "
                "must have the same kind (all continuous or all categorical), "
                f"got {kinds}."
            )
        effect = np.asarray(self.effect, dtype=float)
        if not np.isfinite(effect).all():
            raise ValueError(f"Covariate '{self.name}': effect must be finite, got {self.effect}.")
        if effect.ndim > 2 or (effect.ndim == 2 and self.kind == "continuous"):
            raise ValueError(f"Covariate '{self.name}': effect must be a float or an array of shape (n_features,).")
        if not callable(self.effect_shape) and self.effect_shape not in get_args(EFFECT_SHAPES):
            raise ValueError(
                f"Covariate '{self.name}': effect_shape must be one of {get_args(EFFECT_SHAPES)} or a callable, "
                f"got {self.effect_shape!r}."
            )
        if self.kind == "categorical" and self.effect_shape != "linear":
            raise ValueError(f"Covariate '{self.name}': effect_shape only applies to continuous covariates.")
        if self.depends_on is not None:
            if self.kind != "continuous" or not isinstance(self.depends_on, dict) or len(self.depends_on) != 1:
                raise ValueError(
                    f"Covariate '{self.name}': depends_on must be a dict with one categorical covariate and its "
                    "shifts, for a continuous covariate."
                )
            shifts = np.asarray(next(iter(self.depends_on.values())), dtype=float)
            if shifts.ndim != 1 or not np.isfinite(shifts).all():
                raise ValueError(f"Covariate '{self.name}': depends_on needs one finite shift per category.")

    @property
    def kind(self) -> str:
        """Inferred kind: ``'continuous'`` or ``'categorical'``."""
        return self.site_distributions[0].kind


def make_covariate_site_distributions(
    locs: Sequence[float] | None = None,
    scales: Sequence[float] | float = 1.0,
    clips: Sequence[tuple[float, float] | None] | tuple[float, float] | None = None,
    probs: Sequence[Sequence[float]] | Sequence[float] | None = None,
) -> list[CovariateSiteDistribution]:
    """Build a list of :class:`CovariateSiteDistribution` objects from summary statistics.

    A convenience factory that broadcasts scalar ``scales``/``clips`` across
    all sites and zips everything into a list of :class:`CovariateSiteDistribution`
    instances.

    Parameters
    ----------
    locs : list[float]
        Per-site location parameters (mean for continuous, ignored for
        categorical). Determines ``len(site_distributions)``.

    scales : list[float] | float, default 1.0
        Per-site scale (std). Broadcast to all sites if scalar.

    clips : list[tuple | None] | tuple | None, default None
        Per-site ``(min, max)`` clipping bounds. Broadcast to all sites if a
        single tuple or ``None``.

    probs : list[list[float]] | list[float] | None, default None
        Per-site probability vectors for categorical covariates. A flat list
        is applied to every site.

    Returns
    -------
    list[CovariateSiteDistribution]

    Examples
    --------
    Three sites with different means, same sd, same clip:

    >>> dists = make_covariate_site_distributions(
    ...     locs=[30.0, 45.0, 60.0],
    ...     scales=10.0,
    ...     clips=(18.0, 90.0),
    ... )
    >>> len(dists)
    3
    >>> dists[1].loc
    45.0

    """
    logger.debug(f"locs: {locs} and probs: {probs}")

    # Determine number of sites
    if locs is not None:
        n = len(locs)
    elif probs is not None and probs:
        n = len(probs)
    else:
        raise ValueError(
            "When using a single probability vector, you must provide 'locs' (with None values) to specify the number of sites."
        )

    # Broadcast scales
    scales_list: list[float] = [float(scales)] * n if isinstance(scales, (int, float)) else list(scales)
    if len(scales_list) != n:
        raise ValueError(f"'scales' has {len(scales_list)} entries but 'locs' has {n}.")

    # Broadcast clips
    if clips is None or (isinstance(clips, tuple) and isinstance(clips[0], (int, float))):
        clips_list: list = [clips] * n
    else:
        clips_list = list(clips)
    if len(clips_list) != n:
        raise ValueError(f"'clips' has {len(clips_list)} entries but 'locs' has {n}.")

    # Broadcast probs
    if probs is None:
        probs_list: list = [None] * n
    elif isinstance(probs[0], (int, float)):
        probs_list = [list(probs)] * n
    else:
        probs_list = [list(p) for p in probs]  # type: ignore[arg-type]
    if len(probs_list) != n:
        raise ValueError(f"'probs' has {len(probs_list)} entries but 'locs' has {n}.")
    # Broadcast None
    locs_list: list[float | None] = [None] * n if locs is None else list(locs)
    return [
        CovariateSiteDistribution(loc=locs_list[i], scale=scales_list[i], probs=probs_list[i], clip=clips_list[i])
        for i in range(n)
    ]


def _make_preset_covariate(
    name: str | PRESET_COVARIATES,
    n_sites: int,
) -> Covariate:
    """Build a :class:`Covariate` from a named neuroimaging preset.

    Provides defaults for common covariates of multi-site neuroimaging
    studies. Except for ``"quality"``, the distributions are the same in
    every site; build a :class:`Covariate` for sites that recruit different
    populations.

    Parameters
    ----------
    name : {"age", "sex", "tiv", "quality"}
        Preset identifier:

        - ``"age"``: continuous Gaussian with mean 50 and std 20 at every
          site, clipped to [0, 110], with a linear effect of 0.5 (reference
          SD per SD of age) on every feature.
        - ``"sex"``: binary categorical (0 = male, 1 = female), 50 % at
          every site; category 1 is 0.3 reference SD higher on every
          feature.
        - ``"tiv"``: total intracranial volume (cm³), Gaussian with mean
          1520 for sex 0 and 1360 for sex 1 and SD 120 within each sex
          (approximate adult values), clipped to [900, 2200], with a linear
          effect of 0.5 on every feature. It depends on sex (correlation
          about 0.55 with 50 % of each sex), so ``"sex"`` must also be in the
          ``covariates``.
        - ``"quality"``: continuous IQM-like score with std 0.08 and site
          means spread linearly from 0.1 to 5, clipped to [0, 5]. It is not
          linked to ``noise_strength`` and has no effect on the features.

    n_sites : int
        Number of sites.

    Returns
    -------
    Covariate

    Examples
    --------
    >>> spec = _make_preset_covariate("age", n_sites=3)
    >>> len(spec.site_distributions)
    3
    >>> spec.site_distributions[0].loc  # All sites with the same age range
    50.0

    """
    if name.lower() == "age":
        dists = make_covariate_site_distributions(
            locs=[50.0] * n_sites,
            scales=20.0,
            clips=(0, 110),
        )
        return Covariate(
            name="age",
            site_distributions=dists,
            effect=0.5,
        )

    if name.lower() == "sex":
        # Equally distributed sex
        dists = make_covariate_site_distributions(
            probs=[[0.5, 0.5]] * n_sites,
        )
        return Covariate(
            name="sex",
            site_distributions=dists,
            categories=[0, 1],
            effect=0.3,
        )

    if name.lower() == "tiv":
        dists = make_covariate_site_distributions(locs=[1520.0] * n_sites, scales=120.0, clips=(900.0, 2200.0))
        return Covariate(name="tiv", site_distributions=dists, effect=0.5, depends_on={"sex": [0.0, -160.0]})

    if name.lower() == "quality":
        # Site means spread linearly from 0.1 to 5
        locs = np.linspace(0.1, 5, n_sites).tolist()
        dists = make_covariate_site_distributions(
            locs=locs,
            scales=0.08,
            clips=(0.0, 5.0),
        )
        return Covariate(
            name="quality",
            site_distributions=dists,
        )

    raise ValueError(f"Unknown preset '{name}'. Choose from: {get_args(PRESET_COVARIATES)}.")


def _resolve_covariates(covariates: Sequence[Covariate | str] | None, n_sites: int) -> list[Covariate] | None:
    if covariates is not None:
        resolved_specs = []
        for entry in covariates:
            # Make the covariate given a preset name.
            if isinstance(entry, str):
                resolved_specs.append(
                    _make_preset_covariate(
                        entry,
                        n_sites=n_sites,
                    )
                )
            elif isinstance(entry, Covariate):
                # Copy, so that broadcasting to n_sites does not modify the user's object
                resolved_specs.append(replace(entry, site_distributions=list(entry.site_distributions)))
            else:
                raise TypeError(
                    f"Each entry in 'covariates' must be a CovariateSpec or a str preset name, got {type(entry).__name__}."
                )
        _validate_covariates(resolved_specs, n_sites)
        logger.debug("Covariates requested: %s", [s.name for s in resolved_specs])
    else:
        resolved_specs = None
    return resolved_specs


def _make_covariate(
    covars: list[Covariate],
    site_idx: int,
    n_samples_site: int,
    random_state: np.random.RandomState,
) -> dict[str, np.ndarray]:
    """Draw the covariates of one site from their site distributions.

    Parameters
    ----------
    covars : list[Covariate]
        Covariate specifications.
    site_idx : int
        Index of the current site.
    n_samples_site : int
        Number of samples at this site.
    random_state : np.random.RandomState
        sklearn-compatible random state.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from covariate name to array of shape (n_samples_site,).

    """
    result: dict[str, np.ndarray] = {}

    # Covariates that depend on another one are drawn after it
    for covar in sorted(covars, key=lambda c: c.depends_on is not None):
        sd = covar.site_distributions[site_idx]

        if sd.loc is not None:  # continuous
            values = sd.loc + sd.scale * random_state.normal(size=n_samples_site)
            if covar.depends_on is not None:
                (parent_name, shifts), parent = next(iter(covar.depends_on.items())), _by_name(covars, covar)
                codes = np.array([list(parent.categories).index(v) for v in result[parent_name]], dtype=int)
                values = values + np.asarray(shifts, dtype=float)[codes]

            if sd.clip is not None:
                values = np.clip(values, sd.clip[0], sd.clip[1])

        else:  # categorical
            n_cat = len(covar.categories)
            if sd.probs is None:
                p = np.ones(n_cat) / n_cat
            else:
                p = np.array(sd.probs, dtype=float)
                p = p / p.sum()
            indices = random_state.choice(len(covar.categories), size=n_samples_site, p=p)
            values = np.array(covar.categories)[indices]

        result[covar.name] = values

    return {covar.name: result[covar.name] for covar in covars}


def _validate_dependence(covar: Covariate, covars: list[Covariate]) -> None:
    """Check ``depends_on``: an existing categorical covariate with one shift per category."""
    if covar.depends_on is None:
        return
    parent = _by_name(covars, covar)
    shifts = next(iter(covar.depends_on.values()))
    if parent.kind != "categorical" or parent.depends_on is not None:
        raise ValueError(f"Covariate '{covar.name}' can only depend on a categorical covariate.")
    if len(shifts) != len(parent.categories):
        raise ValueError(
            f"Covariate '{covar.name}': depends_on gives {len(shifts)} shifts but '{parent.name}' has "
            f"{len(parent.categories)} categories."
        )


def _by_name(covars: list[Covariate], dependent: Covariate) -> Covariate:
    """Return the categorical covariate that ``dependent`` depends on."""
    name = next(iter(dependent.depends_on or {}))
    for covar in covars:
        if covar.name == name:
            return covar
    raise ValueError(
        f"Covariate '{dependent.name}' depends on '{name}', which is not in the covariates: add it "
        f"(e.g. covariates=['{name}', '{dependent.name}'])."
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _validate_covariates(
    covars: list[Covariate],
    n_sites: int,
) -> None:
    """Check that every Covariate is consistent with n_sites."""
    duplicated = {c.name for c in covars if sum(other.name == c.name for other in covars) > 1}
    if duplicated:
        raise ValueError(f"Covariate names must be unique, got {sorted(duplicated)} more than once.")
    for covar in covars:
        n_dists = len(covar.site_distributions)

        # Broadcast a single entry to all sites
        if n_dists == 1:
            covar.site_distributions = covar.site_distributions * n_sites
            logger.debug(
                "Covariate '%s': single site_distribution broadcast to %d sites.",
                covar.name,
                n_sites,
            )
        elif n_dists != n_sites:
            raise ValueError(
                f"Covariate '{covar.name}' has {n_dists} site_distributions "
                f"but n_sites={n_sites}. Provide exactly 1 (broadcast to all "
                "sites) or one entry per site."
            )

        _validate_dependence(covar, covars)

        # For categorical covariates, verify probs length matches categories
        if covar.kind == "categorical":
            n_cat = len(covar.categories)
            for i, sd in enumerate(covar.site_distributions):
                if sd.probs is None:
                    continue
                if len(sd.probs) != n_cat:
                    raise ValueError(
                        f"Covariate '{covar.name}', site {i}: probs has {len(sd.probs)} entries but categories has {n_cat}."
                    )


# ---------------------------------------------------------------------------
# Effects of the covariates on the features
# ---------------------------------------------------------------------------
def _covariate_effects(
    covars: list[Covariate],
    values: dict[str, np.ndarray],
    n_features: int,
    feature_scale: np.ndarray,
    random_state: np.random.RandomState,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Additive effect of all covariates on the features, the same in every site.

    Parameters
    ----------
    covars : list[Covariate]
        Covariate specifications.
    values : dict[str, np.ndarray]
        Covariate values of all samples, shape (n_samples,) each.
    n_features : int
        Number of features.
    feature_scale : np.ndarray of shape (n_features,)
        Reference SD of each feature (the unit of ``effect``).
    random_state : np.random.RandomState
        sklearn-compatible random state (draws the per-feature weights).

    Returns
    -------
    effect : np.ndarray of shape (n_samples, n_features)
        Sum of the contributions of all covariates, in data units.
    coefficients : dict[str, np.ndarray]
        Per covariate, its effect on each feature in reference-SD units:
        shape (n_features,) for a continuous covariate (per SD of its
        standardized shape), shape (n_categories, n_features) for a
        categorical one (row 0, the reference category, is 0).

    """
    n_samples = len(next(iter(values.values())))
    total = np.zeros((n_samples, n_features))
    coefficients: dict[str, np.ndarray] = {}
    for covar in covars:
        if covar.kind == "continuous":
            weights = _effect_weights(covar, n_features, n_rows=1, random_state=random_state)[0]
            total += _effect_shape_values(covar, values[covar.name])[:, None] * weights * feature_scale
        else:
            categories = list(covar.categories)
            weights = np.vstack(
                [
                    np.zeros(n_features),
                    _effect_weights(covar, n_features, n_rows=len(categories) - 1, random_state=random_state),
                ]
            )
            codes = np.array([categories.index(v) for v in values[covar.name]], dtype=int)
            total += weights[codes] * feature_scale
        coefficients[covar.name] = weights
    return total, coefficients


def _effect_weights(covar: Covariate, n_features: int, n_rows: int, random_state: np.random.RandomState) -> np.ndarray:
    """Effect of the covariate on each feature, shape (n_rows, n_features), in reference-SD units."""
    effect = np.asarray(covar.effect, dtype=float)
    if effect.ndim == 0:
        weights = np.zeros((n_rows, n_features))
        if effect != 0:
            features = _resolve_effect_features(covar, n_features)
            weights[:, features] = effect * random_state.uniform(0.5, 1.5, size=(n_rows, len(features)))
        return weights
    if effect.shape[-1] != n_features:
        raise ValueError(f"Covariate '{covar.name}': effect has {effect.shape[-1]} values but there are {n_features} features.")
    if effect.ndim == 2 and effect.shape[0] != n_rows:
        raise ValueError(
            f"Covariate '{covar.name}': effect has {effect.shape[0]} rows; one per non-reference category ({n_rows}) is needed."
        )
    return np.broadcast_to(effect, (n_rows, n_features)).copy()


def _resolve_effect_features(covar: Covariate, n_features: int) -> np.ndarray:
    """Return the indices of the features affected by ``covar`` (``effect_features``)."""
    return _resolve_feature_subset(covar.effect_features, n_features, f"Covariate '{covar.name}': effect_features")


def _resolve_feature_subset(spec: int | float | Sequence[int] | None, n_features: int, name: str) -> np.ndarray:
    """Return feature indices from ``None`` (all), an int ``k`` (first k), a fraction, or a sequence of indices."""
    if spec is None:
        return np.arange(n_features)
    if isinstance(spec, bool):
        raise TypeError(f"{name} must be an int, a float, a sequence of int or None.")
    if isinstance(spec, (int, np.integer)):
        if not 1 <= int(spec) <= n_features:
            raise ValueError(f"{name}={spec} must be between 1 and n_features={n_features}.")
        return np.arange(int(spec))
    if isinstance(spec, (float, np.floating)):
        fraction = float(spec)
        if not 0 < fraction <= 1:
            raise ValueError(f"A float {name} must be in (0, 1], got {spec}.")
        return np.arange(max(1, round(fraction * n_features)))
    indices = np.asarray(spec)
    if indices.ndim != 1 or len(indices) == 0 or not np.issubdtype(indices.dtype, np.integer):
        raise TypeError(f"{name} must be an int, a float, a sequence of int or None.")
    if indices.min() < 0 or indices.max() >= n_features or len(np.unique(indices)) != len(indices):
        raise ValueError(f"{name} must be distinct indices in [0, {n_features}), got {indices.tolist()}.")
    return indices


def _effect_shape_values(covar: Covariate, values: np.ndarray) -> np.ndarray:
    """Shape of the effect of a continuous covariate, standardized to mean 0 and SD 1 over all samples."""
    values = np.asarray(values, dtype=float)
    if callable(covar.effect_shape):
        shaped = np.asarray(covar.effect_shape(values), dtype=float)
        if shaped.shape != values.shape or not np.isfinite(shaped).all():
            raise ValueError(
                f"Covariate '{covar.name}': effect_shape must return finite values of shape {values.shape}, got {shaped.shape}."
            )
    else:
        z = _standardize(values)
        shaped = {"linear": z, "quadratic": z**2, "sigmoid": np.tanh(2 * z)}[covar.effect_shape]
    return _standardize(shaped)


def _standardize(values: np.ndarray) -> np.ndarray:
    """Center and scale to SD 1 (a constant gives zeros)."""
    sd = values.std()
    return (values - values.mean()) / sd if sd > 0 else np.zeros_like(values)
