"""Multisite classification dataset generator with covariate support."""

from dataclasses import dataclass, field
from typing import Literal, cast, get_args

import numpy as np
import structlog


logger = structlog.get_logger()


__all__ = ["Covariate", "CovariateSiteDistribution", "_make_covariate", "_resolve_covariates"]

PRESET_COVARIATES = Literal["age", "sex", "quality"]


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

    >>> CovariateSiteDistribution(loc=60.0, scale=8.0, clip=(18.0, 90.0))

    Categorical — 70 % female site:

    >>> CovariateSiteDistribution(probs=[0.3, 0.7])

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

    x_correlation : float, default 0.0
        Strength of the linear correlation between this covariate and the
        *pre-site-effect* feature matrix. ``0.0`` = independent,
        ``1.0`` = perfectly correlated with the first feature of X.

    Raises
    ------
    ValueError
        If ``site_distributions`` is empty, or if entries mix continuous and
        categorical distributions.

    Examples
    --------
    Same distribution at every site (broadcast from one entry):

    >>> cov = Covariate(
    ...     name="age",
    ...     site_distributions=[
    ...         CovariateSiteDistribution(loc=45.0, scale=10.0, clip=(18.0, 90.0)),
    ...     ],
    ...     x_correlation=0.2,
    ... )

    Different distribution per site (two sites, explicit):

    >>> cov = Covariate(
    ...     name="age",
    ...     site_distributions=[
    ...         CovariateSiteDistribution(loc=30.0, scale=5.0, clip=(18.0, 90.0)),
    ...         CovariateSiteDistribution(loc=60.0, scale=8.0, clip=(18.0, 90.0)),
    ...     ],
    ...     x_correlation=0.2,
    ... )

    Categorical covariate — kind inferred from ``probs``:

    >>> cov = Covariate(
    ...     name="sex",
    ...     site_distributions=[
    ...         CovariateSiteDistribution(probs=[0.4, 0.6]),
    ...     ],
    ... )

    """

    name: str
    site_distributions: list[CovariateSiteDistribution] = field(default_factory=list)
    categories: list = field(default_factory=lambda: [0, 1])
    x_correlation: float = 0.0

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

    @property
    def kind(self) -> str:
        """Inferred kind: ``'continuous'`` or ``'categorical'``."""
        return self.site_distributions[0].kind


def make_covariate_site_distributions(
    locs: list[float] | None = None,
    scales: list[float] | float = 1.0,
    clips: list[tuple[float, float] | None] | tuple[float, float] | None = None,
    probs: list[list[float]] | list[float] | None = None,
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

    >>> dists = make_site_distributions(
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
        probs_list = list(probs)
    if len(probs_list) != n:
        raise ValueError(f"'probs' has {len(probs_list)} entries but 'locs' has {n}.")
    # Broadcast None
    if locs is None:
        locs: list = [None] * n
    return [
        CovariateSiteDistribution(loc=locs[i], scale=scales_list[i], probs=probs_list[i], clip=clips_list[i]) for i in range(n)
    ]


def _make_preset_covariate(
    name: str | PRESET_COVARIATES,
    n_sites: int,
) -> Covariate:
    """Build a :class:`Covariate` from a named neuroimaging preset.

    Provides sensible defaults for the three most common covariates in
    multi-site neuroimaging studies. All presets produce mildly different
    distributions per site to reflect realistic recruitment variation.
    Override individual sites by editing the returned ``site_distributions``
    list.

    Parameters
    ----------
    name : {"age", "sex", "quality"}
        Preset identifier:

        - ``"age"``: continuous Gaussian, site means 50 and std 25,
        clipped to [0, 100], mild positive correlation with X.
        - ``"sex"``: binary categorical (0 = male, 1 = female), site
          prevalence spread symmetrically around 50 %.
        - ``"quality"``: continuous IQM-like score in [0, 1]; quality is
          inversely proportional to ``noise_strength`` at each site.

    n_sites : int
        Number of sites.

    Returns
    -------
    Covariate

    Examples
    --------
    >>> spec = make_covariate_spec("age", n_sites=3)
    >>> len(spec.site_distributions)
    3
    >>> spec.site_distributions[0].loc  # All sites with the same age range
    50.0

    """
    if name.lower() == "age":
        # Site means spread linearly from 35 to 60
        dists = make_covariate_site_distributions(
            locs=[50] * n_sites,
            scales=20.0,
            clips=(0, 110),
        )
        return Covariate(
            name="age",
            site_distributions=dists,
            x_correlation=0.2,
        )

    if name.lower() == "sex":
        # Equally distributed sex
        dists = make_covariate_site_distributions(
            probs=[cast("list[float]", [0.5, 0.5])] * n_sites,
        )
        return Covariate(
            name="sex",
            site_distributions=dists,
            categories=[0, 1],
        )

    if name.lower() == "quality":
        # High noise → lower quality mean (shift down by up to -0.4)
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


def _resolve_covariates(covariates: list[Covariate | str] | None, n_sites: int) -> list[Covariate] | None:
    if covariates is not None:
        resolved_specs = []
        for entry in covariates:
            # Make the covariate given a name. Age, sex and quality are supported.
            if isinstance(entry, str):
                resolved_specs.append(
                    _make_preset_covariate(
                        entry,
                        n_sites=n_sites,
                    )
                )
            elif isinstance(entry, Covariate):
                resolved_specs.append(entry)
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
    X: np.ndarray,
    random_state: np.random.RandomState,
) -> dict[str, np.ndarray]:
    """Step 2 — Generate covariates correlated with the pre-effect X.

    Covariates are generated *before* site effects are applied so that any
    ``x_correlation`` reflects a genuine biological relationship rather than
    a scanner artefact.

    Parameters
    ----------
    covars : list[Covariate]
        Covariate specifications.
    site_idx : int
        Index of the current site.
    n_samples_site : int
        Number of samples at this site.
    X : np.ndarray, shape (n_samples_site, n_features)
        Base feature matrix *before* site effects or noise.
    random_state : np.random.RandomState
        sklearn-compatible random state.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from covariate name to array of shape (n_samples_site,).

    """
    result: dict[str, np.ndarray] = {}

    for covar in covars:
        sd = covar.site_distributions[site_idx]

        if covar.kind == "continuous":
            values = random_state.normal(loc=sd.loc, scale=sd.scale, size=n_samples_site)
            # Mix with the first feature of X (mean-centred, unit variance)
            # *before* site effects — this encodes a biological covariance
            # TODO: Now covariates are related with the first feature. We should expand this for other alternatives.
            alpha = float(np.clip(covar.x_correlation, 0.0, 1.0))
            if alpha > 0.0 and X.shape[1] > 0:
                x_col = X[:, 0].copy()
                x_std = x_col.std()
                if x_std > 0:
                    x_col = (x_col - x_col.mean()) / x_std * sd.scale + sd.loc
                    values = (1.0 - alpha) * values + alpha * x_col

            if sd.clip is not None:
                values = np.clip(values, sd.clip[0], sd.clip[1])

        elif covar.kind == "categorical":
            n_cat = len(covar.categories)
            if sd.probs is None:
                p = np.ones(n_cat) / n_cat
            else:
                p = np.array(sd.probs, dtype=float)
                p = p / p.sum()
            indices = random_state.choice(len(covar.categories), size=n_samples_site, p=p)
            values = np.array(covar.categories)[indices]

        else:
            raise ValueError(f"Covariate '{covar.name}': unknown kind='{covar.kind}'. Use 'continuous' or 'categorical'.")

        result[covar.name] = values

    return result


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _validate_covariates(
    covars: list[Covariate],
    n_sites: int,
) -> None:
    """Check that every Covariate is consistent with n_sites."""
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
