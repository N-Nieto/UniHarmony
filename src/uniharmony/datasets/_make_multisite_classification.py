"""Simulation of multi-site classification data."""

import numbers
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast, get_args, overload

import numpy as np
import numpy.typing as npt
from sklearn.datasets import make_blobs, make_circles, make_classification, make_gaussian_quantiles, make_moons

from uniharmony._verbose import get_logger

from ._make_covariates import Covariate
from ._simulation_constants import (
    CLASSIFICATION_SIGNAL_TYPES,
    DEFAULT_CLASSIFICATION_SIGNAL,
    FEW_SAMPLES,
    PRESET_COVARIATES,
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
    "make_multisite_classification",
]

logger = get_logger(src="make_multisite_classification")


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
    random_state: RandomStateLike = 42,
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
    random_state: RandomStateLike = 42,
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
    random_state: RandomStateLike = 42,
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
       noise form the data without site effect (``site_free``).
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
        - ``"dropout"``: replaces values, e.g. by ``np.nan`` to simulate
          missing values. It takes a dict ``{"value": v, "prob": p}``: in
          each site, every affected feature is replaced by ``v`` with
          probability ``p`` (a float or one per site). With ``"per":
          "value"``, single values are replaced instead of whole features.
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
        - ``"dropout_mask"``: True where ``"dropout"`` replaced a value
          (``site_effect`` is then not ``X - site_free`` but
          ``value - site_free`` there, NaN for a NaN value);
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
    shared = s_params.shared
    rng = shared.random_state

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
        shared.covariates, site_rows, s_params.n_features, feature_scale, rng
    )

    # Step 4: noise; signal + covariate effects + noise is the data without site effect
    noise = _draw_noise(site_rows, s_params.n_features, shared.noise_strength, feature_scale, rng)
    site_free = signal_samples + covariate_effect + noise
    center = (signal_samples + noise).mean(axis=0) + covariate_effect  # expected value of each sample without site effect

    # Step 5: site effects
    effects = _apply_site_effects(
        site_free, center, site_rows, feature_scale, shared.site_effect, rng, classes=y, n_classes=s_params.n_classes
    )

    logger.info(f"Generated {len(y)} samples across {s_params.n_sites} sites")
    logger.debug(f"Class counts per site: {s_params.class_counts.tolist()}")

    truth = None
    if return_ground_truth:
        truth = _ground_truth(
            signal=signal_samples,
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
    """Validated and resolved parameters of ``make_multisite_classification``."""

    n_sites: int
    n_features: int
    n_classes: int
    class_counts: npt.NDArray  # (n_sites, n_classes)
    signal_type: str
    signal_strength: float
    shared: _SharedParams


def _validate_inputs(**params: Any) -> _SimParams:
    """Validate every parameter of ``make_multisite_classification`` and resolve defaults and broadcasts."""
    n_sites, n_features, n_classes = _check_integers(
        n_sites=params["n_sites"], n_features=params["n_features"], n_classes=params["n_classes"]
    )
    if n_features <= 0:
        raise ValueError(f"n_features must be positive, got {n_features}")
    if n_classes < 2:
        raise ValueError(f"n_classes must be at least 2, got {n_classes}")
    samples_per_site = _samples_per_site(params["n_samples"], n_sites)
    signal_type, signal_strength, n_classes, n_features = _validate_signal(
        params["signal"], n_classes, n_features, params["kwargs"]
    )
    balance = _validate_balance_per_site(params["balance_per_site"], n_sites, n_classes)
    class_counts = np.array([_allocate(n, proportions) for n, proportions in zip(samples_per_site, balance, strict=True)])
    _check_class_counts(class_counts)
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
        with_classes=True,
    )
    return _SimParams(
        n_sites=n_sites,
        n_features=n_features,
        n_classes=n_classes,
        class_counts=class_counts,
        signal_type=signal_type,
        signal_strength=signal_strength,
        shared=shared,
    )


def _validate_signal(
    signal: Mapping[str, float] | None, n_classes: int, n_features: int, kwargs: dict
) -> tuple[str, float, int, int]:
    """Return the signal type and strength, and the number of classes and features the generator produces."""
    signal_type, strength = _signal_type_and_strength(
        signal, DEFAULT_CLASSIFICATION_SIGNAL, get_args(CLASSIFICATION_SIGNAL_TYPES)
    )

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
    rows: list[Any] = list(balance_per_site) if nested else [balance_per_site] * n_sites
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
    few = np.argwhere((class_counts > 0) & (class_counts < FEW_SAMPLES))
    if len(few) > 0:
        cells = ", ".join(f"site {s} class {c}: {class_counts[s, c]}" for s, c in few[:5])
        logger.warning(
            f"Some classes have fewer than {FEW_SAMPLES} samples in a site ({cells}); many methods need more "
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
        One of ``CLASSIFICATION_SIGNAL_TYPES``.
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
        X, y = make_circles(n_samples=cast("Any", tuple(class_counts)), random_state=random_state, **kwargs)
    elif signal_type == "moons":
        X, y = make_moons(n_samples=cast("Any", tuple(class_counts)), random_state=random_state, **kwargs)
    elif signal_type == "blobs":
        blobs = make_blobs(
            n_samples=cast("Any", list(class_counts)),
            n_features=n_features,
            center_box=(-signal_strength, signal_strength),
            random_state=random_state,
            **kwargs,
        )
        X, y = blobs[0], blobs[1]
    else:  # gaussian_quantiles: classes are equal-sized quantile shells
        X, y = make_gaussian_quantiles(
            cov=signal_strength,
            n_features=n_features,
            n_samples=n_classes * int(class_counts.max()),
            n_classes=n_classes,
            random_state=random_state,
            **kwargs,
        )

    return np.asarray(X, dtype=float), np.asarray(y)


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
