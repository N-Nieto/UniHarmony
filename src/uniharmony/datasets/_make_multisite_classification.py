"""Data simulation module for multi-site data generation."""

import numbers
import warnings
from typing import Literal, get_args, overload

import numpy as np
import numpy.typing as npt
import structlog
from sklearn.datasets import make_blobs, make_circles, make_classification, make_gaussian_quantiles, make_moons
from sklearn.utils import check_random_state

from ._make_covariates import Covariate, _make_covariate, _resolve_covariates


__all__ = [
    "make_multisite_classification",
]

# Currently available base signal types
SIGNAL_TYPES = Literal["linear", "circles", "moons", "blobs", "gaussian_quantiles"]
_SIGNAL_TYPE_ALIASES = {"make_gaussian_quantiles": "gaussian_quantiles"}  # deprecated names
# Currently available EoS signal types
SITE_EFFECT_TYPES = Literal["location", "scale", "location+scale", "variance", "nonlinear", "dropout"]

# Preset covariates
PRESET_COVARS = Literal["age", "sex", "quality"]

ReturnType = (
    tuple[np.ndarray, np.ndarray, np.ndarray]  # covariates None / return_base_data False
    | tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]  # covariates not None / return_base_data False
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]  # covariates None / return_base_data True
    | tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray]  # covariates not None / return_base_data True
)

logger = structlog.get_logger()


@overload
def make_multisite_classification(
    n_sites: int = 2,
    n_samples: int | list[int] = 1000,
    n_features: int = 10,
    n_classes: int = 2,
    balance_per_site: list[float] | list[list[float]] | None = None,
    signal_type: str = "linear",
    signal_strength: float = 1.0,
    noise_strength: list[float] | float = 0.1,
    site_effect_type: str = "location",
    site_effect_strength: list[float] | float = 3.0,
    site_effect_homogeneous: bool = True,
    covariates: None = None,
    return_base_data: Literal[False] = False,
    random_state: int | np.random.RandomState = 42,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


@overload
def make_multisite_classification(
    n_sites: int = 2,
    n_samples: int | list[int] = 1000,
    n_features: int = 10,
    n_classes: int = 2,
    balance_per_site: list[float] | list[list[float]] | None = None,
    signal_type: str = "linear",
    signal_strength: float = 1.0,
    noise_strength: list[float] | float = 0.1,
    site_effect_type: str = "location",
    site_effect_strength: list[float] | float = 3.0,
    site_effect_homogeneous: bool = True,
    *,
    covariates: list,  # no default - must be provided
    return_base_data: Literal[False] = False,
    random_state: int | np.random.RandomState = 42,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]: ...


@overload
def make_multisite_classification(
    n_sites: int = 2,
    n_samples: int | list[int] = 1000,
    n_features: int = 10,
    n_classes: int = 2,
    balance_per_site: list[float] | list[list[float]] | None = None,
    signal_type: str = "linear",
    signal_strength: float = 1.0,
    noise_strength: list[float] | float = 0.1,
    site_effect_type: str = "location",
    site_effect_strength: list[float] | float = 3.0,
    site_effect_homogeneous: bool = True,
    *,
    covariates: None = None,
    return_base_data: Literal[True],  # no default - must be provided
    random_state: int | np.random.RandomState = 42,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...


@overload
def make_multisite_classification(
    n_sites: int = 2,
    n_samples: int | list[int] = 1000,
    n_features: int = 10,
    n_classes: int = 2,
    balance_per_site: list[float] | list[list[float]] | None = None,
    signal_type: str = "linear",
    signal_strength: float = 1.0,
    noise_strength: list[float] | float = 0.1,
    site_effect_type: str = "location",
    site_effect_strength: list[float] | float = 3.0,
    site_effect_homogeneous: bool = True,
    *,
    covariates: list,  # no default
    return_base_data: Literal[True],  # no default
    random_state: int | np.random.RandomState = 42,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray]: ...


def make_multisite_classification(
    n_sites: int = 2,
    n_samples: int | list[int] = 1000,
    n_features: int = 10,
    n_classes: int = 2,
    balance_per_site: list[float] | list[list[float]] | None = None,
    signal_type: str | SIGNAL_TYPES = "linear",
    signal_strength: float = 1.0,
    noise_strength: list[float] | float = 0.1,
    site_effect_type: str | SITE_EFFECT_TYPES = "location",
    site_effect_strength: list[float] | float = 3.0,
    site_effect_homogeneous: bool = True,
    covariates: list[Covariate | str] | PRESET_COVARS | None = None,
    return_base_data: bool = False,
    random_state: int | np.random.RandomState = 42,
    **kwargs,
) -> ReturnType:
    """Simulate multi-site classification data with signal, site effect and noise.

    The data follow the additive model::

        X = Signal(y) + SiteEffect(site) + Noise(site)

    1. A pool of base samples (signal only) is generated with the
       scikit-learn generator selected by ``signal_type``, with exactly as
       many samples of each class as the sites need.
    2. Each site draws its samples from the pool without replacement,
       following its class balance. No sample is used twice.
    3. Optional covariates are drawn per site, correlated with the base
       samples (before the site effect).
    4. The Effect of Site (EoS) selected by ``site_effect_type`` is applied.
    5. Independent Gaussian noise is added.

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
        Each must sum to 1. ``None`` gives balanced classes. Class counts are
        rounded with the largest-remainder method, so a class with
        proportion 0 never gets samples.

    signal_type : {"linear", "moons", "circles", "blobs", "gaussian_quantiles"}, optional (default "linear")
        Generator of the base problem (``sklearn.datasets.make_classification``,
        ``make_moons``, ``make_circles``, ``make_blobs`` or
        ``make_gaussian_quantiles``). ``"moons"`` and ``"circles"`` always
        produce 2 features and 2 classes.

    signal_strength : float, optional (default 1.0)
        Class separation: ``class_sep`` for ``"linear"`` and the half-width
        of the ``center_box`` for ``"blobs"``. For ``"gaussian_quantiles"``
        it is the covariance of the Gaussian, which scales the data but does
        not change how separable the classes are. Ignored by ``"moons"`` and
        ``"circles"``.

    noise_strength : float or list of float, optional (default 0.1)
        Standard deviation of the Gaussian noise, for all sites or per site.

    site_effect_type : str, optional (default "location")
        Type of EoS, with ``s = site_effect_strength`` of the site:

        - ``"location"``: adds a per-feature offset drawn from U(0, s).
        - ``"scale"``: multiplies the deviations from the global mean by a
          per-feature factor exp(U(-0.1 s, 0.1 s)); the means do not move.
        - ``"location+scale"``: ``"scale"`` followed by ``"location"``.
        - ``"variance"``: adds Gaussian noise with a per-feature standard
          deviation drawn from U(0, s).
        - ``"nonlinear"``: adds ``w * X**2`` with ``w`` drawn from U(0, 0.1 s).
        - ``"dropout"``: sets values to 0 with probability ``min(0.1 s, 1)``.

        A strength of 0 means no EoS for that site.

    site_effect_strength : float or list of float, optional (default 3.0)
        Strength of the EoS, for all sites or per site (non-negative).

    site_effect_homogeneous : bool, optional (default True)
        If True, every sample of a site gets the same site effect. If False,
        each sample's effect varies around its site's effect, by as much as
        the site effects vary between sites (for ``"dropout"``, values are
        dropped per sample instead of whole features per site).

    covariates : list of Covariate or str, or None, optional (default None)
        Covariate specifications. Each entry is a :class:`Covariate` or a
        preset name (``"age"``, ``"sex"``, ``"quality"``). When ``None``, no
        covariates are generated.

    return_base_data : bool, optional (default False)
        If True, also return the base samples (signal only, before EoS and
        noise), row-aligned with ``X``. ``X - X_base`` is then the EoS plus
        noise of each sample.

    random_state : int or RandomState instance, optional (default 42)
        The seed of the pseudo random number generator or RandomState for
        reproducibility.

    **kwargs
        Additional keyword arguments passed to the scikit-learn generator.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
        Simulated feature matrix.
    y : np.ndarray of shape (n_samples,)
        Class labels (0 to ``n_classes`` - 1).
    sites : np.ndarray of shape (n_samples,)
        Site labels (0 to ``n_sites`` - 1).
    covariates_dict : dict[str, np.ndarray]
        Only returned when ``covariates`` is not ``None``. Maps covariate name
        to an array of shape (n_samples,).
    X_base : np.ndarray of shape (n_samples, n_features)
        Only returned when ``return_base_data`` is True.

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

    """
    random_state = check_random_state(random_state)

    signal_type = _resolve_signal_type(signal_type)
    if site_effect_type.lower() not in get_args(SITE_EFFECT_TYPES):
        raise ValueError(f"Unsupported site_effect_type: {site_effect_type}, Choose from: {get_args(SITE_EFFECT_TYPES)}.")
    n_classes, n_features = _validate_signal_type(signal_type, n_classes, n_features)
    samples_per_site = _validate_parameters(n_sites=n_sites, n_samples=n_samples, n_features=n_features, n_classes=n_classes)
    signal_strength, site_effect_strength, noise_strength = _validate_components(
        signal_strength, site_effect_strength, noise_strength, n_sites
    )
    balance = _validate_balance_per_site(balance_per_site, n_sites, n_classes)
    resolved_specs = _resolve_covariates(covariates, n_sites)

    # Steps 1-2: a pool with exactly the class counts the sites need, split without replacement.
    class_counts = np.array([_allocate(n, proportions) for n, proportions in zip(samples_per_site, balance, strict=True)])
    X_pool, y_pool = _generate_base_samples(
        class_counts.sum(axis=0), n_features, n_classes, signal_type, signal_strength, random_state, **kwargs
    )
    site_indices = _split_pool_by_site(y_pool, class_counts, random_state)
    X_base = [X_pool[idx] for idx in site_indices]
    center = np.vstack(X_base).mean(axis=0)  # the scale EoS acts around the global mean

    X_parts = []
    cov_parts: dict[str, list[np.ndarray]] = {spec.name: [] for spec in resolved_specs or []}
    for site_idx, X_site in enumerate(X_base):
        logger.debug(f"Site {site_idx}: class counts {class_counts[site_idx].tolist()}, EoS {site_effect_strength[site_idx]}")

        # Step 3: covariates, correlated with the base samples (before the EoS)
        if resolved_specs is not None:
            site_covars = _make_covariate(resolved_specs, site_idx, len(X_site), X_site, random_state)
            for name, values in site_covars.items():
                cov_parts[name].append(values)

        # Steps 4-5: EoS and noise
        X_site = _apply_site_effect(
            X_site, site_effect_type, site_effect_strength[site_idx], site_effect_homogeneous, random_state, center=center
        )
        if noise_strength[site_idx] > 0:
            X_site = _apply_noise(X_site, noise_strength[site_idx], random_state)
        X_parts.append(X_site)

    X = np.vstack(X_parts)
    y = np.concatenate([y_pool[idx] for idx in site_indices])
    sites = np.repeat(np.arange(n_sites), samples_per_site)
    X_base_all = np.vstack(X_base)
    covariates_dict = {name: np.concatenate(parts) for name, parts in cov_parts.items()}

    logger.info(f"Generated {len(X)} samples across {n_sites} sites")
    logger.debug(f"Class distribution: {np.bincount(y, minlength=n_classes)}, site distribution: {samples_per_site}")

    outputs: tuple = (X, y, sites)
    if resolved_specs is not None:
        outputs += (covariates_dict,)
    if return_base_data:
        outputs += (X_base_all,)
    return outputs


##########################################################################################################
############################################### VALIDATIONS ##############################################
##########################################################################################################
def _resolve_signal_type(signal_type: str) -> str:
    """Return the canonical signal type name, warning for deprecated aliases."""
    if signal_type in _SIGNAL_TYPE_ALIASES:
        new_name = _SIGNAL_TYPE_ALIASES[signal_type]
        warnings.warn(
            f"signal_type='{signal_type}' is deprecated, use '{new_name}' instead.",
            FutureWarning,
            stacklevel=3,
        )
        return new_name
    if signal_type not in get_args(SIGNAL_TYPES):
        raise ValueError(f"Unsupported signal_type: {signal_type}. Choose from {get_args(SIGNAL_TYPES)}")
    return signal_type


def _validate_signal_type(signal_type: str, n_classes: int, n_features: int) -> tuple[int, int]:
    """Force 2 features and 2 classes for the 2D-only generators."""
    if signal_type in {"moons", "circles"}:
        if n_features != 2:
            logger.warning(f"signal_type='{signal_type}' always produces 2 features; n_features={n_features} is set to 2.")
        if n_classes != 2:
            logger.warning(f"signal_type='{signal_type}' always produces 2 classes; n_classes={n_classes} is set to 2.")
        return 2, 2
    return n_classes, n_features


def _validate_parameters(n_sites: int, n_samples: int | list[int], n_features: int, n_classes: int) -> npt.NDArray:
    """Validate the problem size and return the number of samples of each site.

    Raises
    ------
    ValueError
        If ``n_sites`` is less than 1, ``n_features`` is not positive,
        ``n_classes`` is less than 2, or a site would get no samples.
    TypeError
        If ``n_samples`` is not an int or a list of ints.

    """
    if n_sites < 1:
        raise ValueError(f"n_sites must be at least 1, got {n_sites}")
    if n_sites == 1:
        logger.warning("n_sites is 1, which may not be suitable for testing multi-site methods.")
    if n_features <= 0:
        raise ValueError(f"n_features must be positive, got {n_features}")
    if n_classes < 2:
        raise ValueError(f"n_classes must be at least 2, got {n_classes}")

    if isinstance(n_samples, numbers.Integral):
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


def _validate_components(
    signal_strength: float,
    site_effect_strength: list[float] | float,
    noise_strength: list[float] | float,
    n_sites: int,
) -> tuple[float, list[float], list[float]]:
    """Validate the strengths of the signal, the EoS and the noise."""
    signal_strength = float(signal_strength)
    if not signal_strength >= 0:
        raise ValueError(f"signal_strength must be non-negative, got {signal_strength}")
    if signal_strength == 0.0:
        logger.warning("signal_strength is 0. Adding a delta (1e-6) to signal_strength to avoid degenerate data.")
        signal_strength = 1e-6

    site_effect_strength = _make_component_list(site_effect_strength, n_sites, "site_effect_strength")
    noise_strength = _make_component_list(noise_strength, n_sites, "noise_strength")
    return signal_strength, site_effect_strength, noise_strength


def _make_component_list(component: float | list[float], n_sites: int, component_name: str) -> list[float]:
    """Broadcast a non-negative strength to all sites, or validate one strength per site."""
    if isinstance(component, numbers.Real):
        if not component >= 0:
            raise ValueError(f"{component_name} must be non-negative, got {component}")
        return [float(component)] * n_sites

    if not isinstance(component, (list, tuple, np.ndarray)):
        raise TypeError(f"Invalid type for {component_name}: must be a float or list[float], got {type(component)}")
    if len(component) != n_sites:
        raise ValueError(f"{component_name} must have length n_sites ({n_sites}), got {len(component)}")
    for i, site_component in enumerate(component):
        if not isinstance(site_component, numbers.Real):
            raise TypeError(f"Invalid type for {component_name}[{i}]: must be a float, got {type(site_component)}")
        if not site_component >= 0:
            raise ValueError(f"{component_name}[{i}] must be non-negative, got {site_component}")
    return [float(x) for x in component]


##########################################################################################################
################################################# BALANCE ################################################
##########################################################################################################
def _validate_balance_per_site(
    balance_per_site: list[float] | list[list[float]] | None,
    n_sites: int,
    n_classes: int,
) -> npt.NDArray:
    """Validate the class proportions and return them as an array of shape (n_sites, n_classes).

    Raises
    ------
    ValueError
        If the shape does not match ``n_sites`` and ``n_classes``, a
        proportion is outside [0, 1], or the proportions of a site do not
        sum to 1.
    TypeError
        If ``balance_per_site`` is not a list (or array) of numbers.

    """
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
    """Split ``n`` into integer counts proportional to ``proportions`` (largest-remainder method)."""
    raw = n * np.asarray(proportions, dtype=float)
    counts = np.floor(raw).astype(int)
    remainder_order = np.argsort(-(raw - counts), kind="stable")
    counts[remainder_order[: n - counts.sum()]] += 1
    return counts


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


##########################################################################################################
############################################# EoS ########################################################
##########################################################################################################
def _apply_site_effect(
    X: npt.NDArray,
    site_effect_type: str,
    site_effect_strength: float,
    site_effect_homogeneous: bool,
    random_state: np.random.RandomState,
    center: npt.NDArray | None = None,
) -> npt.NDArray:
    """Apply the Effect of Site of one site.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples_site, n_features)
        Features of a single site before the site effect.
    site_effect_type : str
        One of ``SITE_EFFECT_TYPES`` (see ``make_multisite_classification``).
    site_effect_strength : float
        Strength ``s`` of the site effect; 0 means no effect.
    site_effect_homogeneous : bool
        If True, all samples of the site get the same effect. If False, the
        effect of each sample varies around the site's effect.
    random_state : RandomState instance
        The RandomState for reproducibility.
    center : np.ndarray of shape (n_features,), optional (default None)
        Point around which ``"scale"`` multiplies the data (the global mean
        of all sites). If None, the mean of ``X`` is used.

    Returns
    -------
    np.ndarray of shape (n_samples_site, n_features)
        Features with the site effect (``X`` is not modified).

    """
    n_samples, n_features = X.shape
    s = site_effect_strength

    def draw(low: float, high: float) -> npt.NDArray:
        """Per-feature site parameter from U(low, high), plus per-sample jitter if heterogeneous."""
        value = random_state.uniform(low, high, size=n_features)
        if not site_effect_homogeneous:
            half_width = (high - low) / 2
            value = value + random_state.uniform(-half_width, half_width, size=(n_samples, n_features))
        return value

    def scale(X: npt.NDArray) -> npt.NDArray:
        origin = X.mean(axis=0) if center is None else center
        return origin + (X - origin) * np.exp(draw(-0.1 * s, 0.1 * s))

    effect_type = site_effect_type.lower()
    if effect_type == "location":
        return X + draw(0.0, s)
    if effect_type == "scale":
        return scale(X)
    if effect_type == "location+scale":
        return scale(X) + draw(0.0, s)
    if effect_type == "variance":
        return X + random_state.normal(size=X.shape) * np.clip(draw(0.0, s), 0, None)
    if effect_type == "nonlinear":
        return X + draw(0.0, 0.1 * s) * np.square(X)
    if effect_type == "dropout":
        keep_probability = float(np.clip(1.0 - 0.1 * s, 0.0, 1.0))
        mask_shape = (n_features,) if site_effect_homogeneous else (n_samples, n_features)
        return X * random_state.binomial(1, keep_probability, size=mask_shape)
    raise ValueError(f"Unsupported site_effect_type: {site_effect_type}, Choose from: {get_args(SITE_EFFECT_TYPES)}.")


##########################################################################################################
############################################# Noise ######################################################
##########################################################################################################
def _apply_noise(
    X: np.ndarray,
    noise_strength: float,
    random_state: np.random.RandomState,
) -> np.ndarray:
    """Add independent Gaussian noise to every feature.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples_site, n_features)
        Data matrix.
    noise_strength : float
        Standard deviation of the noise.
    random_state : np.random.RandomState
        sklearn-compatible random state.

    Returns
    -------
    np.ndarray of the same shape as X.

    """
    return X + random_state.normal(0.0, noise_strength, size=X.shape)
