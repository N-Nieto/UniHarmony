"""Data simulation module for multi-site data generation."""

from typing import Literal, cast, get_args, overload

import numpy as np
import numpy.typing as npt
import structlog
from sklearn.datasets import make_blobs, make_circles, make_classification, make_gaussian_quantiles, make_moons
from sklearn.utils import check_random_state
from sklearn.utils.validation import (
    check_array,
    check_consistent_length,
    check_X_y,
)

from ._make_covariates import Covariate, _make_covariate, _resolve_covariates


__all__ = [
    "make_multisite_classification",
]

# Currently available base signal types
SIGNAL_TYPES = Literal["linear", "circular", "moons", "blobs", "gaussian_quantiles"]
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
    """Simulate multi-site data with signal, noise, and site effect components.

    In the data generation process, first a 'base' problem is generated using sklearn functions, selected with "signal_type".
    Then, each site is simulated and a site effect component is added to X, selected with "site_effect_type".
    The strength of the 'Effect of Site' (EoS) is controlled by `site_effect_strength`.
    If a list is passed, which element corresponds to the `site_effect_strength` in each site. List len musts be equal to n_sites.
    If a single value is passed, all sites has the same EoS
    Finally a gaussian noise  is added to each site, controlled by "noise_strength".

    Generates synthetic multi-centre biomedical data following the additive
    model::

        X = Signal(y) + SiteEffect(site) + Noise(site)

    Optionally generates covariates (age, sex, quality, or custom) that are
    correlated with the *pre-site-effect* feature matrix and vary by site.

    The per-site pipeline is::

        1. _make_base_data      → X_base, y           (signal only)
        2. _make_covariates     → covariates          (from X_base, before EoS)
        3. _apply_site_effect   → X_site              (with EoS)
        4. _apply_noise         → X_final             (independent Gaussian noise)


    Parameters
    ----------
    n_sites : int, optional (default 2)
        Number of sites to simulate.

    n_samples : int or list[int], optional (default 1000)
        If an int is provided, total number of samples across all sites.
        If a list is provided, N for each site, must have the same len as n_sites.

    n_features : int, optional (default 10)
        Number of features per sample.

    n_classes : int, optional (default 2)
        Number of classes to simulate (2 for binary, >2 for multi-class).

    balance_per_site : list of float, list of list of float or None, optional (default None)
        Class balance for each site. If None, uses balanced classes (0.5 for
        binary, equal distribution for multi-class).
        A flat list applies to every site; a list-of-lists gives one weight vector per site.
        Weights must sum to 1 (a warning is issued and sklearn normalizes automatically if not).
        ``None`` = balanced classes.

    signal_type : str, optional (default "linear")
        Which type of signal to generate the base problem. One of ``"linear"``, ``"moons"``, ``"circles"``,
        ``"blobs"``, ``"gaussian_quantiles"``.
        Note: ``"moons"`` and ``"circles"`` always produce 2 features regardless of ``n_features``.

    signal_strength : list of float or float, optional (default 1.0)
        Strength of the signal component separating classes. Passed as 'class_sep` to ``sklearn.datasets.make_classification`.

    noise_strength : list of float or float, optional (default 0.1)
        Strength of the noise component by site. If one component is passed, all sites has the same noise_strength.

    site_effect_type : str, optional (default "location")
        Type of site effect to add to the original data.
        Options: "location", "scale", "location+scale", "variance", "nonlinear", "dropout".

    site_effect_strength : float, optional (default 3.0)
        Strength of site-specific effects.

    site_effect_homogeneous : bool, optional (default True)
        Whether the site effect is homogeneous (same for all samples in a site).

    covariates : list[Covariate | str] | None, default None
        Covariate specifications. Each entry is a :class:`Covariate`
        instance or a preset name string (``"age"``, ``"sex"``, ``"quality"``).
        When ``None``, no covariates are generated and the function returns a 3-tuple.

    return_base_data : bool, default False
        Return base data before applying any change. This represents the ground Truth.

    random_state : int or RandomState instance, (default 42)
        The seed of the pseudo random number generator or RandomState for
        reproducibility.

    kwargs : dict
        Additional keyword arguments passed to ``sklearn.datasets.make_classification``.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
        Simulated feature matrix
    y : np.ndarray of shape (n_samples,)
        Class labels (0 to n_classes-1)
    sites : np.ndarray of shape (n_samples,)
        Site labels (0 to ``n_sites``-1)

    covariates_dict : dict[str, np.ndarray]
        Only returned when ``covariates`` is not ``None``. Maps covariate name
        to array of shape (n_samples_total,).

    X_base : np.ndarray of shape (n_samples, n_features)
        Only returned when ``return_base_data`` is not ``True``
        Simulated base samples (without EoS or noise)

    Examples
    --------
    >>> X, y, sites = make_multisite_classification(
    ...     n_sites=3, n_samples=300, n_features=20, n_classes=3
    ... )
    >>> X.shape, y.shape, sites.shape
    ((300, 20), (300,), (300,))

    """
    random_state = check_random_state(random_state)

    # Validate input parameters
    n_classes, n_features, samples_per_site, n_samples_total = _validate_parameters(
        n_classes=n_classes, n_sites=n_sites, n_samples=n_samples, n_features=n_features, signal_type=signal_type
    )

    signal_strength, site_effect_strength, noise_strength = _validate_components(
        signal_strength, site_effect_strength, noise_strength, n_sites
    )

    balance_per_site, overall_balance = _validate_balance_per_site(balance_per_site, n_sites, n_classes)

    # Generate a base dataset with more samples than needed to allow for site-specific sampling
    # We will sample from this base dataset for each site according to the specified balance and class distribution
    X, y = _generate_base_samples(
        n_samples_total, n_features, overall_balance, n_classes, signal_type, signal_strength, random_state, **kwargs
    )

    resolved_specs = _resolve_covariates(covariates, n_sites)

    X_parts, y_parts, site_labels_part, cov_parts, covariates_dict, X_base = _initialize_output(X, resolved_specs)

    # Create a copy of indices to track available samples
    available_indices = list(range(len(X)))

    if return_base_data:
        X_base = X.copy()

    # Generate data for each site
    for site_idx in range(n_sites):
        n_site_samples = samples_per_site[site_idx]
        site_eos = site_effect_strength[site_idx]
        site_noise = noise_strength[site_idx]
        balance = balance_per_site[site_idx]
        logger.info(f"For site {site_idx}")
        logger.info(f"Generating {n_site_samples} samples")
        logger.debug(f"Balance {balance} for site {site_idx}")

        # Get site-specific samples based on balance and class distribution in the global dataset
        # X and y are sampled without replacement when possible, but if not enough samples of a particular class are available,
        # it falls back to sampling with replacement and issues a warning.
        X_site, y_site = _get_site_samples(X, y, balance, n_classes, n_site_samples, available_indices, random_state)

        # Step 2: covariates (correlated with X_base, before site effects)
        if resolved_specs is not None:
            site_covars = _make_covariate(
                covars=resolved_specs,
                site_idx=site_idx,
                n_samples_site=n_site_samples,
                X=X_site,
                random_state=random_state,
            )
            for name, arr in site_covars.items():
                cov_parts[name].append(arr)
                logger.debug(
                    "Site %d — covariate '%s': mean=%.3f",
                    site_idx,
                    name,
                    float(np.mean(arr.astype(float))),
                )

        X_site = _apply_site_effect(X_site, site_effect_type, site_eos, site_effect_homogeneous, random_state)

        # If the site noise is not 0, apply noise.
        if site_noise != 0:
            # Step 4: noise
            X_site = _apply_noise(
                X=X_site,
                noise_strength=site_noise,
                random_state=random_state,
            )

        X_parts.append(X_site)
        y_parts.append(y_site)
        site_labels_part.extend([site_idx] * n_site_samples)
        logger.debug(f"Site {site_idx}, site effect strength {site_effect_strength}")

    # Concatenate all sites
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    sites = np.array(site_labels_part, dtype=int)

    # Check generated data.
    X, y = check_X_y(X, y)
    sites = check_array(sites, dtype=None, ensure_2d=False)
    check_consistent_length(X, y, sites)

    logger.info(f"Generated {len(X)} samples across {n_sites} sites")
    logger.info(f"Class distribution: {np.bincount(y)}")
    logger.info(f"Site distribution: {np.bincount(sites)}")
    covariates_dict = {name: np.concatenate(parts) for name, parts in cov_parts.items()}

    # Returning possibilities
    if return_base_data:
        if resolved_specs is not None:
            return X, y, sites, covariates_dict, X_base
        else:
            return X, y, sites, X_base

    if resolved_specs is not None:
        return X, y, sites, covariates_dict

    return X, y, sites


##########################################################################################################
##########################################################################################################
############################################### VALIDATIONS ##############################################
##########################################################################################################
##########################################################################################################
def _validate_parameters(  # noqa: C901
    n_sites: int,
    n_samples: int | list[int],
    n_features: int,
    n_classes: int,
    signal_type: str,
) -> tuple[int, int, np.ndarray, int]:
    """Validate all input parameters for data simulation.

    Parameters
    ----------
    n_sites : int
        Number of sites to simulate.
    n_samples : int
        Total number of samples across all sites.
    n_features : int
        Number of features per sample.
    n_classes : int
        Number of classes to simulate (2 for binary, >2 for multi-class).
    signal_type : str
        Type of signal to simulate.

    Raises
    ------
    ValueError
        If ``n_sites`` is less than 1 or
        if ``n_features`` is negative or
        if ``n_classes`` is less than 2 or
        if ``n_samples`` is less than ``n_sites``.

    """
    n_classes, n_features = _validate_signal_type(signal_type, n_classes, n_features)
    if n_sites < 2:
        logger.warning(
            f"n_sites is {n_sites}, which is less than 2."
            " This will result in a single site dataset, which may not be suitable for testing multi-site methods."
        )
    if n_sites < 1:
        raise ValueError(f"n_sites must be at least 1, got {n_sites}")

    if n_features <= 0:
        raise ValueError(f"n_features must be positive, got {n_features}")

    if n_classes < 2:
        raise ValueError(f"n_classes must be at least 2, got {n_classes}")

    if isinstance(n_samples, int):
        if n_samples < n_sites:
            raise ValueError(
                f"n_samples ({n_samples}) is less than n_sites ({n_sites}). Some sites will have 0 samples.",
            )
    elif isinstance(n_samples, list):
        if np.array(n_samples).sum() < n_sites:
            raise ValueError(
                f"n_samples ({n_samples}) is less than n_sites ({n_sites}). Some sites will have 0 samples.",
            )
    else:
        raise TypeError(
            f"n_samples must be int or list[int], got {type(n_samples)}.",
        )

    if isinstance(n_samples, list):
        if len(n_samples) == n_sites:
            samples_per_site = np.array(n_samples)
            n_samples_total = np.array(samples_per_site).sum()
        else:
            raise ValueError(f"n_samples has len {len(n_samples)} and does not match with n_sites={n_sites}")
    else:
        n_samples_total = n_samples
        # Allocate samples per site (even distribution)
        samples_per_site = np.full(n_sites, n_samples // n_sites, dtype=int)
        samples_per_site[: n_samples % n_sites] += 1
    logger.debug(f"Total Samples to generate {n_samples_total}")
    logger.debug("Total Samples to generate per site %s", samples_per_site)

    return n_classes, n_features, samples_per_site, n_samples_total


def _validate_signal_type(signal_type, n_classes, n_features):
    _2d_only = {"moons", "circles"}
    if signal_type in _2d_only and n_features != 2:
        logger.warning(
            f"signal_type='{signal_type}' always produces 2 features; n_features={n_features} is ignored and set to 2.",
        )
        n_features = 2
    if signal_type in _2d_only and n_classes != 2:
        logger.warning(
            f"signal_type='{signal_type}' always produces 2 classes; n_classes={n_classes} is ignored and set to 2.",
        )
        n_classes = 2
    return n_classes, n_features


def _validate_components(
    signal_strength: float,
    site_effect_strength: list[float] | float,
    noise_strength: list[float] | float,
    n_sites: int,
) -> tuple[float, list[float], list[float]]:
    """Component Validation."""
    # Check signal_strength
    signal_strength = float(signal_strength)
    if signal_strength < 0:
        raise ValueError(f"signal_strength must be non-negative, got {signal_strength}")

    if signal_strength == 0.0:
        logger.warning("signal_strength is 0. Adding a delta (1e-6) to signal_strength to avoid degenerate data.")
        signal_strength = 1e-6

    site_effect_strength = _make_component_list(site_effect_strength, n_sites, "site_effect_strength")
    noise_strength = _make_component_list(noise_strength, n_sites, "noise_strength")

    return signal_strength, site_effect_strength, noise_strength


def _make_component_list(component: float | list[float], n_sites: int, component_name) -> list[float]:
    # Check site_effect_strength
    if isinstance(component, (float, int)):
        if component < 0:
            raise ValueError(f"{component_name} must be non-negative, got {component}")
        component_list = [float(component)] * n_sites
    elif isinstance(component, list):
        # Check all elements are numeric
        if len(component) != n_sites:
            raise ValueError(f"{component_name} must have length n_sites ({n_sites}), got {len(component)}")
        for i, site_component in enumerate(component):
            if not isinstance(site_component, (float, int)):
                raise TypeError(
                    f"Invalid type for {component_name}[[{i}]]: must be a class proportion (float), got {type(site_component)}"
                )
            if not 0 < site_component:
                raise ValueError(f"{component_name}[{i}] must be non-negative, got {site_component}")
        component_list = [float(x) for x in component]
    else:
        raise TypeError(f"Invalid type for {component_name}: must be a float or list[float], got {type(component)}")
    return component_list


def _initialize_output(X, resolved_specs) -> tuple[list, list, list, dict, dict, np.ndarray]:
    # output typing
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    site_labels_part: list[np.ndarray] = []
    covariates_dict: dict[str, np.ndarray] = {}
    cov_parts: dict[str, list[np.ndarray]] = {spec.name: [] for spec in resolved_specs} if resolved_specs else {}
    X_base: np.ndarray = np.zeros([1])
    return X_parts, y_parts, site_labels_part, cov_parts, covariates_dict, X_base


##########################################################################################################
##########################################################################################################
################################################# BALANCE ################################################
##########################################################################################################
##########################################################################################################
def _get_default_balance_per_site(n_sites: int, n_classes: int) -> list[float] | list[list[float]]:
    """Get default class balance for each site."""
    equal_prob = 1.0 / n_classes
    return [[equal_prob] * n_classes] * n_sites


def _validate_balance_per_site(
    balance_per_site: list[float] | list[list[float]] | None,
    n_sites: int,
    n_classes: int,
) -> tuple[list[list[float]], list[float]]:
    """Validate balance_per_site parameter for multi-site data generation.

    Parameters
    ----------
    balance_per_site : list or None
        Class balance specification.
    n_sites : int
        Number of sites.
    n_classes : int
        Number of classes.

    Raises
    ------
    ValueError
        If ``balance_per_site`` has invalid structure or values.
    TypeError
        If ``balance_per_site`` has wrong types.

    """
    balance_per_site_normalized: list[list[float]]
    if balance_per_site is None:
        balance_per_site = _get_default_balance_per_site(n_sites, n_classes)
        logger.info(f"Using balanced classes: {balance_per_site}")

    # Check if it's a list
    if not isinstance(balance_per_site, list):
        raise TypeError(f"balance_per_site must be a list, got {type(balance_per_site)}")

    # This means that the same proportion will be use in all sites
    if isinstance(balance_per_site[0], float):
        # It's a flat list [0.3, 0.7], convert to nested list
        balance_per_site_normalized = [cast("list[float]", balance_per_site)] * n_sites
        logger.debug("Using the same balance_per_site for all sites.")
    else:
        # It should be nested already
        balance_per_site_normalized = cast("list[list[float]]", balance_per_site)

    # Check length matches n_sites
    if len(balance_per_site_normalized) != n_sites:
        raise ValueError(f"balance_per_site must have length n_sites ({n_sites}), got {len(balance_per_site_normalized)}")

    _check_balance(balance_per_site_normalized, n_classes)
    overall_balance = np.mean(balance_per_site_normalized, axis=0)

    logger.info(f"`balance_per_site` is: {balance_per_site_normalized}")
    logger.info(f"Overall class balance across sites: {overall_balance}")

    return balance_per_site_normalized, overall_balance


def _check_balance(balance_per_site: list | list[list], n_classes: int) -> None:
    """Check balance for multi-class classification.

    Parameters
    ----------
    balance_per_site: list, list[list] or tuple
        Class balance specification.
    n_classes: int
        Number of classes.

    Raises
    ------
    ValueError
        If ``balance_per_site`` has invalid structure or values.
    TypeError
        If ``balance_per_site`` has wrong types.

    """
    # list of lists
    for i, site_balance in enumerate(balance_per_site):
        # Check it's a list
        if not isinstance(site_balance, (list, np.ndarray)):
            raise TypeError(f"For n_classes > 2, balance_per_site[{i}] must be a list or array, got {type(site_balance)}")

        # Check length matches n_classes
        if len(site_balance) != n_classes:
            raise ValueError(f"balance_per_site[{i}] must have length n_classes ({n_classes}), got {len(site_balance)}")

        # Check all elements are numeric
        for j, class_prob in enumerate(site_balance):
            if not isinstance(class_prob, (float)):
                raise TypeError(f"balance_per_site[{i}][{j}] must be a class proportion (float), got {type(class_prob)}")
            if not 0 <= class_prob <= 1:
                raise ValueError(f"balance_per_site[{i}] must be between 0 and 1, got {class_prob}")

        # Convert to numpy array for sum check
        # site_balance_array = np.array(site_balance, dtype=float)

        # Check sum is approximately 1
        if not np.isclose(sum(site_balance), 1.0, atol=1e-10):
            raise ValueError(f"balance_per_site[{i}] must sum to 1.0, got {sum(site_balance):.6f}")


##########################################################################################################
##########################################################################################################
############################################### BASE SAMPLES #############################################
##########################################################################################################
##########################################################################################################
def _generate_base_samples(
    n_samples: int,
    n_features: int,
    overall_balance: list[float],
    n_classes: int,
    signal_type: str,
    signal_strength: float,
    random_state: np.random.RandomState,
    **kwargs,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Generate base samples using specified signal type.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    n_features : int
        Number of features per sample.
    overall_balance : float or list of float
        Class balance for the base dataset. For binary classification, a float
        representing proportion of class 1. For multi-class, a list of
        probabilities for each class.
    n_classes : int
        Number of classes.
    signal_type : str
        Type of signal to generate. Options: "linear", "circles", "moons",
        "blobs", "make_gaussian_quantiles".
    signal_strength : float
        Strength of the signal component separating classes.
    random_state : RandomState instance
        The RandomState for reproducibility.
    kwargs : dict
        Additional keyword arguments passed to the signal generation function.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
        Generated feature matrix.
    y : np.ndarray of shape (n_samples,)
        Generated class labels.

    Raises
    ------
    ValueError
        If ``signal_type`` is not supported.
        If ``signal_type`` is "moons" but n_classes != 2 or n_features < 2.

    """
    base_samples = int(np.ceil(n_samples * (1.1 + max(overall_balance))) + 1)  # Generate more samples than needed for sampling

    if signal_type == "linear":
        # Replace the default values of sklearn for this variables.
        make_classification_kwargs = {
            "n_redundant": 0,
            "flip_y": 0.0,
            "n_clusters_per_class": 1,
            "n_informative": min(n_features, n_classes * 2),
        }
        make_classification_kwargs.update(kwargs)
        X, y = make_classification(
            n_samples=int(np.ceil(n_samples * 1.1)),
            n_features=n_features,
            n_classes=n_classes,
            return_X_y=True,
            weights=overall_balance,
            class_sep=signal_strength,
            random_state=random_state,
            **make_classification_kwargs,
        )
    elif signal_type == "circles":
        X, y = make_circles(n_samples=base_samples, random_state=random_state, **kwargs)
    elif signal_type == "moons":
        X, y = make_moons(n_samples=base_samples, random_state=random_state, **kwargs)
    elif signal_type == "blobs":
        X, y = make_blobs(
            n_samples=base_samples,
            n_features=n_features,
            centers=n_classes,
            random_state=random_state,
            center_box=(-signal_strength, signal_strength),
            return_centers=False,
            **kwargs,
        )
    elif signal_type == "make_gaussian_quantiles":
        X, y = make_gaussian_quantiles(
            cov=signal_strength,
            n_features=n_features,
            n_samples=base_samples,
            n_classes=n_classes,
            random_state=random_state,
            **kwargs,
        )

    else:
        raise ValueError(f"Unsupported signal_type: {signal_type}. Choose from {get_args(SIGNAL_TYPES)}")

    return X.astype(float), y


def _get_site_samples(
    X: npt.NDArray,
    y: npt.NDArray,
    balance: list[float],
    n_classes: int,
    n_site_samples: int,
    available_indices: list[int],
    random_state: np.random.RandomState,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Sample site-specific data from global dataset according to class balance.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Global feature matrix.
    y : np.ndarray of shape (n_samples,)
        Global class labels.
    balance : list of float
        Class balance for this site. For binary classification, a float
        representing proportion of class 1. For multi-class, a list of
        probabilities for each class.
    n_classes : int
        Number of classes.
    n_site_samples : int
        Number of samples to generate for this site.
    available_indices : list of int
        List of available indices from the global dataset that haven't been used.
    random_state : RandomState instance
        The RandomState for reproducibility.

    Returns
    -------
    X_site : np.ndarray of shape (n_site_samples, n_features)
        Feature matrix for this site.
    y_site : np.ndarray of shape (n_site_samples,)
        Class labels for this site.

    Notes
    -----
    This function samples from the global dataset without replacement when
    possible. If not enough samples of a particular class are available,
    it falls back to sampling with replacement and issues a warning.

    """
    # For multi-class classification
    samples_per_class = [int(n_site_samples * prob) for prob in balance]
    # Adjust for rounding errors by distributing the difference across classes
    diff = n_site_samples - sum(samples_per_class)
    if diff != 0:
        # Distribute the difference evenly, not just to the largest class
        for i in range(abs(diff)):
            idx = i % n_classes
            samples_per_class[idx] += 1 if diff > 0 else -1

    # Randomly select samples for each class
    selected_indices = []
    for class_idx, n_class_samples in enumerate(samples_per_class):
        if n_class_samples == 0:
            continue

        # Get available indices of current class from the remaining pool
        available_class_indices = [idx for idx in available_indices if y[idx] == class_idx]

        if len(available_class_indices) < n_class_samples:
            # Sample with replacement if not enough samples
            logger.error(
                f"Not enough samples of class {class_idx} in global dataset. "
                f"Requested {n_class_samples}, available {len(available_class_indices)}. "
                f"Consider adjusting balance_per_site or generating more samples."
            )
            selected = random_state.choice(available_class_indices, size=n_class_samples, replace=True)
        else:
            # Sample without replacement
            selected = random_state.choice(available_class_indices, size=n_class_samples, replace=False)
            # Remove selected indices from available pool
            for idx in selected:
                available_indices.remove(idx)

        selected_indices.extend(selected)

    # Shuffle the selected indices
    random_state.shuffle(selected_indices)

    # Extract the samples
    X_site = X[selected_indices]
    y_site = y[selected_indices]
    return X_site, y_site


##########################################################################################################
##########################################################################################################
############################################# EoS ########################################################
##########################################################################################################
##########################################################################################################
def _apply_site_effect(
    X: npt.NDArray,
    site_effect_type: str,
    site_effect_strength: float,
    site_effect_homogeneous: bool,
    random_state: np.random.RandomState,
) -> npt.NDArray:
    """Generate site effect component for features.

    Parameters
    ----------
    X : npt.NDArray
        Features for a single site before adding site effect.
    site_effect_homogeneous : bool
        If True, generates same effect for all features in this site.
        If False, generates different effect for each feature.
    site_effect_strength : float
        Magnitude of site effect. For homogeneous case, effects are uniformly
        distributed in [-site_effect_strength, site_effect_strength].
        For heterogeneous case, effects are normally distributed with
        scale = site_effect_strength.
    random_state : RandomState instance
        The RandomState for reproducibility.
    site_effect_type : str, default ("location")
        Type of effect of site added to the original data.

    Returns
    -------
    X = npt.NDArray
        Features with simulated site effect.
    y = npt.NDArray
        Target with simulated site effect (not always applied).

    """
    n_samples, n_features = X.shape

    # If the site are homogeneous, apply the same effect to all samples.
    if site_effect_homogeneous:
        shape = (n_features,)
    else:
        shape = (n_samples, n_features)

    effect_type = site_effect_type.lower()
    if effect_type in ["location"]:
        location_offset = random_state.uniform(0.0, site_effect_strength, size=shape)
        # Add site component to the signal
        X = X + location_offset
    elif effect_type in ["scale"]:
        scale_factor = random_state.uniform(0.0, site_effect_strength * 0.1, size=shape)
        # Add site component to the signal
        X = X * scale_factor
    elif effect_type in ["location+scale"]:
        location_offset = random_state.uniform(0.0, site_effect_strength, size=shape)
        scale_factor = random_state.uniform(0.0, site_effect_strength * 0.1, size=shape)
        X = X * scale_factor + location_offset
    elif effect_type in ["variance"]:
        variance_scale = random_state.uniform(0.0, site_effect_strength, size=shape)
        X = X + random_state.normal(0.0, variance_scale, size=(n_samples, n_features))
    elif effect_type in ["nonlinear"]:
        nonlinear_weight = random_state.uniform(0.0, site_effect_strength * 0.1, size=shape)
        X = X + nonlinear_weight * np.square(X)
    elif effect_type in ["dropout"]:
        keep_probability = float(np.clip(1.0 - site_effect_strength * 0.1, 0.0, 1.0))
        X = X * random_state.binomial(1, keep_probability, size=shape)
    else:
        raise ValueError(f"Unsupported site_effect_type: {site_effect_type}, Choose from: {get_args(SITE_EFFECT_TYPES)}.")

    return X


##########################################################################################################
##########################################################################################################
############################################# Noise ######################################################
##########################################################################################################
##########################################################################################################
def _apply_noise(
    X: np.ndarray,
    noise_strength: float,
    random_state: np.random.RandomState,
) -> np.ndarray:
    """Step 4 — Add independent Gaussian noise to every feature.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples_site, n_features)
        Data matrix.

    noise_strength : float
        Standard deviation of the noise.

    random_state : np.random.RandomState
        sklearn-compatible random state.

    Returns
    -------
    np.ndarray, same shape as X.

    """
    return X + random_state.normal(0.0, noise_strength, size=X.shape)
