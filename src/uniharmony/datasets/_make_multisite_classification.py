"""Data simulation module for multi-site data generation."""

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


__all__ = [
    "make_multisite_classification",
]


logger = structlog.get_logger()


def make_multisite_classification(
    n_sites: int = 2,
    n_samples: int = 1000,
    n_features: int = 10,
    n_classes: int = 2,
    balance_per_site: list[float] | list[list[float]] | None = None,
    signal_type: str = "linear",
    signal_strength: float = 1.0,
    noise_strength: float = 0.1,
    site_effect_type: str = "location",
    site_effect_strength: float = 3.0,
    site_effect_homogeneous: bool = True,
    random_state: int | np.random.RandomState = 42,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate multi-site data with signal, noise, and site effect components.

    The data generation follows: X = signal + noise + site_effect
    All components are sampled from Gaussian distributions.

    Parameters
    ----------
    n_classes : int, optional (default 2)
        Number of classes to simulate (2 for binary, >2 for multi-class).
    n_sites : int, optional (default 2)
        Number of sites to simulate.
    n_samples : int, optional (default 1000)
        Total number of samples across all sites.
    balance_per_site : list of float, list of list of float or None, optional (default None)
        Class balance for each site. If None, uses balanced classes (0.5 for
        binary, equal distribution for multi-class).
    n_features : int, optional (default 10)
        Number of features per sample.
    signal_type : str, optional (default "linear")
        Which type of signal to generate the base problem.
    signal_strength : float, optional (default 1.0)
        Strength of the signal component separating classes. Passed as 'class_sep` to ``sklearn.datasets.make_classification`.
    noise_strength : float, optional (default 0.1)
        Strength of the noise component.
    site_effect_type : str, optional (default "location")
        Type of site effect to add to the original data. Options: "location", "scale", "location+scale".
    site_effect_strength : float, optional (default 3.0)
        Strength of site-specific effects.
    site_effect_homogeneous : bool, optional (default True)
        Whether the site effect is homogeneous (same for all samples in a site).
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
    _validate_parameters(
        n_classes=n_classes,
        n_sites=n_sites,
        n_samples=n_samples,
        n_features=n_features,
        signal_type=signal_type,
        signal_strength=signal_strength,
        noise_strength=noise_strength,
        site_effect_strength=site_effect_strength,
    )

    balance_per_site, overall_balance = _validate_balance_per_site(balance_per_site, n_sites, n_classes)

    # Allocate samples per site (even distribution)
    samples_per_site = np.full(n_sites, n_samples // n_sites, dtype=int)
    samples_per_site[: n_samples % n_sites] += 1

    # Generate a base dataset with more samples than needed to allow for site-specific sampling
    # We will sample from this base dataset for each site according to the specified balance and class distribution
    X, y = _generate_base_samples(
        n_samples, n_features, overall_balance, n_classes, signal_type, signal_strength, random_state, **kwargs
    )
    site_labels_list = []
    X_list = []
    y_list = []

    # Create a copy of indices to track available samples
    available_indices = list(range(len(X)))
    # Generate data for each site
    for site_idx in range(n_sites):
        n_site_samples = samples_per_site[site_idx]

        balance = balance_per_site[site_idx]
        logger.info(f"For site {site_idx}")
        logger.info(f"Generating {n_site_samples} samples")
        logger.debug(f"Balance {balance} for site {site_idx}")

        # Get site-specific samples based on balance and class distribution in the global dataset
        X_site, y_site = _get_site_samples(X, y, balance, n_classes, n_site_samples, available_indices, random_state)
        # Generate site effect component
        X_site, y_site = _generate_site_effect_component(
            X_site, y_site, site_effect_type, site_effect_strength, site_effect_homogeneous, random_state
        )

        if noise_strength != 0:
            # Generate noise component
            noise = random_state.normal(loc=0.0, scale=noise_strength, size=X_site.shape)
            X_site = X_site + noise

        X_list.append(X_site)
        y_list.append(y_site)
        site_labels_list.extend([site_idx] * n_site_samples)
        logger.debug(f"Site {site_idx}, site effect strength {site_effect_strength}")

    # Concatenate all sites
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    sites = np.array(site_labels_list, dtype=int)

    # Shuffle samples across sites
    indices = random_state.permutation(len(X))
    X = X[indices]
    y = y[indices]
    sites = sites[indices]

    # Check generated data.
    X, y = check_X_y(X, y)
    sites = check_array(sites, dtype=None, ensure_2d=False)
    check_consistent_length(X, y, sites)

    logger.info(f"Generated {len(X)} samples across {n_sites} sites")
    logger.info(f"Class distribution: {np.bincount(y)}")
    logger.info(f"Site distribution: {np.bincount(sites)}")

    return X, y, sites


def _validate_parameters(
    n_sites: int,
    n_samples: int,
    n_features: int,
    signal_strength: float,
    noise_strength: float,
    site_effect_strength: float,
    n_classes: int,
) -> None:
    """Validate all input parameters for data simulation.

    Parameters
    ----------
    n_sites : int
        Number of sites to simulate.
    n_samples : int
        Total number of samples across all sites.
    n_features : int
        Number of features per sample.
    signal_strength : float
        Strength of the signal component separating classes.
    noise_strength : float
        Strength of the noise component.
    site_effect_strength : float
        Strength of site-specific effects.
    n_classes : int
        Number of classes to simulate (2 for binary, >2 for multi-class).

    Raises
    ------
    ValueError
        If ``n_sites`` is less than 1 or
        if ``n_features`` is negative or
        if ``n_classes`` is less than 2 or
        if ``signal_strength`` is negative or
        if ``noise_strength`` is negative or
        if ``site_effect_strength`` is negative or
        if ``n_samples`` is less than ``n_sites``.

    """
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

    if signal_strength < 0:
        raise ValueError(f"signal_strength must be non-negative, got {signal_strength}")

    if noise_strength < 0:
        raise ValueError(f"noise_strength must be non-negative, got {noise_strength}")

    if site_effect_strength < 0:
        raise ValueError(f"site_effect_strength must be non-negative, got {site_effect_strength}")

    if n_samples < n_sites:
        raise ValueError(
            f"n_samples ({n_samples}) is less than n_sites ({n_sites}). Some sites will have 0 samples.",
        )


def _get_default_balance_per_site(n_sites: int, n_classes: int) -> list[float] | list[list[float]]:
    """Get default class balance for each site."""
    if n_classes == 2:
        # Binary: 0.5 for each site
        return [0.5] * n_sites
    else:
        # Multi-class: equal distribution for each site
        equal_prob = 1.0 / n_classes
        return [[equal_prob] * n_classes] * n_sites


def _validate_balance_per_site(
    balance_per_site: list | list[list] | None,
    n_sites: int,
    n_classes: int,
) -> tuple[list | list[list], float | list[float]]:
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
    if balance_per_site is None:
        balance_per_site = _get_default_balance_per_site(n_sites, n_classes)
        logger.info(f"Using balanced classes: {balance_per_site}")

    # Check it's a list
    if not isinstance(balance_per_site, list):
        raise TypeError(f"balance_per_site must be a list, got {type(balance_per_site)}")

    # Check length matches n_sites
    if len(balance_per_site) != n_sites:
        raise ValueError(f"balance_per_site must have length n_sites ({n_sites}), got {len(balance_per_site)}")

    # Validate based on number of classes
    if n_classes == 2:
        _check_balance_for_binary_classification(balance_per_site)
    else:
        _check_balance_for_multiclass(balance_per_site, n_classes)

    # Get the overall balance across all sites for logging
    overall_balance = (
        np.mean(balance_per_site, axis=0) if n_classes > 2 else [np.mean(balance_per_site), 1 - np.mean(balance_per_site)]
    )
    logger.info(f"Overall class balance across sites: {overall_balance}")

    return balance_per_site, overall_balance


def _check_balance_for_binary_classification(
    balance_per_site: list[float],
) -> None:
    """Check balance for binary classification.

    Parameters
    ----------
    balance_per_site : list or list[list]
        Class balance specification.

    Raises
    ------
    ValueError
        If ``balance_per_site`` has invalid structure or values.
    TypeError
        If ``balance_per_site`` has wrong types.

    """
    # For binary: list of floats
    for i, p_class1 in enumerate(balance_per_site):
        # Check type
        if not isinstance(p_class1, (float)):
            raise TypeError(f"balance_per_site[{i}] must be numeric class proportion (float), got {type(p_class1)}")

        # Check range
        if not 0 <= p_class1 <= 1:
            raise ValueError(f"balance_per_site[{i}] must be between 0 and 1, got {p_class1}")


def _check_balance_for_multiclass(balance_per_site: list | list[list] | tuple, n_classes: int) -> None:
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
    # For multi-class: list of lists
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
                raise TypeError(f"balance_per_site[{i}][{j}] must be a class proportion (int or float), got {type(class_prob)}")
            if not 0 <= class_prob <= 1:
                raise ValueError(f"balance_per_site[{i}] must be between 0 and 1, got {class_prob}")

        # Convert to numpy array for sum check
        site_balance_array = np.array(site_balance, dtype=float)

        # Check sum is approximately 1
        if not np.isclose(np.sum(site_balance_array), 1.0, atol=1e-10):
            raise ValueError(f"balance_per_site[{i}] must sum to 1.0, got {np.sum(site_balance_array):.6f}")


def _generate_base_samples(
    n_samples: int,
    n_features: int,
    overall_balance: float | list[float],
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

    """
    if signal_strength == 0.0:
        logger.warning(
            "signal_strength is 0. This will result in no separation between classes"
            "making the classification problem very difficult."
            "Adding a delta to signal_strength to avoid degenerate data."
        )
        signal_strength = 1e-6

    base_samples = int(np.ceil(n_samples * 1.05))  # Generate 5% more samples than needed for sampling
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
            n_samples=base_samples,
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
        if n_classes != 2 or n_features != 2:
            raise ValueError("make_moons requires n_classes=2 and n_features>=2")
    elif signal_type == "blobs":
        X, y = make_blobs(
            n_samples=base_samples,
            n_features=n_features,
            centers=n_classes,
            random_state=random_state,
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
        raise ValueError(f"Unsupported signal_type: {signal_type}")

    return X, y


def _generate_site_effect_component(
    X: npt.NDArray,
    y: npt.NDArray,
    site_effect_type: str,
    site_effect_strength: float,
    site_effect_homogeneous: bool,
    random_state: np.random.RandomState,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Generate site effect component for features.

    Parameters
    ----------
    X : npt.NDArray
        Features for a single site before adding site effect.
    y : npt.NDArray
        Target for a single site before adding site effect (not always applied).
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
    n_features = X.shape[1]

    if site_effect_type.lower() in ["location", "l"]:
        site_effect = _site_effect_value(site_effect_strength, site_effect_homogeneous, n_features, random_state)
        # Add site component to the signal
        X = X + site_effect
    elif site_effect_type.lower() in ["scale", "s"]:
        site_effect = _site_effect_value(site_effect_strength, site_effect_homogeneous, n_features, random_state)
        # Add site component to the signal
        X = X * site_effect
    elif site_effect_type.lower() in ["location+scale", "l+s"]:
        site_effect_location = _site_effect_value(site_effect_strength, site_effect_homogeneous, n_features, random_state)
        site_effect_scale = _site_effect_value(site_effect_strength, site_effect_homogeneous, n_features, random_state)
        X = (X + site_effect_location) * (site_effect_scale)
    else:
        raise ValueError(f"Unsupported site_effect_type: {site_effect_type}")

    return X, y


def _site_effect_value(
    site_effect_strength: float,
    site_effect_homogeneous: bool,
    n_features: int,
    random_state: np.random.RandomState,
) -> np.ndarray:
    """Generate site effect values for features.

    Parameters
    ----------
    site_effect_strength : float
        Magnitude of site effect. For homogeneous case, effects are uniformly
        distributed in [-site_effect_strength, site_effect_strength].
        For heterogeneous case, effects are normally distributed with
        scale = site_effect_strength.
    site_effect_homogeneous : bool
        If True, generates same effect for all features in this site.
        If False, generates different effect for each feature.
    n_features : int
        Number of features.
    random_state : RandomState instance
        The RandomState for reproducibility.

    Returns
    -------
    site_effect : np.ndarray of shape (1, n_features)
        Site effect values to be applied to features.

    """
    if site_effect_homogeneous:
        # Single uniform value replicated across all features
        strength = random_state.uniform(-site_effect_strength, site_effect_strength)
        site_effect = np.full((1, n_features), strength)
    else:
        # Different normal value for each feature
        site_effect = random_state.normal(0.0, site_effect_strength, (1, n_features))

    return site_effect


def _get_site_samples(
    X: npt.NDArray,
    y: npt.NDArray,
    balance: float | list[float],
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
    balance : float or list of float
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
    # Determine how many samples per class for this site
    if n_classes == 2:
        # For binary classification
        p_class1 = balance
        n_class1 = int(n_site_samples * p_class1)
        n_class0 = n_site_samples - n_class1
        samples_per_class = [n_class0, n_class1]
    else:
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
            logger.warning(
                f"Not enough samples of class {class_idx} in global dataset. "
                f"Requested {n_class_samples}, available {len(available_class_indices)}. "
                f"Sampling with replacement."
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
