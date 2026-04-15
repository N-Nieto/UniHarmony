"""Data simulation module for multi-site data generation."""

import numpy as np
import structlog
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
    n_classes: int = 2,
    n_sites: int = 2,
    n_samples: int = 1000,
    balance_per_site: list[float] | list[list[float]] | None = None,
    n_features: int = 10,
    signal_strength: float = 1.0,
    noise_strength: float = 1.0,
    site_effect_strength: float = 3.0,
    site_effect_homogeneous: bool = True,
    random_state: int | np.random.RandomState = 42,
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
    signal_strength : float, optional (default 1.0)
        Strength of the signal component separating classes.
    noise_strength : float, optional (default 1.0)
        Strength of the noise component.
    site_effect_strength : float, optional (default 3.0)
        Strength of site-specific effects.
    site_effect_homogeneous : bool, optional (default True)
        Whether the site effect is homogeneous (same for all samples in a site).

    random_state : int or RandomState instance, (default 42)
        The seed of the pseudo random number generator or RandomState for
        reproducibility.

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
    >>> X, y, site_labels = make_multisite_classification(
    ...     n_sites=3, n_samples=300, n_features=20, n_classes=3
    ... )
    >>> X.shape, y.shape, site_labels.shape
    ((300, 20), (300,), (300,))

    """
    random_state = check_random_state(random_state)

    # Validate input parameters
    _validate_parameters(
        n_classes=n_classes,
        n_sites=n_sites,
        n_samples=n_samples,
        n_features=n_features,
        signal_strength=signal_strength,
        noise_strength=noise_strength,
        site_effect_strength=site_effect_strength,
    )

    balance_per_site = _validate_balance_per_site(balance_per_site, n_sites, n_classes)

    # Allocate samples per site (even distribution)
    samples_per_site = np.full(n_sites, n_samples // n_sites, dtype=int)
    samples_per_site[: n_samples % n_sites] += 1

    # Pre-allocate arrays for better performance
    X_list = []
    y_list = []
    site_labels_list = []

    # Generate data for each site
    for site_idx in range(n_sites):
        n_site_samples = samples_per_site[site_idx]

        logger.info(f"Generating {n_site_samples} samples for site {site_idx}")

        balance = balance_per_site[site_idx]
        y_site = _generate_labels(n_classes, n_site_samples, balance, random_state)

        # Generate signal component based on class labels
        signal = _generate_signal_component(
            y_site,
            n_features,
            signal_strength,
            n_classes,
            random_state,
        )

        # Generate noise component
        noise = random_state.normal(loc=0.0, scale=noise_strength, size=(n_site_samples, n_features))

        # Generate site effect component
        site_effect = _generate_site_effect_component(site_effect_homogeneous, site_effect_strength, n_features, random_state)

        # Combine components: X = signal + noise + site_effect
        X_site = signal + noise + site_effect

        X_list.append(X_site)
        y_list.append(y_site)
        site_labels_list.extend([site_idx] * n_site_samples)
        logger.debug(f"Site {site_idx}, site effect strength {site_effect}")

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


def _generate_labels(
    n_classes: int,
    n_site_samples: int,
    balance: float | list[float],
    random_state: np.random.RandomState,
) -> np.ndarray:
    """Generate labels for a single site.

    Parameters
    ----------
    n_classes : int
        Number of classes (2 for binary, >2 for multiclass).
    n_site_samples : int
        Number of samples in this site.
    balance : float or list[float].
        Class balance for a given site. For binary classification, this is the
        probability of class 1. For multiclass, this is a list of probabilities
        for each class (must sum to 1.0).
    site_idx : int
        Index of the current site.
    random_state : RandomState instance
        The RandomState for reproducibility.

    Returns
    -------
    np.ndarray
        Generated labels of shape (n_site_samples,).

    """
    if n_classes == 2:
        return _generate_binary_labels(n_site_samples, balance, random_state)
    else:
        return _generate_multiclass_labels(n_site_samples, balance, n_classes, random_state)


def _generate_binary_labels(
    n_samples: int,
    p_class_1: float,
    random_state: np.random.RandomState,
) -> np.ndarray:
    """Generate binary labels (0 or 1) for given number of samples.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    p_class_1 : float
        Probability of class 1.
    random_state : RandomState instance
        The RandomState for reproducibility.

    Returns
    -------
    np.ndarray
        Binary labels.

    """
    n_class_1 = int(np.round(n_samples * p_class_1))

    y = np.zeros(n_samples, dtype=int)
    y[:n_class_1] = 1  # First n_class_1 samples are class 1
    random_state.shuffle(y)  # Shuffle to randomize order

    return y


def _generate_multiclass_labels(
    n_samples: int,
    class_probs: list[float],
    n_classes: int,
    random_state: np.random.RandomState,
) -> np.ndarray:
    """Generate multi-class labels based on class probabilities.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    class_probs : list of float
        Probability for each class (must sum to 1.0).
    n_classes : int
        Number of classes.
    random_state : RandomState instance
        The RandomState for reproducibility.

    Returns
    -------
    np.ndarray
        Multi-class labels (0 to ``n_classes``-1).

    """
    # Using multinomial for cleaner and potentially faster generation
    samples_per_class = random_state.multinomial(n_samples, class_probs)

    # Handle rounding issues by adjusting the largest class if needed
    diff = n_samples - samples_per_class.sum()
    if diff != 0:
        samples_per_class[np.argmax(samples_per_class)] += diff

    # Create labels using repeat (more efficient than loop with extend)
    y = np.repeat(np.arange(n_classes), samples_per_class)
    random_state.shuffle(y)

    return y


def _generate_signal_component(
    y: np.ndarray,
    n_features: int,
    signal_strength: float,
    n_classes: int,
    random_state: np.random.RandomState,
) -> np.ndarray:
    """Generate signal component that separates classes.

    For binary classification: class 0 mean = -signal_strength/2,
                               class 1 mean = +signal_strength/2
    For multi-class: equally spaced means
        from -signal_strength/2 to +signal_strength/2

    Parameters
    ----------
    y : np.ndarray
        Class labels.
    n_features : int
        Number of features.
    signal_strength : float
        Strength of signal separation.
    n_classes : int
        Number of classes.
    random_state : RandomState instance
        The RandomState for reproducibility.

    Returns
    -------
    np.ndarray
        Signal component matrix.

    """
    n_samples = len(y)
    signal = np.zeros((n_samples, n_features))

    if n_classes == 2:
        # Binary classification
        mean_class0 = -signal_strength / 2
        mean_class1 = signal_strength / 2

        mask_class0 = y == 0
        mask_class1 = y == 1

        if np.any(mask_class0):
            signal[mask_class0] = random_state.normal(
                loc=mean_class0,
                scale=1.0,
                size=(np.sum(mask_class0), n_features),
            )

        if np.any(mask_class1):
            signal[mask_class1] = random_state.normal(
                loc=mean_class1,
                scale=1.0,
                size=(np.sum(mask_class1), n_features),
            )
    else:
        # Multi-class classification
        # Create equally spaced class means
        class_means = np.linspace(-signal_strength / 2, signal_strength / 2, n_classes)

        for class_idx in range(n_classes):
            mask = y == class_idx
            if np.any(mask):
                signal[mask] = random_state.normal(
                    loc=class_means[class_idx],
                    scale=1.0,
                    size=(np.sum(mask), n_features),
                )

    return signal


def _validate_balance_per_site(
    balance_per_site: list | list[list] | None,
    n_sites: int,
    n_classes: int,
) -> list | list[list]:
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

    return balance_per_site


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
                raise TypeError(f"balance_per_site[{i}][{j}] must be a class proprtion (int or float), got {type(class_prob)}")
            if not 0 <= class_prob <= 1:
                raise ValueError(f"balance_per_site[{i}] must be between 0 and 1, got {class_prob}")

        # Convert to numpy array for sum check
        site_balance_array = np.array(site_balance, dtype=float)

        # Check sum is approximately 1
        if not np.isclose(np.sum(site_balance_array), 1.0, atol=1e-10):
            raise ValueError(f"balance_per_site[{i}] must sum to 1.0, got {np.sum(site_balance_array):.6f}")


def _generate_site_effect_component(
    site_effect_homogeneous: bool,
    site_effect_strength: float,
    n_features: int,
    random_state: np.random.RandomState,
) -> np.ndarray:
    """Generate site effect component for features.

    Parameters
    ----------
    site_effect_homogeneous : bool
        If True, generates same effect for all features in this site.
        If False, generates different effect for each feature.
    site_effect_strength : float
        Magnitude of site effect. For homogeneous case, effects are uniformly
        distributed in [-site_effect_strength, site_effect_strength].
        For heterogeneous case, effects are normally distributed with
        scale = site_effect_strength.
    n_features : int
        Number of features.
    random_state : RandomState instance
        The RandomState for reproducibility.

    Returns
    -------
    np.ndarray
        Site effect component of shape (1, n_features).

    """
    if site_effect_homogeneous:
        # Single uniform value replicated across all features
        strength = random_state.uniform(-site_effect_strength, site_effect_strength)
        return np.full((1, n_features), strength)
    else:
        # Different normal value for each feature
        return random_state.normal(0.0, site_effect_strength, (1, n_features))
