"""Data simulation module for multi-site data generation."""

__all__ = [
    "make_multisite_classification",
]

import numpy as np


def make_multisite_classification(
    n_sites: int = 2,
    n_samples: int = 1000,
    balance_per_site: list[float] | None = None,
    n_features: int = 10,
    signal_strength: float = 1.0,
    noise_strength: float = 1.0,
    site_effect_strength: float = 3.0,
    site_effect_homogeneous: bool = True,
    n_classes: int = 2,
    random_state: int | None = 23,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate multi-site data with signal, noise, and site effect components.

    The data generation follows: X = signal + noise + site_effect
    All components are sampled from Gaussian distributions.

    Parameters
    ----------
    n_sites : int, default=2
        Number of sites to simulate
    n_samples : int, default=1000
        Total number of samples across all sites
    balance_per_site : list[float] or None, default=None
        Class balance for each site. If None, uses balanced classes (0.5 for
        binary, equal distribution for multi-class)
    n_features : int, default=10
        Number of features per sample
    signal_strength : float, default=1.0
        Strength of the signal component separating classes
    noise_strength : float, default=1.0
        Strength of the noise component
    site_effect_strength : float, default=3.0
        Strength of site-specific effects
    site_effect_homogeneous : bool, default=True
        Whether the site effect is homogeneous (same for all samples in a site)
    n_classes : int, default=2
        Number of classes to simulate (2 for binary, >2 for multi-class)
    random_state : int or None, default=23
        Random seed for reproducibility
    verbose : bool, default=False
        Whether to print progress information

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
        Simulated feature matrix
    y : np.ndarray of shape (n_samples,)
        Class labels (0 to n_classes-1)
    site_labels : np.ndarray of shape (n_samples,)
        Site labels (0 to n_sites-1)

    Examples
    --------
    >>> X, y, site_labels = simulate_multi_site_data(
    ...     n_sites=3, n_samples=300, n_features=20, n_classes=3
    ... )
    >>> X.shape, y.shape, site_labels.shape
    ((300, 20), (300,), (300,))

    """
    # Set random seed for reproducibility
    if random_state is not None:
        np.random.seed(random_state)

    # Validate input parameters
    _validate_parameters(
        n_sites=n_sites,
        n_samples=n_samples,
        n_features=n_features,
        signal_strength=signal_strength,
        noise_strength=noise_strength,
        site_effect_strength=site_effect_strength,
        n_classes=n_classes,
    )

    # Set default class balance if not provided
    if balance_per_site is None:
        balance_per_site = _get_default_balance_per_site(n_sites, n_classes)  # type: ignore
        if verbose:
            print(f"Using balanced classes: {balance_per_site}")

    _data_generation_parameter_checks(balance_per_site, n_sites, n_classes)

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

        if verbose:
            print(f"Generating {n_site_samples} samples for site {site_idx}")

        # Generate labels for this site
        if n_classes == 2:
            y_site = _generate_binary_labels(
                n_site_samples,
                balance_per_site[site_idx],  # type: ignore
            )
        else:
            y_site = _generate_multiclass_labels(
                n_site_samples,
                balance_per_site[site_idx],  # type: ignore
                n_classes,  # type: ignore
            )

        # Generate signal component based on class labels
        signal = _generate_signal_component(
            y_site, n_features, signal_strength, n_classes
        )

        # Generate noise component
        noise = np.random.normal(
            loc=0.0, scale=noise_strength, size=(n_site_samples, n_features)
        )

        if site_effect_homogeneous:
            # Generate site effect (same for all samples in this site)
            site_effect = np.random.normal(
                loc=0.0,
                scale=site_effect_strength,
                size=(
                    1,
                    n_features,
                ),  # Same effect for all samples in this site
            )
        else:
            # Generate site effect (different for each sample in this site)
            site_effect = np.random.normal(
                loc=0.0,
                scale=site_effect_strength,
                size=(n_site_samples, n_features),
            )

        # Combine components: X = signal + noise + site_effect
        X_site = signal + noise + site_effect

        X_list.append(X_site)
        y_list.append(y_site)
        site_labels_list.extend([site_idx] * n_site_samples)

    # Concatenate all sites
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    site_labels = np.array(site_labels_list, dtype=int)

    # Shuffle samples across sites (optional but recommended)
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    site_labels = site_labels[indices]

    if verbose:
        print(f"Generated {len(X)} samples across {n_sites} sites")
        print(f"Class distribution: {np.bincount(y)}")
        print(f"Site distribution: {np.bincount(site_labels)}")

    return X, y, site_labels


def _validate_parameters(
    n_sites: int,
    n_samples: int,
    n_features: int,
    signal_strength: float,
    noise_strength: float,
    site_effect_strength: float,
    n_classes: int,
) -> None:
    """Validate all input parameters for data simulation."""
    if n_sites < 2:
        raise ValueError(f"n_sites must be at least 2, got {n_sites}")

    if n_features <= 0:
        raise ValueError(f"n_features must be positive, got {n_features}")

    if n_classes < 2:
        raise ValueError(f"n_classes must be at least 2, got {n_classes}")

    if signal_strength < 0:
        raise ValueError(
            f"signal_strength must be non-negative, got {signal_strength}"
        )

    if noise_strength < 0:
        raise ValueError(
            f"noise_strength must be non-negative, got {noise_strength}"
        )

    if site_effect_strength < 0:
        raise ValueError(
            f"site_effect_strength must be non-negative, got {site_effect_strength}"  # noqa: E501
        )

    if n_samples < n_sites:
        raise ValueError(
            f"n_samples ({n_samples}) is less than n_sites ({n_sites}). "
            f"Some sites will have 0 samples.",
        )


def _get_default_balance_per_site(n_sites: int, n_classes: int):
    """Get default class balance for each site."""
    if n_classes == 2:
        # Binary: 0.5 for each site
        return [0.5] * n_sites
    else:
        # Multi-class: equal distribution for each site
        equal_prob = 1.0 / n_classes
        return [[equal_prob] * n_classes] * n_sites


def _generate_binary_labels(n_samples: int, p_class1: float) -> np.ndarray:
    """Generate binary labels (0 or 1) for given number of samples.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    p_class1 : float
        Probability of class 1

    Returns
    -------
    np.ndarray
        Binary labels

    """
    n_class1 = int(np.round(n_samples * p_class1))

    y = np.zeros(n_samples, dtype=int)
    y[:n_class1] = 1  # First n_class1 samples are class 1
    np.random.shuffle(y)  # Shuffle to randomize order

    return y


def _generate_multiclass_labels(
    n_samples: int, class_probs: list[float], n_classes: int
) -> np.ndarray:
    """Generate multi-class labels based on class probabilities.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    class_probs : list[float]
        Probability for each class (must sum to 1.0)
    n_classes : int
        Number of classes

    Returns
    -------
    np.ndarray
        Multi-class labels (0 to n_classes-1)

    """
    # Calculate number of samples per class
    samples_per_class = np.round(np.array(class_probs) * n_samples).astype(int)

    # Adjust to ensure total equals n_samples
    diff = n_samples - samples_per_class.sum()
    if diff != 0:
        samples_per_class[np.argmax(samples_per_class)] += diff

    # Generate labels
    y = []
    for class_idx in range(n_classes):
        y.extend([class_idx] * samples_per_class[class_idx])

    y = np.array(y, dtype=int)
    np.random.shuffle(y)

    return y


def _generate_signal_component(
    y: np.ndarray,
    n_features: int,
    signal_strength: float,
    n_classes: int,
) -> np.ndarray:
    """Generate signal component that separates classes.

    For binary classification: class 0 mean = -signal_strength/2,
                               class 1 mean = +signal_strength/2
    For multi-class: equally spaced means
        from -signal_strength/2 to +signal_strength/2

    Parameters
    ----------
    y : np.ndarray
        Class labels
    n_features : int
        Number of features
    signal_strength : float
        Strength of signal separation
    n_classes : int
        Number of classes

    Returns
    -------
    np.ndarray
        Signal component matrix

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
            signal[mask_class0] = np.random.normal(
                loc=mean_class0,
                scale=1.0,
                size=(np.sum(mask_class0), n_features),
            )

        if np.any(mask_class1):
            signal[mask_class1] = np.random.normal(
                loc=mean_class1,
                scale=1.0,
                size=(np.sum(mask_class1), n_features),
            )
    else:
        # Multi-class classification
        # Create equally spaced class means
        class_means = np.linspace(
            -signal_strength / 2, signal_strength / 2, n_classes
        )

        for class_idx in range(n_classes):
            mask = y == class_idx
            if np.any(mask):
                signal[mask] = np.random.normal(
                    loc=class_means[class_idx],
                    scale=1.0,
                    size=(np.sum(mask), n_features),
                )

    return signal


def _data_generation_parameter_checks(balance_per_site, n_sites, n_classes):
    """Validate balance_per_site structure."""
    # Validate balance_per_site structure
    if len(balance_per_site) != n_sites:
        raise ValueError(
            f"balance_per_site must have length n_sites ({n_sites}), "
            f"got {len(balance_per_site)}"
        )

    # For multi-class, balance_per_site should be a list of lists
    if n_classes > 2:
        for i, site_balance in enumerate(balance_per_site):
            if not isinstance(site_balance, (list, np.ndarray)):
                raise TypeError(
                    f"For n_classes > 2, balance_per_site[{i}] must be a list/array, "  # noqa: E501
                    f"got {type(site_balance)}"
                )
            if len(site_balance) != n_classes:
                raise ValueError(
                    f"balance_per_site[{i}] must have length n_classes ({n_classes}), "  # noqa: E501
                    f"got {len(site_balance)}"
                )
            if not np.isclose(sum(site_balance), 1.0, atol=1e-10):
                raise ValueError(
                    f"balance_per_site[{i}] must sum to 1.0, got {sum(site_balance):.6f}"  # noqa: E501
                )
    else:
        # For binary classification, validate proportions
        for i, p_class1 in enumerate(balance_per_site):
            if not 0 <= p_class1 <= 1:
                raise ValueError(
                    f"balance_per_site[{i}] must be between 0 and 1, got {p_class1}"  # noqa: E501
                )
