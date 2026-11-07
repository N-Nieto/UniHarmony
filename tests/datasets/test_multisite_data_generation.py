"""Test module for make_multisite_classification function."""

import numpy as np
import pytest

from uniharmony.datasets import make_multisite_classification


def test_basic_functionality() -> None:
    """Test basic functionality with default parameters."""
    X, y, sites = make_multisite_classification(n_sites=2, n_samples=100, n_features=10, random_state=42)

    assert X.shape == (100, 10), f"Expected X shape (100, 10), got {X.shape}"
    assert y.shape == (100,), f"Expected y shape (100,), got {y.shape}"
    assert sites.shape == (100,), f"Expected sites shape (100,), got {sites.shape}"

    # Check labels are binary
    assert set(y) == {0, 1}, f"Expected binary labels, got {np.unique(y)}"

    # Check site labels
    assert set(sites) == {0, 1}, f"Expected sites {0, 1}, got {np.unique(sites)}"


def test_sample_distribution() -> None:
    """Test that samples are properly distributed across sites."""
    n_sites = 3
    n_samples = 100
    X, _, sites = make_multisite_classification(n_sites=n_sites, n_samples=n_samples, random_state=42)

    # Check total samples
    assert len(X) == n_samples, f"Expected {n_samples} samples, got {len(X)}"

    # Check samples per site (should be roughly equal)
    site_counts = np.bincount(sites)
    expected_per_site = n_samples // n_sites

    # Allow for rounding differences
    assert all(abs(count - expected_per_site) <= 1 for count in site_counts), f"Uneven site distribution: {site_counts}"


def test_class_balance() -> None:
    """Test custom class balance."""
    balance_per_site = [0.3, 0.7]  # 30% class 1 in site 0, 70% in site 1
    _, y, sites = make_multisite_classification(
        n_sites=2,
        n_samples=200,
        balance_per_site=balance_per_site,
        random_state=42,
    )

    for site_idx in [0, 1]:
        site_mask = sites == site_idx
        y_site = y[site_mask]
        if len(y_site) > 0:
            p_class1 = np.mean(y_site == 1)
            expected = balance_per_site[1]
            # Allow 5% tolerance due to random sampling
            assert abs(p_class1 - expected) < 0.05, f"Site {site_idx}: expected {expected}, got {p_class1}"


def test_multiclass_functionality() -> None:
    """Test multi-class functionality."""
    n_classes = 3
    _, y, _ = make_multisite_classification(n_sites=2, n_samples=150, n_classes=n_classes, random_state=42)

    # Check all classes are present
    unique_classes = np.unique(y)
    assert len(unique_classes) == n_classes, f"Expected {n_classes} classes, got {len(unique_classes)}"
    assert set(unique_classes) == set(range(n_classes)), f"Expected classes {set(range(n_classes))}, got {set(unique_classes)}"


def test_multiclass_balance() -> None:
    """Test multi-class with custom balance."""
    balance_per_site = [
        [0.2, 0.3, 0.5],  # Site 0: 20% class 0, 30% class 1, 50% class 2
        [0.4, 0.4, 0.2],  # Site 1: 40% class 0, 40% class 1, 20% class 2
    ]

    _, y, sites = make_multisite_classification(
        n_sites=2,
        n_samples=200,
        n_classes=3,
        balance_per_site=balance_per_site,
        random_state=42,
    )

    for site_idx, expected_probs in enumerate(balance_per_site):
        site_mask = sites == site_idx
        y_site = y[site_mask]

        if len(y_site) > 0:
            for class_idx in range(3):
                actual_prob = np.mean(y_site == class_idx)
                expected_prob = expected_probs[class_idx]
                # Allow 5% tolerance
                assert np.isclose(actual_prob, expected_prob, rtol=0.5), (
                    f"Site {site_idx}, class {class_idx}: expected {expected_prob}, got {actual_prob}"
                )


def test_signal_strength() -> None:
    """Test that signal strength affects class separation."""
    # Generate data with different signal strengths
    X_weak, y_weak, _ = make_multisite_classification(n_sites=2, n_samples=100, signal={"linear": 0.5}, random_state=42)
    X_strong, y_strong, _ = make_multisite_classification(n_sites=2, n_samples=100, signal={"linear": 2.0}, random_state=42)

    # Calculate mean difference between classes for each feature
    def class_separation(X, y):
        mean_class0 = X[y == 0].mean(axis=0)
        mean_class1 = X[y == 1].mean(axis=0)
        return np.abs(mean_class1 - mean_class0).mean()

    sep_weak = class_separation(X_weak, y_weak)
    sep_strong = class_separation(X_strong, y_strong)

    # Strong signal should have larger separation
    assert sep_strong > sep_weak, "Expected stronger separation with larger signal_strength"


def test_site_effect() -> None:
    """Test that site effect creates differences between sites."""
    X, _, sites = make_multisite_classification(n_sites=2, n_samples=100, site_effect={"location": 5.0}, random_state=42)

    # Calculate mean difference between sites for each feature
    mean_site0 = X[sites == 0].mean(axis=0)
    mean_site1 = X[sites == 1].mean(axis=0)
    site_difference = np.abs(mean_site1 - mean_site0).mean()

    # Site effect should create noticeable differences
    assert site_difference > 1.0, "Expected larger site differences with site_effect_strength=5.0"


def test_reproducibility() -> None:
    """Test that random_state ensures reproducible results."""
    X1, y1, sites1 = make_multisite_classification(random_state=42)
    X2, y2, sites2 = make_multisite_classification(random_state=42)
    X3, _, _ = make_multisite_classification(random_state=43)  # Different seed

    # Same seed should give identical results
    assert np.array_equal(X1, X2), "Results with same seed should be identical"
    assert np.array_equal(y1, y2), "Results with same seed should be identical"
    assert np.array_equal(sites1, sites2), "Results with same seed should be identical"

    # Different seed should give different results (very high probability)
    assert not np.array_equal(X1, X3), "Results with different seeds should differ"


def test_edge_cases() -> None:
    """Test edge cases and error handling."""
    # Test with very few samples
    X, _, _ = make_multisite_classification(n_samples=10, n_sites=3, random_state=42)
    assert len(X) == 10, f"Expected 10 samples, got {len(X)}"

    # Test single feature
    X, _, _ = make_multisite_classification(n_features=1, random_state=42)
    assert X.shape[1] == 1, f"Expected 1 feature, got {X.shape[1]}"

    make_multisite_classification(n_sites=1)

    # Test invalid parameters
    with pytest.raises(ValueError):
        make_multisite_classification(n_sites=0)

    with pytest.raises(ValueError):
        make_multisite_classification(n_classes=1)

    with pytest.raises(ValueError):
        make_multisite_classification(balance_per_site=[0.5, 0.5, 0.5])  # Wrong length
    with pytest.raises(ValueError):
        make_multisite_classification(n_features=0)  # Wrong number of features
    with pytest.raises(ValueError):
        make_multisite_classification(signal={"linear": -1})  # Wrong signal strength
    with pytest.raises(ValueError):
        make_multisite_classification(noise_strength=-1)  # Wrong noise strength
    with pytest.raises(ValueError):
        make_multisite_classification(site_effect={"location": -1})  # Wrong Effect of Site strength
    with pytest.raises(ValueError):
        make_multisite_classification(n_samples=2, n_sites=4)  # Wrong site-samples


def test_balance_combinations_multiclass() -> None:
    """Test invalid parameter for multiclass classification combinations."""
    # Wrong length
    with pytest.raises(ValueError):
        make_multisite_classification(n_classes=2, balance_per_site=[0.1, 0.1, 0.2])
    # Wrong Type
    with pytest.raises(TypeError):
        make_multisite_classification(
            n_classes=4,
            n_sites=4,
            balance_per_site=[None, None, None, None],
        )
    # Not summing 1
    with pytest.raises(ValueError):
        make_multisite_classification(n_classes=4, n_sites=4, balance_per_site=[0.1, 0.1, 0.1, 0.1])
    # Wrong combination,
    # one list has one element more than the number of classes
    with pytest.raises(ValueError):
        balance_per_site = [
            [
                0.2,
                0.3,
                0.9,
                0.2,
            ],
            [0.4, 0.4, 0.2],
            [0.4, 0.4, 0.2],
            [0.4, 0.4, 0.2],
        ]
        make_multisite_classification(
            n_classes=3,
            n_sites=4,
            balance_per_site=balance_per_site,
        )  # Wrong site-samples

    # Proportion do not sum 1
    with pytest.raises(ValueError):
        balance_per_site = [
            [0.2, 0.3, 0.1],
            [0.4, 0.4, 0.2],
            [0.4, 0.4, 0.2],
            [0.4, 0.4, 0.2],
        ]
        make_multisite_classification(
            n_classes=3,
            n_sites=4,
            balance_per_site=balance_per_site,
        )
    # Wrong type, string not accepted
    with pytest.raises(TypeError):
        balance_per_site = [
            [0.2, 0.3, "0.1"],
            [0.4, 0.4, 0.2],
            [0.4, 0.4, 0.2],
            [0.4, 0.4, 0.2],
        ]
        make_multisite_classification(
            n_classes=3,
            n_sites=4,
            balance_per_site=balance_per_site,
        )
    # All elements must be lower than 1
    with pytest.raises(ValueError):
        balance_per_site = [
            [1.1, 0.3, 0.1],
            [1.1, 0.4, 0.2],
            [11.1, 0.4, 0.2],
            [0.4, 0.4, 0.2],
        ]
        make_multisite_classification(
            n_classes=3,
            n_sites=4,
            balance_per_site=balance_per_site,
        )
    # No negative elements allowed
    with pytest.raises(ValueError):
        balance_per_site = [
            [0.2, 0.3, -0.1],
            [0.4, 0.4, 0.2],
            [0.4, 0.4, 0.2],
            [0.4, 0.4, 0.2],
        ]
        make_multisite_classification(
            n_classes=3,
            n_sites=4,
            balance_per_site=balance_per_site,
        )


def test_uneven_match_samples() -> None:
    """Test uneven_samples."""
    # n_samples should be stable, even with not exactly matched
    # probabilities due to rounding problems.
    n_samples = 100
    _, y, _ = make_multisite_classification(
        n_samples=100,
        n_classes=9,
        n_sites=4,
    )
    assert n_samples == len(y)


def test_balance_combinations_binary() -> None:
    """Test balance parameters for binary classification."""
    # Wrong type of arguments
    with pytest.raises(TypeError):
        balance_per_site = "Wrong argument"
        make_multisite_classification(
            n_classes=2,
            n_sites=4,
            balance_per_site=balance_per_site,
        )  # Wrong site-samples
    # Right number, wrong type of arguments
    with pytest.raises(TypeError):
        balance_per_site = ["Wrong", "argument", "1", "0"]
        make_multisite_classification(
            n_classes=2,
            n_sites=4,
            balance_per_site=balance_per_site,
        )
    # Not passing proportions
    with pytest.raises(ValueError):
        balance_per_site = [11.11, 11.11, 0, 0]
        make_multisite_classification(
            n_classes=2,
            n_sites=4,
            balance_per_site=balance_per_site,
        )
    # Integer proportions are accepted, but these values are invalid
    with pytest.raises(ValueError):
        balance_per_site = [
            [1, 1],  # Site 0: 20% class 0, 30% class 1, 50% class 2
            [11, 0],  # Site 1: 40% class 0, 40% class 1, 20% class 2
            [-1, 0.2],  # Site 1: 40% class 0, 40% class 1, 20% class 2
            [0.4, 0.4],  # Site 1: 40% class 0, 40% class 1, 20% class 2
        ]
        make_multisite_classification(
            n_classes=2,
            n_sites=4,
            balance_per_site=balance_per_site,
        )  # Wrong site-samples


def test_heterogeneous_site_effect() -> None:
    """Heterogeneous site effects run with few samples."""
    for heterogeneity in (0.0, 0.5):
        X, _, _ = make_multisite_classification(n_features=2, n_samples=20, site_effect_heterogeneity=heterogeneity)
        assert np.isfinite(X).all()


@pytest.mark.parametrize(
    "n_sites, n_samples, n_features, n_classes, signal_type, site_effect_type, "
    "site_effect_heterogeneity, balance_per_site, noise_strength, signal_strength, "
    "site_effect_strength",
    [
        # Binary classification tests
        (2, 200, 10, 2, "linear", "location", 0.0, None, 0.1, 1.0, 1.0),
        (3, 300, 2, 2, "circles", "scale", 0.5, None, 0.0, 1.0, 2.0),
        (2, 150, 2, 2, "moons", "combat", 0.0, [0.3, 0.7], 0.2, 1.0, 1.5),
        (4, 400, 12, 2, "blobs", "location", 0.5, [0.2, 0.8], 0.15, 1.0, 2.0),
        (2, 250, 8, 2, "gaussian_quantiles", "scale", 0.0, None, 0.05, 0.8, 3.0),
        # Multi-class classification tests
        (2, 300, 15, 3, "linear", "location", 0.0, None, 0.1, 1.0, 1.0),
        (3, 450, 20, 4, "blobs", "scale", 0.5, None, 0.0, 1.2, 2.5),
        (
            2,
            200,
            10,
            5,
            "linear",
            "combat",
            0.0,
            [[0.3, 0.2, 0.2, 0.15, 0.15], [0.1, 0.3, 0.3, 0.15, 0.15]],
            0.2,
            1.0,
            1.0,
        ),
        (3, 600, 25, 6, "gaussian_quantiles", "location", 0.5, None, 0.1, 1.5, 2.0),
        # Edge cases
        (2, 100, 5, 2, "linear", "location", 0.0, [0.5, 0.5], 0.0, 0.0, 0.0),  # No signal, no site effect
        (5, 500, 20, 3, "blobs", "scale", 0.0, None, 0.2, 1.0, 0.0),  # No site effect
    ],
    ids=[
        "binary_linear_location_homogeneous_default_balance",
        "binary_circles_scale_heterogeneous_default_balance_no_noise",
        "binary_moons_combat_homogeneous_custom_balance",
        "binary_blobs_location_heterogeneous_custom_balance",
        "binary_gaussian_quantiles_scale_homogeneous_default_balance",
        "multiclass_linear_location_homogeneous_default_balance",
        "multiclass_blobs_scale_heterogeneous_default_balance_no_noise",
        "multiclass_linear_combat_homogeneous_custom_balance",
        "multiclass_gaussian_quantiles_location_heterogeneous_default_balance",
        "edge_no_signal_no_site_effect",
        "edge_no_site_effect",
    ],
)
def test_make_multisite_classification_parametrized(
    n_sites,
    n_samples,
    n_features,
    n_classes,
    signal_type,
    site_effect_type,
    site_effect_heterogeneity,
    balance_per_site,
    noise_strength,
    signal_strength,
    site_effect_strength,
):
    """Test make_multisite_classification with various parameter combinations."""
    # Generate data
    X, y, sites = make_multisite_classification(
        n_sites=n_sites,
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        balance_per_site=balance_per_site,
        signal={signal_type: signal_strength},
        site_effect={site_effect_type: site_effect_strength},
        noise_strength=noise_strength,
        site_effect_heterogeneity=site_effect_heterogeneity,
        random_state=42,
    )

    # Basic shape and type checks
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(sites, np.ndarray)
    assert X.shape == (n_samples, n_features)
    assert y.shape == (n_samples,)
    assert sites.shape == (n_samples,)

    # Data type checks
    assert X.dtype == np.float64
    assert y.dtype == np.int64 or y.dtype == np.int32
    assert sites.dtype == np.int64 or sites.dtype == np.int32

    # Label range checks
    assert np.all(np.unique(y) == np.arange(n_classes))
    assert np.all(np.unique(sites) == np.arange(n_sites))

    # Class distribution checks
    class_counts = np.bincount(y)
    assert len(class_counts) == n_classes
    assert np.sum(class_counts) == n_samples
    assert np.all(class_counts > 0)  # Each class should have at least one sample

    # Site distribution checks
    site_counts = np.bincount(sites)
    assert len(site_counts) == n_sites
    assert np.sum(site_counts) == n_samples
    assert np.all(site_counts > 0)  # Each site should have at least one sample

    # Check if site effect is correctly applied
    if site_effect_strength > 0:
        # Calculate within-site and between-site variance
        site_means = []
        for site_idx in range(n_sites):
            site_mask = sites == site_idx
            site_means.append(np.mean(X[site_mask], axis=0))

        site_means = np.array(site_means)
        if n_sites > 1:
            # Between-site variance should be larger than within-site variance
            # when site effect is present (unless noise is very high)
            between_var = np.var(site_means, axis=0).mean()

            if signal_strength > 0 and noise_strength < 1.0:
                # This is a soft check - site effect might be subtle
                assert between_var > 0, "Between-site variance should be positive when site effect > 0"

    # Check balance_per_site constraints if provided
    if balance_per_site is not None:
        for site_idx in range(n_sites):
            site_mask = sites == site_idx
            site_y = y[site_mask]
            site_class_counts = np.bincount(site_y, minlength=n_classes)
            site_proportions = site_class_counts / len(site_y)

            if n_classes == 2:
                expected_p_class1 = balance_per_site[1]
                # Allow for rounding errors due to integer sampling
                np.testing.assert_almost_equal(
                    site_proportions[1], expected_p_class1, decimal=1, err_msg=f"Site {site_idx} class 1 proportion mismatch"
                )
            else:
                expected_proportions = np.array(balance_per_site[site_idx])
                # Allow for rounding errors
                np.testing.assert_almost_equal(
                    site_proportions, expected_proportions, decimal=1, err_msg=f"Site {site_idx} class proportions mismatch"
                )


@pytest.mark.parametrize("signal_type", ["linear", "circles", "moons", "blobs", "gaussian_quantiles"])
def test_all_signal_types(signal_type):
    """Test that all signal types work without errors."""
    if signal_type in ["moons", "circles"]:
        X, y, sites = make_multisite_classification(
            n_sites=2,
            n_samples=200,
            n_features=2,
            n_classes=2,
            signal={signal_type: 1.0},
            random_state=42,
        )

        assert X.shape == (200, 2)
        assert y.shape == (200,)
        assert sites.shape == (200,)
    else:
        X, y, sites = make_multisite_classification(
            n_sites=2,
            n_samples=200,
            n_features=10,
            n_classes=2,
            signal={signal_type: 1.0},
            random_state=42,
        )

        assert X.shape == (200, 10)
        assert y.shape == (200,)
        assert sites.shape == (200,)


@pytest.mark.parametrize(
    "site_effect_type", ["location", "scale", "variance", "nonlinear", "dropout", "combat", "class_interaction"]
)
def test_all_site_effect_types(site_effect_type):
    """Test that all site effect types work without errors."""
    X, y, sites = make_multisite_classification(
        n_sites=2,
        n_samples=200,
        n_features=10,
        n_classes=2,
        site_effect={site_effect_type: {"value": 0.0, "prob": 0.2} if site_effect_type == "dropout" else 2.0},
        random_state=42,
    )

    assert X.shape == (200, 10)
    assert y.shape == (200,)
    assert sites.shape == (200,)


@pytest.mark.parametrize(
    "n_sites, n_samples, n_classes",
    [
        (2, 100, 2),
        (3, 300, 3),
        (4, 400, 4),
        (5, 500, 5),
    ],
)
def test_different_sites_samples_classes(n_sites, n_samples, n_classes):
    """Test different combinations of sites, samples, and classes."""
    X, y, sites = make_multisite_classification(
        n_sites=n_sites,
        n_samples=n_samples,
        n_features=10,
        n_classes=n_classes,
        random_state=42,
    )

    assert X.shape == (n_samples, 10)
    assert np.unique(sites).shape[0] == n_sites
    assert np.unique(y).shape[0] == n_classes


@pytest.mark.parametrize(
    "balance_per_site, n_classes",
    [
        ([0.1, 0.9], 2),
        ([0.3, 0.7], 2),
        ([0.5, 0.5], 2),
        ([[0.6, 0.2, 0.2], [0.2, 0.6, 0.2]], 3),
        ([[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1]], 4),
    ],
)
def test_custom_balance_per_site(balance_per_site, n_classes):
    """Test custom class balance per site."""
    n_sites = len(balance_per_site)
    n_samples = 500

    _, y, sites = make_multisite_classification(
        n_sites=n_sites,
        n_samples=n_samples,
        n_features=10,
        n_classes=n_classes,
        balance_per_site=balance_per_site,
        random_state=42,
    )

    # Check each site's class distribution
    for site_idx in range(n_sites):
        site_mask = sites == site_idx
        site_y = y[site_mask]
        site_counts = np.bincount(site_y, minlength=n_classes)
        site_proportions = site_counts / len(site_y)

        if n_classes == 2:
            expected = balance_per_site[site_idx]
            np.testing.assert_almost_equal(site_proportions[site_idx], expected, decimal=1)
        else:
            expected = np.array(balance_per_site[site_idx])
            np.testing.assert_almost_equal(site_proportions, expected, decimal=1)


def test_invalid_parameters():
    """Test that invalid parameters raise appropriate errors."""
    with pytest.raises(ValueError, match="n_sites must be at least 1"):
        make_multisite_classification(n_sites=0, n_samples=100, n_features=10)

    with pytest.raises(ValueError, match="n_features must be positive"):
        make_multisite_classification(n_sites=2, n_samples=100, n_features=0)

    with pytest.raises(ValueError, match="n_classes must be at least 2"):
        make_multisite_classification(n_sites=2, n_samples=100, n_classes=1)

    with pytest.raises(ValueError, match=r"signal\['linear'\] must be non-negative"):
        make_multisite_classification(n_sites=2, n_samples=100, signal={"linear": -1.0})

    with pytest.raises(ValueError, match=r"site_effect\['location'\] must be non-negative"):
        make_multisite_classification(n_sites=2, n_samples=100, site_effect={"location": -1.0})

    with pytest.raises(ValueError, match="Unsupported signal type: 'invalid'"):
        make_multisite_classification(n_sites=2, n_samples=100, signal={"invalid": 1.0})

    with pytest.raises(ValueError, match="Unsupported site effect type: 'invalid'"):
        make_multisite_classification(n_sites=2, n_samples=100, site_effect={"invalid": 3.0})


def test_noise_strength_zero():
    """Test that noise_strength=0 produces no additional noise."""
    X_noisy, _, _ = make_multisite_classification(
        n_sites=2,
        n_samples=20,
        n_features=10,
        n_classes=2,
        noise_strength=0.1,
        random_state=42,
    )

    X_clean, _, _ = make_multisite_classification(
        n_sites=2,
        n_samples=20,
        n_features=10,
        n_classes=2,
        noise_strength=0.0,
        random_state=42,
    )

    # Clean and noisy versions should be different
    assert not np.allclose(X_noisy, X_clean)

    # The difference should be only due to noise (not signal or site effect)
    # since they use the same random_state
    noise = X_noisy - X_clean
    assert np.abs(noise).mean() > 0


def test_site_effect_homogeneous_vs_heterogeneous():
    """Heterogeneity changes the data; without it every sample of a site gets the same offset."""
    params = {"n_sites": 2, "n_samples": 200, "noise_strength": 0, "site_effect": {"location": 2.0}, "return_ground_truth": True}
    *_, homogeneous = make_multisite_classification(**params)
    *_, heterogeneous = make_multisite_classification(site_effect_heterogeneity=0.5, **params)
    np.testing.assert_array_equal(homogeneous["intensity"], 1.0)
    assert not np.allclose(homogeneous["site_effect"], heterogeneous["site_effect"])


@pytest.mark.parametrize(
    ["site_effect_strength", "n_sites"],
    [(1, 2), (1.1, 4), ([1, 2], 2), ([1.4, 1.6, 5.4], 3)],
    ids=["pass integer", "pass float", "pass int list", "pass float list"],
)
def test_site_effect_strength_validation(site_effect_strength, n_sites):
    """Test site_effect_strength options and validations."""
    _, _, _ = make_multisite_classification(
        n_sites=n_sites,
        n_samples=200,
        n_features=10,
        n_classes=2,
        site_effect={"location": site_effect_strength},
        random_state=42,
    )


@pytest.mark.parametrize(
    ["noise_strength", "n_sites"],
    [(1, 2), (1.1, 4), ([1, 2], 2), ([1.4, 1.6, 5.4], 3)],
    ids=["pass integer", "pass float", "pass int list", "pass float list"],
)
def test_noise_strength_validation(noise_strength, n_sites):
    """Test noise_strength options and validations."""
    _, _, _ = make_multisite_classification(n_sites=n_sites, n_samples=100, noise_strength=noise_strength)


def test_noise_strength_invalid():
    """Test noise_strength invalid options."""
    with pytest.raises(TypeError, match="Invalid type for noise_strength"):
        _, _, _ = make_multisite_classification(n_sites=2, n_samples=20, noise_strength="invalid")
    with pytest.raises(TypeError, match="Invalid type for noise_strength"):
        _, _, _ = make_multisite_classification(n_sites=2, n_samples=20, noise_strength=["invalid", "invalid"])
    with pytest.raises(ValueError, match="noise_strength must have length n_sites"):
        _, _, _ = make_multisite_classification(n_sites=2, n_samples=20, noise_strength=[1, 2, 3])


def test_site_effect_strength_invalid():
    """Test site_effect_strength invalid options."""
    with pytest.raises(TypeError, match=r"Invalid type for site_effect\['location'\]"):
        _, _, _ = make_multisite_classification(n_sites=2, n_samples=20, site_effect={"location": "invalid"})
    with pytest.raises(TypeError, match=r"Invalid type for site_effect\['location'\]"):
        _, _, _ = make_multisite_classification(n_sites=2, n_samples=20, site_effect={"location": ["invalid", "invalid"]})
    with pytest.raises(ValueError, match=r"site_effect\['location'\] must have length n_sites"):
        _, _, _ = make_multisite_classification(n_sites=2, n_samples=20, site_effect={"location": [1, 2, 3]})
    with pytest.raises(ValueError, match=r"site_effect\['location'\]\[0\] must be non-negative"):
        _, _, _ = make_multisite_classification(n_sites=2, n_samples=20, site_effect={"location": [-1, 2]})


def test_invalid_moons():
    """Test moons with invalid options."""
    X, _, _ = make_multisite_classification(n_features=10, signal={"moons": 1.0})

    assert X.shape[1] == 2

    _, _, _ = make_multisite_classification(n_classes=3, signal={"moons": 1.0})

    assert X.shape[1] == 2


# --------------------------------------------------------------------------- #
# Regression tests for the 0.0.5 simulator fixes
# --------------------------------------------------------------------------- #


def test_old_gaussian_quantiles_name_is_rejected():
    """Only the documented name is accepted (no deprecated alias before 0.1)."""
    X, _, _ = make_multisite_classification(n_samples=100, signal={"gaussian_quantiles": 1.0})
    assert X.shape == (100, 10)
    with pytest.raises(ValueError, match="Unsupported signal type"):
        make_multisite_classification(n_samples=100, signal={"make_gaussian_quantiles": 1.0})


def _truth(**kwargs):
    *out, truth = make_multisite_classification(return_ground_truth=True, **kwargs)
    return (*out, truth)


def test_signal_is_row_aligned():
    """X - signal is exactly the EoS of each sample's site (no noise here)."""
    X, _, sites, truth = _truth(n_sites=3, n_samples=300, noise_strength=0)
    assert truth["signal"].shape == X.shape
    for site in range(3):
        offsets = X[sites == site] - truth["signal"][sites == site]
        np.testing.assert_allclose(offsets, offsets[[0]].repeat(len(offsets), axis=0))


@pytest.mark.parametrize("signal_type", ["linear", "blobs", "gaussian_quantiles", "moons", "circles"])
def test_no_base_sample_is_used_twice(signal_type):
    """Strong class imbalance and unequal site sizes never reuse a base sample."""
    _, y, _, truth = _truth(n_sites=2, n_samples=[100, 900], balance_per_site=[[0.0, 1.0], [1.0, 0.0]], signal={signal_type: 1.0})
    assert len(np.unique(truth["signal"], axis=0)) == len(truth["signal"])
    np.testing.assert_array_equal(np.bincount(y), [900, 100])


def test_class_counts_use_largest_remainder():
    """A class with proportion 0 gets no samples; counts follow the proportions."""
    _, y, sites = make_multisite_classification(
        n_sites=2, n_samples=[10, 7], n_classes=3, balance_per_site=[[0.0, 0.33, 0.67], [0.5, 0.25, 0.25]]
    )
    np.testing.assert_array_equal(np.bincount(y[sites == 0], minlength=3), [0, 3, 7])
    np.testing.assert_array_equal(np.bincount(y[sites == 1], minlength=3), [3, 2, 2])  # 3.5, 1.75, 1.75


def test_requested_class_is_never_rounded_away():
    """A class with a small non-zero proportion gets at least one sample in that site."""
    _, y, sites = make_multisite_classification(n_sites=2, n_samples=[50, 50], balance_per_site=[[0.995, 0.005], [0.5, 0.5]])
    np.testing.assert_array_equal(np.bincount(y[sites == 0], minlength=2), [49, 1])


def test_site_cannot_hold_its_classes():
    """A site with fewer samples than requested classes raises."""
    with pytest.raises(ValueError, match="cannot hold the 3 classes"):
        make_multisite_classification(n_sites=2, n_samples=[2, 50], n_classes=3, n_features=4)


def test_zero_strength_inside_lists():
    """A site can have no EoS and no noise when strengths are given per site."""
    X, _, sites, truth = _truth(n_samples=100, site_effect={"location": [0, 2]}, noise_strength=[0, 0.1])
    np.testing.assert_array_equal(X[sites == 0], truth["signal"][sites == 0])
    assert not np.allclose(X[sites == 1], truth["signal"][sites == 1])


@pytest.mark.parametrize("n_samples", [np.array([30, 70]), np.int64(100), (30, 70)])
def test_numpy_and_tuple_n_samples(n_samples):
    """NumPy integers and arrays are accepted for n_samples."""
    X, _, _ = make_multisite_classification(n_samples=n_samples)
    assert len(X) == 100


def test_site_without_samples_raises():
    """Every site needs samples."""
    with pytest.raises(ValueError, match="at least 1 sample"):
        make_multisite_classification(n_samples=[100, 0])


@pytest.mark.parametrize("balance", [np.array([[1, 0], [0, 1]]), np.array([0.5, 0.5]), [[1, 0], [0, 1]]])
def test_integer_and_array_balance(balance):
    """Integer proportions and arrays are accepted."""
    _, y, _ = make_multisite_classification(n_samples=100, balance_per_site=balance)
    assert len(y) == 100


def test_scale_effect_keeps_means_and_is_centred():
    """The scale EoS multiplies deviations from the global mean by exp(U(-0.1 s, 0.1 s))."""
    strength = 5.0
    X, _, sites, truth = _truth(n_samples=4000, site_effect={"scale": strength}, noise_strength=0)
    signal = truth["signal"]
    center = signal.mean(axis=0)
    for site in (0, 1):
        ratio = (X[sites == site] - center) / (signal[sites == site] - center)
        factor = np.median(ratio, axis=0)
        np.testing.assert_allclose(ratio, np.broadcast_to(factor, ratio.shape), rtol=1e-8)
        assert ((factor >= np.exp(-0.1 * strength)) & (factor <= np.exp(0.1 * strength))).all()
    np.testing.assert_allclose(X.mean(axis=0), signal.mean(axis=0), atol=0.2)


# ---------------------------------------------------------------------------
# Location and scale strengths set independently (#75), with a dict of strengths
# ---------------------------------------------------------------------------


def _affine_site_effect(X: np.ndarray, X_base: np.ndarray, center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature factor f and offset o of X = center + f (X_base - center) + o (exact without noise)."""
    deviation = X_base - center
    factor = ((X - X.mean(axis=0)) * (deviation - deviation.mean(axis=0))).sum(axis=0) / (
        (deviation - deviation.mean(axis=0)) ** 2
    ).sum(axis=0)
    offset = X.mean(axis=0) - center - factor * deviation.mean(axis=0)
    np.testing.assert_allclose(X, center + factor * deviation + offset, atol=1e-9)  # the EoS is affine per feature
    return factor, offset


def test_site_effect_order_and_per_site_list_are_equivalent():
    """The order of the dict does not matter, and a list of equal strengths equals the scalar."""
    params = {"n_sites": 3, "n_samples": 300}
    reference = make_multisite_classification(site_effect={"location": 2.0, "scale": 2.0}, **params)
    reordered = make_multisite_classification(site_effect={"scale": 2.0, "location": 2.0}, **params)
    per_site = make_multisite_classification(site_effect={"location": [2.0] * 3, "scale": 2.0}, **params)
    for a, b, c in zip(reference, reordered, per_site, strict=True):
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a, c)


def test_empty_site_effect_means_no_site_effect():
    """site_effect={} (or all strengths 0) gives the site-free data."""
    X, _, _, truth = _truth(n_sites=3, n_samples=300, site_effect={})
    np.testing.assert_array_equal(X, truth["site_free"])
    np.testing.assert_array_equal(truth["site_effect"], 0.0)
    X0, *_ = _truth(n_sites=3, n_samples=300, site_effect={"location": 0, "scale": 0})
    np.testing.assert_array_equal(X, X0)


@pytest.mark.parametrize(("location", "scale"), [(5.0, 0.0), (0.0, 5.0), (6.0, 1.0), (1.0, 6.0)])
def test_location_and_scale_strengths_are_independent(location, scale):
    """The offsets follow the location strength and the factors the scale strength."""
    X, _, sites, truth = _truth(
        n_sites=3,
        n_samples=1500,
        n_features=8,
        site_effect={"location": location, "scale": scale},
        noise_strength=0,
    )
    signal = truth["signal"]
    center = signal.mean(axis=0)
    for site in range(3):
        factor, offset = _affine_site_effect(X[sites == site], signal[sites == site], center)
        np.testing.assert_allclose(factor, truth["site_scale"][site], rtol=1e-9)  # the ground truth is exact
        np.testing.assert_allclose(offset, truth["site_location"][site], atol=1e-9)
    factors = truth["site_scale"]
    offsets = truth["site_location"] / truth["feature_scale"]  # in reference-SD units
    tol = 1e-9
    assert (np.abs(np.log(factors)) <= 0.1 * scale + tol).all()
    assert (np.abs(offsets) <= location / 2 + tol).all()  # centred offsets, U(-s/2, s/2)
    if scale == 0:
        np.testing.assert_allclose(factors, 1.0, atol=1e-9)
    else:
        assert np.ptp(np.log(factors)) > 0.05 * scale  # the factors vary between sites and features
    if location == 0:
        np.testing.assert_allclose(offsets, 0.0, atol=1e-9)
    else:
        assert np.abs(offsets).max() > 0.25 * location
        assert offsets.min() < 0 < offsets.max()


def test_strength_per_type_and_per_site():
    """A dict value can be a list: one strength per site for that type."""
    *_, truth = _truth(
        n_sites=2,
        n_samples=800,
        site_effect={"location": 3.0, "scale": [0.0, 8.0]},
        noise_strength=0,
    )
    np.testing.assert_allclose(truth["site_scale"][0], 1.0)
    assert np.abs(np.log(truth["site_scale"][1])).max() > 0.2
    assert (np.abs(truth["site_location"]) > 0).all()


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"site_effect": {"location": -1.0}}, ValueError, r"site_effect\['location'\] must be non-neg"),
        ({"site_effect": {"location": [1.0, 2.0, 3.0]}}, ValueError, "must have length n_sites"),
        ({"site_effect": {"location": "strong"}}, TypeError, "Invalid type"),
        ({"site_effect": {"rotate": 1.0}}, ValueError, "Unsupported site effect type: 'rotate'"),
        ({"site_effect": {"location+scale": 1.0}}, ValueError, "Unsupported site effect type"),
        ({"site_effect": "location"}, TypeError, "site_effect must be a dict"),
        ({"site_effect": 3.0}, TypeError, "site_effect must be a dict"),
        ({"signal": "linear"}, TypeError, "signal must be a dict with one type"),
        ({"signal": {}}, ValueError, "signal must be a dict with one type"),
        ({"signal": {"linear": 1.0, "blobs": 1.0}}, ValueError, "signal must be a dict with one type"),
        ({"signal": {"spiral": 1.0}}, ValueError, "Unsupported signal type: 'spiral'"),
        ({"signal": {"linear": [1.0, 2.0]}}, TypeError, r"Invalid type for signal\['linear'\]"),
    ],
)
def test_signal_and_site_effect_validation(kwargs, error, match):
    """The signal is one {type: strength}; site_effect maps known types to a strength or one per site."""
    with pytest.raises(error, match=match):
        make_multisite_classification(n_sites=2, **kwargs)
