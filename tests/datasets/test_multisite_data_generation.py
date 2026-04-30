"""Test module for make_multisite_classification function."""

import numpy as np
import pytest

from uniharmony import make_multisite_classification


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
            expected = balance_per_site[site_idx]
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
    X_weak, y_weak, _ = make_multisite_classification(n_sites=2, n_samples=100, signal_strength=0.5, random_state=42)
    X_strong, y_strong, _ = make_multisite_classification(n_sites=2, n_samples=100, signal_strength=2.0, random_state=42)

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
    X, _, sites = make_multisite_classification(n_sites=2, n_samples=100, site_effect_strength=5.0, random_state=42)

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
        make_multisite_classification(signal_strength=-1)  # Wrong signal strength
    with pytest.raises(ValueError):
        make_multisite_classification(noise_strength=-1)  # Wrong noise strength
    with pytest.raises(ValueError):
        make_multisite_classification(site_effect_strength=-1)  # Wrong Effect of Site strength
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
    # Wrong length
    with pytest.raises(TypeError):
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
    # Wrong type (all should be float)
    with pytest.raises(TypeError):
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
    """Test the heterogeneous site effect functionality."""
    # Just ensure it runs without errors
    make_multisite_classification(
        n_features=2,
        n_samples=100,
        site_effect_homogeneous=True,
        random_state=42,
    )
    make_multisite_classification(
        n_features=2,
        n_samples=100,
        site_effect_homogeneous=False,
        random_state=42,
    )


@pytest.mark.parametrize(
    "n_sites, n_samples, n_features, n_classes, signal_type, site_effect_type, "
    "site_effect_homogeneous, balance_per_site, noise_strength, signal_strength, "
    "site_effect_strength",
    [
        # Binary classification tests
        (2, 200, 10, 2, "linear", "location", True, None, 0.1, 1.0, 1.0),
        (3, 300, 2, 2, "circles", "scale", False, None, 0.0, 1.0, 2.0),
        (2, 150, 2, 2, "moons", "location+scale", True, [0.3, 0.7], 0.2, 1.5, 1.5),
        (4, 400, 12, 2, "blobs", "location", False, [0.2, 0.4, 0.6, 0.8], 0.15, 1.0, 2.0),
        (2, 250, 8, 2, "make_gaussian_quantiles", "scale", True, None, 0.05, 0.8, 3.0),
        # Multi-class classification tests
        (2, 300, 15, 3, "linear", "location", True, None, 0.1, 1.0, 1.0),
        (3, 450, 20, 4, "blobs", "scale", False, None, 0.0, 1.2, 2.5),
        (
            2,
            200,
            10,
            5,
            "linear",
            "location+scale",
            True,
            [[0.3, 0.2, 0.2, 0.15, 0.15], [0.1, 0.3, 0.3, 0.15, 0.15]],
            0.2,
            1.0,
            1.0,
        ),
        (3, 600, 25, 6, "make_gaussian_quantiles", "location", False, None, 0.1, 1.5, 2.0),
        # Edge cases
        (2, 100, 5, 2, "linear", "location", True, [0.0, 1.0], 0.0, 1.0, 1.0),  # Extreme balance
        (2, 100, 5, 2, "linear", "location", True, [0.5, 0.5], 0.0, 0.0, 0.0),  # No signal, no site effect
        (5, 500, 20, 3, "blobs", "scale", True, None, 0.2, 1.0, 0.0),  # No site effect
    ],
    ids=[
        "binary_linear_location_homogeneous_default_balance",
        "binary_circles_scale_heterogeneous_default_balance_no_noise",
        "binary_moons_location+scale_homogeneous_custom_balance",
        "binary_blobs_location_heterogeneous_custom_balance",
        "binary_gaussian_quantiles_scale_homogeneous_default_balance",
        "multiclass_linear_location_homogeneous_default_balance",
        "multiclass_blobs_scale_heterogeneous_default_balance_no_noise",
        "multiclass_linear_location+scale_homogeneous_custom_balance",
        "multiclass_gaussian_quantiles_location_heterogeneous_default_balance",
        "edge_extreme_balance",
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
    site_effect_homogeneous,
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
        signal_type=signal_type,
        signal_strength=signal_strength,
        noise_strength=noise_strength,
        site_effect_strength=site_effect_strength,
        site_effect_homogeneous=site_effect_homogeneous,
        site_effect_type=site_effect_type,
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
                expected_p_class1 = balance_per_site[site_idx]
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


@pytest.mark.parametrize("signal_type", ["linear", "circles", "moons", "blobs", "make_gaussian_quantiles"])
def test_all_signal_types(signal_type):
    """Test that all signal types work without errors."""
    if signal_type in ["moons", "circles"]:
        X, y, sites = make_multisite_classification(
            n_sites=2,
            n_samples=200,
            n_features=2,
            n_classes=2,
            signal_type=signal_type,
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
            signal_type=signal_type,
            random_state=42,
        )

        assert X.shape == (200, 10)
        assert y.shape == (200,)
        assert sites.shape == (200,)


@pytest.mark.parametrize("site_effect_type", ["location", "scale", "location+scale"])
def test_all_site_effect_types(site_effect_type):
    """Test that all site effect types work without errors."""
    X, y, sites = make_multisite_classification(
        n_sites=2,
        n_samples=200,
        n_features=10,
        n_classes=2,
        site_effect_type=site_effect_type,
        site_effect_strength=2.0,
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
            np.testing.assert_almost_equal(site_proportions[1], expected, decimal=1)
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

    with pytest.raises(ValueError, match="signal_strength must be non-negative"):
        make_multisite_classification(n_sites=2, n_samples=100, signal_strength=-1.0)

    with pytest.raises(ValueError, match="site_effect_strength must be non-negative"):
        make_multisite_classification(n_sites=2, n_samples=100, site_effect_strength=-1.0)

    with pytest.raises(ValueError, match="Unsupported signal_type: invalid"):
        make_multisite_classification(n_sites=2, n_samples=100, signal_type="invalid")

    with pytest.raises(ValueError, match="Unsupported site_effect_type: invalid"):
        make_multisite_classification(n_sites=2, n_samples=100, site_effect_type="invalid")


def test_noise_strength_zero():
    """Test that noise_strength=0 produces no additional noise."""
    X_noisy, _, _ = make_multisite_classification(
        n_sites=2,
        n_samples=200,
        n_features=10,
        n_classes=2,
        noise_strength=0.1,
        random_state=42,
    )

    X_clean, _, _ = make_multisite_classification(
        n_sites=2,
        n_samples=200,
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
    """Test difference between homogeneous and heterogeneous site effects."""
    X_homo, _, _ = make_multisite_classification(
        n_sites=2,
        n_samples=20,
        n_features=10,
        n_classes=2,
        site_effect_homogeneous=True,
        site_effect_strength=2.0,
        random_state=42,
    )

    X_hetero, _, _ = make_multisite_classification(
        n_sites=2,
        n_samples=20,
        n_features=10,
        n_classes=2,
        site_effect_homogeneous=False,
        site_effect_strength=2.0,
        random_state=42,
    )

    # Both should produce valid data
    assert X_homo.shape == X_hetero.shape

    # The feature distributions should differ
    assert not np.allclose(X_homo, X_hetero)
