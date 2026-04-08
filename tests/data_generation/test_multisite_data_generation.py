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
                assert abs(actual_prob - expected_prob) < 0.05, (
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
