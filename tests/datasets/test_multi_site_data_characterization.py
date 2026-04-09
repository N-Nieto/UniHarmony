"""Test suite for MAREoS dataset loading functions."""

from uniharmony.datasets import (
    get_site_data_statistics,
    load_MAREoS,
    make_multisite_classification,
    print_statistics_summary,
)


def test_data_characterization_and_printing() -> None:
    """Test basic functionality."""
    load_MAREoS()
    X, y, sites = make_multisite_classification(
        n_sites=3,
        n_samples=100,
        n_features=10,
        n_classes=3,
        random_state=42,
    )
    # Compute statistics
    stats = get_site_data_statistics(
        x=X,
        y=y,
        site_labels=sites,
        compute_comprehensive=True,
    )

    # Compute statistics
    _ = get_site_data_statistics(
        x=X,
        y=y,
        site_labels=sites,
        compute_comprehensive=False,
    )

    print_statistics_summary(stats)
