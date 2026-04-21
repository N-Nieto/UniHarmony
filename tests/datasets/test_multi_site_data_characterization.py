"""Test suite for data characterization functions."""

from uniharmony.datasets import (
    get_multisite_data_statistics,
    make_multisite_classification,
    print_statistics_summary,
)


def test_data_characterization_and_printing() -> None:
    """Test basic functionality."""
    X, y, sites = make_multisite_classification(
        n_sites=3,
        n_samples=100,
        n_classes=3,
    )
    # Compute statistics
    stats = get_multisite_data_statistics(
        X=X,
        y=y,
        sites=sites,
    )

    # Compute statistics
    _ = get_multisite_data_statistics(
        X=X,
        y=y,
        sites=sites,
    )

    print_statistics_summary(stats)
