"""Test IntraSiteInterpolation transformer."""

import numbers

import numpy as np
import pytest
from imblearn.over_sampling import (
    ADASYN,
    SMOTE,
    SVMSMOTE,
    BorderlineSMOTE,
    KMeansSMOTE,
    RandomOverSampler,
)
from sklearn.linear_model import LogisticRegression

from uniharmony.datasets import make_multisite_classification
from uniharmony.interpolation import IntraSiteInterpolation


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def binary_data():
    """Return a standard binary-classification fixture."""
    x, y, sites = make_multisite_classification(n_samples=200, n_features=4, n_sites=2, n_classes=2, random_state=42)
    return x, y, sites


@pytest.fixture
def multiclass_data():
    """Return a standard multi-class fixture."""
    x, y, sites = make_multisite_classification(n_samples=200, n_features=4, n_sites=2, n_classes=3, random_state=44)
    return x, y, sites


@pytest.fixture
def global_max_data():
    """Return data specifically shaped for global_max testing.

    Site 0: class 0 = 100, class 1 = 50  -> majority = 100
    Site 1: class 0 = 10,  class 1 = 20  -> majority = 20
    Global max = 100.
    """
    rng = np.random.default_rng(46)
    x = rng.standard_normal((180, 4))
    sites = np.array([0] * 150 + [1] * 30)
    y = np.array(
        [0] * 100
        + [1] * 50  # site 0
        + [0] * 10
        + [1] * 20
    )  # site 1
    return x, y, sites


@pytest.fixture
def large_binary_data():
    """Large dataset for unreliable interpolators (ADASYN, KMeansSMOTE, SVMSMOTE).

    Ensures each site has at least 50 samples per class.
    """
    x, y, sites = make_multisite_classification(n_samples=5000, n_features=10, n_sites=5, n_classes=2, random_state=45)
    return x, y, sites


# ==============================================================================
# Interpolator parametrization helpers
# ==============================================================================

# Reliable interpolators that work with small/medium datasets and produce exact counts
RELIABLE_NAMES = [
    "smote",
    "borderline-smote",
    "random",
]

# All built-in interpolator names (for comprehensive testing with appropriate data)
ALL_NAMES = [
    "smote",
    "borderline-smote",
    "svm-smote",
    "adasyn",
    "kmeans-smote",
    "random",
]

# Reliable pre-instantiated interpolators
RELIABLE_INSTANCES = [
    SMOTE(),
    BorderlineSMOTE(),
    RandomOverSampler(),
]

# All pre-instantiated interpolators
ALL_INSTANCES = [
    SMOTE(),
    BorderlineSMOTE(),
    SVMSMOTE(),
    ADASYN(),
    KMeansSMOTE(),
    RandomOverSampler(),
]

VALID_SAMPLING_STRATEGIES = ["auto", "not majority"]


# ==============================================================================
# 1. Smoke / shape tests
# ==============================================================================


@pytest.mark.parametrize("interpolator", RELIABLE_NAMES)
def test_binary_runs_with_reliable_interpolators(interpolator, binary_data):
    """Reliable interpolators should run without error on binary data."""
    x, y, sites = binary_data
    isi = IntraSiteInterpolation(interpolator)
    xr, yr = isi.fit_resample(x, y, sites=sites)
    sr = isi.sites_resampled_
    assert len(xr) == len(yr) == len(sr)
    assert xr.ndim == 2
    assert yr.ndim == 1
    assert sr.ndim == 1


@pytest.mark.parametrize("interpolator", RELIABLE_NAMES)
def test_multiclass_runs_with_reliable_interpolators(interpolator, multiclass_data):
    """Reliable interpolators should run without error on multi-class data."""
    x, y, sites = multiclass_data
    isi = IntraSiteInterpolation(interpolator)
    xr, yr = isi.fit_resample(x, y, sites=sites)
    sr = isi.sites_resampled_
    assert len(xr) == len(yr) == len(sr)


@pytest.mark.parametrize("interpolator", RELIABLE_INSTANCES)
def test_binary_runs_with_reliable_instance(interpolator, binary_data):
    """Passing a pre-instantiated reliable sampler should run without error."""
    x, y, sites = binary_data
    isi = IntraSiteInterpolation(interpolator=interpolator)
    xr, yr = isi.fit_resample(x, y, sites=sites)
    assert len(xr) == len(yr)


# ==============================================================================
# 2. Unreliable interpolators (require larger datasets)
# ==============================================================================


@pytest.mark.parametrize("interpolator", ["svm-smote", "adasyn", "kmeans-smote"])
def test_unreliable_interpolators_with_large_data(interpolator, large_binary_data):
    """ADASYN, KMeansSMOTE, SVMSMOTE need larger datasets to work reliably."""
    x, y, sites = large_binary_data
    isi = IntraSiteInterpolation(interpolator)
    xr, yr = isi.fit_resample(x, y, sites=sites)
    sr = isi.sites_resampled_
    assert len(xr) == len(yr) == len(sr)


@pytest.mark.parametrize("interpolator", [ADASYN(), KMeansSMOTE(), SVMSMOTE()])
def test_unreliable_instances_with_large_data(interpolator, large_binary_data):
    """ADASYN, KMeansSMOTE, SVMSMOTE instances need larger datasets."""
    x, y, sites = large_binary_data
    isi = IntraSiteInterpolation(interpolator=interpolator)
    xr, yr = isi.fit_resample(x, y, sites=sites)
    assert len(xr) == len(yr)


# ==============================================================================
# 3. Balance correctness - per_site (default)
# ==============================================================================


@pytest.mark.parametrize("interpolator", ["smote", "random"])
def test_per_site_balance_binary(interpolator, binary_data):
    """With per_site strategy every site must have identical class counts."""
    x, y, sites = binary_data
    isi = IntraSiteInterpolation(interpolator, balance_strategy="per_site")
    _, yr = isi.fit_resample(x, y, sites=sites)
    sr = isi.sites_resampled_

    for site in np.unique(sr):
        counts = np.unique(yr[sr == site], return_counts=True)[1]
        assert len(set(counts)) == 1, f"Site {site} not balanced: counts={counts}"


@pytest.mark.parametrize("interpolator", ["smote", "random"])
def test_per_site_balance_multiclass(interpolator, multiclass_data):
    """With per_site strategy every site must have identical class counts (multiclass)."""
    x, y, sites = multiclass_data
    isi = IntraSiteInterpolation(interpolator, balance_strategy="per_site")
    _, yr = isi.fit_resample(x, y, sites=sites)
    sr = isi.sites_resampled_

    for site in np.unique(sr):
        counts = np.unique(yr[sr == site], return_counts=True)[1]
        assert len(set(counts)) == 1, f"Site {site} not balanced: counts={counts}"


# ==============================================================================
# 4. Balance correctness - global_max
# ==============================================================================


@pytest.mark.parametrize("interpolator", ["smote", "random"])
def test_global_max_balance_binary(interpolator, global_max_data):
    """With global_max all sites must have the SAME count for every class."""
    x, y, sites = global_max_data
    isi = IntraSiteInterpolation(interpolator, balance_strategy="global_max")
    _, yr = isi.fit_resample(x, y, sites=sites)
    sr = isi.sites_resampled_

    site_counts = {}
    for site in np.unique(sr):
        counts = dict(zip(*np.unique(yr[sr == site], return_counts=True), strict=True))
        site_counts[site] = counts
        # Within-site balance
        assert len(set(counts.values())) == 1, f"Site {site} not internally balanced: {counts}"

    # Cross-site balance
    first = next(iter(site_counts.values()))
    for site, counts in site_counts.items():
        assert counts == first, f"Site {site} counts {counts} != first site {first}"

    # Must equal the global maximum (100 in the fixture)
    assert next(iter(first.values())) == 100


@pytest.mark.parametrize("interpolator", ["smote", "random"])
def test_global_max_target_count_attribute(interpolator, global_max_data):
    """target_count_ must be set to the global maximum when using global_max."""
    x, y, sites = global_max_data
    isi = IntraSiteInterpolation(interpolator, balance_strategy="global_max")
    isi.fit_resample(x, y, sites=sites)
    assert isi.target_count_ == 100


def test_global_max_missing_class_raises():
    """global_max requires every class to be present in every site."""
    rng = np.random.default_rng(47)
    x = rng.standard_normal((120, 4))
    # Site 0 has only class 0; site 1 has class 0 and 1
    sites = np.array([0] * 60 + [1] * 60)
    y = np.array([0] * 60 + [0] * 30 + [1] * 30)
    isi = IntraSiteInterpolation("random", balance_strategy="global_max")
    with pytest.raises(ValueError, match=r"has only one class; cannot resample."):
        isi.fit_resample(x, y, sites=sites)


# ==============================================================================
# 5. samples_created_ attribute
# ==============================================================================


@pytest.mark.parametrize("balance_strategy", ["per_site", "global_max"])
def test_samples_created_attribute_exists(balance_strategy, binary_data):
    """samples_created_ must be a nested dict after fit_resample."""
    x, y, sites = binary_data
    isi = IntraSiteInterpolation("random", balance_strategy=balance_strategy)
    isi.fit_resample(x, y, sites=sites)
    assert hasattr(isi, "samples_created_")
    assert isinstance(isi.samples_created_, dict)
    for _, class_dict in isi.samples_created_.items():
        assert isinstance(class_dict, dict)
        for _, n_created in class_dict.items():
            # Use numbers.Integral to handle both int and np.integer types
            assert isinstance(n_created, numbers.Integral)
            assert n_created >= 0


def test_samples_created_values_per_site(binary_data):
    """samples_created_ should correctly report synthetic samples per class per site."""
    x, y, sites = binary_data
    isi = IntraSiteInterpolation("random", balance_strategy="per_site")
    _, yr = isi.fit_resample(x, y, sites=sites)
    sr = isi.sites_resampled_

    for site in np.unique(sr):
        for cls in np.unique(y):
            original = np.sum((sites == site) & (y == cls))
            resampled = np.sum((sr == site) & (yr == cls))
            expected_created = max(0, resampled - original)
            actual_created = isi.samples_created_[site][cls]
            assert actual_created == expected_created, (
                f"Site {site}, class {cls}: expected {expected_created} created, got {actual_created}"
            )


def test_samples_created_values_global_max(global_max_data):
    """samples_created_ should correctly report synthetic samples with global_max."""
    x, y, sites = global_max_data
    isi = IntraSiteInterpolation("random", balance_strategy="global_max")
    _, yr = isi.fit_resample(x, y, sites=sites)
    sr = isi.sites_resampled_

    for site in np.unique(sr):
        for cls in np.unique(y):
            original = np.sum((sites == site) & (y == cls))
            resampled = np.sum((sr == site) & (yr == cls))
            expected_created = max(0, resampled - original)
            actual_created = isi.samples_created_[site][cls]
            assert actual_created == expected_created, (
                f"Site {site}, class {cls}: expected {expected_created} created, got {actual_created}"
            )


# ==============================================================================
# 6. balance_strategy validation
# ==============================================================================


def test_invalid_balance_strategy_raises(binary_data):
    """An unknown balance_strategy must raise ValueError."""
    x, y, sites = binary_data
    isi = IntraSiteInterpolation("smote", balance_strategy="invalid")
    with pytest.raises(ValueError, match="balance_strategy must be"):
        isi.fit_resample(x, y, sites=sites)


@pytest.mark.parametrize("strategy", ["per_site", "global_max"])
def test_balance_strategy_accepted(strategy, binary_data):
    """Valid balance_strategy values should run without error."""
    x, y, sites = binary_data
    isi = IntraSiteInterpolation("smote", balance_strategy=strategy)
    isi.fit_resample(x, y, sites=sites)


# ==============================================================================
# 7. Input validation errors
# ==============================================================================


def test_shape_mismatch_x_y_raises(binary_data):
    """Mismatched lengths between X and y must raise ValueError."""
    x, y, sites = binary_data
    isi = IntraSiteInterpolation("smote")
    with pytest.raises(ValueError):
        isi.fit_resample(x[:-1], y, sites=sites)


def test_shape_mismatch_x_sites_raises(binary_data):
    """Mismatched lengths between X and sites must raise ValueError."""
    x, y, sites = binary_data
    isi = IntraSiteInterpolation("smote")
    with pytest.raises(ValueError):
        isi.fit_resample(x, y, sites=sites[:-1])


def test_single_site_raises():
    """Only one unique site must raise ValueError (from validate_sites)."""
    rng = np.random.default_rng(49)
    x = rng.standard_normal((20, 2))
    y = np.array([0] * 10 + [1] * 10)
    sites = np.zeros(20)
    isi = IntraSiteInterpolation()
    with pytest.raises(ValueError):
        isi.fit_resample(x, y, sites=sites)


def test_single_class_in_a_site_raises():
    """A site containing only one class must raise ValueError."""
    rng = np.random.default_rng(50)
    x = rng.standard_normal((300, 10))
    y = np.array([0] * 180 + [1] * 80 + [2] * 40)
    sites = np.array([0] * 150 + [1] * 150)
    isi = IntraSiteInterpolation()
    with pytest.raises(ValueError):
        isi.fit_resample(x, y, sites=sites)


# ==============================================================================
# 8. Interpolator validation
# ==============================================================================


@pytest.mark.parametrize("name", RELIABLE_NAMES)
def test_valid_interpolator_names(name, binary_data):
    """Valid string names should instantiate and run."""
    x, y, sites = binary_data
    isi = IntraSiteInterpolation(name)
    isi.fit_resample(x, y, sites=sites)


@pytest.mark.parametrize("name", ["invalid", "wrong_name", ""])
def test_invalid_interpolator_name_raises(name, binary_data):
    """Invalid interpolator string names must raise ValueError."""
    x, y, sites = binary_data
    isi = IntraSiteInterpolation(name)
    with pytest.raises(ValueError, match="Unsupported interpolator"):
        isi.fit_resample(x, y, sites=sites)


def test_valid_interpolator_name_case_insensitive(binary_data):
    """Interpolator names should be case-insensitive (converted to lowercase)."""
    x, y, sites = binary_data
    isi = IntraSiteInterpolation("SMOTE")
    # Should not raise because "SMOTE" -> "smote" via .lower()
    isi.fit_resample(x, y, sites=sites)


@pytest.mark.parametrize("strategy", VALID_SAMPLING_STRATEGIES)
def test_interpolator_with_valid_sampling_strategy(strategy, binary_data):
    """Instances with 'auto' or 'not majority' sampling_strategy should work."""
    x, y, sites = binary_data
    isi = IntraSiteInterpolation(interpolator=SMOTE(sampling_strategy=strategy))
    isi.fit_resample(x, y, sites=sites)


@pytest.mark.parametrize("strategy", ["invalid", "majority", "all"])
def test_interpolator_with_invalid_sampling_strategy_raises(strategy, binary_data):
    """Instances with disallowed sampling_strategy must raise ValueError."""
    x, y, sites = binary_data
    isi = IntraSiteInterpolation(interpolator=SMOTE(sampling_strategy=strategy))
    with pytest.raises(ValueError, match="sampling_strategy='not majority'"):
        isi.fit_resample(x, y, sites=sites)


def test_interpolator_as_wrong_instance_raises(binary_data):
    """Passing a non-SamplerMixin instance must raise ValueError."""
    x, y, sites = binary_data
    isi = IntraSiteInterpolation(interpolator=LogisticRegression())
    with pytest.raises(ValueError, match="SamplerMixin"):
        isi.fit_resample(x, y, sites=sites)


# ==============================================================================
# 9. sklearn compatibility
# ==============================================================================


def test_fit_resample_compatibility(binary_data):
    """_fit_resample must exist for SamplerMixin compatibility."""
    x, y, _ = binary_data
    isi = IntraSiteInterpolation(interpolator="smote")
    # Should not raise (no-op implementation)
    isi._fit_resample(x, y)


def test_sklearn_tags(binary_data):
    """__sklearn_tags__ must return the expected estimator tags."""
    isi = IntraSiteInterpolation()
    tags = isi.__sklearn_tags__()
    assert tags.estimator_type == "sampler"
    assert tags.input_tags.two_d_array is True
    assert tags.input_tags.sparse is False
    assert tags.input_tags.allow_nan is True
    assert tags.requires_fit is False


# ==============================================================================
# 11. Reproducibility
# ==============================================================================


def test_reproducibility_with_random_state(binary_data):
    """Same random_state must yield identical resampled datasets."""
    x, y, sites = binary_data
    isi1 = IntraSiteInterpolation("random", random_state=42)
    isi2 = IntraSiteInterpolation("random", random_state=42)
    xr1, yr1 = isi1.fit_resample(x, y, sites=sites)
    xr2, yr2 = isi2.fit_resample(x, y, sites=sites)
    np.testing.assert_array_equal(xr1, xr2)
    np.testing.assert_array_equal(yr1, yr2)
    np.testing.assert_array_equal(isi1.sites_resampled_, isi2.sites_resampled_)


# ==============================================================================
# 12. Sites attribute
# ==============================================================================


def test_sites_resampled_attribute(binary_data):
    """sites_resampled_ must preserve site identifiers and be correct length."""
    x, y, sites = binary_data
    isi = IntraSiteInterpolation("random")
    xr, _ = isi.fit_resample(x, y, sites=sites)
    sr = isi.sites_resampled_
    assert len(sr) == len(xr)
    assert set(np.unique(sr)) == set(np.unique(sites))
