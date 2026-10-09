"""Test IntraSiteInterpolation transformer."""

import numbers
import warnings

import numpy as np
import pytest
from imblearn.base import BaseSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler, TomekLinks
from sklearn import config_context
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

from uniharmony.datasets import make_multisite_classification
from uniharmony.interpolation import IntraSiteInterpolation
from uniharmony.interpolation._utils import (
    allocate_proportionally,
    create_interpolator,
    create_undersampler,
    effective_dimension,
    variance_ratio,
)


# ==============================================================================
# Fixtures and helpers
# ==============================================================================


@pytest.fixture
def binary_data():
    """Generate binary classification dataset with site imbalance."""
    return make_multisite_classification(
        n_samples=2000,
        n_features=4,
        n_sites=2,
        n_classes=2,
        random_state=42,
        balance_per_site=[0.1, 0.9],
    )


@pytest.fixture
def opposite_sites_data():
    """Two sites with opposite class imbalance (site predicts the target)."""
    rng = np.random.default_rng(3)
    n_features = 5
    X0 = rng.standard_normal((400, n_features))
    X1 = rng.standard_normal((400, n_features)) + 2.0
    y0 = np.r_[np.zeros(320), np.ones(80)].astype(int)
    y1 = np.r_[np.zeros(80), np.ones(320)].astype(int)
    X = np.vstack([X0, X1]) + np.outer(np.r_[y0, y1], np.ones(n_features))
    return X, np.r_[y0, y1], np.repeat([0, 1], 400)


@pytest.fixture
def regression_data():
    """Generate regression dataset with continuous targets."""
    rng = np.random.default_rng(53)
    X = rng.standard_normal((200, 4))
    sites = np.array([0] * 100 + [1] * 100)
    y = rng.standard_normal(200) * 10 + 50
    return X, y, sites


@pytest.fixture
def covariate_data():
    """Generate dataset with categorical and continuous covariates."""
    rng = np.random.default_rng(54)
    n_samples = 2000

    X, y, sites = make_multisite_classification(
        n_samples=n_samples,
        n_features=4,
        n_sites=2,
        n_classes=2,
        random_state=42,
        balance_per_site=[0.1, 0.9],
    )

    sex = rng.integers(0, 2, (n_samples, 1))
    age = rng.standard_normal((n_samples, 1)) * 10 + 50

    return X, y, sites, sex, age


@pytest.fixture
def linear_regression_data():
    """Regression data whose target is an exact linear function of X, with imbalanced sites."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((200, 3))
    weights = np.array([3.0, -2.0, 0.5])
    y = X @ weights + 50.0
    # Site 0 over-represents high targets, so its low bins are oversampled
    sites = np.where(rng.random(200) < np.where(y > 50.0, 0.8, 0.3), 0, 1)
    return X, y, sites, weights


def _counts(yr, sr):
    """``{site: {class: count}}``."""
    return {s: dict(zip(*np.unique(yr[sr == s], return_counts=True), strict=True)) for s in np.unique(sr)}


def _assert_balanced(yr, sr):
    """Every class has the same count within each site."""
    for counts in _counts(yr, sr).values():
        assert len(set(counts.values())) == 1


def _split_original_and_synthetic(X, Xr):
    """Return a mask of the rows of ``Xr`` that are original rows of ``X`` (first occurrence only)."""
    remaining = {tuple(row) for row in X}
    is_original = np.zeros(len(Xr), dtype=bool)
    for i, row in enumerate(map(tuple, Xr)):
        if row in remaining:
            is_original[i] = True
            remaining.discard(row)
    return is_original


def _distance_to_closest_segment(x, X_class):
    """Squared distance from ``x`` to the closest segment between two rows of ``X_class`` (brute force)."""
    best = np.inf
    for a in range(len(X_class)):
        direction = X_class - X_class[a]
        norm2 = np.sum(direction**2, axis=1)
        lam = np.clip(np.divide(direction @ (x - X_class[a]), norm2, out=np.zeros(len(X_class)), where=norm2 > 0), 0, 1)
        best = min(best, np.min(np.sum((X_class[a] + lam[:, np.newaxis] * direction - x) ** 2, axis=1)))
    return best


def _assert_originals_unchanged(X, y, Xr, yr):
    """Every original sample is returned once with its original target."""
    original_target = {tuple(row): target for row, target in zip(X, y, strict=True)}
    is_original = _split_original_and_synthetic(X, Xr)
    assert is_original.sum() == len(X)
    for row, target in zip(Xr[is_original], yr[is_original], strict=True):
        assert target == original_target[tuple(row)]
    return is_original


def _assert_provenance(isi, X, y, sites, Xr, yr):
    """``sample_indices_`` / ``is_synthetic_`` / ``sites_resampled_`` describe the output."""
    real = ~isi.is_synthetic_
    np.testing.assert_array_equal(isi.is_synthetic_, isi.sample_indices_ < 0)
    np.testing.assert_array_equal(Xr[real], X[isi.sample_indices_[real]])
    np.testing.assert_array_equal(yr[real], y[isi.sample_indices_[real]])
    np.testing.assert_array_equal(isi.sites_resampled_[real], sites[isi.sample_indices_[real]])
    assert len(np.unique(isi.sample_indices_[real])) == real.sum()


# ==============================================================================
# Basic functionality
# ==============================================================================


def test_basic_run(binary_data):
    """Model should run and return valid shapes."""
    X, y, sites = binary_data
    isi = IntraSiteInterpolation("random")

    Xr, yr = isi.fit_resample(X, y, sites=sites)

    assert len(Xr) == len(yr) == len(isi.sites_resampled_) == len(isi.sample_indices_)
    assert Xr.ndim == 2
    assert yr.ndim == 1
    _assert_provenance(isi, X, y, sites, Xr, yr)


def test_basic_run_invalid_instance(binary_data):
    """A non-sampler interpolator is rejected."""
    X, y, sites = binary_data
    isi = IntraSiteInterpolation(interpolator=LogisticRegression())
    with pytest.raises(ValueError):
        _, _ = isi.fit_resample(X, y, sites=sites)


def test_basic_run_no_balance():
    """Already balanced sites are returned unchanged."""
    X, y, sites = make_multisite_classification()
    isi = IntraSiteInterpolation(interpolator=SMOTE())
    Xr, _ = isi.fit_resample(X, y, sites=sites)
    assert not isi.is_synthetic_.any()
    assert len(Xr) == len(X)


def test_basic_run_no_balance_small():
    """Small sites work (neighbour counts are adapted); one sample per class cannot be interpolated."""
    X, y, sites = make_multisite_classification(n_samples=[4, 19])
    isi = IntraSiteInterpolation(interpolator=SMOTE())
    _, yr = isi.fit_resample(X, y, sites=sites)
    _assert_balanced(yr, isi.sites_resampled_)
    X, y, sites = make_multisite_classification(n_samples=[2, 19])
    with pytest.raises(ValueError, match="at least 2 samples"):
        isi.fit_resample(X, y, sites=sites)


def test_small_minority_neighbors_adapted():
    """SMOTE (k_neighbors=5) works with a 3-sample minority class."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 3))
    y = np.r_[np.zeros(37), np.ones(3), np.zeros(20), np.ones(20)].astype(int)
    sites = np.repeat([0, 1], 40)
    isi = IntraSiteInterpolation("smote", max_amplification=None, random_state=0)
    _, yr = isi.fit_resample(X, y, sites=sites)
    _assert_balanced(yr, isi.sites_resampled_)
    assert isi.samples_created_[0][1] == 34


# ==============================================================================
# Balance correctness
# ==============================================================================


@pytest.mark.parametrize("strategy", ["per_site", "global_max"])
@pytest.mark.parametrize("max_amplification", ["auto", None, 1.0, 0])
def test_balance_strategy(strategy, max_amplification, binary_data):
    """Each site must be class-balanced after resampling, whatever the cap."""
    X, y, sites = binary_data
    isi = IntraSiteInterpolation("smote", balance_strategy=strategy, max_amplification=max_amplification, random_state=0)

    Xr, yr = isi.fit_resample(X, y, sites=sites)
    sr = isi.sites_resampled_
    _assert_balanced(yr, sr)
    for site, target in isi.target_counts_.items():
        assert all(n == target for n in _counts(yr, sr)[site].values())
    _assert_provenance(isi, X, y, sites, Xr, yr)


def test_global_max_equal_sites_when_uncapped(binary_data):
    """Without cap, global_max gives every site-class the largest class count."""
    X, y, sites = binary_data
    isi = IntraSiteInterpolation("smote", balance_strategy="global_max", max_amplification=None, random_state=0)
    _, yr = isi.fit_resample(X, y, sites=sites)
    largest = max(np.sum((sites == s) & (y == c)) for s in np.unique(sites) for c in np.unique(y))
    assert isi.target_count_ == largest
    assert set(isi.target_counts_.values()) == {largest}
    assert len(yr) == largest * 2 * 2


@pytest.mark.parametrize("strategy", ["uniform", "quantile"])
def test_binning_strategy(strategy, regression_data):
    """Each strategies for binning_strategy."""
    X, y, sites = regression_data
    isi = IntraSiteInterpolation("random", binning_strategy=strategy)
    with pytest.warns(UserWarning, match="present in every site") if strategy == "uniform" else _no_warning():
        _, _ = isi.fit_resample(X, y, sites=sites)


class _no_warning:  # noqa: N801
    """Context manager that does nothing (pytest.warns counterpart)."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_binning_strategy_invalid(regression_data):
    """Invalid strategies for binning."""
    X, y, sites = regression_data
    isi = IntraSiteInterpolation("random", binning_strategy="invalid")
    with pytest.raises(ValueError):
        _, _ = isi.fit_resample(X, y, sites=sites)


@pytest.mark.parametrize("n_bins", [1, 2.5])
def test_invalid_n_bins(regression_data, n_bins):
    """n_bins must be an integer >= 2 for regression."""
    X, y, sites = regression_data
    with pytest.raises(ValueError, match="n_bins"):
        IntraSiteInterpolation("random", n_bins=n_bins).fit_resample(X, y, sites=sites)


# ==============================================================================
# Amplification cap and under-sampling
# ==============================================================================


def test_defaults():
    """No cap by default: ISI never removes samples unless asked to."""
    isi = IntraSiteInterpolation()
    assert isi.max_amplification is None
    assert isi.variance_tolerance == 0.2
    assert isi.undersampler == "cluster-centroids"


@pytest.mark.parametrize("interpolator", ["smote", "random"])
def test_default_keeps_all_originals(opposite_sites_data, interpolator):
    """Without cap ISI only over-samples: every real sample is kept."""
    X, y, sites = opposite_sites_data
    isi = IntraSiteInterpolation(interpolator, random_state=0)
    Xr, yr = isi.fit_resample(X, y, sites=sites)
    _assert_originals_unchanged(X, y, Xr, yr)
    assert all(v == 0 for d in isi.samples_removed_.values() for v in d.values())
    assert isi.samples_created_ == {0: {0: 0, 1: 240}, 1: {0: 240, 1: 0}}
    assert all(np.isinf(v) for d in isi.amplification_cap_.values() for v in d.values())


def _extreme_data():
    """Two sites with 1000 samples of one class and 2 of the other (opposite classes)."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((2004, 10))
    y = np.r_[np.zeros(1000), np.ones(2), np.zeros(2), np.ones(1000)].astype(int)
    return X, y, np.repeat([0, 1], 1002)


def test_default_warns_when_interpolation_is_unsafe():
    """1000 vs 2: no sample is dropped, but a clear warning advises the cap."""
    X, y, sites = _extreme_data()
    isi = IntraSiteInterpolation("smote", random_state=0)
    with pytest.warns(UserWarning, match="too small, compared with the largest class") as record:
        Xr, yr = isi.fit_resample(X, y, sites=sites)
    message = str(record[0].message)
    assert "max_amplification='auto'" in message
    assert "site 0, class 1: 2 real -> 1000 samples" in message
    _assert_originals_unchanged(X, y, Xr, yr)
    _assert_balanced(yr, isi.sites_resampled_)


def test_no_warning_when_interpolation_is_safe():
    """A mild imbalance in a densely sampled, low-dimensional class does not warn."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((4600, 2))
    y = np.r_[np.zeros(1200), np.ones(1100), np.zeros(1100), np.ones(1200)].astype(int)
    sites = np.repeat([0, 1], 2300)
    isi = IntraSiteInterpolation("smote", random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        isi.fit_resample(X, y, sites=sites)
    assert all(isi.amplification_[s][c] <= isi.safe_amplification_[s][c] for s, c in [(0, 1), (1, 0)])


def test_two_samples_keep_one_sixth_of_the_variance():
    """With two samples SMOTE draws on one segment: rho = 1/6, so almost no interpolation is safe."""
    X, y, sites = _extreme_data()
    isi = IntraSiteInterpolation("smote", random_state=0)
    with pytest.warns(UserWarning):
        isi.fit_resample(X, y, sites=sites)
    rho = isi.variance_ratio_[0][1]
    assert rho == pytest.approx(1 / 6, abs=0.03)
    assert isi.safe_amplification_[0][1] == pytest.approx(0.2 / ((1 - rho) - 0.2))
    assert isi.safe_amplification_[0][1] * 2 < 1  # not even one synthetic sample is safe


def test_auto_cap_removes_samples_with_warning():
    """max_amplification='auto' applies the safe amplification and under-samples the rest, with a warning."""
    X, y, sites = _extreme_data()
    isi = IntraSiteInterpolation("smote", max_amplification="auto", random_state=0)
    with pytest.warns(UserWarning, match="removed by under-sampling"):
        Xr, yr = isi.fit_resample(X, y, sites=sites)
    assert len(yr) < 20
    assert isi.amplification_cap_[0][1] == isi.safe_amplification_[0][1]
    _assert_balanced(yr, isi.sites_resampled_)
    _assert_provenance(isi, X, y, sites, Xr, yr)


def test_safe_amplification_follows_variance_rule(opposite_sites_data):
    """r* = eps / (|1 - rho| - eps) from the measured variance ratio; 'auto' applies it."""
    X, y, sites = opposite_sites_data
    eps = 0.1
    isi = IntraSiteInterpolation("smote", max_amplification="auto", variance_tolerance=eps, random_state=0)
    isi.fit_resample(X, y, sites=sites)
    for site, minority in [(0, 1), (1, 0)]:
        rho = isi.variance_ratio_[site][minority]
        assert 0 < rho < 1
        expected = np.inf if abs(1 - rho) <= eps else eps / (abs(1 - rho) - eps)
        assert isi.safe_amplification_[site][minority] == pytest.approx(expected)
        assert isi.amplification_cap_[site][minority] == pytest.approx(expected)
        assert isi.amplification_[site][minority] <= expected + 1e-12
        # majority classes are not over-sampled, so they are neither measured nor capped
        assert np.isnan(isi.safe_amplification_[site][1 - minority])
        assert np.isinf(isi.amplification_cap_[site][1 - minority])


def test_safe_amplification_computed_from_all_samples(opposite_sites_data):
    """The safe amplification is the same with and without cap: it never depends on removed samples."""
    X, y, sites = opposite_sites_data
    plain = IntraSiteInterpolation("smote", random_state=0)
    capped = IntraSiteInterpolation("smote", max_amplification="auto", random_state=0)
    plain.fit_resample(X, y, sites=sites)
    capped.fit_resample(X, y, sites=sites)
    assert plain.safe_amplification_ == capped.safe_amplification_
    assert plain.variance_ratio_ == capped.variance_ratio_


@pytest.mark.parametrize("cap", [0, 0.5, 1, 2])
def test_numeric_cap(opposite_sites_data, cap):
    """A numeric cap bounds the synthetic samples per real sample; the majority is under-sampled to meet."""
    X, y, sites = opposite_sites_data
    isi = IntraSiteInterpolation("smote", max_amplification=cap, random_state=0)
    Xr, yr = isi.fit_resample(X, y, sites=sites)
    expected = min(320, int(np.floor(80 * (1 + cap))))
    assert isi.target_counts_ == {0: expected, 1: expected}
    assert isi.samples_created_[0][1] == expected - 80
    assert isi.samples_removed_[0][0] == 320 - expected
    assert all(a <= cap + 1e-12 for d in isi.amplification_.values() for a in d.values())
    _assert_balanced(yr, isi.sites_resampled_)
    _assert_provenance(isi, X, y, sites, Xr, yr)
    if cap == 0:
        assert not isi.is_synthetic_.any()


def test_safe_amplification_grows_with_samples():
    """More real samples per effective dimension -> better variance preservation -> larger safe amplification."""
    rng = np.random.default_rng(2)
    safe = []
    for n_minority in (10, 1000):
        X = rng.standard_normal((2 * (n_minority + 4000), 5))
        y = np.r_[np.ones(n_minority), np.zeros(4000), np.zeros(n_minority), np.ones(4000)].astype(int)
        sites = np.repeat([0, 1], n_minority + 4000)
        isi = IntraSiteInterpolation("smote", random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            isi.fit_resample(X, y, sites=sites)
        safe.append(isi.safe_amplification_[0][1])
    assert safe[0] < safe[1]


def test_random_oversampling_not_limited_by_variance_rule(opposite_sites_data):
    """Duplication preserves the variance, so the variance rule does not limit it."""
    X, y, sites = opposite_sites_data
    isi = IntraSiteInterpolation("random", random_state=0)
    isi.fit_resample(X, y, sites=sites)
    assert np.isinf(isi.safe_amplification_[0][1])
    assert isi.variance_ratio_[0][1] == pytest.approx(1.0, abs=0.1)


def test_callable_cap(opposite_sites_data):
    """A callable receives the real samples of the cell and returns its cap."""
    X, y, sites = opposite_sites_data
    seen = []

    def cap(X_cell):
        seen.append(len(X_cell))
        return 0.25

    isi = IntraSiteInterpolation("smote", max_amplification=cap, random_state=0)
    isi.fit_resample(X, y, sites=sites)
    assert seen == [80, 80]
    assert isi.target_counts_ == {0: 100, 1: 100}


def _single_sample_data():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((60, 3))
    y = np.r_[np.zeros(29), np.ones(1), np.zeros(15), np.ones(15)].astype(int)
    return X, y, np.repeat([0, 1], 30)


def test_single_sample_class_raises():
    """A single sample cannot be interpolated: a class needs at least two samples in every site."""
    X, y, sites = _single_sample_data()
    with pytest.raises(ValueError, match="at least 2 samples"):
        IntraSiteInterpolation("smote").fit_resample(X, y, sites=sites)


def test_single_sample_class_can_be_duplicated():
    """Random over-sampling (duplication) only needs one sample."""
    X, y, sites = _single_sample_data()
    isi = IntraSiteInterpolation("random", random_state=0)
    _, yr = isi.fit_resample(X, y, sites=sites)
    _assert_balanced(yr, isi.sites_resampled_)


def test_cap_without_undersampler_raises(opposite_sites_data):
    """If the cap requires removing samples but no undersampler is set, fail loudly."""
    X, y, sites = opposite_sites_data
    isi = IntraSiteInterpolation("smote", max_amplification=1, undersampler=None)
    with pytest.raises(ValueError, match="undersampler"):
        isi.fit_resample(X, y, sites=sites)


@pytest.mark.parametrize(
    "undersampler",
    [
        "cluster-centroids",
        ClusterCentroids(voting="hard"),
        "nearmiss",
        "nearmiss-2",
        "nearmiss-3",
        "instance-hardness",
        "random",
        RandomUnderSampler(replacement=False),
    ],
)
def test_undersamplers(opposite_sites_data, undersampler):
    """Count-controlled imblearn under-samplers close the remaining imbalance; kept rows are real samples."""
    X, y, sites = opposite_sites_data
    isi = IntraSiteInterpolation("smote", undersampler=undersampler, max_amplification=0.5, random_state=0)
    Xr, yr = isi.fit_resample(X, y, sites=sites)
    assert isi.target_counts_ == {0: 120, 1: 120}
    _assert_balanced(yr, isi.sites_resampled_)
    _assert_provenance(isi, X, y, sites, Xr, yr)
    assert isi.is_synthetic_.sum() == 80


def test_cluster_centroids_soft_prototypes_flagged(opposite_sites_data):
    """ClusterCentroids with soft voting returns centroids (new samples), flagged as synthetic."""
    X, y, sites = opposite_sites_data
    isi = IntraSiteInterpolation("smote", undersampler=ClusterCentroids(voting="soft"), max_amplification=0.5, random_state=0)
    Xr, yr = isi.fit_resample(X, y, sites=sites)
    _assert_balanced(yr, isi.sites_resampled_)
    assert isi.is_synthetic_.sum() == 2 * 40 + 2 * 120
    _assert_provenance(isi, X, y, sites, Xr, yr)


def test_cleaning_undersampler_rejected(opposite_sites_data):
    """Cleaning methods cannot reach a count, so they are rejected."""
    X, y, sites = opposite_sites_data
    with pytest.raises(ValueError, match="cleaning method"):
        IntraSiteInterpolation(undersampler=TomekLinks(), max_amplification=0.5).fit_resample(X, y, sites=sites)


def test_undersampler_as_interpolator_rejected(opposite_sites_data):
    """An under-sampler is not a valid interpolator."""
    X, y, sites = opposite_sites_data
    with pytest.raises(ValueError, match="over-sampler"):
        IntraSiteInterpolation(interpolator=RandomUnderSampler()).fit_resample(X, y, sites=sites)


class _FailingOverSampler(BaseSampler):
    """Over-sampler that always fails, like ADASYN or KMeans-SMOTE on tiny classes."""

    _sampling_type = "over-sampling"
    _parameter_constraints: dict = {}  # noqa: RUF012

    def __init__(self, sampling_strategy="auto"):
        self.sampling_strategy = sampling_strategy

    def _fit_resample(self, X, y):
        raise RuntimeError("No clusters found with sufficient samples.")


def test_interpolator_failure_suggests_smote(opposite_sites_data):
    """When an interpolator fails, the error suggests SMOTE."""
    X, y, sites = opposite_sites_data
    with pytest.raises(RuntimeError, match="interpolator='smote'"):
        IntraSiteInterpolation(_FailingOverSampler()).fit_resample(X, y, sites=sites)


@pytest.mark.parametrize("value", ["max", -1, "1"])
def test_invalid_max_amplification(opposite_sites_data, value):
    """Invalid caps raise."""
    X, y, sites = opposite_sites_data
    with pytest.raises(ValueError, match="max_amplification"):
        IntraSiteInterpolation(max_amplification=value).fit_resample(X, y, sites=sites)


@pytest.mark.parametrize("value", [0, 1, -0.1])
def test_invalid_variance_tolerance(opposite_sites_data, value):
    """variance_tolerance must be in (0, 1)."""
    X, y, sites = opposite_sites_data
    with pytest.raises(ValueError, match="variance_tolerance"):
        IntraSiteInterpolation(variance_tolerance=value).fit_resample(X, y, sites=sites)


@pytest.mark.parametrize("interpolator", ["smote", "borderline-smote", "svm-smote", "adasyn", "random"])
@pytest.mark.parametrize("max_amplification", [None, "auto"])
def test_interpolators(opposite_sites_data, interpolator, max_amplification):
    """All built-in interpolators work, with and without cap."""
    X, y, sites = opposite_sites_data
    isi = IntraSiteInterpolation(interpolator, max_amplification=max_amplification, random_state=0)
    Xr, yr = isi.fit_resample(X, y, sites=sites)
    _assert_balanced(yr, isi.sites_resampled_)
    _assert_provenance(isi, X, y, sites, Xr, yr)


# ==============================================================================
# samples_created_ and summary
# ==============================================================================


def test_samples_created(binary_data):
    """samples_created_ should be a dict with non-negative integers."""
    X, y, sites = binary_data
    isi = IntraSiteInterpolation("random")

    isi.fit_resample(X, y, sites=sites)

    assert isinstance(isi.samples_created_, dict)

    for d in isi.samples_created_.values():
        for v in d.values():
            assert isinstance(v, numbers.Integral)
            assert v >= 0


def test_summary(opposite_sites_data):
    """summary() reports one consistent row per site-class cell."""
    X, y, sites = opposite_sites_data
    isi = IntraSiteInterpolation("smote", max_amplification=1, random_state=0)
    _, yr = isi.fit_resample(X, y, sites=sites)
    df = isi.summary()
    assert len(df) == 4
    assert (df.n_real - df.n_removed + df.n_created == df.n_final).all()
    assert df.n_final.sum() == len(yr)
    assert df.n_real.sum() == len(y)
    assert 1 <= isi.effective_dim_ <= X.shape[1]


# ==============================================================================
# Validation
# ==============================================================================


def test_invalid_balance_strategy(binary_data):
    """Invalid balance_strategy should raise ValueError."""
    X, y, sites = binary_data
    isi = IntraSiteInterpolation(balance_strategy="invalid")

    with pytest.raises(ValueError):
        isi.fit_resample(X, y, sites=sites)


def test_invalid_interpolator(binary_data):
    """Invalid interpolator name should raise ValueError."""
    X, y, sites = binary_data
    isi = IntraSiteInterpolation("invalid")

    with pytest.raises(ValueError):
        isi.fit_resample(X, y, sites=sites)


def test_invalid_task(binary_data):
    """Invalid task raises."""
    X, y, sites = binary_data
    with pytest.raises(ValueError, match="task"):
        IntraSiteInterpolation(task="clustering").fit_resample(X, y, sites=sites)


def test_missing_class_in_site_raises():
    """A class absent from a site keeps the site predictive: classification raises."""
    rng = np.random.default_rng(5)
    X = rng.standard_normal((90, 3))
    y = np.r_[np.repeat([0, 1], 15), np.repeat([0, 1, 2], 20)]
    sites = np.repeat([0, 1], [30, 60])
    with pytest.raises(ValueError, match="present in every site"):
        IntraSiteInterpolation().fit_resample(X, y, sites=sites)


def test_no_parameter_mutation(binary_data):
    """fit_resample does not modify the constructor parameters."""
    X, y, sites = binary_data
    isi = IntraSiteInterpolation("smote", random_state=42)
    before = isi.get_params()
    isi.fit_resample(X, y, sites=sites)
    assert isi.get_params() == before
    assert isinstance(isi.interpolator_, SMOTE)


# ==============================================================================
# Covariates
# ==============================================================================


@pytest.mark.parametrize("strategy", ["per_site", "global_max"])
def test_covariates(strategy, covariate_data):
    """Covariate stratification works with binning-based grouping."""
    X, y, sites, sex, age = covariate_data

    isi = IntraSiteInterpolation(
        interpolator="random",
        balance_strategy=strategy,
        n_bins_cont_cov=5,
        binning_strategy_cont_cov="quantile",
    )

    Xr, yr = isi.fit_resample(X, y, sites=sites, categorical_covariate=sex, continuous_covariate=age)

    assert len(Xr) == len(yr)
    _assert_balanced(yr, isi.sites_resampled_)


@pytest.mark.parametrize("strategy", ["uniform", "quantile"])
def test_binning_strategy_cont_cov(strategy, covariate_data):
    """Each strategies for binning_strategy_cont_cov."""
    X, y, sites, sex, age = covariate_data
    isi = IntraSiteInterpolation("random", n_bins_cont_cov=2, binning_strategy_cont_cov=strategy)
    _, yr = isi.fit_resample(X, y, sites=sites, categorical_covariate=sex, continuous_covariate=age)
    _assert_balanced(yr, isi.sites_resampled_)


def test_binning_strategy_invalid_cont_cov(covariate_data):
    """Invalid strategies for binning_strategy_cont_cov."""
    X, y, sites, sex, age = covariate_data
    isi = IntraSiteInterpolation("random", n_bins_cont_cov=2, binning_strategy_cont_cov="invalid")
    with pytest.raises(ValueError):
        _, _ = isi.fit_resample(X, y, sites=sites, categorical_covariate=sex, continuous_covariate=age)


def test_deprecated_covariate_options_in_fit_resample(covariate_data):
    """Passing the covariate binning options to fit_resample still works, with a FutureWarning."""
    X, y, sites, _, age = covariate_data
    isi = IntraSiteInterpolation("random")
    with pytest.warns(FutureWarning, match="deprecated"):
        _, yr = isi.fit_resample(X, y, sites=sites, continuous_covariate=age, n_bins_cont_cov=3)
    _assert_balanced(yr, isi.sites_resampled_)


def test_covariates_categorical(covariate_data):
    """Categorical covariates alone, also as a 1D array."""
    X, y, sites, sex, _ = covariate_data
    isi = IntraSiteInterpolation(interpolator="random")
    Xr, yr = isi.fit_resample(X, y, sites=sites, categorical_covariate=sex.ravel())
    assert len(Xr) == len(yr)


def test_covariates_requires_bins(covariate_data):
    """Continuous covariates without n_bins_cont_cov must raise ValueError."""
    X, y, sites, _, age = covariate_data
    isi = IntraSiteInterpolation(interpolator="random")
    with pytest.raises(ValueError, match="n_bins_cont_cov"):
        isi.fit_resample(X, y, sites=sites, continuous_covariate=age)


@pytest.mark.parametrize("max_amplification", [None, "auto"])
def test_global_max_covariates_no_explosion(covariate_data, max_amplification):
    """Strata share the (site, class) target: covariates do not multiply the output size."""
    X, y, sites, sex, age = covariate_data
    kwargs = {"balance_strategy": "global_max", "max_amplification": max_amplification, "random_state": 0}
    _, y_plain = IntraSiteInterpolation("smote", **kwargs).fit_resample(X, y, sites=sites)
    isi = IntraSiteInterpolation("smote", n_bins_cont_cov=3, **kwargs)
    _, y_cov = isi.fit_resample(X, y, sites=sites, categorical_covariate=sex, continuous_covariate=age)
    if max_amplification is None:
        assert len(y_cov) == len(y_plain) == isi.target_count_ * 2 * 2
    assert len(y_cov) == sum(isi.target_counts_.values()) * 2


@pytest.mark.parametrize("max_amplification", [None, 1])
def test_covariates_preserve_distribution(max_amplification):
    """Covariates restrict interpolation partners but keep P(covariate | class, site)."""
    rng = np.random.default_rng(6)
    n = 2000
    X, y, sites = make_multisite_classification(
        n_sites=2, n_samples=n, n_features=4, balance_per_site=[[0.7, 0.3], [0.3, 0.7]], random_state=3
    )
    sex = (rng.random(n) < np.where(y == 1, 0.8, 0.3)).astype(int)
    isi = IntraSiteInterpolation("random", max_amplification=max_amplification, random_state=0)
    Xr, yr = isi.fit_resample(X, y, sites=sites, categorical_covariate=sex)
    # random over-sampling duplicates real rows, so the covariate of every output row is known
    lut = {tuple(r): s for r, s in zip(X, sex, strict=True)}
    sex_r = np.array([lut[tuple(r)] for r in Xr])
    sr = isi.sites_resampled_
    for site in (0, 1):
        for c in (0, 1):
            before = sex[(sites == site) & (y == c)].mean()
            after = sex_r[(sr == site) & (yr == c)].mean()
            assert after == pytest.approx(before, abs=0.03)


def test_covariates_interpolate_within_strata():
    """SMOTE samples lie on segments between two samples of the same class, site and stratum."""
    rng = np.random.default_rng(8)
    X = rng.standard_normal((400, 3))
    y = np.r_[np.zeros(160), np.ones(40), np.zeros(40), np.ones(160)].astype(int)
    sites = np.repeat([0, 1], 200)
    sex = rng.integers(0, 2, 400)
    isi = IntraSiteInterpolation("smote", max_amplification=None, random_state=0)
    Xr, yr = isi.fit_resample(X, y, sites=sites, categorical_covariate=sex)
    sr = isi.sites_resampled_
    for x, label, site in zip(Xr[isi.is_synthetic_][:40], yr[isi.is_synthetic_][:40], sr[isi.is_synthetic_][:40], strict=True):
        distances = [
            _distance_to_closest_segment(x, X[(y == label) & (sites == site) & (sex == s)])
            for s in (0, 1)
            if np.sum((y == label) & (sites == site) & (sex == s)) >= 2
        ]
        assert min(distances) < 1e-10


@pytest.mark.parametrize("interpolator", ["smote", "random", "borderline-smote", "adasyn"])
def test_small_strata_do_not_crash(interpolator):
    """Strata with too few samples are skipped for interpolation instead of crashing."""
    X, y, sites = make_multisite_classification(
        n_sites=2, n_samples=[200, 40], n_features=5, balance_per_site=[[0.8, 0.2], [0.8, 0.2]], random_state=4
    )
    groups = np.random.default_rng(1).integers(0, 4, len(y))
    isi = IntraSiteInterpolation(interpolator, random_state=0)
    Xr, yr = isi.fit_resample(X, y, sites=sites, categorical_covariate=groups)
    _assert_balanced(yr, isi.sites_resampled_)
    _assert_provenance(isi, X, y, sites, Xr, yr)


# ==============================================================================
# Regression
# ==============================================================================


@pytest.mark.parametrize("strategy", ["per_site", "global_max"])
def test_regression_runs(strategy, regression_data):
    """Regression should work when n_bins is provided."""
    X, y, sites = regression_data
    isi = IntraSiteInterpolation(interpolator="random", task="regression", n_bins=5, balance_strategy=strategy)
    Xr, yr = isi.fit_resample(X, y, sites=sites)
    assert len(Xr) == len(yr)
    assert yr.dtype.kind == "f"


@pytest.mark.parametrize("interpolator", ["smote", "random", "borderline-smote", "svm-smote"])
@pytest.mark.parametrize("strategy", ["per_site", "global_max"])
def test_regression_keeps_original_targets(linear_regression_data, interpolator, strategy):
    """Original samples keep their exact original target (regression, no cap)."""
    X, y, sites, _ = linear_regression_data
    isi = IntraSiteInterpolation(
        interpolator, task="regression", n_bins=4, balance_strategy=strategy, max_amplification=None, random_state=0
    )
    Xr, yr = isi.fit_resample(X, y, sites=sites)
    is_original = _assert_originals_unchanged(X, y, Xr, yr)
    assert (~is_original).sum() > 0


def test_regression_hybrid_keeps_targets_of_kept_samples(linear_regression_data):
    """With a cap, the kept real samples keep their targets."""
    X, y, sites, _ = linear_regression_data
    isi = IntraSiteInterpolation("smote", task="regression", n_bins=4, max_amplification="auto", random_state=0)
    Xr, yr = isi.fit_resample(X, y, sites=sites)
    _assert_provenance(isi, X, y, sites, Xr, yr)


def test_regression_keeps_original_targets_with_covariates(linear_regression_data):
    """Original targets are kept when interpolating within covariate strata."""
    X, y, sites, _ = linear_regression_data
    sex = (np.arange(len(y)) % 2)[:, np.newaxis]
    isi = IntraSiteInterpolation("smote", task="regression", n_bins=3, max_amplification=None, random_state=0)
    Xr, yr = isi.fit_resample(X, y, sites=sites, categorical_covariate=sex)
    _assert_originals_unchanged(X, y, Xr, yr)


@pytest.mark.parametrize("max_amplification", [None, "auto"])
def test_regression_smote_targets_follow_parents(linear_regression_data, max_amplification):
    """SMOTE targets are interpolated like their features: y = X @ w holds for synthetic samples too."""
    X, y, sites, weights = linear_regression_data
    isi = IntraSiteInterpolation("smote", task="regression", n_bins=4, max_amplification=max_amplification, random_state=0)
    Xr, yr = isi.fit_resample(X, y, sites=sites)
    synthetic = isi.is_synthetic_
    assert synthetic.sum() > 0
    np.testing.assert_allclose(yr[synthetic], Xr[synthetic] @ weights + 50.0, atol=1e-8)


def test_regression_random_targets_copy_parents(linear_regression_data):
    """Random over-sampling duplicates samples together with their targets."""
    X, y, sites, _ = linear_regression_data
    isi = IntraSiteInterpolation("random", task="regression", n_bins=4, random_state=0)
    Xr, yr = isi.fit_resample(X, y, sites=sites)
    original_target = {tuple(row): target for row, target in zip(X, y, strict=True)}
    for row, target in zip(Xr, yr, strict=True):
        assert target == original_target[tuple(row)]


@pytest.mark.parametrize("interpolator", ["smote", "random", "svm-smote"])
def test_regression_synthetic_targets_stay_in_bin(regression_data, interpolator):
    """Synthetic targets stay within the range of their bin; samples_created_ counts them."""
    X, y, sites = regression_data
    isi = IntraSiteInterpolation(interpolator, task="regression", n_bins=5, random_state=0)
    _, yr = isi.fit_resample(X, y, sites=sites)
    synthetic = isi.is_synthetic_
    original_bins = np.clip(np.digitize(y, isi.bins_[1:-1]), 0, len(isi.bins_) - 2)
    synthetic_bins = np.clip(np.digitize(yr[synthetic], isi.bins_[1:-1]), 0, len(isi.bins_) - 2)
    for b in np.unique(synthetic_bins):
        in_bin = y[original_bins == b]
        assert np.all(yr[synthetic][synthetic_bins == b] >= in_bin.min())
        assert np.all(yr[synthetic][synthetic_bins == b] <= in_bin.max())
    n_created = sum(sum(per_class.values()) for per_class in isi.samples_created_.values())
    n_removed = sum(sum(per_class.values()) for per_class in isi.samples_removed_.values())
    assert n_created == synthetic.sum()
    assert n_created - n_removed == len(yr) - len(y)


def test_regression_missing_bin_warns_and_balances_present_bins(regression_data):
    """A target bin absent from a site is reported; the site is balanced over the bins it has."""
    X, y, sites = regression_data
    y = np.where(sites == 0, y - 60.0, y)  # site 0 has much lower targets: its top bins are empty
    isi = IntraSiteInterpolation("smote", task="regression", n_bins=5, random_state=0)
    with pytest.warns(UserWarning, match="present in every site"):
        _, yr = isi.fit_resample(X, y, sites=sites)
    assert set(isi.samples_created_[0]) != set(range(5))
    _assert_balanced(np.clip(np.digitize(yr, isi.bins_[1:-1]), 0, 4), isi.sites_resampled_)


def test_regression_single_sample_bin_left_as_is(regression_data):
    """A target bin with a single sample in a site cannot be interpolated: it is kept as is, with a warning."""
    X, y, sites = regression_data
    y = np.where(sites == 0, y - 60.0, y)
    y[0] = y[sites == 1].max()  # one site-0 sample in the top bin
    isi = IntraSiteInterpolation("smote", task="regression", n_bins=5, random_state=0)
    with pytest.warns(UserWarning, match="fewer than 2 samples"):
        isi.fit_resample(X, y, sites=sites)
    top_bin = int(np.clip(np.digitize(y[0], isi.bins_[1:-1]), 0, 4))
    assert isi.class_counts_[0][top_bin] == 1
    assert isi.samples_created_[0][top_bin] == 0
    assert 0 in isi.sample_indices_


# ==============================================================================
# Targets of classification samples
# ==============================================================================


@pytest.mark.parametrize("interpolator", ["smote", "random"])
@pytest.mark.parametrize(
    "labels",
    [np.array([0, 1]), np.array([3, 7], dtype=np.int32), np.array(["AD", "CN"]), np.array([False, True])],
)
@pytest.mark.parametrize("max_amplification", [None, "auto"])
def test_classification_keeps_labels_and_dtype(binary_data, interpolator, labels, max_amplification):
    """Classification labels are neither reconstructed nor cast to float; real samples are kept unchanged."""
    X, y, sites = binary_data
    y_labels = labels[y]
    isi = IntraSiteInterpolation(interpolator, max_amplification=max_amplification, random_state=0)
    Xr, yr = isi.fit_resample(X, y_labels, sites=sites)
    assert isi.task_ == "classification"
    assert yr.dtype == y_labels.dtype
    assert set(np.unique(yr)) <= set(labels)
    _assert_provenance(isi, X, y_labels, sites, Xr, yr)
    if max_amplification is None:
        _assert_originals_unchanged(X, y_labels, Xr, yr)


def test_classification_synthetic_labels_from_parents(binary_data):
    """SMOTE samples lie between two samples of the class they are labeled with."""
    X, y, sites = binary_data
    isi = IntraSiteInterpolation("smote", random_state=0)
    Xr, yr = isi.fit_resample(X, y, sites=sites)
    synthetic = isi.is_synthetic_
    sr = isi.sites_resampled_
    for x, label, site in zip(Xr[synthetic][:50], yr[synthetic][:50], sr[synthetic][:50], strict=True):
        assert _distance_to_closest_segment(x, X[(y == label) & (sites == site)]) < 1e-10


class _ShuffledOverSampler(BaseSampler):
    """Over-sampler that does not return the original samples first."""

    _sampling_type = "over-sampling"
    _parameter_constraints: dict = {}  # noqa: RUF012

    def __init__(self, sampling_strategy="auto"):
        self.sampling_strategy = sampling_strategy

    def _fit_resample(self, X, y):
        idx = np.concatenate([np.arange(len(X)), np.flatnonzero(y == np.unique(y)[0])])[::-1]
        return X[idx], y[idx]


def test_interpolator_must_keep_originals_first(binary_data):
    """Interpolators that reorder the original samples are rejected with a clear error."""
    X, y, sites = binary_data
    isi = IntraSiteInterpolation(_ShuffledOverSampler(), max_amplification=None)
    with pytest.raises(ValueError, match="original samples first"):
        isi.fit_resample(X, y, sites=sites)


# ==============================================================================
# sklearn / imblearn integration
# ==============================================================================


def test___sklearn_tags__():
    """Test __sklearn_tags__."""
    tags = IntraSiteInterpolation(interpolator="random").__sklearn_tags__()
    assert tags.estimator_type == "sampler"


def test_compatibility(binary_data):
    """_fit_resample requires sites and otherwise delegates to fit_resample."""
    X, y, sites = binary_data
    isi = IntraSiteInterpolation("random", random_state=0)
    with pytest.raises(TypeError, match="sites"):
        isi._fit_resample(X, y)
    Xr, _ = isi._fit_resample(X, y, sites=sites)
    assert len(Xr) == len(isi.sites_resampled_)


def test_pipeline_with_metadata_routing(opposite_sites_data):
    """ISI is applied to the training folds only, with sites routed through an imblearn Pipeline."""
    X, y, sites = opposite_sites_data
    with config_context(enable_metadata_routing=True):
        isi = IntraSiteInterpolation("smote", random_state=0).set_fit_resample_request(sites=True)
        pipe = Pipeline([("isi", isi), ("clf", LogisticRegression())])
        scores = cross_validate(
            pipe, X, y, cv=StratifiedKFold(3, shuffle=True, random_state=0), params={"sites": sites}, error_score="raise"
        )
    assert len(scores["test_score"]) == 3


def test_nested_tuning_with_metadata_routing(opposite_sites_data):
    """ISI inside a tuned pipeline: re-applied to every inner split, with sites routed to it."""
    from sklearn.linear_model import RidgeClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler

    X, y, sites = opposite_sites_data
    with config_context(enable_metadata_routing=True):
        isi = IntraSiteInterpolation("smote", random_state=0).set_fit_resample_request(sites=True)
        pipe = Pipeline([("isi", isi), ("scale", StandardScaler()), ("clf", RidgeClassifier())])
        search = GridSearchCV(pipe, {"clf__alpha": [0.1, 1000.0]}, cv=StratifiedKFold(3, shuffle=True, random_state=0))
        scores = cross_validate(
            search, X, y, cv=StratifiedKFold(3, shuffle=True, random_state=1), params={"sites": sites}, error_score="raise"
        )
    assert len(scores["test_score"]) == 3


# ==============================================================================
# Reproducibility
# ==============================================================================


@pytest.mark.parametrize("max_amplification", [None, "auto"])
def test_reproducibility(binary_data, max_amplification):
    """Same random_state must produce identical results, also when refitting the same object."""
    X, y, sites = binary_data
    isi1 = IntraSiteInterpolation("smote", max_amplification=max_amplification, random_state=42)
    isi2 = IntraSiteInterpolation("smote", max_amplification=max_amplification, random_state=42)
    X1, y1 = isi1.fit_resample(X, y, sites=sites)
    X2, y2 = isi2.fit_resample(X, y, sites=sites)
    X3, y3 = isi1.fit_resample(X, y, sites=sites)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)
    np.testing.assert_array_equal(X1, X3)
    np.testing.assert_array_equal(y1, y3)


def test_reproducibility_regression(regression_data):
    """Same random_state gives identical results for regression."""
    X, y, sites = regression_data
    isi = IntraSiteInterpolation("smote", task="regression", n_bins=5, random_state=42)
    X1, y1 = isi.fit_resample(X, y, sites=sites)
    X2, y2 = IntraSiteInterpolation("smote", task="regression", n_bins=5, random_state=42).fit_resample(X, y, sites=sites)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)
    assert isi.random_state == 42


def test_sites_get_different_random_streams():
    """Identical sites do not receive identical synthetic samples."""
    rng = np.random.default_rng(9)
    X0 = rng.standard_normal((100, 3))
    y0 = np.r_[np.zeros(80), np.ones(20)].astype(int)
    X, y, sites = np.vstack([X0, X0]), np.r_[y0, y0], np.repeat([0, 1], 100)
    isi = IntraSiteInterpolation("smote", max_amplification=None, random_state=0)
    Xr, _ = isi.fit_resample(X, y, sites=sites)
    syn = isi.is_synthetic_
    sr = isi.sites_resampled_
    assert not np.allclose(Xr[syn & (sr == 0)], Xr[syn & (sr == 1)])


# ==============================================================================
# Utilities
# ==============================================================================


def test_allocate_proportionally():
    """Largest-remainder allocation sums to the total and respects capacity."""
    np.testing.assert_array_equal(allocate_proportionally(10, [3, 1, 0, 6]), [3, 1, 0, 6])
    alloc = allocate_proportionally(7, [1, 1, 1], capacity=[1, 10, 10])
    assert alloc.sum() == 7
    assert alloc[0] <= 1
    np.testing.assert_array_equal(allocate_proportionally(0, [1, 2]), [0, 0])
    with pytest.raises(ValueError):
        allocate_proportionally(5, [1, 1], capacity=[1, 1])


def test_variance_ratio():
    """Duplicates keep the variance; segment interpolation between two points loses most of it."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 4))
    assert variance_ratio(X[rng.integers(0, 500, 5000)], X) == pytest.approx(1.0, abs=0.05)
    a, b = np.zeros(3), np.ones(3)
    lam = rng.random((5000, 1))
    assert variance_ratio(a + lam * (b - a), np.vstack([a, b])) == pytest.approx(1 / 6, abs=0.02)
    assert np.isnan(variance_ratio(X[:1], X))


def test_effective_dimension():
    """Participation ratio: about p for independent features, small for collinear ones."""
    rng = np.random.default_rng(1)
    assert effective_dimension(rng.standard_normal((2000, 10))) == pytest.approx(10, rel=0.1)
    low_rank = rng.standard_normal((2000, 2)) @ rng.standard_normal((2, 30))
    assert effective_dimension(low_rank + 1e-3 * rng.standard_normal((2000, 30))) < 3


def test_effective_dimension_matches_eigenvalues_and_subsamples():
    """Trace / Frobenius formula equals the eigenvalue definition; the row cap gives a close estimate."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((3000, 8)) @ rng.standard_normal((8, 60)) + 0.5 * rng.standard_normal((3000, 60))
    Z = (X - X.mean(0)) / X.std(0)
    eig = np.linalg.eigvalsh(Z.T @ Z)
    exact = eig.sum() ** 2 / np.sum(eig**2)
    assert effective_dimension(X, max_samples=None) == pytest.approx(exact, rel=1e-9)
    assert effective_dimension(X, max_samples=1000) == pytest.approx(exact, rel=0.1)
    # p > n: the smaller Gram matrix gives the same value
    W = rng.standard_normal((50, 400))
    Zw = (W - W.mean(0)) / W.std(0)
    ew = np.linalg.eigvalsh(Zw @ Zw.T)
    assert effective_dimension(W) == pytest.approx(ew.sum() ** 2 / np.sum(ew**2), rel=1e-9)


def test_sampler_factories():
    """Factories are case-insensitive and reject unknown names."""
    assert isinstance(create_interpolator("SMOTE"), SMOTE)
    assert isinstance(create_undersampler("Random", random_state=0), RandomUnderSampler)
    assert create_undersampler("nearmiss-3").version == 3
    with pytest.raises(ValueError):
        create_undersampler("tomek")
