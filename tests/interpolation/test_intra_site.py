"""Test IntraSiteInterpolation transformer."""

import numbers

import numpy as np
import pytest
from imblearn.base import BaseSampler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

from uniharmony.datasets import make_multisite_classification
from uniharmony.interpolation import IntraSiteInterpolation


# ==============================================================================
# Fixtures
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


# ==============================================================================
# Basic functionality
# ==============================================================================


def test_basic_run(binary_data):
    """Model should run and return valid shapes."""
    X, y, sites = binary_data
    isi = IntraSiteInterpolation("random")

    Xr, yr = isi.fit_resample(X, y, sites=sites)

    assert len(Xr) == len(yr)
    assert Xr.ndim == 2
    assert yr.ndim == 1


def test_basic_run_invalid_instance(binary_data):
    """Model should run and return valid shapes."""
    X, y, sites = binary_data
    interpolator = LogisticRegression()
    isi = IntraSiteInterpolation(interpolator=interpolator)
    with pytest.raises(ValueError):
        _, _ = isi.fit_resample(X, y, sites=sites)


def test_basic_run_no_balance():
    """Model should run but no resampling."""
    X, y, sites = make_multisite_classification()
    interpolator = SMOTE()

    isi = IntraSiteInterpolation(interpolator=interpolator)
    _, _ = isi.fit_resample(X, y, sites=sites)


def test_basic_run_no_balance_small():
    """Model should run but no resampling with small data."""
    X, y, sites = make_multisite_classification(n_samples=[2, 19])
    interpolator = SMOTE()

    isi = IntraSiteInterpolation(interpolator=interpolator)
    _, _ = isi.fit_resample(X, y, sites=sites)


# ==============================================================================
# Balance correctness
# ==============================================================================


@pytest.mark.parametrize("strategy", ["per_site", "global_max"])
def test_balance_strategy(strategy, binary_data):
    """Each site must be class-balanced after resampling."""
    X, y, sites = binary_data
    isi = IntraSiteInterpolation("random", balance_strategy=strategy)

    _, yr = isi.fit_resample(X, y, sites=sites)
    sr = isi.sites_resampled_

    for site in np.unique(sr):
        counts = np.unique(yr[sr == site], return_counts=True)[1]
        assert len(set(counts)) == 1


## Binning strategies
@pytest.mark.parametrize("strategy", ["uniform", "quantile"])
def test_binning_strategy(strategy, regression_data):
    """Each strategies for binning_strategy."""
    X, y, sites = regression_data
    isi = IntraSiteInterpolation("random", binning_strategy=strategy)

    _, _ = isi.fit_resample(X, y, sites=sites)


def test_binning_strategy_invalid(regression_data):
    """Invalid strategies for binning."""
    X, y, sites = regression_data
    isi = IntraSiteInterpolation("random", binning_strategy="invalid")
    with pytest.raises(ValueError):
        _, _ = isi.fit_resample(X, y, sites=sites)


## continuos Binning strategies
@pytest.mark.parametrize("strategy", ["uniform", "quantile"])
def test_binning_strategy_cont_cov(strategy, covariate_data):
    """Each strategies for binning_strategy_cont_cov(."""
    X, y, sites, sex, age = covariate_data
    isi = IntraSiteInterpolation("random")

    _, _ = isi.fit_resample(
        X,
        y,
        sites=sites,
        categorical_covariate=sex,
        continuous_covariate=age,
        n_bins_cont_cov=2,
        binning_strategy_cont_cov=strategy,
    )


def test_binning_strategy_invalid_cont_cov(covariate_data):
    """Invalid strategies for binning_strategy_cont_cov."""
    X, y, sites, sex, age = covariate_data
    isi = IntraSiteInterpolation("random")
    with pytest.raises(ValueError):
        _, _ = isi.fit_resample(
            X,
            y,
            sites=sites,
            categorical_covariate=sex,
            continuous_covariate=age,
            n_bins_cont_cov=2,
            binning_strategy_cont_cov="invalid",
        )


# ==============================================================================
# samples_created_
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


# ==============================================================================
# Covariates
# ==============================================================================


@pytest.mark.parametrize("strategy", ["per_site", "global_max"])
def test_covariates(strategy, covariate_data):
    """Covariate stratification should work with binning-based grouping."""
    X, y, sites, sex, age = covariate_data

    isi = IntraSiteInterpolation(
        interpolator="random",
        balance_strategy=strategy,
    )

    Xr, yr = isi.fit_resample(
        X,
        y,
        sites=sites,
        categorical_covariate=sex,
        continuous_covariate=age,
        n_bins_cont_cov=5,
        binning_strategy_cont_cov="quantile",
    )

    assert len(Xr) == len(yr)


def test___sklearn_tags__():
    """Test __sklearn_tags__."""
    isi = IntraSiteInterpolation(
        interpolator="random",
    )
    isi.__sklearn_tags__()


def test_compatibility(binary_data):
    """Test compatibility."""
    X, y, _ = binary_data
    isi = IntraSiteInterpolation()
    isi._fit_resample(X, y)


def test_covariates_categorical(covariate_data):
    """Covariate stratification should work with binning-based grouping."""
    X, y, sites, sex, _ = covariate_data

    isi = IntraSiteInterpolation(
        interpolator="random",
    )

    Xr, yr = isi.fit_resample(
        X,
        y,
        sites=sites,
        categorical_covariate=sex,
    )

    assert len(Xr) == len(yr)


def test_covariates_requires_bins(covariate_data):
    """Continuous covariates without n_bins_cont_cov must raise ValueError."""
    X, y, sites, _, age = covariate_data

    isi = IntraSiteInterpolation(
        interpolator="random",
    )

    with pytest.raises(ValueError):
        isi.fit_resample(
            X,
            y,
            sites=sites,
            continuous_covariate=age,
        )


# ==============================================================================
# Regression
# ==============================================================================


@pytest.mark.parametrize("strategy", ["per_site", "global_max"])
def test_regression_runs(strategy, regression_data):
    """Regression should work when n_bins is provided."""
    X, y, sites = regression_data

    isi = IntraSiteInterpolation(
        interpolator="random",
        task="regression",
        n_bins=5,
        balance_strategy=strategy,
    )

    Xr, yr = isi.fit_resample(X, y, sites=sites)

    assert len(Xr) == len(yr)
    assert yr.dtype.kind == "f"


# ==============================================================================
# Targets of original and synthetic samples
# ==============================================================================


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


@pytest.mark.parametrize("interpolator", ["smote", "random", "borderline-smote", "svm-smote"])
@pytest.mark.parametrize("strategy", ["per_site", "global_max"])
def test_regression_keeps_original_targets(linear_regression_data, interpolator, strategy):
    """Original samples keep their exact original target (regression)."""
    X, y, sites, _ = linear_regression_data
    isi = IntraSiteInterpolation(interpolator, task="regression", n_bins=4, balance_strategy=strategy, random_state=0)
    Xr, yr = isi.fit_resample(X, y, sites=sites)
    is_original = _assert_originals_unchanged(X, y, Xr, yr)
    assert (~is_original).sum() > 0


def test_regression_keeps_original_targets_with_covariates(linear_regression_data):
    """Original targets are kept when balancing within covariate strata."""
    X, y, sites, _ = linear_regression_data
    sex = (np.arange(len(y)) % 2)[:, np.newaxis]
    isi = IntraSiteInterpolation("smote", task="regression", n_bins=3, random_state=0)
    Xr, yr = isi.fit_resample(X, y, sites=sites, categorical_covariate=sex)
    _assert_originals_unchanged(X, y, Xr, yr)


def test_regression_smote_targets_follow_parents(linear_regression_data):
    """SMOTE targets are interpolated like their features: y = X @ w holds for synthetic samples too."""
    X, y, sites, weights = linear_regression_data
    isi = IntraSiteInterpolation("smote", task="regression", n_bins=4, random_state=0)
    Xr, yr = isi.fit_resample(X, y, sites=sites)
    synthetic = ~_split_original_and_synthetic(X, Xr)
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
    Xr, yr = isi.fit_resample(X, y, sites=sites)
    synthetic = ~_split_original_and_synthetic(X, Xr)
    original_bins = np.clip(np.digitize(y, isi.bins_[1:-1]), 0, len(isi.bins_) - 2)
    synthetic_bins = np.clip(np.digitize(yr[synthetic], isi.bins_[1:-1]), 0, len(isi.bins_) - 2)
    for b in np.unique(synthetic_bins):
        in_bin = y[original_bins == b]
        assert np.all(yr[synthetic][synthetic_bins == b] >= in_bin.min())
        assert np.all(yr[synthetic][synthetic_bins == b] <= in_bin.max())
    n_created = sum(sum(per_class.values()) for per_class in isi.samples_created_.values())
    assert n_created == len(yr) - len(y)


@pytest.mark.parametrize("interpolator", ["smote", "random"])
@pytest.mark.parametrize(
    "labels",
    [np.array([0, 1]), np.array([3, 7], dtype=np.int32), np.array(["AD", "CN"]), np.array([False, True])],
)
def test_classification_keeps_labels_and_dtype(binary_data, interpolator, labels):
    """Classification labels are neither reconstructed nor cast to float; originals are kept."""
    X, y, sites = binary_data
    y_labels = labels[y]
    isi = IntraSiteInterpolation(interpolator, random_state=0)
    Xr, yr = isi.fit_resample(X, y_labels, sites=sites)
    assert isi.task_ == "classification"
    assert yr.dtype == y_labels.dtype
    assert set(np.unique(yr)) <= set(labels)
    _assert_originals_unchanged(X, y_labels, Xr, yr)


def test_classification_synthetic_labels_from_parents(binary_data):
    """SMOTE samples lie between two samples of the class they are labeled with."""
    X, y, sites = binary_data
    isi = IntraSiteInterpolation("smote", random_state=0)
    Xr, yr = isi.fit_resample(X, y, sites=sites)
    synthetic = ~_split_original_and_synthetic(X, Xr)
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
    isi = IntraSiteInterpolation(_ShuffledOverSampler())
    with pytest.raises(ValueError, match="original samples first"):
        isi.fit_resample(X, y, sites=sites)


# ==============================================================================
# Reproducibility
# ==============================================================================


def test_reproducibility(binary_data):
    """Same random_state must produce identical results."""
    X, y, sites = binary_data

    isi1 = IntraSiteInterpolation("random", random_state=42)
    isi2 = IntraSiteInterpolation("random", random_state=42)

    X1, y1 = isi1.fit_resample(X, y, sites=sites)
    X2, y2 = isi2.fit_resample(X, y, sites=sites)

    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_reproducibility_regression(regression_data):
    """Same random_state gives identical results for regression, also when refitting."""
    X, y, sites = regression_data
    isi = IntraSiteInterpolation("smote", task="regression", n_bins=5, random_state=42)
    X1, y1 = isi.fit_resample(X, y, sites=sites)
    X2, y2 = IntraSiteInterpolation("smote", task="regression", n_bins=5, random_state=42).fit_resample(X, y, sites=sites)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)
    assert isi.random_state == 42


# ==============================================================================
# Monkeypatch robustness
# ==============================================================================
