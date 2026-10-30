"""Test SiteBalancedResampler (within-site over-sampling with ISI combined with under-sampling)."""

import warnings

import numpy as np
import pytest
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler, TomekLinks
from sklearn import config_context
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

from uniharmony.datasets import make_multisite_classification
from uniharmony.interpolation import IntraSiteInterpolation, SiteBalancedResampler


# ==============================================================================
# Helpers (fixtures are in conftest.py)
# ==============================================================================


def _counts(yr, sr):
    """``{site: {class: count}}``."""
    return {s: dict(zip(*np.unique(yr[sr == s], return_counts=True), strict=True)) for s in np.unique(sr)}


def _assert_balanced(yr, sr):
    """Every class has the same count within each site."""
    for counts in _counts(yr, sr).values():
        assert len(set(counts.values())) == 1


def _assert_provenance(sbr, X, y, sites, Xr, yr):
    """``sample_indices_`` / ``is_synthetic_`` / ``sites_resampled_`` describe the output; real rows are unique."""
    real = ~sbr.is_synthetic_
    np.testing.assert_array_equal(sbr.is_synthetic_, sbr.sample_indices_ < 0)
    np.testing.assert_array_equal(Xr[real], X[sbr.sample_indices_[real]])
    np.testing.assert_array_equal(yr[real], y[sbr.sample_indices_[real]])
    np.testing.assert_array_equal(sbr.sites_resampled_[real], sites[sbr.sample_indices_[real]])
    assert len(np.unique(sbr.sample_indices_[real])) == real.sum()


def _assert_accounting(sbr, y, yr):
    """Check that removed and created samples add up to the output size."""
    n_removed = sum(sum(d.values()) for d in sbr.samples_removed_.values())
    n_created = sum(sum(d.values()) for d in sbr.samples_created_.values())
    assert n_created == sbr.is_synthetic_.sum()
    assert len(yr) == len(y) - n_removed + n_created
    assert (~sbr.is_synthetic_).sum() == len(y) - n_removed


def _extreme_data():
    """Two sites with 1000 samples of one class and 2 of the other (opposite classes)."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((2004, 10))
    y = np.r_[np.zeros(1000), np.ones(2), np.zeros(2), np.ones(1000)].astype(int)
    return X, y, np.repeat([0, 1], 1002)


# ==============================================================================
# Defaults and limiting cases
# ==============================================================================


def test_defaults():
    """Default: ISI with SMOTE, cap at the safe amplification, cluster-centroids under-sampling."""
    sbr = SiteBalancedResampler()
    assert sbr.oversampler is None
    assert sbr.max_amplification == "auto"
    assert sbr.undersampler == "cluster-centroids"


@pytest.mark.parametrize("interpolator", ["smote", "random"])
def test_no_cap_equals_oversampler_alone(opposite_sites_data, interpolator):
    """max_amplification=None removes nothing and gives exactly the output of the over-sampler."""
    X, y, sites = opposite_sites_data
    isi = IntraSiteInterpolation(interpolator, random_state=0)
    X_isi, y_isi = isi.fit_resample(X, y, sites=sites)
    sbr = SiteBalancedResampler(IntraSiteInterpolation(interpolator, random_state=0), max_amplification=None)
    X_sbr, y_sbr = sbr.fit_resample(X, y, sites=sites)
    np.testing.assert_array_equal(X_sbr, X_isi)
    np.testing.assert_array_equal(y_sbr, y_isi)
    np.testing.assert_array_equal(sbr.sample_indices_, isi.sample_indices_)
    assert all(v == 0 for d in sbr.samples_removed_.values() for v in d.values())
    assert sbr.samples_created_ == isi.samples_created_


def test_zero_cap_is_pure_undersampling(opposite_sites_data):
    """max_amplification=0 creates nothing: within-site under-sampling."""
    X, y, sites = opposite_sites_data
    sbr = SiteBalancedResampler(max_amplification=0, undersampler="random", random_state=0)
    Xr, yr = sbr.fit_resample(X, y, sites=sites)
    assert not sbr.is_synthetic_.any()
    assert len(yr) == 4 * 80
    assert sbr.samples_removed_ == {0: {0: 240, 1: 0}, 1: {0: 0, 1: 240}}
    _assert_balanced(yr, sbr.sites_resampled_)
    _assert_provenance(sbr, X, y, sites, Xr, yr)


def test_zero_cap_handles_single_sample_class():
    """A class with one sample cannot be interpolated, but under-sampling to one sample per class works."""
    rng = np.random.default_rng(4)
    X = rng.standard_normal((60, 3))
    y = np.r_[np.zeros(29), np.ones(1), np.zeros(15), np.ones(15)].astype(int)
    sites = np.repeat([0, 1], 30)
    sbr = SiteBalancedResampler(max_amplification="auto", undersampler="random", random_state=0)
    with pytest.warns(UserWarning, match="removed by under-sampling"):
        Xr, yr = sbr.fit_resample(X, y, sites=sites)
    assert sbr.target_counts_[0] == 1
    assert sbr.safe_amplification_[0][1] == 0
    _assert_balanced(yr, sbr.sites_resampled_)
    _assert_provenance(sbr, X, y, sites, Xr, yr)


# ==============================================================================
# Amplification cap
# ==============================================================================


@pytest.mark.parametrize("strategy", ["per_site", "global_max"])
@pytest.mark.parametrize("max_amplification", ["auto", None, 1.0, 0])
def test_balance_strategy(strategy, max_amplification, binary_data):
    """Each site is class-balanced at its target, whatever the cap and the balance strategy."""
    X, y, sites = binary_data
    sbr = SiteBalancedResampler(
        IntraSiteInterpolation("smote", balance_strategy=strategy, random_state=0),
        max_amplification=max_amplification,
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        Xr, yr = sbr.fit_resample(X, y, sites=sites)
    sr = sbr.sites_resampled_
    for site, target in sbr.target_counts_.items():
        assert all(n == target for n in _counts(yr, sr)[site].values())
    _assert_provenance(sbr, X, y, sites, Xr, yr)
    _assert_accounting(sbr, y, yr)


def test_auto_cap_removes_samples_with_warning():
    """'auto' applies the safe amplification and under-samples the rest; the removal is reported."""
    X, y, sites = _extreme_data()
    sbr = SiteBalancedResampler(IntraSiteInterpolation("smote", random_state=0), random_state=0)
    with pytest.warns(UserWarning, match="removed by under-sampling") as record:
        Xr, yr = sbr.fit_resample(X, y, sites=sites)
    # the over-sampler stays within the safe amount, so it does not warn
    assert not any("too small, compared with the largest class" in str(w.message) for w in record)
    assert len(yr) < 20
    assert sbr.amplification_cap_[0][1] == sbr.safe_amplification_[0][1]
    _assert_balanced(yr, sbr.sites_resampled_)
    _assert_provenance(sbr, X, y, sites, Xr, yr)
    _assert_accounting(sbr, y, yr)


def test_safe_amplification_follows_variance_rule(opposite_sites_data):
    """r* = eps / (|1 - rho| - eps) from the measured variance ratio; 'auto' applies it."""
    X, y, sites = opposite_sites_data
    eps = 0.1
    sbr = SiteBalancedResampler(IntraSiteInterpolation("smote", variance_tolerance=eps, random_state=0), random_state=0)
    sbr.fit_resample(X, y, sites=sites)
    for site, minority in [(0, 1), (1, 0)]:
        rho = sbr.variance_ratio_[site][minority]
        assert 0 < rho < 1
        expected = np.inf if abs(1 - rho) <= eps else eps / (abs(1 - rho) - eps)
        assert sbr.safe_amplification_[site][minority] == pytest.approx(expected)
        assert sbr.amplification_cap_[site][minority] == pytest.approx(expected)
        assert sbr.amplification_[site][minority] <= expected + 1e-12
        # majority classes are not over-sampled, so they are neither measured nor capped
        assert np.isnan(sbr.safe_amplification_[site][1 - minority])
        assert np.isinf(sbr.amplification_cap_[site][1 - minority])


def test_safe_amplification_measured_on_all_samples(opposite_sites_data):
    """The cap comes from all real samples of the site, as ISI alone measures it, not from the kept ones."""
    X, y, sites = opposite_sites_data
    isi = IntraSiteInterpolation("smote", random_state=0)
    isi.fit_resample(X, y, sites=sites)
    sbr = SiteBalancedResampler(IntraSiteInterpolation("smote", random_state=0), random_state=0)
    sbr.fit_resample(X, y, sites=sites)
    # same seed, same first pilot draw from the same samples: identical (later sites use other draws)
    assert sbr.variance_ratio_[0][1] == isi.variance_ratio_[0][1]
    assert sbr.safe_amplification_[0][1] == isi.safe_amplification_[0][1]
    # the over-sampler then only sees the kept samples
    assert sbr.oversampler_.class_counts_[0][0] == sbr.target_counts_[0]


@pytest.mark.parametrize("cap", [0, 0.5, 1, 2])
def test_numeric_cap(opposite_sites_data, cap):
    """A numeric cap bounds the synthetic samples per real sample; the majority is under-sampled to meet."""
    X, y, sites = opposite_sites_data
    sbr = SiteBalancedResampler(IntraSiteInterpolation("smote", random_state=0), max_amplification=cap, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # a cap above the safe amplification warns (ISI)
        Xr, yr = sbr.fit_resample(X, y, sites=sites)
    expected = min(320, int(np.floor(80 * (1 + cap))))
    assert sbr.target_counts_ == {0: expected, 1: expected}
    assert sbr.samples_created_[0][1] == expected - 80
    assert sbr.samples_removed_[0][0] == 320 - expected
    assert all(a <= cap + 1e-12 for d in sbr.amplification_.values() for a in d.values())
    _assert_balanced(yr, sbr.sites_resampled_)
    _assert_provenance(sbr, X, y, sites, Xr, yr)
    _assert_accounting(sbr, y, yr)


def test_cap_above_safe_amplification_warns():
    """A numeric cap above the safe amplification warns, pointing to the caller and advising a lower cap."""
    X, y, sites = _extreme_data()
    sbr = SiteBalancedResampler(max_amplification=None, random_state=0)
    with pytest.warns(UserWarning, match="Lower `max_amplification`") as record:
        sbr.fit_resample(X, y, sites=sites)
    assert any(w.filename == __file__ for w in record)
    assert not any("Consider SiteBalancedResampler" in str(w.message) for w in record)


def test_callable_cap(opposite_sites_data):
    """A callable receives the real samples of the class and returns its cap."""
    X, y, sites = opposite_sites_data
    seen = []

    def cap(X_cell):
        seen.append(len(X_cell))
        return 0.25

    sbr = SiteBalancedResampler(max_amplification=cap, random_state=0)
    sbr.fit_resample(X, y, sites=sites)
    assert seen == [80, 80]
    assert sbr.target_counts_ == {0: 100, 1: 100}


def test_random_oversampler_with_cap(opposite_sites_data):
    """Duplication can be combined with under-sampling too ('auto' does not limit it: rho = 1)."""
    X, y, sites = opposite_sites_data
    sbr = SiteBalancedResampler(IntraSiteInterpolation("random", random_state=0), max_amplification=1, random_state=0)
    Xr, yr = sbr.fit_resample(X, y, sites=sites)
    assert sbr.target_counts_ == {0: 160, 1: 160}
    _assert_provenance(sbr, X, y, sites, Xr, yr)
    sbr_auto = SiteBalancedResampler(IntraSiteInterpolation("random", random_state=0), random_state=0)
    sbr_auto.fit_resample(X, y, sites=sites)
    assert np.isinf(sbr_auto.safe_amplification_[0][1])
    assert sum(sum(d.values()) for d in sbr_auto.samples_removed_.values()) == 0


# ==============================================================================
# Under-samplers
# ==============================================================================


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
    sbr = SiteBalancedResampler(
        IntraSiteInterpolation("smote", random_state=0), undersampler=undersampler, max_amplification=0.5, random_state=0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        Xr, yr = sbr.fit_resample(X, y, sites=sites)
    assert sbr.target_counts_ == {0: 120, 1: 120}
    _assert_balanced(yr, sbr.sites_resampled_)
    _assert_provenance(sbr, X, y, sites, Xr, yr)
    assert sbr.is_synthetic_.sum() == 80


def test_undersampler_kwargs(opposite_sites_data):
    """Keyword arguments are passed to an under-sampler given by name."""
    X, y, sites = opposite_sites_data
    sbr = SiteBalancedResampler(undersampler="nearmiss", undersampler_kwargs={"n_neighbors": 2}, max_amplification=0)
    sbr.fit_resample(X, y, sites=sites)
    assert sbr.undersampler_.n_neighbors == 2


def test_soft_cluster_centroids_rejected(opposite_sites_data):
    """ClusterCentroids with soft voting creates centroids, not real samples: rejected."""
    X, y, sites = opposite_sites_data
    sbr = SiteBalancedResampler(undersampler=ClusterCentroids(voting="soft"), max_amplification=0.5, random_state=0)
    with pytest.raises(ValueError, match="creates new samples"):
        sbr.fit_resample(X, y, sites=sites)


def test_cleaning_undersampler_rejected(opposite_sites_data):
    """Cleaning methods cannot reach a count, so they are rejected."""
    X, y, sites = opposite_sites_data
    with pytest.raises(ValueError, match="cleaning method"):
        SiteBalancedResampler(undersampler=TomekLinks(), max_amplification=0.5).fit_resample(X, y, sites=sites)


def test_oversampler_as_undersampler_rejected(opposite_sites_data):
    """An over-sampler is not a valid under-sampler."""
    X, y, sites = opposite_sites_data
    with pytest.raises(ValueError, match="under-sampler"):
        SiteBalancedResampler(undersampler=SMOTE()).fit_resample(X, y, sites=sites)


@pytest.mark.parametrize("oversampler", [SMOTE(), "smote", RandomUnderSampler()])
def test_invalid_oversampler(opposite_sites_data, oversampler):
    """The over-sampler must be an IntraSiteInterpolation instance."""
    X, y, sites = opposite_sites_data
    with pytest.raises(ValueError, match="IntraSiteInterpolation"):
        SiteBalancedResampler(oversampler).fit_resample(X, y, sites=sites)


@pytest.mark.parametrize("value", ["max", -1, "1", [1]])
def test_invalid_max_amplification(opposite_sites_data, value):
    """Invalid caps raise."""
    X, y, sites = opposite_sites_data
    with pytest.raises(ValueError, match="max_amplification"):
        SiteBalancedResampler(max_amplification=value).fit_resample(X, y, sites=sites)


def test_no_parameter_mutation(opposite_sites_data):
    """fit_resample does not modify the given over-sampler or the parameters."""
    X, y, sites = opposite_sites_data
    isi = IntraSiteInterpolation("smote", random_state=0)
    sbr = SiteBalancedResampler(isi, max_amplification=1, random_state=0)
    params = sbr.get_params(deep=False)
    sbr.fit_resample(X, y, sites=sites)
    assert isi.balance_strategy == "per_site"
    assert not hasattr(isi, "target_counts_")
    assert sbr.get_params(deep=False) == params
    assert sbr.oversampler_ is not isi


def test_summary(opposite_sites_data):
    """summary() reports every site-class cell; n_final equals the target."""
    X, y, sites = opposite_sites_data
    sbr = SiteBalancedResampler(max_amplification=1, random_state=0)
    sbr.fit_resample(X, y, sites=sites)
    report = sbr.summary()
    assert len(report) == 4
    assert {"n_real", "n_removed", "n_created", "n_final", "amplification_cap"} <= set(report.columns)
    assert (report["n_final"] == 160).all()


# ==============================================================================
# Covariates and regression
# ==============================================================================


@pytest.mark.parametrize("max_amplification", [None, 1, 0])
def test_covariates_preserve_distribution(max_amplification):
    """Both steps are stratified by the covariates: P(covariate | class, site) is kept."""
    rng = np.random.default_rng(6)
    n = 2000
    X, y, sites = make_multisite_classification(
        n_sites=2, n_samples=n, n_features=4, balance_per_site=[[0.7, 0.3], [0.3, 0.7]], random_state=3
    )
    sex = (rng.random(n) < np.where(y == 1, 0.8, 0.3)).astype(int)
    sbr = SiteBalancedResampler(
        IntraSiteInterpolation("random", random_state=0),
        undersampler="random",
        max_amplification=max_amplification,
        random_state=0,
    )
    Xr, yr = sbr.fit_resample(X, y, sites=sites, categorical_covariate=sex)
    # random over-sampling duplicates real rows, so the covariate of every output row is known
    lut = {tuple(r): s for r, s in zip(X, sex, strict=True)}
    sex_r = np.array([lut[tuple(r)] for r in Xr])
    sr = sbr.sites_resampled_
    for site in (0, 1):
        for c in (0, 1):
            before = sex[(sites == site) & (y == c)].mean()
            after = sex_r[(sr == site) & (yr == c)].mean()
            assert after == pytest.approx(before, abs=0.03)


def test_continuous_covariate(covariate_data):
    """Continuous covariates are binned by the over-sampler's options and passed on for the kept samples."""
    X, y, sites, sex, age = covariate_data
    sbr = SiteBalancedResampler(IntraSiteInterpolation("smote", n_bins_cont_cov=3, random_state=0), random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        Xr, yr = sbr.fit_resample(X, y, sites=sites, categorical_covariate=sex, continuous_covariate=age)
    _assert_balanced(yr, sbr.sites_resampled_)
    _assert_provenance(sbr, X, y, sites, Xr, yr)


@pytest.mark.parametrize("max_amplification", ["auto", 0.5, 0])
def test_regression(linear_regression_data, max_amplification):
    """Regression: bins of the full data are used for both steps; kept samples keep their targets."""
    X, y, sites, weights = linear_regression_data
    sbr = SiteBalancedResampler(
        IntraSiteInterpolation("smote", task="regression", n_bins=4, random_state=0),
        max_amplification=max_amplification,
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        Xr, yr = sbr.fit_resample(X, y, sites=sites)
    assert sbr.task_ == "regression"
    np.testing.assert_array_equal(sbr.oversampler_.bins_, sbr.bins_)
    _assert_provenance(sbr, X, y, sites, Xr, yr)
    _assert_accounting(sbr, y, yr)
    bins = np.clip(np.digitize(yr, sbr.bins_[1:-1]), 0, len(sbr.bins_) - 2)
    _assert_balanced(bins, sbr.sites_resampled_)
    synthetic = sbr.is_synthetic_
    np.testing.assert_allclose(yr[synthetic], Xr[synthetic] @ weights + 50.0, atol=1e-8)


def test_regression_with_tied_quantiles():
    """A discrete regression target (tied quantile edges) works: the same edges are passed to the over-sampler."""
    rng = np.random.default_rng(5)
    X = rng.standard_normal((400, 6))
    y = np.r_[np.repeat([0.0, 1.0], [160, 40]), np.repeat([0.0, 1.0], [40, 160])]
    sites = np.repeat([0, 1], 200)
    sbr = SiteBalancedResampler(max_amplification=0, undersampler="random", random_state=0)
    Xr, yr = sbr.fit_resample(X, y, sites=sites)
    assert sbr.task_ == "regression"
    assert np.any(np.diff(sbr.bins_) == 0)
    np.testing.assert_array_equal(sbr.oversampler_.bins_, sbr.bins_)
    _assert_balanced(yr, sbr.sites_resampled_)
    _assert_provenance(sbr, X, y, sites, Xr, yr)


@pytest.mark.parametrize(
    "labels",
    [np.array([0, 1]), np.array([3, 7], dtype=np.int32), np.array(["AD", "CN"]), np.array([False, True])],
)
def test_classification_keeps_labels_and_dtype(binary_data, labels):
    """Labels are neither reconstructed nor cast to float."""
    X, y, sites = binary_data
    y_labels = labels[y]
    sbr = SiteBalancedResampler(max_amplification=1, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        Xr, yr = sbr.fit_resample(X, y_labels, sites=sites)
    assert sbr.task_ == "classification"
    assert yr.dtype == y_labels.dtype
    assert set(np.unique(yr)) <= set(labels)
    _assert_provenance(sbr, X, y_labels, sites, Xr, yr)


# ==============================================================================
# sklearn / imblearn integration and reproducibility
# ==============================================================================


def test___sklearn_tags__():
    """Test __sklearn_tags__."""
    assert SiteBalancedResampler().__sklearn_tags__().estimator_type == "sampler"


def test_compatibility(opposite_sites_data):
    """_fit_resample requires sites and otherwise delegates to fit_resample."""
    X, y, sites = opposite_sites_data
    sbr = SiteBalancedResampler(random_state=0)
    with pytest.raises(TypeError, match="sites"):
        sbr._fit_resample(X, y)
    Xr, _ = sbr._fit_resample(X, y, sites=sites)
    assert len(Xr) == len(sbr.sites_resampled_)


def test_pipeline_with_metadata_routing(opposite_sites_data):
    """The resampler is applied to the training folds only, with sites routed through an imblearn Pipeline."""
    X, y, sites = opposite_sites_data
    with config_context(enable_metadata_routing=True):
        sbr = SiteBalancedResampler(random_state=0).set_fit_resample_request(sites=True)
        pipe = Pipeline([("sbr", sbr), ("clf", LogisticRegression())])
        scores = cross_validate(
            pipe, X, y, cv=StratifiedKFold(3, shuffle=True, random_state=0), params={"sites": sites}, error_score="raise"
        )
    assert len(scores["test_score"]) == 3


@pytest.mark.parametrize("max_amplification", ["auto", 1])
def test_reproducibility(binary_data, max_amplification):
    """Same random_state gives identical results, also when refitting the same object."""
    X, y, sites = binary_data
    sbr1 = SiteBalancedResampler(max_amplification=max_amplification, random_state=42)
    sbr2 = SiteBalancedResampler(max_amplification=max_amplification, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        X1, y1 = sbr1.fit_resample(X, y, sites=sites)
        X2, y2 = sbr2.fit_resample(X, y, sites=sites)
        X3, y3 = sbr1.fit_resample(X, y, sites=sites)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)
    np.testing.assert_array_equal(X1, X3)
    np.testing.assert_array_equal(y1, y3)
