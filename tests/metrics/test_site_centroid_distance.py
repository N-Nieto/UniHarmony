"""Tests for ``uniharmony.metrics.site_centroid_distance``."""

import numpy as np
import pytest
from scipy.stats import ortho_group
from sklearn.feature_selection import f_classif
from sklearn.metrics import calinski_harabasz_score

from uniharmony.metrics import site_centroid_distance


METRICS = ["euclidean", "standardized", "mahalanobis"]


@pytest.fixture
def null_data() -> tuple[np.ndarray, np.ndarray]:
    """Three sites drawn from the same distribution (no Effect of Site)."""
    rng = np.random.default_rng(0)
    return rng.normal(size=(150, 8)), np.repeat(["a", "b", "c"], 50)


@pytest.fixture
def shifted_data() -> tuple[np.ndarray, np.ndarray]:
    """Two isotropic sites whose means differ by 1 standard deviation along one feature."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2000, 5))
    sites = np.repeat([0, 1], 1000)
    X[sites == 1, 0] += 1.0
    return X, sites


# --------------------------------------------------------------------------- #
# The statistic matches its textbook definitions
# --------------------------------------------------------------------------- #


def test_euclidean_statistic_is_calinski_harabasz(null_data):
    """With the euclidean metric, F is the Calinski-Harabasz index of the site partition."""
    X, sites = null_data
    result = site_centroid_distance(X, sites, metric="euclidean", n_permutations=0)
    assert result["statistic"] == pytest.approx(calinski_harabasz_score(X, sites))


def test_euclidean_r2_is_share_of_variance(null_data):
    """With the euclidean metric, R² is tr(B) / tr(T)."""
    X, sites = null_data
    centered = X - X.mean(axis=0)
    between = sum(np.sum(sites == s) * np.sum(centered[sites == s].mean(axis=0) ** 2) for s in np.unique(sites))
    result = site_centroid_distance(X, sites, metric="euclidean", n_permutations=0)
    assert result["r2"] == pytest.approx(between / np.sum(centered**2))


def test_standardized_statistic_is_mean_univariate_f(null_data):
    """With the standardized metric, F is the mean of the per-feature ANOVA F."""
    X, sites = null_data
    result = site_centroid_distance(X, sites, metric="standardized", n_permutations=0)
    assert result["statistic"] == pytest.approx(f_classif(X, sites)[0].mean())


def test_euclidean_is_invariant_to_rotation_and_translation(null_data):
    """The euclidean geometry does not depend on the orientation or origin of the feature space."""
    X, sites = null_data
    X_rotated = X @ ortho_group.rvs(X.shape[1], random_state=0) + 5.0
    result = site_centroid_distance(X, sites, metric="euclidean", n_permutations=0)
    result_rotated = site_centroid_distance(X_rotated, sites, metric="euclidean", n_permutations=0)
    assert result_rotated["statistic"] == pytest.approx(result["statistic"])
    np.testing.assert_allclose(result_rotated["pairwise_distance"], result["pairwise_distance"], atol=1e-10)


@pytest.mark.parametrize("metric", ["standardized", "mahalanobis"])
def test_invariant_to_feature_scaling(null_data, metric):
    """Standardized and Mahalanobis geometries do not depend on the units of the features."""
    X, sites = null_data
    scales = np.arange(1, X.shape[1] + 1) ** 2
    result = site_centroid_distance(X, sites, metric=metric, n_permutations=0)
    result_scaled = site_centroid_distance(X * scales - 3.0, sites, metric=metric, n_permutations=0)
    assert result_scaled["statistic"] == pytest.approx(result["statistic"])
    np.testing.assert_allclose(result_scaled["pairwise_distance"], result["pairwise_distance"], atol=1e-10)


def test_mahalanobis_without_shrinkage_is_lawley_hotelling(shifted_data):
    """With many samples (no shrinkage), F is the Lawley-Hotelling trace tr(W^-1 B) / (df_between * n_features)."""
    X, sites = shifted_data
    centered = X - X.mean(axis=0)
    means = np.array([centered[sites == s].mean(axis=0) for s in (0, 1)])
    between = 1000 * means.T @ means
    within = centered.T @ centered - between
    lawley_hotelling = np.trace(np.linalg.solve(within / (len(X) - 2), between)) / X.shape[1]
    result = site_centroid_distance(X, sites, metric="mahalanobis", n_permutations=0)
    assert result["statistic"] == pytest.approx(lawley_hotelling, rel=1e-2)


@pytest.mark.parametrize("metric", METRICS)
def test_constant_features_are_ignored(null_data, metric):
    """Adding a constant feature changes nothing."""
    X, sites = null_data
    X_constant = np.column_stack([X, np.full(len(X), 3.0)])
    result = site_centroid_distance(X, sites, metric=metric, n_permutations=0)
    result_constant = site_centroid_distance(X_constant, sites, metric=metric, n_permutations=0)
    assert result_constant["statistic"] == pytest.approx(result["statistic"])


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("metric", METRICS)
def test_no_site_effect_gives_chance_level(null_data, metric):
    """Without an Effect of Site, F is close to its null mean and is not significant."""
    X, sites = null_data
    result = site_centroid_distance(X, sites, metric=metric, n_permutations=200)
    assert result["p_value"] > 0.05
    assert abs(result["z_score"]) < 3


def test_permutation_test_is_calibrated():
    """Under the null, p <= 0.05 happens in about 5% of the data sets."""
    rng = np.random.default_rng(2)
    p_values = [
        site_centroid_distance(rng.normal(size=(60, 4)), np.repeat([0, 1, 2], 20), n_permutations=99, random_state=i)["p_value"]
        for i in range(100)
    ]
    assert np.mean(np.array(p_values) <= 0.05) < 0.12


@pytest.mark.parametrize("metric", METRICS)
def test_site_effect_is_detected(shifted_data, metric):
    """A location Effect of Site gives a large F and a small p-value."""
    X, sites = shifted_data
    result = site_centroid_distance(X, sites, metric=metric, n_permutations=100)
    assert result["statistic"] > 10 * result["null_mean"]
    assert result["p_value"] == pytest.approx(1 / 101)


def test_conditioning_on_y_removes_biology():
    """When diagnosis proportions differ by site, the diagnosis effect is not counted as EoS once y is given."""
    rng = np.random.default_rng(3)
    sites = np.repeat([0, 1], 300)
    y = np.concatenate([rng.random(300) < 0.8, rng.random(300) < 0.2]).astype(int)
    X = rng.normal(size=(600, 6)) + 2.0 * y[:, None]

    assert site_centroid_distance(X, sites, n_permutations=200)["p_value"] < 0.01
    assert site_centroid_distance(X, sites, y=y, n_permutations=200)["p_value"] > 0.05


def test_no_permutations_returns_nan(null_data):
    """With n_permutations=0 the observed statistic is returned without a null."""
    X, sites = null_data
    result = site_centroid_distance(X, sites, n_permutations=0)
    assert np.isfinite(result["statistic"])
    assert all(np.isnan(result[key]) for key in ("null_mean", "null_std", "z_score", "p_value"))
    assert result["null_distribution"].shape == (0,)


def test_random_state_makes_results_reproducible(null_data):
    """The same seed gives the same null distribution."""
    X, sites = null_data
    first = site_centroid_distance(X, sites, n_permutations=20, random_state=7)
    second = site_centroid_distance(X, sites, n_permutations=20, random_state=7)
    np.testing.assert_array_equal(first["null_distribution"], second["null_distribution"])


# --------------------------------------------------------------------------- #
# Pairwise distances
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("metric", METRICS)
def test_pairwise_distance_is_a_labelled_distance_matrix(null_data, metric):
    """The pairwise output is symmetric, non-negative, zero on the diagonal and labelled by site."""
    X, sites = null_data
    distance = site_centroid_distance(X, sites, metric=metric, n_permutations=0)["pairwise_distance"]
    assert list(distance.index) == list(distance.columns) == ["a", "b", "c"]
    np.testing.assert_allclose(distance, distance.T)
    assert (distance.to_numpy() >= 0).all()
    assert (np.diag(distance) == 0).all()


def test_mahalanobis_distance_recovers_true_shift(shifted_data):
    """For a 1-SD shift between two sites, the Mahalanobis centroid distance is about 1."""
    X, sites = shifted_data
    distance = site_centroid_distance(X, sites, metric="mahalanobis", n_permutations=0)["pairwise_distance"]
    assert distance.loc[0, 1] == pytest.approx(1.0, abs=0.1)


def test_pairwise_distance_is_bias_corrected():
    """Without EoS, the squared distance between small sites averages 0 instead of tr(S_a)/n_a + tr(S_b)/n_b."""
    rng = np.random.default_rng(4)
    sites = np.repeat([0, 1], 10)
    sq_distances = [
        site_centroid_distance(rng.normal(size=(20, 50)), sites, metric="euclidean", n_permutations=0)["pairwise_distance"].loc[
            0, 1
        ]
        ** 2
        for _ in range(200)
    ]
    # The naive estimate would average 50 * (1/10 + 1/10) = 10; clipping at 0 keeps the corrected one small.
    assert np.mean(sq_distances) < 2


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"metric": "cosine"}, "metric must be one of"),
        ({"n_permutations": -1}, "n_permutations must be >= 0"),
        ({"sites": np.zeros(150)}, "At least 2 sites"),
        ({"sites": np.r_[np.zeros(149), 1]}, "at least 2 samples"),
        ({"y": np.repeat([0, 1, 2], 50)}, "No class of y contains more than one site"),
        ({"sites": np.zeros(10)}, "inconsistent numbers of samples"),
    ],
)
def test_invalid_inputs_raise(null_data, kwargs, match):
    """Invalid arguments raise a clear ValueError."""
    X, sites = null_data
    arguments = {"X": X, "sites": sites, "n_permutations": 0, **kwargs}
    with pytest.raises(ValueError, match=match):
        site_centroid_distance(**arguments)
