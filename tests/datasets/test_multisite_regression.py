"""Tests of make_multisite_regression: target model, target ranges per site, shared site effects."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from uniharmony.datasets import make_multisite_regression


def _truth(**kwargs):
    return make_multisite_regression(return_ground_truth=True, **kwargs)


def test_shapes_and_defaults():
    """Default call: two sites, 1000 samples, 10 features, a continuous target."""
    X, y, sites = make_multisite_regression()
    assert X.shape == (1000, 10) and y.shape == (1000,) and sites.shape == (1000,)
    assert np.issubdtype(y.dtype, np.floating)
    np.testing.assert_array_equal(np.bincount(sites), [500, 500])


@pytest.mark.parametrize("signal_type", ["linear", "friedman1"])
@pytest.mark.parametrize("strength", [0.0, 1.0, 3.0])
def test_population_r2_is_s2_over_1_plus_s2(signal_type, strength):
    """F is standardized, so the explained variance of y is s**2 / (1 + s**2) whatever the generator."""
    _, y, _, truth = _truth(n_samples=40000, signal={signal_type: strength}, site_effect={})
    r2 = 1 - np.var(y - truth["target_signal"]) / np.var(y)
    assert r2 == pytest.approx(strength**2 / (1 + strength**2), abs=0.01)
    assert np.std(y - truth["target_signal"]) == pytest.approx(1.0, abs=0.02)


@pytest.mark.parametrize(("signal_type", "sd"), [("linear", 1.0), ("friedman1", 1 / np.sqrt(12))])
def test_feature_scale_is_the_population_sd_of_the_signal(signal_type, sd):
    """Strengths are in units of the population SD of the features of the signal."""
    *_, truth = _truth(n_samples=20000, signal={signal_type: 1.0})
    np.testing.assert_allclose(truth["feature_scale"], sd)
    np.testing.assert_allclose(truth["signal"].std(axis=0), sd, rtol=0.03)


def test_linear_coefficients_are_exact():
    """For "linear", target_signal = signal @ target_coefficients, with n_informative non-zero coefficients."""
    *_, truth = _truth(n_features=8, n_informative=3, signal={"linear": 2.0})
    np.testing.assert_allclose(truth["signal"] @ truth["target_coefficients"], truth["target_signal"], atol=1e-12)
    assert np.count_nonzero(truth["target_coefficients"]) == 3
    assert np.linalg.norm(truth["target_coefficients"]) == pytest.approx(2.0)
    *_, friedman = _truth(signal={"friedman1": 1.0})
    assert friedman["target_coefficients"] is None


@pytest.mark.parametrize("signal_type", ["linear", "friedman1"])
def test_target_ranges_are_population_quantiles(signal_type):
    """Each site keeps the targets between its quantiles of the population distribution."""
    ranges = [(0.0, 0.2), (0.4, 0.6), (0.9, 1.0)]
    _, y, sites = make_multisite_regression(n_sites=3, n_samples=30000, signal={signal_type: 2.0}, target_range_per_site=ranges)
    _, population, _ = make_multisite_regression(n_sites=1, n_samples=200000, signal={signal_type: 2.0}, random_state=1)
    population = np.sort(population)
    for site, (low, high) in enumerate(ranges):
        quantiles = np.searchsorted(population, [y[sites == site].min(), y[sites == site].max()]) / len(population)
        np.testing.assert_allclose(quantiles, [low, high], atol=0.005)


def test_linear_target_range_matches_the_normal_quantiles():
    """For "linear", y ~ N(0, 1 + s**2): the site of the upper half has its targets above 0."""
    _, y, sites = make_multisite_regression(target_range_per_site=[(0, 0.5), (0.5, 1)], signal={"linear": 1.0})
    assert y[sites == 0].max() <= 0 <= y[sites == 1].min()
    _, y, sites = make_multisite_regression(n_samples=20000, target_range_per_site=(0.1, 0.3), signal={"linear": 1.0})
    bounds = norm.ppf([0.1, 0.3]) * np.sqrt(2)
    assert bounds[0] <= y.min() and y.max() <= bounds[1]


def test_signal_follows_the_target_range():
    """Selecting by target shifts the signal: the sites differ in X even without site effect (target confounding)."""
    X, _, sites = make_multisite_regression(
        n_samples=4000, target_range_per_site=[(0, 0.5), (0.5, 1)], site_effect={}, noise_strength=0
    )
    assert np.abs(X[sites == 0].mean(axis=0) - X[sites == 1].mean(axis=0)).max() > 0.3


def test_site_effect_does_not_change_the_target():
    """The site effect acts on the features only: y is the same with or without it."""
    _, y_with, _ = make_multisite_regression(site_effect={"location": 5, "scale": 2})
    _, y_without, _ = make_multisite_regression(site_effect={"location": 5, "scale": 2, "variance": 1})
    np.testing.assert_array_equal(y_with, y_without)


@pytest.mark.parametrize("site_effect_type", ["location", "scale", "variance", "nonlinear", "combat"])
def test_ground_truth_decomposes_x(site_effect_type):
    """X = site_free + site_effect, and site_free = signal + covariate effects + noise."""
    X, _, _, truth = _truth(
        n_sites=3, covariates=["age", "sex"], site_effect={site_effect_type: 3.0}, site_effect_heterogeneity=0.3
    )
    np.testing.assert_allclose(truth["site_free"], truth["signal"] + truth["covariate_effect"] + truth["noise"], atol=1e-12)
    np.testing.assert_allclose(X, truth["site_free"] + truth["site_effect"], atol=1e-12)
    assert "site_class_location" not in truth


def test_dropout_and_frames():
    """Missing values and the DataFrame output work as in classification."""
    X, y, sites, truth = _truth(
        site_effect={"dropout": {"value": np.nan, "prob": 0.5, "per": "value"}},
        site_names=["young", "old"],
        as_frame=True,
    )
    assert isinstance(X, pd.DataFrame) and y.name == "target" and sites.name == "site"
    np.testing.assert_array_equal(X.isna().to_numpy(), truth["dropout_mask"])
    assert list(truth["site_location"].index) == ["young", "old"]


def test_reproducible():
    """The same seed gives the same data, with and without target ranges."""
    for ranges in (None, [(0, 0.3), (0.6, 1)]):
        first = make_multisite_regression(target_range_per_site=ranges, signal={"friedman1": 1.0})
        second = make_multisite_regression(target_range_per_site=ranges, signal={"friedman1": 1.0})
        for a, b in zip(first, second, strict=True):
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"signal": {"blobs": 1.0}}, ValueError, "Unsupported signal type: 'blobs'"),
        ({"signal": "linear"}, TypeError, "signal must be a dict"),
        ({"signal": {"friedman1": 1.0}, "n_features": 4}, ValueError, "n_features >= 5"),
        ({"signal": {"friedman1": 1.0}, "n_informative": 3}, ValueError, "only applies to the 'linear'"),
        ({"n_informative": 11}, ValueError, "n_informative must be between 1 and n_features"),
        ({"n_informative": 2.5}, TypeError, "n_informative must be an int"),
        ({"site_effect": {"class_interaction": 1.0}}, ValueError, "needs class labels"),
        ({"target_range_per_site": (0.5, 0.2)}, ValueError, "0 <= low < high <= 1"),
        ({"target_range_per_site": (0.5, 0.505)}, ValueError, "at least 0.01"),
        ({"target_range_per_site": [(0, 0.5)] * 3}, ValueError, "one pair per site"),
        ({"target_range_per_site": "young"}, TypeError, "target_range_per_site must be"),
        ({"n_features": 0}, ValueError, "n_features must be positive"),
    ],
)
def test_invalid_inputs(kwargs, error, match):
    """Informative errors for invalid signals, ranges and class-only site effects."""
    with pytest.raises(error, match=match):
        make_multisite_regression(**kwargs)
