"""Tests of make_traveling_subjects: the same subjects measured in every site."""

import numpy as np
import pandas as pd
import pytest

from uniharmony.datasets import Covariate, CovariateSiteDistribution, make_traveling_subjects


def _truth(**kwargs):
    return make_traveling_subjects(return_ground_truth=True, **kwargs)


def test_design_every_subject_in_every_site():
    """Rows are ordered by site, repeat and subject; each subject has n_repeats measurements per site."""
    X, sites, subjects, truth = _truth(n_subjects=5, n_sites=3, n_repeats=2)
    assert X.shape == (30, 10)
    np.testing.assert_array_equal(sites, np.repeat([0, 1, 2], 10))
    np.testing.assert_array_equal(subjects, np.tile(np.arange(5), 6))
    np.testing.assert_array_equal(truth["repeat"], np.tile(np.repeat([0, 1], 5), 3))
    counts = pd.crosstab(subjects, sites).to_numpy()
    np.testing.assert_array_equal(counts, 2)


def test_trait_and_covariates_are_the_same_in_every_site():
    """The signal (trait) and the covariates of a subject do not change between measurements."""
    _, _, subjects, truth = _truth(n_subjects=8, covariates=["age", "sex", "tiv"])
    for subject in range(8):
        rows = subjects == subject
        assert np.ptp(truth["signal"][rows], axis=0).max() == 0
        assert np.ptp(truth["covariate_effect"][rows], axis=0).max() == 0
        for values in truth["covariates"].values():
            assert len(np.unique(values[rows])) == 1


def test_without_site_effect_measurements_differ_by_noise_only():
    """site_free = trait + covariate effects + noise, and X = site_free without site effect."""
    X, _, _, truth = _truth(site_effect={}, covariates=["age"])
    np.testing.assert_array_equal(X, truth["site_free"])
    np.testing.assert_allclose(truth["site_free"], truth["signal"] + truth["covariate_effect"] + truth["noise"], atol=1e-12)


def test_reliability_follows_the_noise_strength():
    """Within a site, the correlation between two repeats is 1 / (1 + r**2)."""
    X, _, _, truth = _truth(n_subjects=3000, n_sites=1, n_repeats=2, noise_strength=0.5, site_effect={})
    first, second = X[truth["repeat"] == 0], X[truth["repeat"] == 1]
    icc = np.mean([np.corrcoef(first[:, j], second[:, j])[0, 1] for j in range(X.shape[1])])
    assert icc == pytest.approx(1 / 1.25, abs=0.02)


def test_location_effect_is_the_same_for_every_subject_of_a_site():
    """A homogeneous location effect shifts every measurement of a site by the site offset."""
    _, sites, _, truth = _truth(n_sites=3, site_effect={"location": 2.0})
    for site in range(3):
        effect = truth["site_effect"][sites == site]
        np.testing.assert_allclose(effect, np.broadcast_to(truth["site_location"][site], effect.shape), atol=1e-12)


def test_heterogeneity_is_a_property_of_the_subject():
    """With heterogeneity, the intensity of a subject is the same in every site."""
    _, _, subjects, truth = _truth(n_subjects=30, site_effect_heterogeneity=0.5)
    intensity = truth["intensity"]
    for subject in range(30):
        assert np.ptp(intensity[subjects == subject]) == 0
    assert np.std(intensity[:30]) > 0.2


def test_per_site_noise_and_frames():
    """noise_strength can differ between sites; as_frame returns a DataFrame and named Series."""
    X, sites, subjects, truth = _truth(n_subjects=200, n_sites=2, noise_strength=[0.1, 1.0], site_names=["A", "B"], as_frame=True)
    assert isinstance(X, pd.DataFrame) and sites.name == "site" and subjects.name == "subject"
    noise = truth["noise"]
    assert noise[(sites == "A").to_numpy()].std() == pytest.approx(0.1, rel=0.1)
    assert noise[(sites == "B").to_numpy()].std() == pytest.approx(1.0, rel=0.1)
    assert list(truth["site_location"].index) == ["A", "B"]


def test_covariate_with_several_distributions_raises():
    """The subjects are the same in every site, so a covariate has a single (population) distribution."""
    age = Covariate(name="age", site_distributions=[CovariateSiteDistribution(loc=30.0), CovariateSiteDistribution(loc=60.0)])
    with pytest.raises(ValueError, match="needs a single distribution"):
        make_traveling_subjects(n_sites=2, covariates=[age])


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"n_subjects": 1}, ValueError, "n_subjects must be at least 2"),
        ({"n_repeats": 0}, ValueError, "n_repeats must be at least 1"),
        ({"n_sites": 2.0}, TypeError, "n_sites must be an int"),
        ({"site_effect": {"class_interaction": 1.0}}, ValueError, "needs class labels"),
        ({"noise_strength": [0.1, 0.2]}, ValueError, "must have length n_sites"),
    ],
)
def test_invalid_inputs(kwargs, error, match):
    """Informative errors for invalid designs."""
    with pytest.raises(error, match=match):
        make_traveling_subjects(**kwargs)
