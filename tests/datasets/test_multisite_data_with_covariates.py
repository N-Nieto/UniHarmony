"""Tests for _make_multisite_classification.py.

Run with:
    pytest tests/test_make_multisite_classification.py -v
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from sklearn.utils import check_random_state

from uniharmony.datasets import (
    Covariate,
    CovariateSiteDistribution,
    make_multisite_classification,
)
from uniharmony.datasets._make_covariates import (
    _make_covariate,
    _make_preset_covariate,
    _validate_covariates,
    make_covariate_site_distributions,
)


def _make_with_covariates(**kwargs):
    """Simulate and return X, y, sites and the covariates (from the ground truth)."""
    X, y, sites, truth = make_multisite_classification(return_ground_truth=True, **kwargs)
    return X, y, sites, truth["covariates"]


if TYPE_CHECKING:
    from numpy.random import RandomState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rng(seed: int = 0) -> RandomState:
    return check_random_state(seed)


def _age_cov(n_sites: int = 2) -> Covariate:
    """Continuous covariate with one distribution broadcast to all sites."""
    return Covariate(
        name="age",
        site_distributions=[CovariateSiteDistribution(loc=45.0, scale=10.0, clip=(18.0, 90.0))],
    )


def _sex_cov(n_sites: int = 2) -> Covariate:
    """Categorical covariate with one distribution broadcast to all sites."""
    return Covariate(
        name="sex",
        site_distributions=[CovariateSiteDistribution(probs=[0.4, 0.6])],
        categories=[0, 1],
    )


# ===========================================================================
# CovariateSiteDistribution
# ===========================================================================


def test_csd_continuous_kind():
    """Test."""
    sd = CovariateSiteDistribution(loc=45.0, scale=10.0)
    assert sd.kind == "continuous"


def test_csd_categorical_kind():
    """Test."""
    sd = CovariateSiteDistribution(probs=[0.3, 0.7])
    assert sd.kind == "categorical"


def test_csd_both_loc_and_probs_raises():
    """Test."""
    with pytest.raises(ValueError, match="not both"):
        CovariateSiteDistribution(loc=1.0, probs=[0.5, 0.5])


def test_csd_neither_loc_nor_probs_raises():
    """Test."""
    with pytest.raises(ValueError, match="supply"):
        CovariateSiteDistribution()


def test_csd_clip_stored():
    """Test."""
    sd = CovariateSiteDistribution(loc=50.0, clip=(18.0, 90.0))
    assert sd.clip == (18.0, 90.0)


def test_csd_scale_default():
    """Test."""
    sd = CovariateSiteDistribution(loc=0.0)
    assert sd.scale == 1.0


# ===========================================================================
# Covariate
# ===========================================================================


def test_covariate_kind_continuous():
    """Test."""
    cov = _age_cov()
    assert cov.kind == "continuous"


def test_covariate_kind_categorical():
    """Test."""
    cov = _sex_cov()
    assert cov.kind == "categorical"


def test_covariate_empty_site_distributions_raises():
    """Test."""
    with pytest.raises(ValueError, match="must not be empty"):
        Covariate(name="x", site_distributions=[])


def test_covariate_mixed_kinds_raises():
    """Test."""
    with pytest.raises(ValueError, match="same kind"):
        Covariate(
            name="x",
            site_distributions=[
                CovariateSiteDistribution(loc=1.0),
                CovariateSiteDistribution(probs=[0.5, 0.5]),
            ],
        )


# ===========================================================================
# _validate_covariate_specs — broadcast and length checks
# ===========================================================================


def test_validate_covariate_specs_broadcast():
    """Single site_distribution should be broadcast to n_sites."""
    cov = _age_cov()
    _validate_covariates([cov], n_sites=4)
    assert len(cov.site_distributions) == 4
    # All entries are the same object repeated
    assert all(sd is cov.site_distributions[0] for sd in cov.site_distributions)


def test_validate_covariate_specs_exact_match():
    """Exact n_sites distributions should pass without modification."""
    cov = Covariate(
        name="age",
        site_distributions=[
            CovariateSiteDistribution(loc=30.0),
            CovariateSiteDistribution(loc=50.0),
        ],
    )
    _validate_covariates([cov], n_sites=2)
    assert len(cov.site_distributions) == 2


@pytest.mark.parametrize("n_dists,n_sites", [(2, 3), (3, 2), (4, 5)])
def test_validate_covariate_specs_wrong_length_raises(n_dists, n_sites):
    """Test."""
    cov = Covariate(
        name="x",
        site_distributions=[CovariateSiteDistribution(loc=1.0)] * n_dists,
    )
    with pytest.raises(ValueError, match="site_distributions"):
        _validate_covariates([cov], n_sites=n_sites)


def test_validate_covariate_specs_categorical_probs_length_mismatch_raises():
    """Test."""
    cov = Covariate(
        name="sex",
        site_distributions=[CovariateSiteDistribution(probs=[0.5, 0.3, 0.2])],
        categories=[0, 1],  # 2 categories but 3 probs
    )
    with pytest.raises(ValueError, match="probs has"):
        _validate_covariates([cov], n_sites=1)


# ===========================================================================
# make_site_distributions
# ===========================================================================


def test_make_site_distributions_length():
    """Test."""
    dists = make_covariate_site_distributions(locs=[10.0, 20.0, 30.0])
    assert len(dists) == 3


def test_make_site_distributions_locs():
    """Test."""
    dists = make_covariate_site_distributions(locs=[10.0, 20.0])
    assert dists[0].loc == 10.0
    assert dists[1].loc == 20.0


def test_make_site_distributions_scale_broadcast():
    """Test."""
    dists = make_covariate_site_distributions(locs=[10.0, 20.0], scales=5.0)
    assert dists[0].scale == 5.0
    assert dists[1].scale == 5.0


def test_make_site_distributions_clip_broadcast():
    """Test."""
    dists = make_covariate_site_distributions(locs=[10.0, 20.0], clips=(0.0, 100.0))
    assert dists[0].clip == (0.0, 100.0)
    assert dists[1].clip == (0.0, 100.0)


def test_make_site_distributions_scales_wrong_length_raises():
    """Test."""
    with pytest.raises(ValueError, match="scales"):
        make_covariate_site_distributions(locs=[10.0, 20.0], scales=[1.0, 2.0, 3.0])


# ===========================================================================
# make_covariate_spec (presets)
# ===========================================================================


@pytest.mark.parametrize("preset", ["age", "sex", "quality"])
def test_make_covariate_spec_returns_covariate(preset):
    """Test."""
    spec = _make_preset_covariate(preset, n_sites=3)
    assert isinstance(spec, Covariate)


@pytest.mark.parametrize(
    "preset,n_sites",
    [
        ("age", 1),
        ("age", 4),
        ("sex", 2),
        ("quality", 3),
    ],
)
def test_make_covariate_spec_n_distributions(preset, n_sites):
    """Test."""
    spec = _make_preset_covariate(preset, n_sites=n_sites)
    assert len(spec.site_distributions) == n_sites


def test_make_covariate_spec_age_kind():
    """Test."""
    assert _make_preset_covariate("age", n_sites=2).kind == "continuous"


def test_make_covariate_spec_sex_kind():
    """Test."""
    assert _make_preset_covariate("sex", n_sites=2).kind == "categorical"


def test_make_covariate_spec_quality_kind():
    """Test."""
    assert _make_preset_covariate("quality", n_sites=2).kind == "continuous"


def test_make_covariate_spec_unknown_raises():
    """Test."""
    with pytest.raises(ValueError, match="Unknown preset"):
        _make_preset_covariate("banana", n_sites=2)


# ===========================================================================
# _make_covariates
# ===========================================================================


def test_make_covariates_continuous_shape():
    """Test."""
    cov = _age_cov()
    _validate_covariates([cov], n_sites=1)
    result = _make_covariate([cov], site_idx=0, n_samples_site=50, random_state=_rng(0))
    assert result["age"].shape == (50,)


def test_make_covariates_continuous_clip_respected():
    """Test."""
    cov = _age_cov()
    _validate_covariates([cov], n_sites=1)
    result = _make_covariate([cov], site_idx=0, n_samples_site=500, random_state=_rng(0))
    assert result["age"].min() >= 18.0
    assert result["age"].max() <= 90.0


def test_make_covariates_categorical_values():
    """Test."""
    cov = _sex_cov()
    _validate_covariates([cov], n_sites=1)
    result = _make_covariate([cov], site_idx=0, n_samples_site=200, random_state=_rng(0))
    assert set(np.unique(result["sex"])).issubset({0, 1})


def test_make_covariates_categorical_shape():
    """Test."""
    cov = _sex_cov()
    _validate_covariates([cov], n_sites=1)
    result = _make_covariate([cov], site_idx=0, n_samples_site=100, random_state=_rng(0))
    assert result["sex"].shape == (100,)


def test_make_covariates_multiple_keys():
    """Test."""
    covs = [_age_cov(), _sex_cov()]
    _validate_covariates(covs, n_sites=1)
    result = _make_covariate(covs, site_idx=0, n_samples_site=100, random_state=_rng(0))
    assert set(result.keys()) == {"age", "sex"}


def test_make_covariates_site_idx_selects_correct_distribution():
    """Each site should draw from its own distribution, not a shared one."""
    cov = Covariate(
        name="age",
        site_distributions=[
            CovariateSiteDistribution(loc=20.0, scale=1.0),  # site 0: young
            CovariateSiteDistribution(loc=80.0, scale=1.0),  # site 1: old
        ],
    )
    r0 = _make_covariate([cov], site_idx=0, n_samples_site=500, random_state=_rng(0))
    r1 = _make_covariate([cov], site_idx=1, n_samples_site=500, random_state=_rng(1))
    assert r0["age"].mean() < r1["age"].mean()


# ===========================================================================
# Noise (in reference-SD units)
# ===========================================================================


def _noise(noise_strength, **kwargs):
    *_, truth = make_multisite_classification(n_samples=4000, noise_strength=noise_strength, return_ground_truth=True, **kwargs)
    return truth


def test_noise_zero_strength_adds_nothing():
    """noise_strength=0 adds no noise."""
    np.testing.assert_array_equal(_noise(0.0)["noise"], 0.0)


@pytest.mark.parametrize("strength", [0.1, 1.0, 5.0])
@pytest.mark.parametrize("signal_type", ["linear", "moons"])
def test_noise_sd_is_strength_times_feature_scale(strength, signal_type):
    """The noise SD of each feature is noise_strength times its reference SD."""
    truth = _noise(strength, signal={signal_type: 1.0})
    ratio = truth["noise"].std(axis=0) / (strength * truth["feature_scale"])
    np.testing.assert_allclose(ratio, 1.0, atol=0.06)


def test_noise_per_site():
    """A list gives one noise strength per site."""
    _, _, sites, truth = make_multisite_classification(
        n_sites=2, n_samples=4000, noise_strength=[0.0, 1.0], return_ground_truth=True
    )
    np.testing.assert_array_equal(truth["noise"][sites == 0], 0.0)
    assert truth["noise"][sites == 1].std() > 0.5


@pytest.mark.parametrize(
    "random_state",
    [
        0,
        42,
        np.random.RandomState(7),
        np.random.default_rng(7),
        None,
    ],
)
def test_random_state_types_accepted(random_state):
    """Test."""
    X, _, _ = make_multisite_classification(
        n_sites=2,
        n_samples=100,
        random_state=random_state,
    )
    assert X.shape == (100, 10)


# ===========================================================================
# make_multisite_classification — outputs
# ===========================================================================


def test_no_covariates_returns_3_tuple():
    """Test."""
    result = make_multisite_classification(n_sites=2, n_samples=100, random_state=0)
    assert len(result) == 3


def test_with_covariates_returns_3_tuple():
    """Covariates do not change the outputs; their values are in the ground truth."""
    result = make_multisite_classification(n_sites=2, n_samples=100, covariates=[_age_cov()], random_state=0)
    assert len(result) == 3
    *_, truth = make_multisite_classification(
        n_sites=2, n_samples=100, covariates=[_age_cov()], random_state=0, return_ground_truth=True
    )
    assert truth["covariates"]["age"].shape == (100,)


# ===========================================================================
# make_multisite_classification — covariates output
# ===========================================================================


def test_covariates_dict_keys():
    """Test."""
    _, _, _, covs = _make_with_covariates(
        n_sites=2,
        n_samples=200,
        covariates=[_age_cov(), _sex_cov()],
        random_state=0,
    )
    assert set(covs.keys()) == {"age", "sex"}


def test_covariates_array_lengths_match_x():
    """Test."""
    X, _, _, covs = _make_with_covariates(
        n_sites=2,
        n_samples=200,
        covariates=[_age_cov()],
        random_state=0,
    )
    assert covs["age"].shape == (len(X),)


def test_covariate_continuous_clip_in_full_output():
    """Test."""
    _, _, _, covs = _make_with_covariates(
        n_sites=2,
        n_samples=1000,
        covariates=[_age_cov()],
        random_state=0,
    )
    assert covs["age"].min() >= 18.0
    assert covs["age"].max() <= 90.0


def test_covariate_categorical_values_in_full_output():
    """Test."""
    _, _, _, covs = _make_with_covariates(
        n_sites=2,
        n_samples=500,
        covariates=[_sex_cov()],
        random_state=0,
    )
    assert set(np.unique(covs["sex"])).issubset({0, 1})


def test_covariate_per_site_distribution_different_means():
    """Explicit per-site distributions should produce different sample means."""
    cov = Covariate(
        name="age",
        site_distributions=[
            CovariateSiteDistribution(loc=10.0, scale=1.0, clip=None),
            CovariateSiteDistribution(loc=70.0, scale=1.0, clip=None),
        ],
    )
    _, _, sites, covs = _make_with_covariates(
        n_sites=2,
        n_samples=1000,
        covariates=[cov],
        random_state=0,
    )
    mean_s0 = covs["age"][sites == 0].mean()
    mean_s1 = covs["age"][sites == 1].mean()
    print(mean_s0)
    print(mean_s1)
    assert abs(mean_s0 - mean_s1) > 20.0  # clearly different


def test_covariate_broadcast_single_distribution():
    """A single site_distribution broadcast to 3 sites produces the same per-site mean."""
    cov = Covariate(
        name="age",
        site_distributions=[CovariateSiteDistribution(loc=45.0, scale=10.0)],
    )
    _, _, sites, covs = _make_with_covariates(
        n_sites=3,
        n_samples=3000,
        covariates=[cov],
        random_state=0,
    )
    means = [covs["age"][sites == i].mean() for i in range(3)]
    # All sites use the same distribution, so means should be close
    assert max(means) - min(means) < 5.0


@pytest.mark.parametrize("preset", ["age", "sex", "quality"])
def test_preset_string_covariates(preset):
    """Test."""
    *_, covs = _make_with_covariates(n_sites=2, n_samples=200, covariates=[preset], random_state=0)
    assert preset in covs


def test_invalid_covariate_type_raises():
    """Test."""
    with pytest.raises(TypeError, match="Covariate"):
        make_multisite_classification(
            n_sites=2,
            n_samples=100,
            covariates=[42],  # not a Covariate or str
            random_state=0,
        )


def test_invalid_preset_string_raises():
    """Test."""
    with pytest.raises(ValueError, match="Unknown preset"):
        make_multisite_classification(
            n_sites=2,
            n_samples=100,
            covariates=["banana"],
            random_state=0,
        )


def test_covariate_wrong_n_distributions_raises():
    """2 site_distributions for 3 sites must raise."""
    cov = Covariate(
        name="age",
        site_distributions=[
            CovariateSiteDistribution(loc=30.0),
            CovariateSiteDistribution(loc=50.0),
        ],
    )
    with pytest.raises(ValueError, match="site_distributions"):
        make_multisite_classification(
            n_sites=3,
            n_samples=300,
            covariates=[cov],
            random_state=0,
        )


# ===========================================================================
# make_multisite_classification — base data is shared across sites
# ===========================================================================


def test_base_y_is_shared_across_sites():
    """Y must not change between sites — it comes from a single base call."""
    _, y, _ = make_multisite_classification(
        n_sites=3,
        n_samples=300,
        random_state=0,
    )
    # Total samples = sum of per-site samples
    assert len(y) == 300
    # y covers all expected classes
    assert set(np.unique(y)) == {0, 1}


def test_site_effect_does_not_alter_y():
    """Changing site_effect_strength must not change y."""
    _, y1, _ = make_multisite_classification(
        n_sites=2,
        n_samples=11,
        site_effect={"location": 1},
        random_state=0,
    )
    _, y2, _ = make_multisite_classification(
        n_sites=2,
        n_samples=11,
        site_effect={"location": 0},
        random_state=0,
    )
    np.testing.assert_array_equal(y1, y2)


def test_covariate_is_not_modified_and_can_be_reused():
    """Broadcasting to n_sites works on a copy, so the same Covariate can be reused."""
    cov = _age_cov()
    make_multisite_classification(n_sites=2, n_samples=100, covariates=[cov])
    assert len(cov.site_distributions) == 1
    _, _, _, covariates = _make_with_covariates(n_sites=3, n_samples=90, covariates=[cov])
    assert covariates["age"].shape == (90,)
