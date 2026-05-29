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
from uniharmony.datasets._make_multisite_classification import _apply_noise, _apply_site_effect


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
    X = np.random.randn(50, 5)
    result = _make_covariate([cov], site_idx=0, n_samples_site=50, X=X, random_state=_rng(0))
    assert result["age"].shape == (50,)


def test_make_covariates_continuous_clip_respected():
    """Test."""
    cov = _age_cov()
    _validate_covariates([cov], n_sites=1)
    X = np.random.randn(500, 5)
    result = _make_covariate([cov], site_idx=0, n_samples_site=500, X=X, random_state=_rng(0))
    assert result["age"].min() >= 18.0
    assert result["age"].max() <= 90.0


def test_make_covariates_categorical_values():
    """Test."""
    cov = _sex_cov()
    _validate_covariates([cov], n_sites=1)
    X = np.random.randn(200, 5)
    result = _make_covariate([cov], site_idx=0, n_samples_site=200, X=X, random_state=_rng(0))
    assert set(np.unique(result["sex"])).issubset({0, 1})


def test_make_covariates_categorical_shape():
    """Test."""
    cov = _sex_cov()
    _validate_covariates([cov], n_sites=1)
    X = np.random.randn(100, 5)
    result = _make_covariate([cov], site_idx=0, n_samples_site=100, X=X, random_state=_rng(0))
    assert result["sex"].shape == (100,)


def test_make_covariates_multiple_keys():
    """Test."""
    covs = [_age_cov(), _sex_cov()]
    _validate_covariates(covs, n_sites=1)
    X = np.random.randn(100, 5)
    result = _make_covariate(covs, site_idx=0, n_samples_site=100, X=X, random_state=_rng(0))
    assert set(result.keys()) == {"age", "sex"}


@pytest.mark.parametrize("x_correlation", [0.0, 0.5, 1.0])
def test_make_covariates_x_correlation_produces_finite(x_correlation):
    """Test."""
    cov = Covariate(
        name="age",
        site_distributions=[CovariateSiteDistribution(loc=45.0, scale=10.0)],
        x_correlation=x_correlation,
    )
    _validate_covariates([cov], n_sites=1)
    X = np.random.randn(100, 5)
    result = _make_covariate([cov], site_idx=0, n_samples_site=100, X=X, random_state=_rng(0))
    assert np.all(np.isfinite(result["age"]))


def test_make_covariates_site_idx_selects_correct_distribution():
    """Each site should draw from its own distribution, not a shared one."""
    cov = Covariate(
        name="age",
        site_distributions=[
            CovariateSiteDistribution(loc=20.0, scale=1.0),  # site 0: young
            CovariateSiteDistribution(loc=80.0, scale=1.0),  # site 1: old
        ],
    )
    X = np.random.randn(500, 5)
    r0 = _make_covariate([cov], site_idx=0, n_samples_site=500, X=X, random_state=_rng(0))
    r1 = _make_covariate([cov], site_idx=1, n_samples_site=500, X=X, random_state=_rng(1))
    assert r0["age"].mean() < r1["age"].mean()


# ===========================================================================
# _apply_site_effect
# ===========================================================================


@pytest.mark.parametrize("effect_type", ["location", "scale", "location+scale"])
def test_apply_site_effect_shape_preserved(effect_type):
    """Test."""
    X = np.ones((50, 8))
    X_out, _ = _apply_site_effect(
        X=X,
        y=X,
        site_effect_type=effect_type,
        site_effect_strength=1.0,
        site_effect_homogeneous=True,
        random_state=_rng(0),
    )
    assert X_out.shape == X.shape


@pytest.mark.parametrize("effect_type", ["location", "scale", "location+scale"])
def test_apply_site_effect_modifies_data(effect_type):
    """Test."""
    X = np.ones((50, 8))
    y = np.ones((50, 8))
    X_out, _ = _apply_site_effect(
        X=X,
        y=y,
        site_effect_type=effect_type,
        site_effect_strength=5.0,
        site_effect_homogeneous=True,
        random_state=_rng(0),
    )
    assert not np.allclose(X, X_out)


def test_apply_site_effect_location_shifts_mean():
    """Test."""
    X = np.zeros((200, 5))
    X_out, _ = _apply_site_effect(
        X=X,
        y=np.ones((500, 8)),
        site_effect_type="location",
        site_effect_strength=3.0,
        site_effect_homogeneous=True,
        random_state=_rng(0),
    )
    # All samples get the same offset, so std across samples should be ~0
    assert X_out.std(axis=0).mean() < 1e-8


def test_apply_site_effect_heterogeneous_has_within_site_variance():
    """Test."""
    X = np.zeros((500, 5))

    X_out, _ = _apply_site_effect(
        X=X,
        y=np.ones((500, 8)),
        site_effect_type="location",
        site_effect_strength=3.0,
        site_effect_homogeneous=False,
        random_state=_rng(0),
    )
    # Per-sample draws → samples differ from each other
    assert X_out.std(axis=0).mean() > 0.1


def test_apply_site_effect_unknown_type_raises():
    """Test."""
    with pytest.raises(ValueError, match="Unsupported site_effect_type"):
        _apply_site_effect(
            X=np.ones((10, 4)),
            y=np.ones((500, 8)),
            site_effect_type="rotate",
            site_effect_strength=1.0,
            site_effect_homogeneous=True,
            random_state=_rng(0),
        )


def test_apply_site_effect_does_not_mutate_input() -> None:
    """Test."""
    X = np.ones((50, 8))
    X_copy = X.copy()
    _apply_site_effect(
        X=X,
        y=np.ones((500, 8)),
        site_effect_type="location",
        site_effect_strength=3.0,
        site_effect_homogeneous=True,
        random_state=_rng(0),
    )
    np.testing.assert_array_equal(X, X_copy)


# ===========================================================================
# _apply_noise
# ===========================================================================


def test_apply_noise_shape_preserved():
    """Test."""
    X = np.zeros((50, 8))
    X_out = _apply_noise(X, noise_strength=0.5, random_state=_rng(0))
    assert X_out.shape == X.shape


def test_apply_noise_modifies_data():
    """Test."""
    X = np.zeros((50, 8))
    X_out = _apply_noise(X, noise_strength=1.0, random_state=_rng(0))
    assert not np.allclose(X, X_out)


def test_apply_noise_zero_strength_is_identity():
    """Test."""
    X = np.random.randn(50, 8)
    X_out = _apply_noise(X, noise_strength=0.0, random_state=_rng(0))
    np.testing.assert_array_equal(X, X_out)


def test_apply_noise_does_not_mutate_input():
    """Test."""
    X = np.ones((50, 8))
    X_copy = X.copy()
    _apply_noise(X, noise_strength=1.0, random_state=_rng(0))
    np.testing.assert_array_equal(X, X_copy)


@pytest.mark.parametrize("strength", [0.01, 0.1, 1.0, 5.0])
def test_apply_noise_std_scales_with_strength(strength):
    """Test."""
    X = np.zeros((5000, 1))
    X_out = _apply_noise(X, noise_strength=strength, random_state=_rng(0))
    observed_std = X_out.std()
    assert abs(observed_std - strength) / strength < 0.1  # within 10 %


# ===========================================================================
# make_multisite_classification — output shapes and types
# ===========================================================================


@pytest.mark.parametrize(
    "n_sites,n_samples,n_features,n_classes",
    [
        (2, 200, 10, 2),
        (3, 300, 20, 3),
        (5, 500, 5, 2),
        (1, 100, 10, 2),
    ],
)
def test_output_shapes(n_sites, n_samples, n_features, n_classes):
    """Test."""
    X, y, sites = make_multisite_classification(
        n_sites=n_sites,
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        random_state=0,
    )
    assert X.shape == (n_samples, n_features)
    assert y.shape == (n_samples,)
    assert sites.shape == (n_samples,)


def test_output_dtypes():
    """Test."""
    X, y, sites = make_multisite_classification(n_sites=2, n_samples=100, random_state=0)
    assert X.dtype == float
    assert y.dtype in (np.int32, np.int64)
    assert sites.dtype in (np.int32, np.int64)


def test_sites_labels_range():
    """Test."""
    n_sites = 4
    _, _, sites = make_multisite_classification(n_sites=n_sites, n_samples=400, random_state=0)
    assert set(np.unique(sites)) == set(range(n_sites))


def test_y_labels_range():
    """Test."""
    n_classes = 3
    _, y, _ = make_multisite_classification(n_sites=2, n_samples=300, n_classes=n_classes, random_state=0)
    assert set(np.unique(y)) == set(range(n_classes))


# ===========================================================================
# make_multisite_classification — n_samples distribution
# ===========================================================================


def test_n_samples_integer_total_is_exact():
    """Test."""
    X, _, _ = make_multisite_classification(n_sites=3, n_samples=300, random_state=0)
    assert len(X) == 300


def test_n_samples_list():
    """Test."""
    X, _, sites = make_multisite_classification(n_sites=3, n_samples=[100, 200, 150], random_state=0)
    assert X.shape[0] == 450
    for site_idx, expected in enumerate([100, 200, 150]):
        assert np.sum(sites == site_idx) == expected


def test_n_samples_list_wrong_length_raises():
    """Test."""
    with pytest.raises(ValueError, match="n_samples"):
        make_multisite_classification(n_sites=3, n_samples=[100, 200], random_state=0)


def test_n_samples_uneven_split_distributes_remainder():
    """301 samples over 3 sites → at least one site gets an extra sample."""
    X, _, sites = make_multisite_classification(n_sites=3, n_samples=301, random_state=0)
    assert len(X) == 301
    counts = [np.sum(sites == i) for i in range(3)]
    assert sorted(counts) == [100, 100, 101]


def test_class_weights_valid():
    """Test."""
    _, y, _ = make_multisite_classification(
        n_sites=2,
        n_samples=1000,
        balance_per_site=[0.8, 0.2],
        random_state=0,
    )
    counts = np.bincount(y)
    # Minority class should be noticeably smaller
    assert counts[1] < counts[0]


def test_class_weights_wrong_length_raises():
    """Test."""
    with pytest.raises(ValueError, match="balance_per_site"):
        make_multisite_classification(
            n_sites=2,
            n_samples=200,
            n_classes=2,
            balance_per_site=[0.5, 0.3, 0.2],
            random_state=0,
        )


# ===========================================================================
# make_multisite_classification — reproducibility
# ===========================================================================


def test_same_random_state_reproduces_output():
    """Test."""
    X1, y1, s1 = make_multisite_classification(n_sites=2, n_samples=100, random_state=42)
    X2, y2, s2 = make_multisite_classification(n_sites=2, n_samples=100, random_state=42)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)
    np.testing.assert_array_equal(s1, s2)


def test_different_random_state_produces_different_output():
    """Test."""
    X1, _, _ = make_multisite_classification(n_sites=2, n_samples=100, random_state=0)
    X2, _, _ = make_multisite_classification(n_sites=2, n_samples=100, random_state=99)
    assert not np.allclose(X1, X2)


@pytest.mark.parametrize(
    "random_state",
    [
        0,
        42,
        np.random.RandomState(7),
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
# make_multisite_classification — backward compatibility (3-tuple return)
# ===========================================================================


def test_no_covariates_returns_3_tuple():
    """Test."""
    result = make_multisite_classification(n_sites=2, n_samples=100, random_state=0)
    assert len(result) == 3


def test_with_covariates_returns_4_tuple():
    """Test."""
    result = make_multisite_classification(
        n_sites=2,
        n_samples=100,
        covariates=[_age_cov()],
        random_state=0,
    )
    assert len(result) == 4


# ===========================================================================
# make_multisite_classification — covariates output
# ===========================================================================


def test_covariates_dict_keys():
    """Test."""
    _, _, _, covs = make_multisite_classification(
        n_sites=2,
        n_samples=200,
        covariates=[_age_cov(), _sex_cov()],
        random_state=0,
    )
    assert set(covs.keys()) == {"age", "sex"}


def test_covariates_array_lengths_match_x():
    """Test."""
    X, _, _, covs = make_multisite_classification(
        n_sites=2,
        n_samples=200,
        covariates=[_age_cov()],
        random_state=0,
    )
    assert covs["age"].shape == (len(X),)


def test_covariate_continuous_clip_in_full_output():
    """Test."""
    _, _, _, covs = make_multisite_classification(
        n_sites=2,
        n_samples=1000,
        covariates=[_age_cov()],
        random_state=0,
    )
    assert covs["age"].min() >= 18.0
    assert covs["age"].max() <= 90.0


def test_covariate_categorical_values_in_full_output():
    """Test."""
    _, _, _, covs = make_multisite_classification(
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
    _, _, sites, covs = make_multisite_classification(
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
    _, _, sites, covs = make_multisite_classification(
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
    result = make_multisite_classification(
        n_sites=2,
        n_samples=200,
        covariates=[preset],
        random_state=0,
    )
    assert len(result) == 4
    _, _, _, covs = result
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
        site_effect_strength=1,
        random_state=0,
    )
    _, y2, _ = make_multisite_classification(
        n_sites=2,
        n_samples=11,
        site_effect_strength=0,
        random_state=0,
    )
    np.testing.assert_array_equal(y1, y2)
