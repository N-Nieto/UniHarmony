"""ComBatGAM-specific tests (smooth covariates).

Behaviour shared by all ComBat variants (API, site validation, covariates,
harmonization effect, sklearn compatibility) is tested in ``test_combat_common.py``.
Data comes from the shared fixtures in ``tests/conftest.py``.
"""

import numpy as np
import pytest

from uniharmony.combat import ComBatGAM


# GAM fitting is slow: use few features to keep these tests light
N_GAM_FEATURES = 3


def test_combat_gam_multiple_smooth_covariates(multisite_data) -> None:
    """ComBatGAM fits one B-spline smoother per smooth covariate."""
    d = multisite_data
    X = d.X[:, :N_GAM_FEATURES]
    smooth = d.get("age", "extra", "education").astype(float)
    model = ComBatGAM().fit(X, d.sites, smooth, df=5)
    X_harmonized = model.transform(X, d.sites, smooth)
    assert X_harmonized.shape == X.shape
    assert np.all(np.isfinite(X_harmonized))
    assert len(model._bsplines.smoothers) == 3
    single = ComBatGAM().fit(X, d.sites, d.get("age"), df=5)
    assert model._bsplines.basis.shape[1] == 3 * single._bsplines.basis.shape[1]


def test_combat_gam_multiple_continuous_covariates(multisite_data) -> None:
    """ComBatGAM keeps one design column per continuous covariate."""
    d = multisite_data
    X = d.X[:, :N_GAM_FEATURES]
    continuous = d.get("sex", "education", "extra").astype(float)
    model = ComBatGAM().fit(X, d.sites, d.get("age"), continuous_covariates=continuous, df=5)
    X_harmonized = model.transform(X, d.sites, d.get("age"), continuous_covariates=continuous)
    assert X_harmonized.shape == X.shape
    # sites + 3 continuous covariates + spline basis columns
    assert model._beta_hat.shape[0] == d.n_sites + 3 + model._bsplines.basis.shape[1]


def test_combat_gam_transform_rejects_different_number_of_smooth_columns(multisite_data) -> None:
    """Transform fails clearly if the number of smooth covariates differs from fit."""
    d = multisite_data
    X = d.X[:, :2]
    model = ComBatGAM().fit(X, d.sites, d.get("age", "extra"), df=5)
    with pytest.raises(ValueError, match="smooth_covariates has 1 columns, but 2 were seen during fit"):
        model.transform(X, d.sites, d.get("age"))


def test_combat_gam_bounds_with_multiple_smooth_covariates_raises(multisite_data) -> None:
    """Custom bounds are only supported for a single smooth covariate."""
    d = multisite_data
    with pytest.raises(ValueError, match="only supported for a single smooth covariate"):
        ComBatGAM().fit(d.X[:, :2], d.sites, d.get("age", "extra"), smooth_covariates_bounds=(0.0, 100.0))


# ---------------------------------------------------------------------------
# Reproducibility (random_state)
# ---------------------------------------------------------------------------


def test_combat_gam_random_state_reproducible(multisite_data) -> None:
    """The same random_state gives identical results."""
    d = multisite_data
    X = d.X[:, :N_GAM_FEATURES]
    out_1 = ComBatGAM(random_state=0).fit_transform(X, d.sites, d.get("age"))
    out_2 = ComBatGAM(random_state=0).fit_transform(X, d.sites, d.get("age"))
    np.testing.assert_array_equal(out_1, out_2)


def test_combat_gam_random_state_instance(multisite_data) -> None:
    """A RandomState instance is accepted and gives reproducible results when re-seeded."""
    d = multisite_data
    X = d.X[:, :N_GAM_FEATURES]
    out_1 = ComBatGAM(random_state=np.random.RandomState(1)).fit_transform(X, d.sites, d.get("age"))
    out_2 = ComBatGAM(random_state=np.random.RandomState(1)).fit_transform(X, d.sites, d.get("age"))
    np.testing.assert_array_equal(out_1, out_2)


# ---------------------------------------------------------------------------
# Smooth covariates outside the fitted range
# ---------------------------------------------------------------------------


def test_combat_gam_smooth_covariate_outside_range_is_clipped(multisite_data) -> None:
    """Values outside the fitted range are clipped to its boundary instead of failing."""
    d = multisite_data
    X = d.X[:, :N_GAM_FEATURES]
    age = d.covariates["age"]
    model = ComBatGAM(random_state=0).fit(X, d.sites, age)

    age_out = age.copy()
    age_out[0] = age.min() - 10  # below the fitted range
    age_out[1] = age.max() + 10  # above the fitted range
    age_clipped = np.clip(age_out, age.min(), age.max())

    out = model.transform(X, d.sites, age_out)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, model.transform(X, d.sites, age_clipped))


def test_combat_gam_smooth_covariate_bounds_extend_range(multisite_data) -> None:
    """With smooth_covariates_bounds, values inside the bounds are not clipped."""
    d = multisite_data
    X = d.X[:, :N_GAM_FEATURES]
    age = d.covariates["age"]
    model = ComBatGAM(random_state=0).fit(X, d.sites, age, smooth_covariates_bounds=(0.0, 100.0))

    age_out = age.copy()
    age_out[0] = age.max() + 10  # outside the data range, inside the bounds
    age_clipped = np.clip(age_out, age.min(), age.max())

    out = model.transform(X, d.sites, age_out)
    assert np.all(np.isfinite(out))
    assert not np.allclose(out[0], model.transform(X, d.sites, age_clipped)[0])
