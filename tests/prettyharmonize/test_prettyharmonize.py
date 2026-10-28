"""Tests for the harmonization step of PrettYharmonize (any ComBat-based model)."""

import numpy as np
import pytest
import structlog
from numpy.testing import assert_allclose

from uniharmony.combat import ComBatGAM, CovBat, NeuroComBat
from uniharmony.combat._available_models import COMBAT_MODELS
from uniharmony.prettyharmonize._prettyharmonize import TargetHarmonizer, target_argument


def test_target_argument() -> None:
    """Categorical targets go to the categorical covariates when possible; continuous ones to the smooth ones if free."""
    neuro, gam = COMBAT_MODELS[NeuroComBat], COMBAT_MODELS[ComBatGAM]
    assert target_argument(neuro, categorical=True, covariates={}) == "categorical_covariates"
    assert target_argument(neuro, categorical=False, covariates={}) == "continuous_covariates"
    assert target_argument(COMBAT_MODELS[CovBat], categorical=True, covariates={}) == "categorical_covariates"
    assert target_argument(gam, categorical=True, covariates={}) == "continuous_covariates"
    assert target_argument(gam, categorical=False, covariates={}) == "smooth_covariates"
    assert target_argument(gam, categorical=False, covariates={"smooth_covariates": None}) == "continuous_covariates"


@pytest.mark.parametrize("harm_model", [NeuroComBat(), NeuroComBat(mean_only=True), CovBat()])
def test_equals_harm_model_with_target_covariate(clf_data, harm_model) -> None:
    """The harmonizer is the wrapped model with the class as first categorical covariate."""
    X, y, sites = clf_data
    sex = np.random.default_rng(0).choice(["F", "M"], len(y))
    harmonizer = TargetHarmonizer(harm_model, "categorical_covariates", n_classes=2, categorical_argument=True)
    harmonizer.fit(X, y, sites, {"categorical_covariates": sex}, {})
    X_h = harmonizer.transform(X, y, sites, {"categorical_covariates": sex})
    reference = type(harm_model)(**harm_model.get_params())
    expected = reference.fit_transform(X, sites, categorical_covariates=np.c_[y, sex].astype(object))
    assert_allclose(X_h, expected)
    assert not hasattr(harm_model, "sites_")  # the given instance is not fitted


def test_one_hot_target_in_numeric_covariates(clf_data) -> None:
    """A multi-class target is one-hot encoded (first class dropped) in a numeric covariate argument."""
    X, _, sites = clf_data
    y3 = np.arange(len(sites)) % 3
    harmonizer = TargetHarmonizer(NeuroComBat(), "continuous_covariates", n_classes=3, categorical_argument=False)
    harmonizer.fit(X, y3, sites, {}, {})
    X_h = harmonizer.transform(X, y3, sites, {})
    one_hot = np.c_[y3 == 1, y3 == 2].astype(float)
    assert_allclose(X_h, NeuroComBat().fit_transform(X, sites, continuous_covariates=one_hot))
    categorical = TargetHarmonizer(NeuroComBat(), "categorical_covariates", n_classes=3, categorical_argument=True)
    categorical.fit(X, y3, sites, {}, {})
    assert_allclose(X_h, categorical.transform(X, y3, sites, {}))


def test_iter_pretend(clf_data) -> None:
    """Pretending harmonizes the same data with each given target."""
    X, y, sites = clf_data
    harmonizer = TargetHarmonizer(NeuroComBat(), "categorical_covariates", n_classes=2, categorical_argument=True)
    harmonizer.fit(X, y, sites, {}, {})
    targets = [0, 1, np.arange(len(y)) % 2]
    for target, X_t in zip(targets, harmonizer.iter_pretend(X, sites, targets, {}), strict=True):
        expected = harmonizer.harm_model.transform(X, sites, categorical_covariates=np.broadcast_to(target, y.shape))
        assert_allclose(X_t, expected)


def test_pretending_acts_through_the_scale_correction(clf_data) -> None:
    """With ComBat, pretended versions differ only through the scale correction (none with mean_only)."""
    X, y, sites = clf_data
    for mean_only, differ in ((True, False), (False, True)):
        harmonizer = TargetHarmonizer(NeuroComBat(mean_only=mean_only), "categorical_covariates", 2, categorical_argument=True)
        harmonizer.fit(X, y, sites, {}, {})
        X0, X1 = harmonizer.iter_pretend(X, sites, [0, 1], {})
        assert np.allclose(X0, X1, atol=1e-10) is not differ


def test_smooth_target_outside_bounds(reg_data) -> None:
    """A smooth target outside the spline range raises a clear error."""
    X, y, sites = reg_data
    harmonizer = TargetHarmonizer(ComBatGAM(), "smooth_covariates", n_classes=None, categorical_argument=False)
    harmonizer.fit(X, y, sites, {}, {"smooth_covariates_bounds": (y.min(), y.max())})
    assert harmonizer.transform(X, np.full_like(y, y.max()), sites, {}).shape == X.shape
    with pytest.raises(ValueError, match="cannot extrapolate"):
        harmonizer.transform(X, np.full_like(y, y.max() + 10), sites, {})


def test_no_leakage_warning(clf_data) -> None:
    """The ComBat target-leakage warning is silenced inside PrettYharmonize."""
    X, y, sites = clf_data
    with structlog.testing.capture_logs() as logs:
        NeuroComBat().fit(X, sites, categorical_covariates=y)
    assert any("DO *NOT* preserve" in entry["event"] for entry in logs)
    with structlog.testing.capture_logs() as logs:
        TargetHarmonizer(NeuroComBat(), "categorical_covariates", 2, categorical_argument=True).fit(X, y, sites, {}, {})
    assert not any("DO *NOT* preserve" in entry["event"] for entry in logs)
