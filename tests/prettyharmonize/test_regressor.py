"""Tests for PrettYharmonizeRegressor."""

import numpy as np
import pytest
import structlog
from numpy.testing import assert_allclose
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split

from uniharmony.combat import ComBatGAM
from uniharmony.prettyharmonize import PrettYharmonizeRegressor


def test_default_models(reg_data) -> None:
    """Regression with the default models."""
    X, y, sites = reg_data
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(X, y, sites, test_size=0.25, random_state=0)
    reg = PrettYharmonizeRegressor(random_state=0).fit(X_train, y_train, s_train)
    assert reg.predict(X_test, s_test).shape == y_test.shape
    assert reg.score_matrix_.shape == (len(y_train), 10)
    assert_allclose(reg.regression_points_, np.linspace(y_train.min(), y_train.max(), 10))
    assert reg.score(X_test, y_test, s_test) > 0.3


def test_regression_points(reg_data) -> None:
    """Regression points: defaults, int, array, out-of-range warning and errors."""
    X, y, sites = reg_data
    assert PrettYharmonizeRegressor(regression_points=4).fit(X, y, sites).score_matrix_.shape[1] == 4
    points = [y.min(), y.mean(), y.max()]
    assert_allclose(PrettYharmonizeRegressor(regression_points=points).fit(X, y, sites).regression_points_, points)
    with structlog.testing.capture_logs() as logs:
        PrettYharmonizeRegressor(regression_points=[0, 1000]).fit(X, y, sites)
    assert any("outside the training target range" in entry["event"] for entry in logs)
    with pytest.raises(ValueError, match=">= 2"):
        PrettYharmonizeRegressor(regression_points=1).fit(X, y, sites)
    with pytest.raises(TypeError, match="regression_points"):
        PrettYharmonizeRegressor(regression_points=10.0).fit(X, y, sites)


def test_refit_does_not_mutate_params(reg_data) -> None:
    """Fitting does not modify the regression_points parameter."""
    X, y, sites = reg_data
    reg = PrettYharmonizeRegressor()
    first = reg.fit(X, y, sites).regression_points_.copy()
    reg.fit(X, 10 * y, sites)
    assert reg.regression_points is None
    assert_allclose(reg.regression_points_, 10 * first)


def _reference_search(reg, X, sites):
    """Per-sample search loop of the original PrettYharmonize implementation."""
    preds = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        x, s = X[[i]], sites[[i]]
        cur1, cur2 = np.array([reg.y_min_]), np.array([reg.y_max_])
        ntries, cur_dif, d1, d2 = 0, np.inf, np.inf, np.inf
        while cur_dif > reg.regression_search_tol and ntries < 20:
            pred1 = reg.base_model_.predict(reg.harm_model_.transform(x, s, continuous_covariates=cur1.reshape(-1, 1)))
            pred2 = reg.base_model_.predict(reg.harm_model_.transform(x, s, continuous_covariates=cur2.reshape(-1, 1)))
            d1, d2 = abs(cur1 - pred1), abs(cur2 - pred2)
            if d1 < d2:
                cur_dif, cur2 = d1, (cur1 + cur2) / 2
            else:
                cur_dif, cur1 = d2, (cur1 + cur2) / 2
            ntries += 1
        preds[i] = pred1[0] if d1 < d2 else pred2[0]
    return preds


@pytest.mark.parametrize("tol", [0.0, 0.5])
def test_search_matches_reference_loop(reg_data, tol) -> None:
    """The vectorized search equals the original per-sample loop."""
    X, y, sites = reg_data
    reg = PrettYharmonizeRegressor(
        base_model=Ridge(), stack_model=LinearRegression(), regression_search=True, regression_search_tol=tol, random_state=0
    ).fit(X, y, sites)
    assert reg.score_matrix_.shape == (len(y), 1)
    fast = reg._search(reg._harmonizers[0], reg.base_model_, X[:30], sites[:30], {})
    assert_allclose(fast, _reference_search(reg, X[:30], sites[:30]), atol=1e-10)


def test_search_options(reg_data) -> None:
    """Search without final model and parameter validation."""
    X, y, sites = reg_data
    reg = PrettYharmonizeRegressor(regression_search=True, fit_final_model=False, cv=3).fit(X, y, sites)
    assert reg.score_matrix_.shape == (len(y), 1)
    assert reg.predict(X[:4], sites[:4]).shape == (4,)
    for kwargs, match in (
        ({"regression_search_max_iter": 0}, "max_iter"),
        ({"regression_search_tol": -1}, "tol"),
    ):
        with pytest.raises(ValueError, match=match):
            PrettYharmonizeRegressor(regression_search=True, **kwargs).fit(X, y, sites)


def test_covariates_and_transform(reg_data, rng) -> None:
    """Additional covariates in the regressor; transform uses the predicted target."""
    X, y, sites = reg_data
    age = rng.uniform(20, 80, len(y))
    reg = PrettYharmonizeRegressor(random_state=0)
    assert reg.fit_transform(X, y, sites, {"continuous_covariates": age}).shape == X.shape
    y_pred = reg.predict(X, sites, {"continuous_covariates": age})
    assert_allclose(
        reg.transform(X, sites, {"continuous_covariates": age}),
        reg.harm_model_.transform(X, sites, continuous_covariates=np.c_[y_pred, age]),
    )


def test_errors(reg_data) -> None:
    """Invalid inputs raise informative errors."""
    X, y, sites = reg_data
    with pytest.raises(ValueError, match="constant"):
        PrettYharmonizeRegressor().fit(X, np.ones(len(y)), sites)
    with pytest.raises(ValueError, match="numeric"):
        PrettYharmonizeRegressor().fit(X, np.array(["a"] * len(y)), sites)


def test_clone() -> None:
    """The estimator follows the scikit-learn parameter API."""
    params = clone(PrettYharmonizeRegressor(regression_points=5, cv=3)).get_params()
    assert params["regression_points"] == 5
    assert params["cv"] == 3


def test_combat_gam_smooth_target(reg_data, rng) -> None:
    """With ComBatGAM the target is a smooth covariate (spline bounds set to the target range)."""
    X, y, sites = reg_data
    reg = PrettYharmonizeRegressor(harm_model=ComBatGAM(), cv=3, random_state=0).fit(X[:, :3], y, sites)
    assert reg._target_argument == "smooth_covariates"
    assert reg.predict(X[:5, :3], sites[:5]).shape == (5,)
    # If the smooth argument is used by another covariate, the target is a continuous covariate.
    age = rng.uniform(20, 80, len(y))
    reg.fit(X[:, :3], y, sites, {"smooth_covariates": age})
    assert reg._target_argument == "continuous_covariates"
    assert reg.predict(X[:5, :3], sites[:5], {"smooth_covariates": age[:5]}).shape == (5,)
