"""Test the SoftBART forest translated from the SoftBart R package."""

import math

import numpy as np
import pytest

from uniharmony.iqm._softbart import (
    SoftBARTForest,
    _get_perturb_limits,
    _leaf_weights,
    _leaves,
    _Node,
    _rlgam,
    _TreeLikelihood,
    _update_sigma,
    quantile_normalize_bart,
)


@pytest.fixture
def friedman_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Friedman-like regression data with two irrelevant predictors (z-scored response)."""
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(400, 6))
    f = 10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3]
    y = f + rng.normal(size=400)
    scale = y.std(ddof=1)
    return X, (y - y.mean()) / scale, (f - y.mean()) / scale


def _split(tree: _Node, var: int, val: float) -> None:
    tree.add_leaves()
    tree.var = var
    tree.get_limits()
    tree.val = val


def test_quantile_normalize_bart() -> None:
    """Values map to their rank among unique values, scaled to [0, 1] (``trank``)."""
    X = np.array([[3.0, 1.0], [1.0, 1.0], [2.0, 1.0], [2.0, 1.0], [5.0, 1.0]])
    out = quantile_normalize_bart(X)
    np.testing.assert_allclose(out[:, 0], [2 / 3, 0, 1 / 3, 1 / 3, 1])
    np.testing.assert_array_equal(out[:, 1], 0.0)
    np.testing.assert_allclose(quantile_normalize_bart(np.array([2.0, 1.0])), [[1.0], [0.0]])


def test_forest_fits_smooth_function(friedman_data) -> None:
    """The forest recovers the regression function and selects the relevant predictors."""
    X, y, f = friedman_data
    forest = SoftBARTForest(X.shape[1], num_tree=50, num_burn=100, random_state=0)
    forest.set_sigma(np.std(y - f))  # the true noise level
    predictions = [forest.do_gibbs(X, y) for _ in range(200)]
    assert forest.num_gibbs == 200
    posterior_mean = np.mean(predictions[100:], axis=0)
    assert np.sqrt(np.mean((posterior_mean - f) ** 2)) < 0.1
    # The sparsity prior puts little mass on the irrelevant predictors 4 and 5
    assert forest.get_s()[[4, 5]].sum() < 0.1


def test_forest_predictions_are_consistent(friedman_data) -> None:
    """``do_gibbs`` returns the prediction of the updated forest; snapshots predict the same."""
    X, y, _ = friedman_data
    forest = SoftBARTForest(X.shape[1], num_tree=10, random_state=0)
    for _ in range(5):
        prediction = forest.do_gibbs(X, y)
    np.testing.assert_allclose(prediction, forest.predict(X), atol=1e-10)
    np.testing.assert_allclose(forest.snapshot().predict(X), forest.predict(X), atol=1e-10)
    X_new = np.random.default_rng(1).uniform(size=(7, X.shape[1]))
    np.testing.assert_allclose(forest.snapshot().predict(X_new), forest.predict(X_new), atol=1e-10)


def test_forest_weighted(friedman_data) -> None:
    """Weighted backfitting runs and differs from the unweighted one."""
    X, y, _ = friedman_data
    weights = np.where(np.arange(len(y)) % 2 == 0, 4.0, 0.25)
    unweighted = SoftBARTForest(X.shape[1], num_tree=10, random_state=0)
    weighted = SoftBARTForest(X.shape[1], num_tree=10, random_state=0)
    assert not np.allclose(unweighted.do_gibbs(X, y, num_iter=3), weighted.do_gibbs(X, y, weights, num_iter=3))


def test_s_and_alpha_fixed_during_warmup(friedman_data) -> None:
    """Splitting probabilities are only updated after floor(num_burn / 2) iterations."""
    X, y, _ = friedman_data
    forest = SoftBARTForest(X.shape[1], num_tree=5, num_burn=10, random_state=0)
    forest.do_gibbs(X, y, num_iter=6)  # num_gibbs 0..5 <= 5
    np.testing.assert_array_equal(forest.get_s(), np.full(X.shape[1], 1 / X.shape[1]))
    assert forest.hypers.alpha == 1.0
    forest.do_gibbs(X, y)
    assert not np.allclose(forest.get_s(), 1 / X.shape[1])


def test_leaf_weights() -> None:
    """Leaf weights follow ``1 - expit((x - c) / tau)`` and sum to one."""
    tree = _Node.root(width=0.1)
    _split(tree, var=0, val=0.5)
    _split(tree.right, var=1, val=0.3)
    X = np.array([[0.5, 0.3], [0.1, 0.9], [0.9, 0.0]])
    W = _leaf_weights(tree, X)
    assert W.shape == (3, 3)
    np.testing.assert_allclose(W.sum(axis=1), 1.0)
    np.testing.assert_allclose(W[0], [0.5, 0.25, 0.25])
    go_left = 1.0 - 1.0 / (1.0 + math.exp(-(0.1 - 0.5) / 0.1))
    np.testing.assert_allclose(W[1, 0], go_left)


def test_tree_loglik_matches_marginal_likelihood() -> None:
    """``LogLT`` equals the log density of the residual with the leaf parameters integrated out."""
    rng = np.random.default_rng(0)
    tree = _Node.root(width=0.2)
    _split(tree, var=0, val=0.4)
    X = rng.uniform(size=(30, 1))
    r = rng.normal(size=30)
    weights = rng.uniform(0.5, 2.0, size=30)
    forest = SoftBARTForest(1, num_tree=4)
    forest.set_sigma(0.7)
    hypers = forest.hypers
    W = _leaf_weights(tree, X)
    # r ~ N(0, sigma^2 diag(1 / weights) + sigma_mu^2 W W^T)
    cov = hypers.sigma**2 * np.diag(1 / weights) + hypers.sigma_mu**2 * W @ W.T
    _, log_det = np.linalg.slogdet(2 * np.pi * cov)
    expected = -0.5 * log_det - 0.5 * r @ np.linalg.solve(cov, r)
    assert _TreeLikelihood(r, weights, hypers, unit_weights=False).loglik(W) == pytest.approx(expected)
    ones = np.ones(30)
    assert _TreeLikelihood(r, ones, hypers, unit_weights=True).loglik(W) == pytest.approx(
        _TreeLikelihood(r, ones, hypers, unit_weights=False).loglik(W)
    )


def test_get_limits_and_perturb_limits() -> None:
    """Cut point ranges follow the ancestors (and, for perturbations, the whole tree)."""
    tree = _Node.root(width=0.1)
    _split(tree, var=0, val=0.6)
    _split(tree.left, var=0, val=0.2)
    leaf = tree.left.right
    leaf.add_leaves()
    leaf.var = 0
    leaf.get_limits()
    assert (leaf.lower, leaf.upper) == (0.2, 0.6)
    # The root's limits scan all branches of its left subtree for the lower bound
    assert _get_perturb_limits(tree) == (0.2, 1.0)


def test_rlgam_small_shape() -> None:
    """``rlgam`` draws log-gamma variables, also below shape 0.1."""
    rng = np.random.default_rng(0)
    for shape in (0.05, 2.0):
        draws = np.exp([_rlgam(shape, rng) for _ in range(20000)])
        assert np.mean(draws) == pytest.approx(shape, rel=0.1)


def test_update_sigma_concentrates() -> None:
    """The half-Cauchy sigma update concentrates around the sample standard deviation."""
    rng = np.random.default_rng(0)
    r = rng.normal(scale=0.3, size=2000)
    sigma = 1.0
    draws = []
    for _ in range(200):
        sigma = _update_sigma(r, 1.0, sigma, rng)
        draws.append(sigma)
    assert np.mean(draws[50:]) == pytest.approx(0.3, rel=0.05)


def test_leaves_order() -> None:
    """Leaves are listed from left to right."""
    tree = _Node.root(width=0.1)
    _split(tree, var=0, val=0.5)
    _split(tree.left, var=0, val=0.2)
    assert _leaves(tree) == [tree.left.left, tree.left.right, tree.right]


def test_invalid_forest() -> None:
    """Invalid sizes raise errors."""
    with pytest.raises(ValueError, match="n_features"):
        SoftBARTForest(0)
    with pytest.raises(ValueError, match="num_tree"):
        SoftBARTForest(2, num_tree=0)
    forest = SoftBARTForest(2)
    with pytest.raises(ValueError, match="X must have shape"):
        forest.do_gibbs(np.zeros((5, 3)), np.zeros(5))
    with pytest.raises(ValueError, match="Y must have shape"):
        forest.do_gibbs(np.zeros((5, 2)), np.zeros(4))
