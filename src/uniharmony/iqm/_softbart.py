"""Provide the soft Bayesian additive regression tree (SoftBART) forest used by BARTharm."""

# Translated from the SoftBart R package, version 1.0.3:
# https://github.com/theodds/SoftBART (src/soft_bart.cpp, src/soft_bart.h,
# src/functions.h, R/SoftBart.R, R/quantile_normalize_bart.R)
# licensed under GPL (>= 2).
#
# Linero, A. R., & Yang, Y. (2018). Bayesian regression tree ensembles that
# adapt to smoothness and sparsity. Journal of the Royal Statistical Society:
# Series B, 80(5), 1087-1110. https://doi.org/10.1111/rssb.12293
#
# Only the parts that BARTharm uses are translated: the ``Forest`` object
# (``MakeForest()`` in R) with its ``do_gibbs()``, ``do_gibbs_weighted()``,
# ``set_sigma()`` and ``do_predict()`` methods, under the default ``Opts()``
# except ``update_sigma`` (``update_beta``, ``update_gamma``, ``update_tau_mean``
# and ``update_num_tree`` are off by default and not translated). Every
# predictor is its own group (the default of ``Hypers()``).
#
# Differences to the C++ code that do not change the sampled distributions:
#
# * Leaf weights, sufficient statistics and predictions are computed for all
#   samples at once (vectorized) instead of one sample at a time.
# * The leaf weights and the log marginal likelihood of the current state of
#   the tree being updated are cached instead of being recomputed for every
#   proposal, and so are the per-tree predictions at the training data between
#   iterations.
# * The leaf parameters are drawn as ``mu_hat + solve(C^T, z)``, with
#   ``Omega_inv = C C^T``, instead of ``mu_hat + chol(inv(Omega_inv)) z``.
# * Random numbers come from a :class:`numpy.random.Generator`, so draws do not
#   match R's random number stream.

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.linalg import solve_triangular
from scipy.special import betaln, expit, gammaln, logsumexp


__all__ = ["ForestSnapshot", "SoftBARTForest", "quantile_normalize_bart"]

_LOG_2PI = math.log(2.0 * math.pi)
# Probability of proposing a new tree from the prior (MH_PRIOR in TreeBackfit)
_MH_PRIOR = 0.4
# Probability of a birth/death move instead of a perturbation (MH_BD in TreeBackfit)
_MH_BD = 0.7
# Grid used to update the sparsity parameter alpha (GRID_SIZE in Hypers)
_RHO_GRID = np.arange(1, 1000) / 1000.0


def quantile_normalize_bart(X: npt.ArrayLike) -> npt.NDArray:
    """Quantile normalize each column of ``X`` to lie in [0, 1].

    Translation of ``quantile_normalize_bart()`` from SoftBart: every value is
    replaced by the rank of its value among the unique values of its column,
    rescaled so that the smallest value maps to 0 and the largest to 1.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data.

    Returns
    -------
    ndarray, shape (n_samples, n_features)
        The normalized data. Constant columns are mapped to 0 (they are
        undefined in SoftBart).

    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, np.newaxis]
    out = np.zeros_like(X)
    for j in range(X.shape[1]):
        unique_values, ranks = np.unique(X[:, j], return_inverse=True)
        if len(unique_values) > 1:
            out[:, j] = ranks / (len(unique_values) - 1)
    return out


# ---------------------------------------------------------------------------
# Trees
# ---------------------------------------------------------------------------


class _Node:
    """Node of a soft decision tree (``Node`` in SoftBart)."""

    __slots__ = ("is_leaf", "is_root", "left", "lower", "mu", "parent", "right", "tau", "upper", "val", "var")

    def __init__(self) -> None:
        self.is_leaf = True
        self.is_root = True
        self.left: _Node | None = None
        self.right: _Node | None = None
        self.parent: _Node | None = None
        # Branch parameters
        self.var = 0
        self.val = 0.0
        self.lower = 0.0
        self.upper = 1.0
        self.tau = 1.0
        # Leaf parameter
        self.mu = 0.0

    @classmethod
    def root(cls, width: float) -> "_Node":
        """Make a tree with a single leaf (``Node::Root``)."""
        node = cls()
        node.tau = width
        return node

    def is_left(self) -> bool:
        """Whether the node is the left child of its parent."""
        return self is self.parent.left

    def add_leaves(self) -> None:
        """Turn the leaf into a branch with two leaves (``Node::AddLeaves``)."""
        for side in ("left", "right"):
            child = _Node()
            child.is_root = False
            child.parent = self
            child.tau = self.tau
            setattr(self, side, child)
        self.is_leaf = False

    def birth_leaves(self, hypers: "_Hypers", rng: np.random.Generator) -> None:
        """Split the leaf with a decision rule drawn from the prior (``Node::BirthLeaves``)."""
        if self.is_leaf:
            self.add_leaves()
            self.var = hypers.sample_var(rng)
            self.get_limits()
            self.val = (self.upper - self.lower) * rng.random() + self.lower

    def gen_below(self, hypers: "_Hypers", rng: np.random.Generator) -> None:
        """Grow the subtree below the node from the prior (``Node::GenBelow``)."""
        if rng.random() < hypers.growth_prior(_depth(self)):
            self.birth_leaves(hypers, rng)
            self.left.gen_below(hypers, rng)
            self.right.gen_below(hypers, rng)

    def get_limits(self) -> None:
        """Find the range of cut points allowed for the node's variable (``Node::GetLimits``)."""
        y = self
        self.lower = 0.0
        self.upper = 1.0
        searching = not y.is_root
        while searching:
            is_left = y.is_left()
            y = y.parent
            searching = not y.is_root
            if y.var == self.var:
                searching = False
                if is_left:
                    self.upper = y.val
                    self.lower = y.lower
                else:
                    self.upper = y.upper
                    self.lower = y.val

    def delete_leaves(self) -> None:
        """Turn the branch back into a leaf (``Node::DeleteLeaves``)."""
        self.left = None
        self.right = None
        self.is_leaf = True

    def set_tau(self, tau: float) -> None:
        """Set the bandwidth of the node and all nodes below it (``Node::SetTau``)."""
        self.tau = tau
        if not self.is_leaf:
            self.left.set_tau(tau)
            self.right.set_tau(tau)


def _depth(node: _Node) -> int:
    depth = 0
    while not node.is_root:
        node = node.parent
        depth += 1
    return depth


def _leaves(node: _Node, out: list[_Node] | None = None) -> list[_Node]:
    """Leaves from left to right (``leaves``)."""
    if out is None:
        out = []
    if node.is_leaf:
        out.append(node)
    else:
        _leaves(node.left, out)
        _leaves(node.right, out)
    return out


def _branches(node: _Node, out: list[_Node] | None = None) -> list[_Node]:
    """Non-leaf nodes in pre-order (``branches``)."""
    if out is None:
        out = []
    if not node.is_leaf:
        out.append(node)
        _branches(node.left, out)
        _branches(node.right, out)
    return out


def _not_grand_branches(node: _Node, out: list[_Node] | None = None) -> list[_Node]:
    """Branches whose children are both leaves (``not_grand_branches``)."""
    if out is None:
        out = []
    if not node.is_leaf:
        if node.left.is_leaf and node.right.is_leaf:
            out.append(node)
        else:
            _not_grand_branches(node.left, out)
            _not_grand_branches(node.right, out)
    return out


def _probability_node_birth(tree: _Node) -> float:
    return 1.0 if tree.is_leaf else 0.5


def _calc_cutpoint_likelihood(node: _Node) -> float:
    """Product of ``1 / (upper - lower)`` over the branches (``calc_cutpoint_likelihood``)."""
    if node.is_leaf:
        return 1.0
    out = _reciprocal(node.upper - node.lower)
    return out * _calc_cutpoint_likelihood(node.left) * _calc_cutpoint_likelihood(node.right)


def _get_perturb_limits(branch: _Node) -> tuple[float, float]:
    """Range of cut points for a perturbed decision rule (``get_perturb_limits``).

    Notes
    -----
    As in SoftBart, the cut points of the same variable are searched among the
    ancestors of ``branch`` and among *all* branches in the left (for the lower
    limit) and right (for the upper limit) subtrees of the root, not only among
    the descendants of ``branch``. The lower limit can therefore exceed the
    upper limit; the proposal is then accepted (the Metropolis-Hastings ratio
    is NaN), as in SoftBart.

    """
    lower = 0.0
    upper = 1.0
    node = branch
    while not node.is_root:
        if node.is_left():
            node = node.parent
            if node.var == branch.var and node.val > lower:
                lower = node.val
        else:
            node = node.parent
            if node.var == branch.var and node.val < upper:
                upper = node.val
    for other in _branches(node.left):
        if other.var == branch.var and other.val > lower:
            lower = other.val
    for other in _branches(node.right):
        if other.var == branch.var and other.val < upper:
            upper = other.val
    return lower, upper


def _get_limits_below(node: _Node) -> None:
    node.get_limits()
    if not node.left.is_leaf:
        _get_limits_below(node.left)
    if not node.right.is_leaf:
        _get_limits_below(node.right)


def _reciprocal(x: float) -> float:
    """``1 / x`` with IEEE semantics (``inf`` for 0), as in C++."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.float64(1.0) / np.float64(x))


def _log(x: float) -> float:
    """``log(x)`` with IEEE semantics (``-inf`` for 0, NaN for negative), as in C++."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.log(np.float64(x)))


def _leaf_weights(tree: _Node, X: npt.NDArray) -> npt.NDArray:
    """Weight of every leaf for every sample (``Node::GetW``), shape (n_samples, n_leaves).

    The probability of going left at a branch is ``1 - expit((x - val) / tau)``.
    """
    if tree.is_leaf:
        return np.ones((X.shape[0], 1))
    columns: list[npt.NDArray] = []

    def descend(node: _Node, weight: npt.NDArray | None) -> None:
        if node.is_leaf:
            columns.append(weight)
            return
        go_left = expit((node.val - X[:, node.var]) / node.tau)
        if weight is None:
            descend(node.left, go_left)
            descend(node.right, 1.0 - go_left)
        else:
            descend(node.left, weight * go_left)
            descend(node.right, weight * (1.0 - go_left))

    descend(tree, None)
    return np.column_stack(columns)


def _sample_class(probs: npt.NDArray, rng: np.random.Generator) -> int:
    """Draw an index with the given cumulative probabilities (``sample_class``)."""
    index = int(np.searchsorted(probs, rng.random(), side="right"))
    return min(index, len(probs) - 1)


def _uniform(rng: np.random.Generator) -> float:
    """Draw from Uniform(0, 1], so that its log is finite (R's ``unif_rand()`` never returns 0)."""
    return 1.0 - rng.random()


def _log_uniform(rng: np.random.Generator) -> float:
    """Draw ``log(U)`` with ``U ~ Uniform(0, 1]``, for Metropolis-Hastings acceptance tests."""
    return math.log(_uniform(rng))


def _rand_index(n: int, rng: np.random.Generator) -> int:
    """Uniformly draw an index in ``range(n)`` (``rand`` / ``sample_class(n)``)."""
    return min(int(rng.random() * n), n - 1)


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------


@dataclass
class _Hypers:
    """Hyperparameters of a forest (``Hypers`` in SoftBart)."""

    n_features: int
    num_tree: int
    alpha: float = 1.0
    beta: float = 2.0
    gamma: float = 0.95
    k: float = 2.0
    sigma: float = 1.0
    sigma_hat: float = 1.0
    width: float = 0.1
    tau_rate: float = 10.0
    temperature: float = 1.0
    alpha_scale: float | None = None
    alpha_shape_1: float = 0.5
    alpha_shape_2: float = 1.0
    sigma_mu: float = field(init=False)
    sigma_mu_hat: float = field(init=False)
    s: npt.NDArray = field(init=False)
    logs: npt.NDArray = field(init=False)
    s_cumulative: npt.NDArray = field(init=False)

    def __post_init__(self) -> None:
        if self.alpha_scale is None:
            self.alpha_scale = float(self.n_features)
        self.sigma_mu = 0.5 / (self.k * math.sqrt(self.num_tree))
        self.sigma_mu_hat = self.sigma_mu
        self.set_s(np.full(self.n_features, 1.0 / self.n_features))

    def set_s(self, s: npt.NDArray) -> None:
        """Set the splitting probabilities."""
        self.s = s
        with np.errstate(divide="ignore"):
            self.logs = np.log(s)
        self.s_cumulative = np.cumsum(s)

    def sample_var(self, rng: np.random.Generator) -> int:
        """Draw a splitting variable (``Hypers::SampleVar``)."""
        return _sample_class(self.s_cumulative, rng)

    def growth_prior(self, depth: int) -> float:
        """Prior probability that a node at ``depth`` is a branch (``growth_prior``)."""
        return self.gamma * (1.0 + depth) ** (-self.beta)

    def update_sigma_mu(self, means: npt.NDArray, rng: np.random.Generator) -> None:
        """Update the standard deviation of the leaf parameters (``Hypers::UpdateSigmaMu``)."""
        self.sigma_mu = _update_sigma(means, self.sigma_mu_hat, self.sigma_mu, rng)

    def update_s(self, var_counts: npt.NDArray, rng: np.random.Generator) -> None:
        """Update the splitting probabilities from their Dirichlet full conditional (``UpdateS``)."""
        shape = self.alpha / self.n_features + var_counts
        logs = np.array([_rlgam(a, rng) for a in shape])
        logs = logs - logsumexp(logs)
        self.logs = logs
        self.s = np.exp(logs)
        self.s_cumulative = np.cumsum(self.s)

    def update_alpha(self, rng: np.random.Generator) -> None:
        """Update the Dirichlet concentration on a grid (``Hypers::UpdateAlpha``)."""
        p = float(self.n_features)
        alpha = _rho_to_alpha(_RHO_GRID, self.alpha_scale)
        loglik = (
            alpha * np.mean(self.logs)
            + gammaln(alpha)
            - p * gammaln(alpha / p)
            + (self.alpha_shape_1 - 1.0) * np.log(_RHO_GRID)
            + (self.alpha_shape_2 - 1.0) * np.log1p(-_RHO_GRID)
            - betaln(self.alpha_shape_1, self.alpha_shape_2)
        )
        probs = np.exp(loglik - logsumexp(loglik))
        rho = _RHO_GRID[_sample_class(np.cumsum(probs), rng)]
        self.alpha = float(_rho_to_alpha(rho, self.alpha_scale))


def _rho_to_alpha(rho: float | npt.NDArray, scale: float) -> float | npt.NDArray:
    return scale * rho / (1.0 - rho)


def _cauchy_jacobian(tau: float, sigma_hat: float) -> float:
    """Log half-Cauchy prior density of ``sigma = tau^(-1/2)`` on the ``tau`` scale (``cauchy_jacobian``)."""
    sigma = tau**-0.5
    log_density = -math.log(math.pi * sigma_hat * (1.0 + (sigma / sigma_hat) ** 2))
    return log_density - math.log(2.0) - 1.5 * math.log(tau)


def _update_sigma(
    r: npt.NDArray,
    sigma_hat: float,
    sigma_old: float,
    rng: np.random.Generator,
    temperature: float = 1.0,
) -> float:
    """Metropolis-Hastings update of a standard deviation with a half-Cauchy prior (``update_sigma``)."""
    sse = float(r @ r) * temperature
    n = r.size * temperature
    shape = 0.5 * n + 1.0
    scale = 2.0 / sse
    sigma_prop = rng.gamma(shape, scale) ** -0.5
    log_ratio = _cauchy_jacobian(sigma_prop**-2.0, sigma_hat) - _cauchy_jacobian(sigma_old**-2.0, sigma_hat)
    return sigma_prop if _log_uniform(rng) < log_ratio else sigma_old


def _rlgam(shape: float, rng: np.random.Generator) -> float:
    """Draw ``log(Gamma(shape, 1))``, stable for small shapes (``rlgam``).

    Uses the method of Liu, Martin and Syring (2017) for ``shape < 0.1``.
    """
    if shape >= 0.1:
        return _log(rng.gamma(shape, 1.0))
    a = shape
    lam = 1.0 / a - 1.0
    w = math.exp(-1.0) * a / (1.0 - a)
    ww = 1.0 / (1.0 + w)
    while True:
        u = _uniform(rng)
        z = -math.log(u / ww) if u <= ww else _log_uniform(rng) / lam
        eta = -z if z >= 0 else math.log(w) + math.log(lam) + lam * z
        h = -z - math.exp(-z / a)
        if h - eta > _log_uniform(rng):
            return -z / a


# ---------------------------------------------------------------------------
# Likelihood of a tree
# ---------------------------------------------------------------------------


# Trees have few leaves, so the leaf-parameter posterior is factorized in plain
# Python below this size, which is much faster than LAPACK calls on tiny matrices.
_MAX_SMALL_SOLVE = 12


def _cholesky(matrix: npt.NDArray) -> list[list[float]]:
    """Lower Cholesky factor of a small symmetric positive definite matrix."""
    a = matrix.tolist()
    size = len(a)
    chol = [[0.0] * size for _ in range(size)]
    for j in range(size):
        row_j = chol[j]
        diag = a[j][j] - sum(row_j[k] * row_j[k] for k in range(j))
        if not diag > 0.0:
            raise np.linalg.LinAlgError("Matrix is not positive definite")
        row_j[j] = math.sqrt(diag)
        for i in range(j + 1, size):
            row_i = chol[i]
            row_i[j] = (a[i][j] - sum(row_i[k] * row_j[k] for k in range(j))) / row_j[j]
    return chol


def _solve_lower(chol: list[list[float]], b: list[float]) -> list[float]:
    """Solve ``chol @ x = b`` for lower triangular ``chol``."""
    x: list[float] = []
    for i, row in enumerate(chol):
        x.append((b[i] - sum(row[k] * x[k] for k in range(i))) / row[i])
    return x


def _solve_upper_transposed(chol: list[list[float]], b: list[float]) -> list[float]:
    """Solve ``chol.T @ x = b`` for lower triangular ``chol``."""
    size = len(chol)
    x = [0.0] * size
    for i in range(size - 1, -1, -1):
        x[i] = (b[i] - sum(chol[k][i] * x[k] for k in range(i + 1, size))) / chol[i][i]
    return x


class _LeafPosterior:
    """Posterior of the leaf parameters: ``N(mu_hat, Omega_inv^-1)`` with ``Omega_inv = C C^T``."""

    __slots__ = ("_chol", "_small", "_z")

    def __init__(self, precision: npt.NDArray, rhs: npt.NDArray) -> None:
        self._small = precision.shape[0] <= _MAX_SMALL_SOLVE
        if self._small:
            self._chol = _cholesky(precision)
            self._z = _solve_lower(self._chol, rhs.tolist())
        else:
            self._chol = np.linalg.cholesky(precision)
            self._z = solve_triangular(self._chol, rhs, lower=True)

    def log_det(self) -> float:
        """``log(det(Omega_inv))``."""
        if self._small:
            return 2.0 * sum(math.log(row[i]) for i, row in enumerate(self._chol))
        return 2.0 * float(np.sum(np.log(np.diag(self._chol))))

    def quadratic(self) -> float:
        """``mu_hat^T Omega_inv mu_hat``."""
        return math.fsum(v * v for v in self._z) if self._small else float(self._z @ self._z)

    def sample(self, rng: np.random.Generator) -> npt.NDArray:
        """Draw from the posterior (``rmvnorm``)."""
        noise = rng.standard_normal(len(self._z))
        if self._small:
            b = [z + e for z, e in zip(self._z, noise.tolist(), strict=True)]
            return np.asarray(_solve_upper_transposed(self._chol, b))
        return solve_triangular(self._chol.T, self._z + noise, lower=False)


class _TreeLikelihood:
    """Marginal likelihood of a tree given partial residuals (``LogLT`` and ``GetSuffStats``).

    The terms that do not depend on the tree are computed once per residual.
    """

    __slots__ = ("constant", "hypers", "unit_weights", "weighted_residual", "weights")

    def __init__(self, residual: npt.NDArray, weights: npt.NDArray, hypers: _Hypers, unit_weights: bool) -> None:
        sigma2 = hypers.sigma**2
        self.hypers = hypers
        self.weights = weights
        self.unit_weights = unit_weights
        self.weighted_residual = residual if unit_weights else residual * weights
        self.constant = (
            0.5 * float(np.sum(np.log(weights / (2.0 * math.pi) / sigma2))) * hypers.temperature
            - 0.5 * float(residual @ self.weighted_residual) / sigma2 * hypers.temperature
        )

    def posterior(self, W: npt.NDArray) -> _LeafPosterior:
        """Posterior of the leaf parameters of a tree with leaf weights ``W`` (``GetSuffStats``)."""
        h = self.hypers
        scale = h.temperature / h.sigma**2
        Wt = W.T
        precision = (Wt @ W if self.unit_weights else (Wt * self.weights) @ W) * scale
        precision.flat[:: W.shape[1] + 1] += 1.0 / h.sigma_mu**2
        rhs = (Wt @ self.weighted_residual) * scale
        return _LeafPosterior(precision, rhs)

    def loglik(self, W: npt.NDArray) -> float:
        """Log marginal likelihood of the tree with leaf weights ``W`` (``LogLT``)."""
        n_leaves = W.shape[1]
        posterior = self.posterior(W)
        return (
            self.constant
            - 0.5 * n_leaves * (_LOG_2PI + 2.0 * math.log(self.hypers.sigma_mu))
            - 0.5 * (posterior.log_det() - n_leaves * _LOG_2PI)
            + 0.5 * posterior.quadratic()
        )

    def sample_mu(self, W: npt.NDArray, rng: np.random.Generator) -> npt.NDArray:
        """Draw the leaf parameters from their full conditional (``Node::UpdateMu``)."""
        return self.posterior(W).sample(rng)


@dataclass
class _TreeState:
    """A tree with the leaf weights and log marginal likelihood of its current state."""

    tree: _Node
    W: npt.NDArray
    loglik: float


# ---------------------------------------------------------------------------
# Forest snapshots (for prediction after fitting)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForestSnapshot:
    """Compact, immutable copy of the trees of a forest.

    Nodes of all trees are stored in pre-order.

    Attributes
    ----------
    var : ndarray of int32, shape (n_nodes,)
        Splitting variable of each branch, -1 for leaves.
    val : ndarray, shape (n_nodes,)
        Cut point of each branch, leaf parameter of each leaf.
    tau : ndarray, shape (n_trees,)
        Bandwidth of each tree.
    tree_start : ndarray of int64, shape (n_trees,)
        Index of the root of each tree.

    """

    var: npt.NDArray
    val: npt.NDArray
    tau: npt.NDArray
    tree_start: npt.NDArray

    @classmethod
    def from_trees(cls, trees: list[_Node]) -> "ForestSnapshot":
        """Copy the trees."""
        var: list[int] = []
        val: list[float] = []
        starts: list[int] = []

        def visit(node: _Node) -> None:
            if node.is_leaf:
                var.append(-1)
                val.append(node.mu)
            else:
                var.append(node.var)
                val.append(node.val)
                visit(node.left)
                visit(node.right)

        for tree in trees:
            starts.append(len(var))
            visit(tree)
        return cls(
            var=np.asarray(var, dtype=np.int32),
            val=np.asarray(val, dtype=float),
            tau=np.asarray([tree.tau for tree in trees], dtype=float),
            tree_start=np.asarray(starts, dtype=np.int64),
        )

    def predict(self, X: npt.NDArray) -> npt.NDArray:
        """Predict with the forest.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Quantile-normalized predictors.

        Returns
        -------
        ndarray, shape (n_samples,)
            The sum of the trees' predictions.

        """
        out = np.zeros(X.shape[0])
        var, val = self.var, self.val

        def descend(index: int, weight: npt.NDArray | None, tau: float) -> int:
            if var[index] < 0:
                out[:] += val[index] if weight is None else weight * val[index]
                return index + 1
            go_left = expit((val[index] - X[:, var[index]]) / tau)
            left_weight = go_left if weight is None else weight * go_left
            right_weight = 1.0 - go_left if weight is None else weight * (1.0 - go_left)
            next_index = descend(index + 1, left_weight, tau)
            return descend(next_index, right_weight, tau)

        for start, tau in zip(self.tree_start, self.tau, strict=True):
            descend(int(start), None, float(tau))
        return out


# ---------------------------------------------------------------------------
# Forest
# ---------------------------------------------------------------------------


class SoftBARTForest:
    """Soft BART forest updated by Bayesian backfitting (``Forest`` in SoftBart).

    Translation of the object returned by ``MakeForest(Hypers(X, Y, ...), Opts(update_sigma = FALSE))``
    in the SoftBart R package. The error standard deviation is set from outside
    with :meth:`set_sigma`.

    Parameters
    ----------
    n_features : int
        Number of predictors. Predictors must be normalized to [0, 1]
        (e.g., with :func:`quantile_normalize_bart`).
    num_tree : int, optional (default 20)
        Number of trees.
    beta : float, optional (default 2.0)
        Power of the tree depth prior, ``gamma * (1 + depth)^(-beta)``.
    gamma : float, optional (default 0.95)
        Base of the tree depth prior.
    k : float, optional (default 2.0)
        The initial standard deviation of the leaf parameters is
        ``0.5 / (k * sqrt(num_tree))``.
    width : float, optional (default 0.1)
        Initial bandwidth of the trees.
    tau_rate : float, optional (default 10.0)
        Rate of the exponential prior on the bandwidths.
    alpha : float, optional (default 1.0)
        Initial concentration of the Dirichlet prior on the splitting probabilities.
    alpha_scale : float or None, optional (default None)
        Scale of the prior on ``alpha``; ``n_features`` if None.
    alpha_shape_1 : float, optional (default 0.5)
        First shape of the beta prior on ``alpha / (alpha + alpha_scale)``.
    alpha_shape_2 : float, optional (default 1.0)
        Second shape of the beta prior on ``alpha / (alpha + alpha_scale)``.
    temperature : float, optional (default 1.0)
        Power of the likelihood.
    update_sigma_mu : bool, optional (default True)
        Whether to update the standard deviation of the leaf parameters.
    update_s : bool, optional (default True)
        Whether to update the splitting probabilities (sparsity).
    update_alpha : bool, optional (default True)
        Whether to update the Dirichlet concentration (only together with ``update_s``).
    update_tau : bool, optional (default True)
        Whether to update the bandwidths.
    num_burn : int, optional (default 2500)
        ``num_burn`` of ``Opts()``. The splitting probabilities and ``alpha`` are
        only updated after ``floor(num_burn / 2)`` Gibbs iterations.
    random_state : numpy.random.Generator or int or None, optional (default None)
        Random number generator or seed.

    Attributes
    ----------
    num_gibbs : int
        Number of Gibbs iterations done so far.

    """

    def __init__(
        self,
        n_features: int,
        num_tree: int = 20,
        beta: float = 2.0,
        gamma: float = 0.95,
        k: float = 2.0,
        width: float = 0.1,
        tau_rate: float = 10.0,
        alpha: float = 1.0,
        alpha_scale: float | None = None,
        alpha_shape_1: float = 0.5,
        alpha_shape_2: float = 1.0,
        temperature: float = 1.0,
        update_sigma_mu: bool = True,
        update_s: bool = True,
        update_alpha: bool = True,
        update_tau: bool = True,
        num_burn: int = 2500,
        random_state: np.random.Generator | int | None = None,
    ) -> None:
        if n_features < 1:
            raise ValueError(f"n_features must be >= 1, got {n_features}")
        if num_tree < 1:
            raise ValueError(f"num_tree must be >= 1, got {num_tree}")
        self.hypers = _Hypers(
            n_features=n_features,
            num_tree=num_tree,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            k=k,
            width=width,
            tau_rate=tau_rate,
            temperature=temperature,
            alpha_scale=alpha_scale,
            alpha_shape_1=alpha_shape_1,
            alpha_shape_2=alpha_shape_2,
        )
        self.update_sigma_mu = update_sigma_mu
        self.update_s = update_s
        self.update_alpha = update_alpha
        self.update_tau = update_tau
        self.num_burn = num_burn
        self.rng = np.random.default_rng(random_state)
        self.trees = [_Node.root(width) for _ in range(num_tree)]
        self.num_gibbs = 0
        # Leaf weights and predictions of every tree at the last training data
        self._cache_X: npt.NDArray | None = None
        self._cache_W: list[npt.NDArray] = []
        self._cache_pred: list[npt.NDArray] = []

    # -- Public API ---------------------------------------------------------

    @property
    def n_features(self) -> int:
        """Number of predictors."""
        return self.hypers.n_features

    def set_sigma(self, sigma: float) -> None:
        """Set the error standard deviation (``Forest::set_sigma``)."""
        self.hypers.sigma = float(sigma)

    def get_sigma_mu(self) -> float:
        """Return the standard deviation of the leaf parameters (``Forest::get_sigma_mu``)."""
        return self.hypers.sigma_mu

    def get_s(self) -> npt.NDArray:
        """Return the splitting probabilities (``Forest::get_s``)."""
        return self.hypers.s.copy()

    def get_counts(self) -> npt.NDArray:
        """Return the number of branches splitting on each predictor (``Forest::get_counts``)."""
        counts = np.zeros(self.n_features, dtype=np.int64)
        for tree in self.trees:
            for branch in _branches(tree):
                counts[branch.var] += 1
        return counts

    def do_gibbs(
        self,
        X: npt.NDArray,
        Y: npt.NDArray,
        weights: npt.NDArray | None = None,
        num_iter: int = 1,
    ) -> npt.NDArray:
        """Run Gibbs iterations of Bayesian backfitting (``Forest::do_gibbs[_weighted]``).

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Predictors normalized to [0, 1]. The leaf weights at ``X`` are
            cached between calls with the same array object, so do not modify
            ``X`` in place between calls.
        Y : ndarray, shape (n_samples,)
            Response.
        weights : ndarray, shape (n_samples,) or None, optional (default None)
            Known precision weights of the samples (heteroskedastic errors with
            variance ``sigma^2 / weights``); all ones if None.
        num_iter : int, optional (default 1)
            Number of Gibbs iterations.

        Returns
        -------
        ndarray, shape (n_samples,)
            Prediction of the forest at ``X`` after the last iteration.

        """
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(f"X must have shape (n_samples, {self.n_features}), got {X.shape}")
        if Y.shape != (X.shape[0],):
            raise ValueError(f"Y must have shape ({X.shape[0]},), got {Y.shape}")
        weights = np.ones_like(Y) if weights is None else np.asarray(weights, dtype=float)
        self._prepare_cache(X)
        num_warmup = self.num_burn // 2
        for _ in range(num_iter):
            self._iterate_gibbs(X, Y, weights, with_s=self.update_s and self.num_gibbs > num_warmup)
            self.num_gibbs += 1
        return self._predict_cached()

    def predict(self, X: npt.NDArray) -> npt.NDArray:
        """Predict with the current trees (``Forest::do_predict``).

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Predictors normalized to [0, 1].

        Returns
        -------
        ndarray, shape (n_samples,)
            The prediction.

        """
        X = np.asarray(X, dtype=float)
        out = np.zeros(X.shape[0])
        for tree in self.trees:
            out += _leaf_weights(tree, X) @ np.array([leaf.mu for leaf in _leaves(tree)])
        return out

    def snapshot(self) -> ForestSnapshot:
        """Copy the current trees into a compact :class:`ForestSnapshot`."""
        return ForestSnapshot.from_trees(self.trees)

    # -- Gibbs sampler ------------------------------------------------------

    def _prepare_cache(self, X: npt.NDArray) -> None:
        if self._cache_X is X:
            return
        self._cache_X = X
        self._cache_W = [_leaf_weights(tree, X) for tree in self.trees]
        self._cache_pred = [
            W @ np.array([leaf.mu for leaf in _leaves(tree)]) for W, tree in zip(self._cache_W, self.trees, strict=True)
        ]

    def _predict_cached(self) -> npt.NDArray:
        return np.sum(self._cache_pred, axis=0)

    def _iterate_gibbs(self, X: npt.NDArray, Y: npt.NDArray, weights: npt.NDArray, with_s: bool) -> None:
        """One Gibbs iteration (``IterateGibbsNoS`` / ``IterateGibbsWithS``)."""
        self._tree_backfit(X, Y, weights)
        hypers = self.hypers
        if self.update_sigma_mu:
            means = np.array([leaf.mu for tree in self.trees for leaf in _leaves(tree)])
            hypers.update_sigma_mu(means, self.rng)
        if with_s:
            if self.update_s:
                hypers.update_s(self.get_counts(), self.rng)
            if self.update_alpha:
                hypers.update_alpha(self.rng)

    def _tree_backfit(self, X: npt.NDArray, Y: npt.NDArray, weights: npt.NDArray) -> None:
        """Update every tree given the others (``TreeBackfit``)."""
        rng = self.rng
        hypers = self.hypers
        unit_weights = bool(np.all(weights == 1.0))
        Y_hat = self._predict_cached()
        for t in range(hypers.num_tree):
            Y_star = Y_hat - self._cache_pred[t]
            likelihood = _TreeLikelihood(Y - Y_star, weights, hypers, unit_weights)
            W = self._cache_W[t]
            state = _TreeState(self.trees[t], W, likelihood.loglik(W))

            if rng.random() < _MH_PRIOR:
                self._draw_prior(state, X, likelihood)
            if state.tree.is_leaf or rng.random() < _MH_BD:
                self._birth_death(state, X, likelihood)
            else:
                self._perturb_decision_rule(state, X, likelihood)
            if self.update_tau:
                self._update_tau(state, X, likelihood)

            leaves = _leaves(state.tree)
            mu = likelihood.sample_mu(state.W, rng)
            for leaf, value in zip(leaves, mu, strict=True):
                leaf.mu = float(value)

            self.trees[t] = state.tree
            self._cache_W[t] = state.W
            self._cache_pred[t] = state.W @ mu
            Y_hat = Y_star + self._cache_pred[t]

    def _draw_prior(self, state: _TreeState, X: npt.NDArray, likelihood: _TreeLikelihood) -> None:
        """Propose a whole new tree from the prior (``draw_prior``)."""
        new_tree = _Node.root(self.hypers.width)
        new_tree.gen_below(self.hypers, self.rng)
        W = _leaf_weights(new_tree, X)
        loglik = likelihood.loglik(W)
        if _log_uniform(self.rng) < loglik - state.loglik:
            state.tree, state.W, state.loglik = new_tree, W, loglik

    def _birth_death(self, state: _TreeState, X: npt.NDArray, likelihood: _TreeLikelihood) -> None:
        """Grow or prune the tree (``birth_death``)."""
        if self.rng.random() < _probability_node_birth(state.tree):
            self._node_birth(state, X, likelihood)
        else:
            self._node_death(state, X, likelihood)

    def _node_birth(self, state: _TreeState, X: npt.NDArray, likelihood: _TreeLikelihood) -> None:
        """Split a leaf (``node_birth``)."""
        hypers = self.hypers
        tree = state.tree
        leaves = _leaves(tree)
        leaf = leaves[_rand_index(len(leaves), self.rng)]
        leaf_probability = 1.0 / len(leaves)

        leaf_depth = _depth(leaf)
        leaf_prior = hypers.growth_prior(leaf_depth)
        ll_before = state.loglik + math.log(1.0 - leaf_prior)
        p_forward = math.log(_probability_node_birth(tree) * leaf_probability)

        leaf.birth_leaves(hypers, self.rng)

        W = _leaf_weights(tree, X)
        loglik = likelihood.loglik(W)
        ll_after = loglik + math.log(leaf_prior) + 2.0 * math.log(1.0 - hypers.growth_prior(leaf_depth + 1))
        p_not_grand = 1.0 / len(_not_grand_branches(tree))
        p_backward = math.log((1.0 - _probability_node_birth(tree)) * p_not_grand)

        log_trans_prob = ll_after + p_backward - ll_before - p_forward
        if _log_uniform(self.rng) > log_trans_prob:
            leaf.delete_leaves()
            leaf.var = 0
        else:
            state.W, state.loglik = W, loglik

    def _node_death(self, state: _TreeState, X: npt.NDArray, likelihood: _TreeLikelihood) -> None:
        """Prune a branch whose children are leaves (``node_death``)."""
        hypers = self.hypers
        tree = state.tree
        candidates = _not_grand_branches(tree)
        branch = candidates[_rand_index(len(candidates), self.rng)]
        p_not_grand = 1.0 / len(candidates)

        leaf_depth = _depth(branch.left)
        leaf_prob = hypers.growth_prior(leaf_depth - 1)
        child_prior = hypers.growth_prior(leaf_depth)
        ll_before = state.loglik + 2.0 * math.log(1.0 - child_prior) + math.log(leaf_prob)
        p_forward = math.log(p_not_grand * (1.0 - _probability_node_birth(tree)))

        left, right = branch.left, branch.right
        branch.left = branch.right = None
        branch.is_leaf = True

        W = _leaf_weights(tree, X)
        loglik = likelihood.loglik(W)
        ll_after = loglik + math.log(1.0 - leaf_prob)
        p_backward = math.log(1.0 / len(_leaves(tree)) * _probability_node_birth(tree))

        log_trans_prob = ll_after + p_backward - ll_before - p_forward
        if _log_uniform(self.rng) > log_trans_prob:
            branch.left, branch.right = left, right
            branch.is_leaf = False
        else:
            state.W, state.loglik = W, loglik

    def _perturb_decision_rule(self, state: _TreeState, X: npt.NDArray, likelihood: _TreeLikelihood) -> None:
        """Change the variable and cut point of a branch (``perturb_decision_rule``)."""
        tree = state.tree
        branches = _branches(tree)
        if not branches:
            return
        branch = branches[_rand_index(len(branches), self.rng)]

        cutpoint_likelihood = _calc_cutpoint_likelihood(tree)
        lower, upper = _get_perturb_limits(branch)
        backward_trans = _reciprocal(upper - lower)

        old = (branch.var, branch.val, branch.lower, branch.upper)

        branch.var = self.hypers.sample_var(self.rng)
        lower, upper = _get_perturb_limits(branch)
        branch.val = lower + (upper - lower) * self.rng.random()
        _get_limits_below(branch)

        W = _leaf_weights(tree, X)
        loglik = likelihood.loglik(W)
        cutpoint_likelihood_after = _calc_cutpoint_likelihood(tree)
        forward_trans = _reciprocal(upper - lower)

        log_trans_prob = (
            loglik
            + _log(cutpoint_likelihood_after)
            + _log(backward_trans)
            - state.loglik
            - _log(cutpoint_likelihood)
            - _log(forward_trans)
        )
        # A NaN ratio is accepted, as in SoftBart
        if _log_uniform(self.rng) > log_trans_prob:
            branch.var, branch.val, branch.lower, branch.upper = old
            _get_limits_below(branch)
        else:
            state.W, state.loglik = W, loglik

    def _update_tau(self, state: _TreeState, X: npt.NDArray, likelihood: _TreeLikelihood) -> None:
        """Metropolis-Hastings update of the bandwidth of the tree (``Node::UpdateTau``)."""
        tree = state.tree
        tau_rate = self.hypers.tau_rate
        tau_old = tree.tau
        tau_new = 5.0 ** (2.0 * self.rng.random() - 1.0) * tau_old

        tree.set_tau(tau_new)
        W = _leaf_weights(tree, X)
        loglik = likelihood.loglik(W)

        # Exponential prior with rate tau_rate; the proposal is symmetric on the log scale
        loglik_new = loglik + math.log(tau_rate) - tau_rate * tau_new
        loglik_old = state.loglik + math.log(tau_rate) - tau_rate * tau_old
        new_to_old = -math.log(tau_old)
        old_to_new = -math.log(tau_new)

        if _log_uniform(self.rng) < loglik_new + new_to_old - loglik_old - old_to_new:
            state.W, state.loglik = W, loglik
        else:
            tree.set_tau(tau_old)
