"""Conformance of NeuroComBat with the reference implementations and with the ComBat model.

UniHarmony differs from R ``sva::ComBat`` (3.50.0) and ``neuroCombat`` (0.2.12)
in exactly two documented places:

1. Pooled variance: UniHarmony divides the residual sum of squares by ``n - S``
   (``S`` sites) instead of ``n``. Outside ``mean_only`` ComBat is
   scale-equivariant in the pooled variance, so the harmonized deviations from
   the standardized mean are those of the references times ``sqrt(n / (n - S))``.
2. ``mean_only``: the posterior mean of the site effect uses the site sample
   size ``n_i`` instead of ``n = 1``.

``neurocombat_0.2.12_reference.npz`` holds a small simulated data set (3 sites
of 6, 8 and 10 subjects, 8 features, age and sex covariates) and the output of
``neuroCombat.neuroCombat`` 0.2.12 for each mode, together with its
standardized mean (``stand.mean + mod.mean``). The outputs match R
``sva::ComBat`` 3.50.0 to 4e-8.
"""

from pathlib import Path

import numpy as np
import pytest

from uniharmony.combat import NeuroComBat


REFERENCE = Path(__file__).parent / "neurocombat_0.2.12_reference.npz"


@pytest.fixture(scope="module")
def reference() -> dict[str, np.ndarray]:
    """Frozen neuroCombat 0.2.12 output."""
    with np.load(REFERENCE) as data:
        return dict(data)


def _harmonize(reference: dict[str, np.ndarray], **params) -> np.ndarray:
    return NeuroComBat(**params).fit_transform(
        reference["X"],
        reference["sites"],
        categorical_covariates=reference["sex"][:, None],
        continuous_covariates=reference["age"][:, None],
    )


@pytest.mark.parametrize(
    ("mode", "params"),
    [
        ("eb_parametric", {}),
        ("eb_non_parametric", {"parametric_adjustments": False}),
        ("no_eb", {"empirical_bayes": False}),
    ],
)
def test_matches_reference_up_to_variance_correction(reference, mode: str, params: dict) -> None:
    """Outside mean_only, the only difference with neuroCombat is the factor sqrt(n / (n - S))."""
    n, n_sites = len(reference["sites"]), len(np.unique(reference["sites"]))
    expected = reference["mean"] + (reference[mode] - reference["mean"]) * np.sqrt(n / (n - n_sites))
    np.testing.assert_allclose(_harmonize(reference, **params), expected, rtol=0, atol=1e-6)


def _closed_form(X: np.ndarray, sites: np.ndarray, mean_only_n: str | None) -> np.ndarray:
    """ComBat without covariates, written from the model equations.

    ``mean_only_n`` is None (no empirical Bayes, location and scale), ``"n_i"``
    or ``"1"`` (mean_only with empirical Bayes and the given n in the posterior).
    """
    labels = np.unique(sites)
    n, n_sites = len(sites), len(labels)
    site_means = np.array([X[sites == s].mean(axis=0) for s in labels])
    n_i = np.array([(sites == s).sum() for s in labels])
    grand_mean = (n_i / n) @ site_means
    residuals = X - site_means[np.searchsorted(labels, sites)]
    var_pooled = (residuals**2).sum(axis=0) / (n - n_sites)
    z = (X - grand_mean) / np.sqrt(var_pooled)
    out = np.empty_like(X)
    for k, s in enumerate(labels):
        z_s = z[sites == s]
        gamma_hat = z_s.mean(axis=0)
        if mean_only_n is None:
            delta_hat = z_s.var(axis=0, ddof=1)
            z_adj = (z_s - gamma_hat) / np.sqrt(delta_hat)
        else:
            gamma_bar, tau_2 = gamma_hat.mean(), gamma_hat.var(ddof=1)
            n_post = n_i[k] if mean_only_n == "n_i" else 1
            gamma_star = (tau_2 * n_post * gamma_hat + gamma_bar) / (tau_2 * n_post + 1)
            z_adj = z_s - gamma_star
        out[sites == s] = z_adj * np.sqrt(var_pooled) + grand_mean
    return out


@pytest.fixture
def small_data() -> tuple[np.ndarray, np.ndarray]:
    """Fewer than 30 subjects, where the removed n - 1 switch used to apply."""
    rng = np.random.default_rng(0)
    sites = np.repeat(np.array(["a", "b", "c"]), [5, 8, 12])
    shift = np.array([0.0, 1.0, -0.7])[np.searchsorted(["a", "b", "c"], sites)]
    X = rng.normal(size=(25, 20)) + shift[:, None] + rng.normal(0, 0.4, size=(3, 20))[np.searchsorted(["a", "b", "c"], sites)]
    return X, sites


def test_no_eb_closed_form(small_data) -> None:
    """Without empirical Bayes the output follows the model equations with var_pooled = RSS / (n - S)."""
    X, sites = small_data
    np.testing.assert_allclose(
        NeuroComBat(empirical_bayes=False).fit_transform(X, sites), _closed_form(X, sites, None), atol=1e-10
    )


def test_mean_only_uses_site_sample_size(small_data) -> None:
    """mean_only shrinks with n_i, not with n = 1 as R sva / neuroCombat do."""
    X, sites = small_data
    harmonized = NeuroComBat(mean_only=True).fit_transform(X, sites)
    np.testing.assert_allclose(harmonized, _closed_form(X, sites, "n_i"), atol=1e-10)
    # The test is discriminative: the reference rule gives a clearly different result
    assert np.abs(harmonized - _closed_form(X, sites, "1")).max() > 0.1


def test_pooled_variance_has_no_switch_at_30() -> None:
    """The pooled-variance denominator is n - S for every n (no discontinuity at n = 30)."""
    rng = np.random.default_rng(1)
    for n_per_site in (9, 10, 11):  # n = 27, 30, 33
        sites = np.repeat([0, 1, 2], n_per_site)
        X = rng.normal(size=(len(sites), 4)) + sites[:, None]
        np.testing.assert_allclose(
            NeuroComBat(empirical_bayes=False).fit_transform(X, sites), _closed_form(X, sites, None), atol=1e-10
        )


def test_harmonized_variance_is_unbiased() -> None:
    """With var_pooled = RSS / (n - S), the harmonized pooled variance matches the true one on average."""
    rng = np.random.default_rng(2)
    ratios = []
    for _ in range(200):
        sites = np.repeat([0, 1, 2], 5)
        scale = np.array([0.7, 1.0, 1.4])[sites][:, None]
        X = rng.normal(size=(15, 10)) * scale + sites[:, None]
        true_var = np.mean([0.7**2, 1.0**2, 1.4**2])
        out = NeuroComBat(empirical_bayes=False).fit_transform(X, sites)
        residuals = out - np.array([out[sites == s].mean(axis=0) for s in range(3)])[sites]
        ratios.append(((residuals**2).sum(axis=0) / (15 - 3) / true_var).mean())
    assert abs(np.mean(ratios) - 1) < 0.03
    # The reference denominator n would give (n - S) / n = 0.8 here
