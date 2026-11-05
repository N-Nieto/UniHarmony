"""Conformance of NeuroComBat with the reference implementations and with the ComBat model.

UniHarmony differs from R ``sva::ComBat`` (3.50.0) and ``neuroCombat`` (0.2.12)
in three documented places:

1. Pooled variance: UniHarmony divides the residual sum of squares by ``n - S``
   (``S`` sites) instead of ``n``. Outside ``mean_only`` ComBat is
   scale-equivariant in the pooled variance, so the harmonized deviations from
   the standardized mean are those of the references times ``sqrt(n / (n - S))``.
2. ``mean_only``: the posterior mean of the site effect uses the site sample
   size ``n_i`` instead of ``n = 1``.
3. Stopping rule of the parametric empirical Bayes solver: the relative change
   is divided by ``|previous estimate|``. The references divide by the signed
   estimate, so features with a negative location effect never block the stop
   and their output is not converged to the stated tolerance (1e-4).

Everything else must reproduce the reference. These tests guard it for any
future change: :class:`_ReferenceChoices` puts the reference choices 1 and 2
back, and its output must equal the frozen neuroCombat output in every mode and
design, up to floating-point rounding. For the parametric mode the comparison
uses both solvers run to convergence (``eb_parametric_converged``), so it does
not depend on the stopping rule; the default output is checked separately.

``neurocombat_0.2.12_reference.npz`` is built by
``make_neurocombat_reference.py`` (see its docstring): six designs (one
categorical and one continuous covariate, several covariates, no covariates,
imbalanced sites, covariates on a 1e4 scale, a covariate confounded with site)
and the neuroCombat 0.2.12 output of each in four modes (plus the converged
parametric output). Its inputs are exact in
float32, the precision neuroCombat converts them to, so the tolerance can be
tight (``RTOL``, relative to the data standard deviation). The frozen
outputs (default modes) also match R ``sva::ComBat`` 3.50.0 to 1e-13. The two
implementations solve the same equations with different but equivalent
linear algebra; their rounding differences are below 1e-10 of the data scale.

``neurocombat_0.2.12_ref_batch_reference.npz`` (same script) holds the
neuroCombat output with a reference batch (``ref_batch``, UniHarmony's
``ref_site``) for three of the designs. neuroCombat divides the reference
residual sum of squares by ``n_ref`` and UniHarmony by ``n_ref - 1``; with
that choice restored the outputs are equal, and the reference site is left
unchanged by both.
"""

from pathlib import Path

import numpy as np
import pytest

from uniharmony.combat import NeuroComBat


REFERENCE = Path(__file__).parent / "neurocombat_0.2.12_reference.npz"
REF_BATCH_REFERENCE = Path(__file__).parent / "neurocombat_0.2.12_ref_batch_reference.npz"
REF_BATCH_DESIGNS = ["one_categorical_one_continuous", "several_covariates", "imbalanced_sites"]

DESIGNS = [
    "one_categorical_one_continuous",
    "several_covariates",
    "no_covariates",
    "imbalanced_sites",
    "large_scale_covariates",
    "covariate_confounded_with_site",
]

#: UniHarmony options of each frozen neuroCombat output compared exactly
MODES = {
    "eb_parametric_converged": {},
    "eb_non_parametric": {"parametric_adjustments": False},
    "no_eb": {"empirical_bayes": False},
    "mean_only": {"mean_only": True},
}

#: Tolerance of the iterative solver when it is run to convergence
CONVERGED = 1e-13

#: Largest allowed difference with the reference, relative to the standard deviation of the data
RTOL = 1e-8


class _Converged(NeuroComBat):
    """NeuroComBat with the parametric empirical Bayes solver run to convergence."""

    def _iteration_solver(self, *args, **kwargs):
        return super()._iteration_solver(*args, **{**kwargs, "convergence": CONVERGED})


class _ReferenceChoices(_Converged):
    """Converged NeuroComBat with the R sva / neuroCombat choices 1 and 2 restored.

    The pooled variance is divided by the number of samples it is computed on
    (``n``, or ``n_ref`` with a reference site) instead of the degrees of
    freedom (``n - S``, ``n_ref - S_ref``), and mean_only uses n = 1.
    """

    def fit_standardize(self, X, design, n_samples, n_samples_per_site, *args, **kwargs):
        super().fit_standardize(X, design, n_samples, n_samples_per_site, *args, **kwargs)
        self._var_pooled = self._var_pooled * self._var_pooled_dof / self._var_pooled_n
        return self.transform_standardize(X=X, design=design, n_samples=n_samples)[0]

    def _postmean(self, gamma_hat, gamma_bar, n, delta_star, tau_2):
        return super()._postmean(gamma_hat, gamma_bar, 1 if self.mean_only else n, delta_star, tau_2)


@pytest.fixture(scope="module")
def reference() -> dict[str, dict[str, np.ndarray]]:
    """Frozen neuroCombat 0.2.12 output, by design."""
    designs: dict[str, dict[str, np.ndarray]] = {}
    with np.load(REFERENCE) as data:
        for key in data.files:
            design, name = key.split("__")
            designs.setdefault(design, {})[name] = data[key]
    assert sorted(designs) == sorted(DESIGNS)
    return designs


@pytest.fixture(scope="module")
def ref_batch_reference() -> dict[str, dict[str, np.ndarray]]:
    """Frozen neuroCombat 0.2.12 output with a reference batch, by design."""
    designs: dict[str, dict[str, np.ndarray]] = {}
    with np.load(REF_BATCH_REFERENCE) as data:
        for key in data.files:
            design, name = key.split("__")
            designs.setdefault(design, {})[name] = data[key]
    assert sorted(designs) == sorted(REF_BATCH_DESIGNS)
    return designs


def _harmonize(design: dict[str, np.ndarray], estimator: type = NeuroComBat, **params) -> np.ndarray:
    return estimator(**params).fit_transform(
        design["X"],
        sites=design["sites"],
        categorical_covariates=design.get("categorical"),
        continuous_covariates=design.get("continuous"),
    )


def _assert_close(actual: np.ndarray, expected: np.ndarray) -> None:
    np.testing.assert_allclose(actual, expected, rtol=0, atol=RTOL * np.std(expected))


@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize("design_name", DESIGNS)
def test_reproduces_reference_with_reference_choices(reference, design_name: str, mode: str) -> None:
    """With the two documented choices restored, the output equals neuroCombat's in every mode and design."""
    design = reference[design_name]
    _assert_close(_harmonize(design, _ReferenceChoices, **MODES[mode]), design[mode])


@pytest.mark.parametrize("mode", ["eb_parametric_converged", "eb_non_parametric", "no_eb"])
@pytest.mark.parametrize("design_name", DESIGNS)
def test_matches_reference_up_to_variance_correction(reference, design_name: str, mode: str) -> None:
    """Outside mean_only, the only other difference with neuroCombat is the factor sqrt(n / (n - S))."""
    design = reference[design_name]
    n, n_sites = len(design["sites"]), len(np.unique(design["sites"]))
    expected = design["mean"] + (design[mode] - design["mean"]) * np.sqrt(n / (n - n_sites))
    _assert_close(_harmonize(design, _Converged, **MODES[mode]), expected)


@pytest.mark.parametrize("design_name", DESIGNS)
def test_default_parametric_stopping_rule(reference, design_name: str) -> None:
    """With default settings, UniHarmony stops at least as close to the converged solution as neuroCombat.

    The difference with neuroCombat's default output comes only from where each
    solver stops, so it is bounded by the solver tolerance (1e-4).
    """
    design = reference[design_name]
    converged = _harmonize(design, _ReferenceChoices)
    default = _harmonize(design, type("_Default", (_ReferenceChoices,), {"_iteration_solver": NeuroComBat._iteration_solver}))
    scale = np.std(design["eb_parametric_converged"])
    error_uniharmony = np.abs(default - converged).max() / scale
    error_reference = np.abs(design["eb_parametric"] - design["eb_parametric_converged"]).max() / scale
    assert error_uniharmony <= error_reference + RTOL
    assert error_reference < 1e-4


@pytest.mark.parametrize("design_name", DESIGNS)
def test_mean_only_differs_from_reference(reference, design_name: str) -> None:
    """mean_only deliberately differs from neuroCombat (posterior with n_i): the correction is active."""
    design = reference[design_name]
    difference = np.abs(_harmonize(design, mean_only=True) - design["mean_only"]).max()
    assert difference > 1e3 * RTOL * np.std(design["mean_only"])


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
        NeuroComBat(empirical_bayes=False).fit_transform(X, sites=sites), _closed_form(X, sites, None), atol=1e-10
    )


def test_mean_only_uses_site_sample_size(small_data) -> None:
    """mean_only shrinks with n_i, not with n = 1 as R sva / neuroCombat do."""
    X, sites = small_data
    harmonized = NeuroComBat(mean_only=True).fit_transform(X, sites=sites)
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
            NeuroComBat(empirical_bayes=False).fit_transform(X, sites=sites), _closed_form(X, sites, None), atol=1e-10
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
        out = NeuroComBat(empirical_bayes=False).fit_transform(X, sites=sites)
        residuals = out - np.array([out[sites == s].mean(axis=0) for s in range(3)])[sites]
        ratios.append(((residuals**2).sum(axis=0) / (15 - 3) / true_var).mean())
    assert abs(np.mean(ratios) - 1) < 0.03
    # The reference denominator n would give (n - S) / n = 0.8 here


# ---------------------------------------------------------------------------
# Reference site (neuroCombat ref_batch)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize("design_name", REF_BATCH_DESIGNS)
def test_ref_site_reproduces_reference_with_reference_choices(ref_batch_reference, design_name: str, mode: str) -> None:
    """With a reference site and the documented choices restored, the output equals neuroCombat's ref_batch output."""
    design = ref_batch_reference[design_name]
    ref_site = str(design["ref_site"])
    _assert_close(_harmonize(design, _ReferenceChoices, ref_site=ref_site, **MODES[mode]), design[mode])


@pytest.mark.parametrize("mode", ["eb_parametric_converged", "eb_non_parametric", "no_eb", "mean_only"])
@pytest.mark.parametrize("design_name", REF_BATCH_DESIGNS)
def test_ref_site_matches_reference_up_to_variance_correction(ref_batch_reference, design_name: str, mode: str) -> None:
    """The reference site is unchanged; the other sites differ from neuroCombat by the factor sqrt(n_ref / (n_ref - 1)).

    In mean_only the scale is fixed, so the rescaling also changes the
    empirical Bayes shrinkage: only the reference site is compared there.
    """
    design = ref_batch_reference[design_name]
    ref_site = str(design["ref_site"])
    in_ref = design["sites"] == ref_site
    harmonized = _harmonize(design, _Converged, ref_site=ref_site, **MODES[mode])
    np.testing.assert_array_equal(harmonized[in_ref], design["X"][in_ref])
    np.testing.assert_array_equal(design[mode][in_ref], design["X"][in_ref])
    if mode == "mean_only":
        return
    n_ref = in_ref.sum()
    expected = design["mean"] + (design[mode] - design["mean"]) * np.sqrt(n_ref / (n_ref - 1))
    _assert_close(harmonized[~in_ref], expected[~in_ref])
