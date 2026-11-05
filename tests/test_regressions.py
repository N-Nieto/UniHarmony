"""Regression tests of fixed bugs (one test, or a few, per bug).

Each test names the bug it guards. The bugs were found while implementing the
pipeline API (0.0.6) and the reference site; see the audit notes.
"""

import copy
import warnings
from pathlib import Path

import numpy as np
import pytest

from uniharmony import verbosity_context
from uniharmony.combat import ComBatGAM, CovBat, NeuroComBat
from uniharmony.interpolation import InterSiteMatchedInterpolation


@pytest.fixture(autouse=True)
def _quiet():
    """Silence the logs and warnings of the estimators."""
    with verbosity_context("error"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.fixture
def data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three sites with location and scale effects and an age effect."""
    rng = np.random.default_rng(0)
    sites = np.repeat(np.array(["a", "b", "c"]), [40, 50, 60])
    shift = np.array([0.0, 2.0, -1.0])[np.searchsorted(["a", "b", "c"], sites)][:, None]
    scale = np.array([1.0, 1.7, 0.6])[np.searchsorted(["a", "b", "c"], sites)][:, None]
    age = rng.uniform(20, 80, len(sites))[:, None]
    X = shift + scale * rng.normal(size=(len(sites), 5)) + 0.02 * age
    return X, sites, age


def _state(estimator) -> dict:
    """Deep copy of the fitted state, the inner models included."""
    return copy.deepcopy(estimator.__dict__)


def _assert_same_state(before: dict, after: dict, path: str = "") -> None:
    assert before.keys() == after.keys(), f"{path}: attributes added or removed: {set(before) ^ set(after)}"
    for name, value in before.items():
        other = after[name]
        where = f"{path}.{name}"
        if hasattr(value, "__dict__") and not isinstance(value, np.ndarray) and hasattr(other, "__dict__"):
            _assert_same_state(value.__dict__, other.__dict__, where)
        elif isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, other, err_msg=where)


# ---------------------------------------------------------------------------
# transform must not modify the fitted state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make",
    [
        lambda: NeuroComBat(),
        lambda: NeuroComBat(ref_site="b"),
        lambda: ComBatGAM(random_state=0),
        lambda: CovBat(),
        lambda: CovBat(ref_site=["a", "b"]),
    ],
    ids=["NeuroComBat", "NeuroComBat-ref", "ComBatGAM", "CovBat", "CovBat-ref"],
)
def test_transform_does_not_modify_the_fitted_state(data, make) -> None:
    """Bug: transform overwrote the pooled variance (ComBat) and the inner standardized mean (CovBat).

    The fitted state, the inner models of CovBat included, must be identical
    before and after transforming other data.
    """
    X, sites, age = data
    model = make()
    covariates = {"smooth_covariates": age} if isinstance(model, ComBatGAM) else {"continuous_covariates": age}
    model.fit(X, sites=sites, **covariates)
    before = _state(model)
    model.transform(X[::3] + 1.0, sites=sites[::3], **{k: v[::3] for k, v in covariates.items()})
    _assert_same_state(before, _state(model))


def test_covbat_transform_is_reproducible(data) -> None:
    """Bug (CovBat): transforming a subset changed the mean used for the next transform."""
    X, sites, _ = data
    model = CovBat().fit(X, sites=sites)
    first = model.transform(X, sites=sites)
    model.transform(X[:7], sites=sites[:7])
    np.testing.assert_array_equal(model.transform(X, sites=sites), first)


# ---------------------------------------------------------------------------
# CovBat
# ---------------------------------------------------------------------------


def test_covbat_sets_sites(data) -> None:
    """Bug: CovBat documented ``sites_`` but never set it."""
    X, sites, _ = data
    model = CovBat().fit(X, sites=sites)
    np.testing.assert_array_equal(model.sites_, ["a", "b", "c"])
    np.testing.assert_array_equal(model.ref_sites_, ["a", "b", "c"])


def test_covbat_solver_options_reach_both_combat_steps(data) -> None:
    """Bug: CovBat's first ComBat step ignored var_epsilon, tau_2_epsilon and max_iter."""
    X, sites, _ = data
    options = {"var_epsilon": 1e-6, "delta_epsilon": 1e-7, "tau_2_epsilon": 1e-9, "max_iter": 7}
    model = CovBat(**options).fit(X, sites=sites)
    for step in (model._first_combat, model._second_combat):
        params = step.get_params()
        assert {name: params[name] for name in options} == options


def test_covbat_mean_only(data) -> None:
    """Bug: CovBat could not adjust only the mean in its first ComBat step (``mean.only`` of the original code)."""
    X, sites, _ = data
    default = CovBat().fit(X, sites=sites)
    mean_only = CovBat(mean_only=True).fit(X, sites=sites)
    assert mean_only._first_combat.mean_only
    assert not default._first_combat.mean_only
    assert not np.allclose(mean_only.transform(X, sites=sites), default.transform(X, sites=sites))
    np.testing.assert_array_equal(mean_only._first_combat._delta_star, 1.0)


# ---------------------------------------------------------------------------
# NeuroComBat / ComBatGAM transform
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("estimator", [NeuroComBat, ComBatGAM])
def test_transform_uses_delta_epsilon(data, estimator, monkeypatch) -> None:
    """Bug: transform ignored delta_epsilon (the lower bound of the site scales) and used the default."""
    X, sites, age = data
    covariates = {"smooth_covariates": age} if estimator is ComBatGAM else {}
    model = estimator(delta_epsilon=1e-3).fit(X, sites=sites, **covariates)
    seen = {}
    original = type(model).harmonize

    def spy(self, *args, **kwargs):
        seen["epsilon"] = kwargs.get("epsilon")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(model), "harmonize", spy)
    model.transform(X, sites=sites, **covariates)
    assert seen["epsilon"] == 1e-3


# ---------------------------------------------------------------------------
# InterSiteMatchedInterpolation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("k", "mode", "resolved"),
    [("MAX", "pairwise", "max"), ("Average", "pairwise", "average"), (2.0, "pairwise", 2), (1, "base_to_others", "average")],
)
def test_ismi_does_not_modify_k(k, mode: str, resolved) -> None:
    """Bug: fit_resample overwrote the parameter ``k`` (lower-cased, cast, or forced to "average")."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(90, 3))
    y = np.tile([0, 1, 1], 30)
    sites = np.repeat(["a", "b", "c"], 30)
    ismi = InterSiteMatchedInterpolation(k=k, mode=mode, random_state=0)
    ismi.fit_resample(X, y, sites=sites)
    assert ismi.k == k
    assert ismi.get_params()["k"] == k
    assert ismi.k_ == resolved


# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------

DOCS = Path(__file__).parents[1] / "docs"


@pytest.mark.skipif(not DOCS.is_dir(), reason="documentation sources not available")
def test_utilities_toctree_lists_existing_pages() -> None:
    """Bug: the utilities toctree listed a page that does not exist (load_onharmony_dataset)."""
    lines = (DOCS / "utilities" / "index.md").read_text().splitlines()
    start = lines.index("```{toctree}") + 1
    for entry in lines[start : lines.index("```", start)]:
        if entry.strip():
            assert (DOCS / "utilities" / f"{entry.strip()}.md").is_file(), entry


@pytest.mark.skipif(not DOCS.is_dir(), reason="documentation sources not available")
@pytest.mark.parametrize("page", ["methods/combat_based/combat.md", "methods/ot_based/otda.md"])
def test_footnotes_are_defined(page: str) -> None:
    """Bug: footnotes cited without a definition ([^6] in combat.md, [^1] in otda.md), unclosed code block."""
    import re

    text = (DOCS / page).read_text()
    cited = set(re.findall(r"\[\^(\w+)\](?!:)", text))
    defined = set(re.findall(r"^\[\^(\w+)\]:", text, flags=re.MULTILINE))
    assert cited <= defined, cited - defined
    # Bug: an unclosed code block (otda.md) swallowed the rest of the page
    assert text.count("```") % 2 == 0


def test_docstring_types_do_not_reference_attributes() -> None:
    """Bug: ``shape (n_features_in_,)`` in a docstring type was rendered as a broken RST reference."""
    from uniharmony.iqm import BARTharm
    from uniharmony.ot import OptimalTransportDomainAdaptation

    for cls in (NeuroComBat, CovBat, ComBatGAM, OptimalTransportDomainAdaptation, BARTharm):
        doc = cls.__doc__
        assert "(n_features_in_,)" not in doc, cls.__name__
        assert "from ot_obj_)" not in doc, cls.__name__
