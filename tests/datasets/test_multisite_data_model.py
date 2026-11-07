"""The generative model of ``make_multisite_classification``.

* Covariate effects: the same in every site, added before the site effect,
  so covariates whose distribution differs between sites confound the sites.
* Ground truth: every component of ``X`` is returned, exactly.
* Units: all strengths are in units of the pooled within-class SD of each
  feature; location offsets are centred.
* Site effects: types and their combinations, ComBat's generative model,
  affected features, heterogeneity, site-by-class interaction.
* Outputs (``as_frame``, ``site_names``) and input validation.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from structlog.testing import capture_logs

from uniharmony import verbosity_context
from uniharmony.datasets import Covariate, CovariateSiteDistribution, make_multisite_classification


@pytest.fixture(autouse=True)
def _quiet():
    with verbosity_context("error"):
        yield


def _age(locs=(30.0, 70.0), **kwargs) -> Covariate:
    return Covariate(name="age", site_distributions=[CovariateSiteDistribution(loc=loc, scale=5.0) for loc in locs], **kwargs)


def _simulate(covariates=None, **kwargs):
    """X, y, sites and the ground truth; no site effect and no noise unless given."""
    params = {"n_sites": 2, "n_samples": 2000, "site_effect": {}, "noise_strength": 0.0, **kwargs}
    return make_multisite_classification(covariates=covariates, return_ground_truth=True, **params)


def _strength(site_effect_type, strength):
    """Return the strength of a site effect type; dropout takes a dict (here: 0 with probability 0.1 s)."""
    return {"value": 0.0, "prob": min(0.1 * strength, 1.0)} if site_effect_type == "dropout" else strength


def _within_class_sd(X, y):
    residuals = np.vstack([X[y == c] - X[y == c].mean(axis=0) for c in np.unique(y)])
    return np.sqrt((residuals**2).sum(axis=0) / (len(X) - len(np.unique(y))))


# ---------------------------------------------------------------------------
# Covariate effects
# ---------------------------------------------------------------------------


def test_covariate_effect_is_the_same_in_every_site_and_confounds_the_sites():
    """An age effect has the same slope in every site, so older sites have shifted features."""
    X, _, sites, truth = _simulate([_age(effect=0.5)])
    age = truth["covariates"]["age"]
    np.testing.assert_allclose(X, truth["signal"] + truth["covariate_effect"], atol=1e-12)
    slopes = [np.polyfit(age[sites == s], truth["covariate_effect"][sites == s], 1)[0] for s in (0, 1)]
    np.testing.assert_allclose(slopes[0], slopes[1], rtol=1e-9)  # exactly linear, one slope for all sites
    # slope per year = coefficient * reference SD / pooled SD of age
    np.testing.assert_allclose(slopes[0], truth["covariate_coefficients"]["age"] * truth["feature_scale"] / age.std(), rtol=1e-9)
    # the 40-year gap moves every feature by slope * 40 (the signal does not differ between sites)
    gap = X[sites == 1].mean(axis=0) - X[sites == 0].mean(axis=0)
    expected = slopes[0] * (age[sites == 1].mean() - age[sites == 0].mean())
    np.testing.assert_allclose(gap, expected, atol=0.15 * truth["feature_scale"].max())
    # z-gap = 40 / pooled SD of age (~20.6); coefficients 0.5 x U(0.5, 1.5) -> gap of 0.49-1.46 reference SD
    assert (expected / truth["feature_scale"] > 0.45).all()


@pytest.mark.parametrize("shape", ["linear", "quadratic", "sigmoid", np.log])
def test_continuous_effect_sd_is_the_coefficient(shape):
    """For every shape, the SD of a covariate's contribution is its coefficient (reference-SD units)."""
    *_, truth = _simulate([_age(locs=(30.0, 50.0), effect=0.7, effect_shape=shape)])
    contribution = truth["covariate_effect"] / truth["feature_scale"]
    np.testing.assert_allclose(contribution.std(axis=0), np.abs(truth["covariate_coefficients"]["age"]), rtol=1e-9)
    np.testing.assert_allclose(contribution.mean(axis=0), 0.0, atol=1e-9)


def test_quadratic_shape_is_quadratic_in_the_standardized_covariate():
    """effect_shape='quadratic' is a linear function of z**2."""
    *_, truth = _simulate([_age(effect=0.5, effect_shape="quadratic")])
    age = truth["covariates"]["age"]
    z = (age - age.mean()) / age.std()
    assert np.corrcoef(truth["covariate_effect"][:, 0], z**2)[0, 1] == pytest.approx(1.0, abs=1e-9)


def test_scalar_effect_weights():
    """A scalar effect is multiplied per feature by U(0.5, 1.5)."""
    *_, truth = _simulate([_age(effect=-0.4)], n_features=200)
    coefficients = truth["covariate_coefficients"]["age"]
    assert ((coefficients <= -0.2) & (coefficients >= -0.6)).all()
    assert coefficients.mean() == pytest.approx(-0.4, abs=0.03)


def test_array_effect_is_used_exactly():
    """An array gives the coefficient of every feature."""
    effect = np.array([0.0, 0.5, -1.0, 2.0])
    *_, truth = _simulate([_age(effect=effect)], n_features=4)
    np.testing.assert_array_equal(truth["covariate_coefficients"]["age"], effect)
    np.testing.assert_array_equal(truth["covariate_effect"][:, 0], 0.0)


@pytest.mark.parametrize(("effect_features", "affected"), [(3, [0, 1, 2]), (0.5, [0, 1, 2, 3, 4]), ([1, 7], [1, 7])])
def test_effect_features(effect_features, affected):
    """effect_features selects the first k, a fraction, or listed features."""
    *_, truth = _simulate([_age(effect=1.0, effect_features=effect_features)], n_features=10)
    assert np.flatnonzero(truth["covariate_coefficients"]["age"]).tolist() == affected
    unaffected = np.setdiff1d(np.arange(10), affected)
    np.testing.assert_array_equal(truth["covariate_effect"][:, unaffected], 0.0)


def test_categorical_effect_is_the_mean_difference():
    """For a categorical covariate, each category differs from the first by its coefficient (reference category 0)."""
    sex = Covariate(name="sex", site_distributions=[CovariateSiteDistribution(probs=[0.5, 0.5])], effect=0.3)
    *_, truth = _simulate([sex])
    coefficients = truth["covariate_coefficients"]["sex"]
    np.testing.assert_array_equal(coefficients[0], 0.0)
    contribution = truth["covariate_effect"] / truth["feature_scale"]
    female = truth["covariates"]["sex"] == 1
    np.testing.assert_allclose(contribution[female], np.broadcast_to(coefficients[1], (int(female.sum()), 10)))
    np.testing.assert_allclose(contribution[~female], 0.0)


def test_categorical_effect_per_category():
    """A (n_categories - 1, n_features) array gives the effect of every non-reference category."""
    effect = np.array([[1.0, 0.0], [0.0, -2.0]])
    group = Covariate(
        name="group", categories=["a", "b", "c"], site_distributions=[CovariateSiteDistribution(probs=[1, 1, 1])], effect=effect
    )
    *_, truth = _simulate([group], n_features=2)
    np.testing.assert_array_equal(truth["covariate_coefficients"]["group"], np.vstack([[0.0, 0.0], effect]))
    for row, category in enumerate(["a", "b", "c"]):
        mask = truth["covariates"]["group"] == category
        expected = truth["covariate_coefficients"]["group"][row] * truth["feature_scale"]
        np.testing.assert_allclose(truth["covariate_effect"][mask], np.broadcast_to(expected, (mask.sum(), 2)))


def test_presets_have_effects():
    """The age and sex presets change the features; quality does not."""
    *_, truth = _simulate(["age", "sex", "quality"])
    coefficients = truth["covariate_coefficients"]
    assert coefficients["age"].mean() == pytest.approx(0.5, abs=0.2)
    assert coefficients["sex"][1].mean() == pytest.approx(0.3, abs=0.12)
    np.testing.assert_array_equal(coefficients["quality"], 0.0)


def test_covariates_are_only_in_the_ground_truth():
    """Covariates do not change the outputs: without the ground truth, X, y and sites are returned."""
    out = make_multisite_classification(covariates=["age"])
    assert len(out) == 3
    *_, truth = make_multisite_classification(covariates=["age"], return_ground_truth=True)
    assert set(truth["covariates"]) == {"age"}
    *_, truth = make_multisite_classification(return_ground_truth=True)
    assert truth["covariates"] == {}


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"effect": float("nan")}, ValueError, "effect must be finite"),
        ({"effect_shape": "cubic"}, ValueError, "effect_shape must be one of"),
        ({"effect": np.ones((2, 3))}, ValueError, "array of shape"),
    ],
)
def test_invalid_covariate_effect(kwargs, error, match):
    """Invalid effect parameters are rejected when the Covariate is created."""
    with pytest.raises(error, match=match):
        _age(**kwargs)


def test_categorical_covariate_rejects_a_nonlinear_shape():
    """effect_shape only applies to continuous covariates."""
    with pytest.raises(ValueError, match="only applies to continuous"):
        Covariate(name="sex", site_distributions=[CovariateSiteDistribution(probs=[1, 1])], effect=1.0, effect_shape="quadratic")


def test_x_correlation_was_removed():
    """x_correlation is no longer a Covariate parameter."""
    with pytest.raises(TypeError):
        _age(x_correlation=0.5)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"effect": 1.0, "effect_features": 11}, ValueError, "between 1 and n_features"),
        ({"effect": 1.0, "effect_features": 1.5}, ValueError, r"in \(0, 1\]"),
        ({"effect": 1.0, "effect_features": [0, 10]}, ValueError, "distinct indices"),
        ({"effect": 1.0, "effect_features": True}, TypeError, "effect_features must be"),
        ({"effect": np.ones(3)}, ValueError, "3 values but there are 10 features"),
        ({"effect": 1.0, "effect_shape": lambda a: a[:5]}, ValueError, "must return finite values"),
    ],
)
def test_invalid_covariate_effect_at_simulation(kwargs, error, match):
    """Effect parameters that depend on the data are checked when simulating."""
    with pytest.raises(error, match=match):
        _simulate([_age(**kwargs)], n_features=10)


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("site_effect_type", ["location", "scale", "variance", "nonlinear", "dropout", "combat"])
@pytest.mark.parametrize("heterogeneity", [0.0, 0.5])
def test_ground_truth_decomposes_x(site_effect_type, heterogeneity):
    """X = site_free + site_effect and site_free = signal + covariate effects + noise, exactly."""
    X, y, _, truth = make_multisite_classification(
        n_sites=3,
        covariates=["age", "sex"],
        site_effect={site_effect_type: _strength(site_effect_type, 3.0), "class_interaction": 0.5},
        site_effect_heterogeneity=heterogeneity,
        return_ground_truth=True,
    )
    np.testing.assert_allclose(truth["site_free"], truth["signal"] + truth["covariate_effect"] + truth["noise"], atol=1e-12)
    np.testing.assert_allclose(X, truth["site_free"] + truth["site_effect"], atol=1e-12)
    np.testing.assert_allclose(truth["feature_scale"], _within_class_sd(truth["signal"], y), rtol=1e-12)


def test_ground_truth_location_is_the_site_offset():
    """For a homogeneous location effect, every sample of a site is shifted by its site_location."""
    _, _, sites, truth = make_multisite_classification(n_sites=4, return_ground_truth=True)
    for site in range(4):
        mask = sites == site
        expected = np.broadcast_to(truth["site_location"][site], (int(mask.sum()), 10))
        np.testing.assert_allclose(truth["site_effect"][mask], expected, atol=1e-12)
    np.testing.assert_array_equal(truth["site_scale"], 1.0)


def test_scale_multiplies_signal_and_noise_but_not_covariate_effects():
    """The scale effect multiplies the residual (signal + noise) around its mean: ComBat's model, covariates kept."""
    X, _, sites, truth = make_multisite_classification(
        n_sites=3,
        covariates=[_age(locs=(30, 50, 70), effect=1.0)],
        site_effect={"scale": 5},
        noise_strength=0.5,
        return_ground_truth=True,
    )
    residual = truth["signal"] + truth["noise"]
    mean = residual.mean(axis=0)
    for site in range(3):
        mask = sites == site
        expected = mean + (residual[mask] - mean) * truth["site_scale"][site] + truth["covariate_effect"][mask]
        np.testing.assert_allclose(X[mask], expected, atol=1e-12)
    np.testing.assert_array_equal(truth["site_location"], 0.0)


# ---------------------------------------------------------------------------
# Units and centred location offsets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("signal_type", ["linear", "blobs", "gaussian_quantiles", "moons", "circles"])
def test_location_offsets_are_centred_and_in_reference_sd_units(signal_type):
    """Offsets / reference SD lie in [-s/2, s/2] and are centred, for every generator."""
    *_, truth = make_multisite_classification(
        n_sites=60,
        n_samples=600,
        n_features=10,
        signal={signal_type: 1.0},
        site_effect={"location": 4.0},
        return_ground_truth=True,
    )
    offsets = truth["site_location"] / truth["feature_scale"]
    assert (np.abs(offsets) <= 2.0 + 1e-12).all()
    assert abs(offsets.mean()) < 0.15  # E = 0 (60 sites x n_features draws of U(-2, 2))
    assert offsets.std() == pytest.approx(4.0 / np.sqrt(12), rel=0.15)


def test_location_effect_is_a_cohens_d():
    """Between two sites, the standardized mean difference of a feature is the difference of their offsets (in SD)."""
    X, _, sites, truth = make_multisite_classification(
        n_sites=2,
        n_samples=20000,
        signal={"moons": 1.0},
        site_effect={"location": 2.0},
        noise_strength=0,
        return_ground_truth=True,
    )
    d = (X[sites == 1].mean(axis=0) - X[sites == 0].mean(axis=0)) / truth["feature_scale"]
    expected = (truth["site_location"][1] - truth["site_location"][0]) / truth["feature_scale"]
    np.testing.assert_allclose(d, expected, atol=0.05)


def test_feature_scale_does_not_depend_on_class_separation():
    """The reference SD is the within-class SD, so stronger signals do not inflate the site effects."""
    scales = [
        make_multisite_classification(n_samples=4000, signal={"blobs": s}, return_ground_truth=True)[-1]["feature_scale"]
        for s in (0.5, 5.0)
    ]
    np.testing.assert_allclose(scales[0], scales[1], rtol=0.1)


# ---------------------------------------------------------------------------
# ComBat's generative model, affected features, heterogeneity, interaction
# ---------------------------------------------------------------------------


def test_combat_type_follows_the_combat_priors():
    """Offsets ~ N(g, (s/4)^2) with g in [-s/2, s/2]; squared factors ~ inverse gamma of CV 0.5, mean in exp([-0.2 s, 0.2 s])."""
    s = 4.0
    *_, truth = make_multisite_classification(
        n_sites=3, n_samples=150, n_features=3000, site_effect={"combat": s}, return_ground_truth=True
    )
    offsets = truth["site_location"] / truth["feature_scale"]
    variance_factors = truth["site_scale"] ** 2
    for site in range(3):
        assert abs(offsets[site].mean()) <= s / 2 + 0.05
        assert offsets[site].std() == pytest.approx(s / 4, rel=0.05)
        mean = variance_factors[site].mean()
        assert np.exp(-0.2 * s) * 0.95 <= mean <= np.exp(0.2 * s) * 1.05
        assert variance_factors[site].std() / mean == pytest.approx(0.5, rel=0.12)  # CV of an inverse gamma of shape 6


def test_combat_with_zero_strength_is_no_effect():
    """A ComBat strength of 0 leaves the data unchanged."""
    X, _, _, truth = make_multisite_classification(site_effect={"combat": 0.0}, return_ground_truth=True)
    np.testing.assert_array_equal(X, truth["site_free"])


@pytest.mark.parametrize("site_effect_type", ["location", "scale", "variance", "nonlinear", "dropout", "combat"])
def test_site_effect_features(site_effect_type):
    """Only the selected features get a site effect; the others are exactly unchanged."""
    X, _, _, truth = make_multisite_classification(
        n_features=8,
        site_effect={site_effect_type: _strength(site_effect_type, 9.0), "class_interaction": 1.0},
        site_effect_features=[1, 4],
        site_effect_heterogeneity=0.3,
        return_ground_truth=True,
    )
    unaffected = [0, 2, 3, 5, 6, 7]
    np.testing.assert_array_equal(X[:, unaffected], truth["site_free"][:, unaffected])
    assert not np.allclose(X[:, [1, 4]], truth["site_free"][:, [1, 4]])
    np.testing.assert_array_equal(truth["site_effect_features"], [1, 4])


@pytest.mark.parametrize("heterogeneity", [0.3, 1.0])
def test_heterogeneity_is_a_per_sample_intensity_along_the_site_effect(heterogeneity):
    """Each sample's offset is its intensity times the site offset: a rank-1 structure, not iid noise."""
    _, _, sites, truth = make_multisite_classification(
        n_sites=2, n_samples=20000, site_effect_heterogeneity=heterogeneity, return_ground_truth=True
    )
    intensity = truth["intensity"]
    for site in range(2):
        mask = sites == site
        np.testing.assert_allclose(truth["site_effect"][mask], intensity[mask, None] * truth["site_location"][site], atol=1e-12)
    # log-normal intensity with mean 1 and SD sqrt(exp(h^2) - 1)
    assert intensity.min() > 0
    assert intensity.mean() == pytest.approx(1.0, abs=0.02)
    assert intensity.std() == pytest.approx(np.sqrt(np.expm1(heterogeneity**2)), rel=0.05)


def test_site_class_interaction():
    """Each class of each site gets its own extra offset; without interaction the classes share the site offset."""
    _, y, sites, truth = make_multisite_classification(
        n_sites=2, n_classes=3, n_samples=900, site_effect={"location": 3.0, "class_interaction": 2.0}, return_ground_truth=True
    )
    for site in range(2):
        for c in range(3):
            mask = (sites == site) & (y == c)
            expected = truth["site_location"][site] + truth["site_class_location"][site, c]
            np.testing.assert_allclose(truth["site_effect"][mask], np.broadcast_to(expected, (int(mask.sum()), 10)), atol=1e-12)
    interaction = truth["site_class_location"] / truth["feature_scale"]
    assert (np.abs(interaction) <= 1.0 + 1e-12).all()
    *_, no_interaction = make_multisite_classification(n_sites=2, n_classes=3, n_samples=900, return_ground_truth=True)
    np.testing.assert_array_equal(no_interaction["site_class_location"], 0.0)


def test_effect_types_combine():
    """Several types can be combined, each with its own strength."""
    X, _, _, truth = make_multisite_classification(
        site_effect={"location": 2.0, "scale": 3.0, "nonlinear": 1.0},
        return_ground_truth=True,
    )
    assert (truth["site_location"] != 0).all()
    assert (truth["site_scale"] != 1).all()
    np.testing.assert_allclose(X, truth["site_free"] + truth["site_effect"], atol=1e-12)


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def test_as_frame_and_site_names():
    """as_frame returns a DataFrame and Series; site_names labels the sites."""
    X, y, sites, truth = make_multisite_classification(
        n_sites=3,
        n_features=4,
        covariates=["age"],
        site_names=["scanner A", "scanner B", "scanner C"],
        as_frame=True,
        return_ground_truth=True,
    )
    assert isinstance(X, pd.DataFrame) and list(X.columns) == ["feature_0", "feature_1", "feature_2", "feature_3"]
    assert isinstance(y, pd.Series) and y.name == "target"
    assert isinstance(sites, pd.Series) and sites.name == "site"
    assert sorted(sites.unique()) == ["scanner A", "scanner B", "scanner C"]
    assert isinstance(truth["covariates"], pd.DataFrame) and list(truth["covariates"].columns) == ["age"]
    for key in ("site_location", "site_scale"):
        assert list(truth[key].index) == ["scanner A", "scanner B", "scanner C"]
        assert list(truth[key].columns) == list(X.columns)
    arrays = make_multisite_classification(n_sites=3, n_features=4, covariates=["age"])  # same data as arrays
    np.testing.assert_array_equal(X.to_numpy(), arrays[0])


@pytest.mark.parametrize(("site_names", "match"), [(["a", "b", "c"], "one label per site"), (["a", "a"], "distinct")])
def test_invalid_site_names(site_names, match):
    """site_names needs one distinct label per site."""
    with pytest.raises(ValueError, match=match):
        make_multisite_classification(n_sites=2, site_names=site_names)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"n_sites": 2.0}, TypeError, "n_sites must be an int"),
        ({"n_sites": True}, TypeError, "n_sites must be an int"),
        ({"n_features": 3.5}, TypeError, "n_features must be an int"),
        ({"n_classes": 2.0}, TypeError, "n_classes must be an int"),
        ({"n_classes": 5, "n_features": 2}, ValueError, "hypercube"),
        ({"noise_strength": float("inf")}, ValueError, "noise_strength must be non-negative and finite"),
        (
            {"site_effect": {"location": [1.0, float("nan")]}},
            ValueError,
            r"site_effect\['location'\]\[1\] must be non-negative and finite",
        ),
        ({"signal": {"linear": float("inf")}}, ValueError, r"signal\['linear'\] must be non-negative and finite"),
        ({"site_effect_heterogeneity": -0.1}, ValueError, "site_effect_heterogeneity must be non-negative"),
        ({"site_effect": {"class_interaction": [1.0, 2.0, 3.0]}}, ValueError, "class_interaction'\\] must have length"),
        ({"site_effect_features": 0}, ValueError, "site_effect_features=0 must be between 1"),
        ({"balance_per_site": [1.0, 0.0]}, ValueError, r"Class\(es\) \[1\] have proportion 0 in every site"),
        ({"n_samples": 2, "n_classes": 3, "n_features": 4}, ValueError, "cannot hold the 3 classes"),
    ],
)
def test_invalid_inputs(kwargs, error, match):
    """Invalid sizes, non-finite strengths, impossible problems and empty classes raise informative errors."""
    with pytest.raises(error, match=match):
        make_multisite_classification(**kwargs)


def test_few_samples_of_a_class_in_a_site_are_reported():
    """A class with fewer than 6 samples in a site is logged as a warning (many methods need more)."""
    with verbosity_context("warning"), capture_logs() as logs:
        make_multisite_classification(n_samples=[40, 40], balance_per_site=[[0.9, 0.1], [0.5, 0.5]])
    assert any("fewer than 6 samples" in entry["event"] for entry in logs)
    with verbosity_context("warning"), capture_logs() as logs:
        make_multisite_classification(n_samples=[40, 40])
    assert not any("fewer than 6 samples" in entry["event"] for entry in logs)


def test_numpy_generator_as_random_state():
    """A NumPy Generator is accepted, and the same seed reproduces the data."""
    first = make_multisite_classification(random_state=np.random.default_rng(3))
    second = make_multisite_classification(random_state=np.random.default_rng(3))
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("signal_type", ["moons", "circles"])
def test_signal_strength_ignored_by_2d_generators_warns(signal_type):
    """Moons and circles have no strength parameter: a value other than 1 warns."""
    with pytest.warns(UserWarning, match="has no strength parameter"):
        make_multisite_classification(signal={signal_type: 0.5})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        make_multisite_classification(signal={signal_type: 1.0})


def test_heterogeneity_keeps_the_site_parameters():
    """Changing the heterogeneity changes the per-sample intensity only, not the site offsets."""
    params = {"n_sites": 3, "return_ground_truth": True}
    *_, homogeneous = make_multisite_classification(**params)
    *_, heterogeneous = make_multisite_classification(site_effect_heterogeneity=0.7, **params)
    np.testing.assert_array_equal(homogeneous["site_location"], heterogeneous["site_location"])
    np.testing.assert_array_equal(homogeneous["intensity"], 1.0)


# ---------------------------------------------------------------------------
# Dependent covariates: TIV depends on sex
# ---------------------------------------------------------------------------


def test_tiv_preset_depends_on_sex():
    """TIV is 160 cm3 lower in sex 1, with SD 120 within sex: correlation about -0.55 with balanced sexes."""
    *_, truth = _simulate(covariates=["sex", "tiv"], n_samples=20000)
    sex, tiv = truth["covariates"]["sex"], truth["covariates"]["tiv"]
    assert tiv[sex == 0].mean() == pytest.approx(1520, abs=5)
    assert tiv[sex == 1].mean() == pytest.approx(1360, abs=5)
    assert tiv[sex == 0].std() == pytest.approx(120, rel=0.05)
    assert np.corrcoef(sex, tiv)[0, 1] == pytest.approx(-0.55, abs=0.03)


def test_dependence_does_not_depend_on_list_order():
    """The covariates are returned in the order given, whatever the order of the dependence."""
    *_, truth = _simulate(covariates=["tiv", "sex"], n_samples=500)
    assert list(truth["covariates"]) == ["tiv", "sex"]
    assert np.corrcoef(truth["covariates"]["sex"], truth["covariates"]["tiv"])[0, 1] < -0.4


def test_tiv_without_sex_raises():
    """A dependent covariate needs its categorical covariate in the same list."""
    with pytest.raises(ValueError, match="depends on 'sex', which is not in the covariates"):
        make_multisite_classification(covariates=["tiv"])


def test_custom_dependence_shifts_by_category():
    """depends_on adds one shift per category of the parent to the continuous covariate."""
    group = Covariate(
        name="group", site_distributions=[CovariateSiteDistribution(probs=[0.5, 0.3, 0.2])] * 2, categories=["a", "b", "c"]
    )
    score = Covariate(
        name="score",
        site_distributions=[CovariateSiteDistribution(loc=0.0, scale=0.0)] * 2,
        depends_on={"group": [0.0, 10.0, -5.0]},
    )
    *_, truth = _simulate(covariates=[group, score], n_samples=300)
    expected = pd.Series(truth["covariates"]["group"]).map({"a": 0.0, "b": 10.0, "c": -5.0}).to_numpy()
    np.testing.assert_array_equal(truth["covariates"]["score"], expected)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"depends_on": {"sex": [0.0, 1.0]}, "kind": "categorical"}, "depends_on must be a dict"),
        ({"depends_on": {"sex": [0.0, 1.0], "site": [0.0]}}, "depends_on must be a dict"),
        ({"depends_on": {"sex": [0.0, float("nan")]}}, "one finite shift per category"),
    ],
)
def test_invalid_depends_on(kwargs, match):
    """depends_on is one categorical parent with finite shifts, for a continuous covariate."""
    kind = kwargs.pop("kind", "continuous")
    dist = CovariateSiteDistribution(probs=[0.5, 0.5]) if kind == "categorical" else CovariateSiteDistribution(loc=0.0)
    with pytest.raises(ValueError, match=match):
        Covariate(name="x", site_distributions=[dist] * 2, **kwargs)


@pytest.mark.parametrize(
    ("shifts", "parent", "match"),
    [
        ([0.0, 1.0, 2.0], "sex", "gives 3 shifts but 'sex' has 2 categories"),
        ([0.0], "age", "can only depend on a categorical covariate"),
    ],
)
def test_dependence_checked_against_parent(shifts, parent, match):
    """The parent must be categorical and get one shift per category."""
    child = Covariate(name="x", site_distributions=[CovariateSiteDistribution(loc=0.0)] * 2, depends_on={parent: shifts})
    with pytest.raises(ValueError, match=match):
        make_multisite_classification(covariates=["age", "sex", child])


def test_duplicate_covariate_names_raise():
    """Two covariates with the same name would overwrite each other."""
    with pytest.raises(ValueError, match="must be unique"):
        make_multisite_classification(covariates=["age", "age"])


# ---------------------------------------------------------------------------
# Dropout: replaced values, e.g. missing values
# ---------------------------------------------------------------------------


def test_dropout_nan_marks_whole_features_of_a_site_as_missing():
    """With per="feature", every affected feature is NaN for all samples of a site with probability prob."""
    X, _, sites, truth = make_multisite_classification(
        n_sites=6,
        n_features=40,
        site_effect={"location": 2.0, "dropout": {"value": np.nan, "prob": 0.3}},
        site_effect_features=30,
        return_ground_truth=True,
    )
    mask = truth["dropout_mask"]
    np.testing.assert_array_equal(np.isnan(X), mask)
    assert not mask[:, 30:].any()  # only the affected features
    for site in range(6):
        site_mask = mask[sites == site]
        np.testing.assert_array_equal(site_mask, np.broadcast_to(site_mask[0], site_mask.shape))  # whole features
    assert 0.15 < mask[:, :30].mean() < 0.45
    kept = ~mask
    np.testing.assert_allclose(X[kept], (truth["site_free"] + truth["site_effect"])[kept], atol=1e-12)


def test_dropout_per_value_and_per_site_probability():
    """With per="value", single values are replaced at rate prob; prob can be given per site."""
    X, _, sites, truth = make_multisite_classification(
        n_sites=2,
        n_samples=4000,
        site_effect={"dropout": {"value": -1.0, "prob": [0.0, 0.25], "per": "value"}},
        return_ground_truth=True,
    )
    mask = truth["dropout_mask"]
    assert not mask[sites == 0].any()
    assert mask[sites == 1].mean() == pytest.approx(0.25, abs=0.01)
    np.testing.assert_array_equal(X[mask], -1.0)
    np.testing.assert_array_equal(X[~mask], truth["site_free"][~mask])


def test_without_dropout_the_mask_is_empty():
    """The ground truth always has the mask."""
    *_, truth = make_multisite_classification(return_ground_truth=True)
    assert truth["dropout_mask"].shape == (1000, 10) and not truth["dropout_mask"].any()


@pytest.mark.parametrize(
    ("dropout", "error", "match"),
    [
        (0.3, TypeError, r"site_effect\['dropout'\] must be a dict"),
        ({"prob": 0.3}, ValueError, "needs the keys 'value' and 'prob'"),
        ({"value": 0.0, "prob": 0.3, "rate": 1}, ValueError, "needs the keys 'value' and 'prob'"),
        ({"value": "missing", "prob": 0.3}, TypeError, r"\['value'\] must be a float"),
        ({"value": 0.0, "prob": 1.5}, ValueError, "must be between 0 and 1"),
        ({"value": 0.0, "prob": [0.1, 0.2, 0.3]}, ValueError, "must have length n_sites"),
        ({"value": 0.0, "prob": 0.3, "per": "sample"}, ValueError, r"\['per'\] must be one of"),
    ],
)
def test_invalid_dropout(dropout, error, match):
    """Dropout takes {"value": float, "prob": float or one per site} and optionally "per"."""
    with pytest.raises(error, match=match):
        make_multisite_classification(n_sites=2, site_effect={"dropout": dropout})
