"""Input validation of the ComBat variants: feature count, covariates given to fit and transform, tags."""

import numpy as np
import pandas as pd
import pytest

from uniharmony.combat import ComBatGAM, CovBat, NeuroComBat


N_SAMPLES = 90


@pytest.fixture
def data() -> dict[str, np.ndarray]:
    """Three sites, four features, a categorical and a continuous covariate."""
    rng = np.random.default_rng(0)
    return {
        "X": rng.normal(size=(N_SAMPLES, 4)),
        "sites": np.repeat(["a", "b", "c"], N_SAMPLES // 3),
        "sex": rng.integers(0, 2, N_SAMPLES)[:, None],
        "age": rng.uniform(20, 80, N_SAMPLES)[:, None],
    }


@pytest.fixture(params=[NeuroComBat, CovBat], ids=lambda cls: cls.__name__)
def linear_combat(request: pytest.FixtureRequest) -> type:
    """ComBat variants with categorical and continuous covariates."""
    return request.param


def _fit_gam(data, **kwargs) -> ComBatGAM:
    return ComBatGAM(random_state=0).fit(data["X"], data["sites"], smooth_covariates=data["age"], **kwargs)


# ---------------------------------------------------------------------------
# Number of features (audit #14)
# ---------------------------------------------------------------------------


def test_n_features_in_is_set(linear_combat, data) -> None:
    """Fit stores the number of features."""
    assert linear_combat().fit(data["X"], data["sites"]).n_features_in_ == 4


@pytest.mark.parametrize("n_features", [3, 5])
def test_transform_with_other_number_of_features_raises(linear_combat, data, n_features: int) -> None:
    """A different number of features at transform gives a clear error instead of a broadcasting error."""
    harmonizer = linear_combat().fit(data["X"], data["sites"])
    X_other = np.random.default_rng(1).normal(size=(N_SAMPLES, n_features))
    with pytest.raises(ValueError, match=f"X has {n_features} features, but .* is expecting 4 features"):
        harmonizer.transform(X_other, data["sites"])


def test_gam_transform_with_other_number_of_features_raises(data) -> None:
    """ComBatGAM checks the number of features too."""
    with pytest.raises(ValueError, match="X has 3 features, but ComBatGAM is expecting 4 features"):
        _fit_gam(data).transform(data["X"][:, :3], data["sites"], smooth_covariates=data["age"])


def test_feature_names_are_checked(linear_combat, data) -> None:
    """Feature names of a DataFrame are stored at fit and checked at transform."""
    columns = ["thickness", "volume", "area", "curvature"]
    X = pd.DataFrame(data["X"], columns=columns)
    harmonizer = linear_combat().fit(X, data["sites"])
    assert list(harmonizer.feature_names_in_) == columns
    with pytest.raises(ValueError, match="feature names"):
        harmonizer.transform(X[columns[::-1]], data["sites"])


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_missing_or_infinite_values_raise(linear_combat, data, value: float) -> None:
    """Missing and infinite values are rejected (and the tags say so)."""
    X = data["X"].copy()
    X[0, 0] = value
    with pytest.raises(ValueError, match=r"NaN|infinity"):
        linear_combat().fit(X, data["sites"])
    assert linear_combat().__sklearn_tags__().input_tags.allow_nan is False


def test_tags_describe_sites(linear_combat) -> None:
    """``sites`` is required, one label per sample, of any value."""
    target_tags = linear_combat().__sklearn_tags__().target_tags
    assert target_tags.required
    assert not target_tags.two_d_labels
    assert not target_tags.positive_only


def test_gam_tags_follow_random_state() -> None:
    """ComBatGAM is deterministic only with a fixed random_state."""
    assert ComBatGAM().__sklearn_tags__().non_deterministic
    assert not ComBatGAM(random_state=0).__sklearn_tags__().non_deterministic


# ---------------------------------------------------------------------------
# Covariates given to fit and transform (audit #15)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing", ["categorical_covariates", "continuous_covariates"])
def test_covariate_given_to_fit_but_not_to_transform_raises(linear_combat, data, missing: str) -> None:
    """A covariate used in fit is required by transform."""
    covariates = {"categorical_covariates": data["sex"], "continuous_covariates": data["age"]}
    harmonizer = linear_combat().fit(data["X"], data["sites"], **covariates)
    del covariates[missing]
    with pytest.raises(ValueError, match=f"{missing} were given to fit, so they must also be given to transform"):
        harmonizer.transform(data["X"], data["sites"], **covariates)


@pytest.mark.parametrize("extra", ["categorical_covariates", "continuous_covariates"])
def test_covariate_given_to_transform_but_not_to_fit_raises(linear_combat, data, extra: str) -> None:
    """A covariate not used in fit cannot be used by transform."""
    harmonizer = linear_combat().fit(data["X"], data["sites"])
    value = data["sex"] if extra == "categorical_covariates" else data["age"]
    with pytest.raises(ValueError, match=f"{extra} were not given to fit, so transform cannot use them"):
        harmonizer.transform(data["X"], data["sites"], **{extra: value})


def test_gam_continuous_covariates_must_match_fit(data) -> None:
    """ComBatGAM checks its continuous covariates the same way."""
    with pytest.raises(ValueError, match="continuous_covariates were not given to fit"):
        _fit_gam(data).transform(data["X"], data["sites"], smooth_covariates=data["age"], continuous_covariates=data["age"])
    with pytest.raises(ValueError, match="continuous_covariates were given to fit"):
        _fit_gam(data, continuous_covariates=data["sex"].astype(float)).transform(
            data["X"], data["sites"], smooth_covariates=data["age"]
        )


def test_gam_requires_smooth_covariates(data) -> None:
    """ComBatGAM needs smooth covariates in fit and transform."""
    with pytest.raises(ValueError, match="smooth_covariates are required"):
        ComBatGAM().fit(data["X"], data["sites"], smooth_covariates=None)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="smooth_covariates are required"):
        _fit_gam(data).transform(data["X"], data["sites"], smooth_covariates=None)  # pyright: ignore[reportArgumentType]
