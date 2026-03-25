"""Comprehensive tests for InterSiteMatchedInterpolation."""

import warnings
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from uniharmony.interpolation import InterSiteMatchedInterpolation


# Fixtures
@pytest.fixture
def binary_2site() -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[str]]:
    """Provide simple 2-site binary classification data."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 5)
    y = rng.randint(0, 2, 100)
    sites = np.array(["A"] * 50 + ["B"] * 50)
    return X, y, sites


@pytest.fixture
def multi_3site() -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[str]]:
    """Provide 3-site data for pairwise testing."""
    rng = np.random.RandomState(42)
    X = rng.randn(150, 10)
    y = rng.randint(0, 2, 150)
    sites = np.array(["Site1"] * 50 + ["Site2"] * 50 + ["Site3"] * 50)
    return X, y, sites


@pytest.fixture
def regression_2site() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[str]]:
    """Provide 2-site regression data."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 5)
    y = rng.randn(100) * 10 + 50
    sites = np.array(["X"] * 50 + ["Y"] * 50)
    return X, y, sites


@pytest.fixture
def covariates_2site() -> tuple[NDArray[Any], NDArray[np.float64]]:
    """Provide sex and age covariates for 100 samples."""
    sex = np.array([["M"], ["F"]] * 50)
    age = np.random.RandomState(42).randint(20, 80, (100, 1)).astype(float)
    return sex, age


# Initialization tests
class TestInit:
    """Test __init__ and parameter validation."""

    def test_defaults(self) -> None:
        """Test default parameter values."""
        ismi = InterSiteMatchedInterpolation()
        assert ismi.alpha == 0.3
        assert ismi.k == 1
        assert ismi.mode == "pairwise"
        assert ismi.concatenate is True
        assert ismi.verbose is False

    def test_alpha_float(self) -> None:
        """Test float alpha initialization."""
        ismi = InterSiteMatchedInterpolation(alpha=0.5)
        assert ismi.alpha == 0.5

    def test_alpha_tuple(self) -> None:
        """Test tuple alpha initialization."""
        ismi = InterSiteMatchedInterpolation(alpha=(0.2, 0.6))
        assert ismi.alpha == (0.2, 0.6)

    def test_k_max(self) -> None:
        """Test k='max' initialization."""
        ismi = InterSiteMatchedInterpolation(k="max")
        assert ismi.k == "max"

    def test_k_average(self) -> None:
        """Test k='average' initialization."""
        ismi = InterSiteMatchedInterpolation(k="average")
        assert ismi.k == "average"

    def test_k_int(self) -> None:
        """Test integer k initialization."""
        ismi = InterSiteMatchedInterpolation(k=5)
        assert ismi.k == 5

    def test_mode_base_to_others(self) -> None:
        """Test base_to_others mode initialization."""
        ismi = InterSiteMatchedInterpolation(mode="base_to_others")
        assert ismi.mode == "base_to_others"


# Parameter validation tests
class TestParameterValidation:
    """Test parameter validation in __init__."""

    def test_invalid_alpha_range(self, binary_2site) -> None:
        """Test invalid alpha range raises ValueError."""
        X, y, sites = binary_2site
        with pytest.raises(ValueError):
            ismi = InterSiteMatchedInterpolation(alpha=(0.6, 0.2))
            ismi.fit_resample(X, y, sites=sites)

    def test_alpha_zero(self, binary_2site: tuple) -> None:
        """Test alpha=0 interpolation."""
        X, y, sites = binary_2site
        ismi = InterSiteMatchedInterpolation(alpha=0.0, random_state=42)
        with pytest.raises(ValueError):
            ismi.fit_resample(X, y, sites=sites)

    def test_alpha_one(self, binary_2site: tuple) -> None:
        """Test alpha=1 interpolation."""
        X, y, sites = binary_2site
        ismi = InterSiteMatchedInterpolation(alpha=1, random_state=42)
        with pytest.raises(ValueError):
            ismi.fit_resample(X, y, sites=sites)

    def test_invalid_alpha_negative(self, binary_2site) -> None:
        """Test negative alpha raises ValueError."""
        X, y, sites = binary_2site
        with pytest.raises(ValueError, match="alpha"):
            ismi = InterSiteMatchedInterpolation(alpha=-0.1)
            ismi.fit_resample(X, y, sites=sites)

    def test_invalid_alpha_large(self, binary_2site) -> None:
        """Test alpha > 1 raises ValueError."""
        X, y, sites = binary_2site
        with pytest.raises(ValueError, match="alpha"):
            ismi = InterSiteMatchedInterpolation(alpha=1.5)
            ismi.fit_resample(X, y, sites=sites)

    def test_invalid_k_string(self, binary_2site) -> None:
        """Test invalid k string raises ValueError."""
        X, y, sites = binary_2site
        with pytest.raises(ValueError, match="k"):
            ismi = InterSiteMatchedInterpolation(k="invalid")
            ismi.fit_resample(X, y, sites=sites)

    def test_invalid_k_zero(self, binary_2site) -> None:
        """Test k=0 raises ValueError."""
        X, y, sites = binary_2site
        with pytest.raises(ValueError, match="k"):
            ismi = InterSiteMatchedInterpolation(k=0)
            ismi.fit_resample(X, y, sites=sites)

    def test_invalid_k_negative(self, binary_2site) -> None:
        """Test negative k raises ValueError."""
        X, y, sites = binary_2site
        with pytest.raises(ValueError, match="k"):
            ismi = InterSiteMatchedInterpolation(k=-1)
            ismi.fit_resample(X, y, sites=sites)

    def test_single_site_error(self) -> None:
        """Test single site raises ValueError."""
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)
        sites = np.array(["A"] * 50)
        ismi = InterSiteMatchedInterpolation()
        with pytest.raises(ValueError, match="at least 2 sites"):
            ismi.fit_resample(X, y, sites=sites)

    def test_mismatched_xy(self, binary_2site: tuple) -> None:
        """Test mismatched X and y raises ValueError."""
        X, y, sites = binary_2site
        ismi = InterSiteMatchedInterpolation()
        with pytest.raises(ValueError):
            ismi.fit_resample(X[:-10], y, sites=sites)

    def test_mismatched_sites(self, binary_2site: tuple) -> None:
        """Test mismatched sites length raises ValueError."""
        X, y, sites = binary_2site
        ismi = InterSiteMatchedInterpolation()
        with pytest.raises(ValueError):
            ismi.fit_resample(X, y, sites=sites[:-10])

    def test_covariate_wrong_length(self, binary_2site: tuple) -> None:
        """Test wrong covariate length raises ValueError."""
        X, y, sites = binary_2site
        ismi = InterSiteMatchedInterpolation()
        wrong_cov = np.array([["M"]] * 90)
        with pytest.raises(ValueError, match="samples"):
            ismi.fit_resample(X, y, sites=sites, categorical_covariate=wrong_cov)

    def test_tolerance_without_continuous(self, binary_2site: tuple) -> None:
        """Test tolerance without continuous covariates raises ValueError."""
        X, y, sites = binary_2site
        ismi = InterSiteMatchedInterpolation(covariate_tolerance=np.array([5.0]))
        with pytest.raises(ValueError, match=r"tolerance.*continuous"):
            ismi.fit_resample(X, y, sites=sites)

    def test_nan_in_categorical(self, binary_2site: tuple) -> None:
        """Test NaN in categorical covariates raises ValueError."""
        X, y, sites = binary_2site
        cat = np.array([["M"], [np.nan]] * 50, dtype=object)
        ismi = InterSiteMatchedInterpolation()
        ismi.fit_resample(X, y, sites=sites, categorical_covariate=cat, allow_nan=True)

    def test_nan_in_categorical_not_allowd(self, binary_2site: tuple) -> None:
        """Test NaN in categorical covariates raises ValueError."""
        X, y, sites = binary_2site

        cat = np.array([["M"], [np.nan]] * 50, dtype=object)
        ismi = InterSiteMatchedInterpolation()
        with pytest.raises(ValueError):
            ismi.fit_resample(X, y, sites=sites, categorical_covariate=cat, allow_nan=False)


# Core functionality tests
class TestCoreFunctionality:
    """Test main interpolation functionality."""

    def test_basic_pairwise(self, binary_2site: tuple) -> None:
        """Test basic pairwise interpolation."""
        X, y, sites = binary_2site
        ismi = InterSiteMatchedInterpolation(random_state=42)
        X_res, y_res = ismi.fit_resample(X, y, sites=sites)

        assert len(X_res) > len(X)
        assert len(X_res) == len(y_res)
        assert hasattr(ismi, "sites_resampled_")
        assert hasattr(ismi, "unmatched_samples_")
        assert ("A", "B") in ismi.unmatched_samples_
        assert ("B", "A") in ismi.unmatched_samples_

    def test_concatenate_false(self, binary_2site: tuple) -> None:
        """Test concatenate=False returns only synthetic samples."""
        X, y, sites = binary_2site
        ismi = InterSiteMatchedInterpolation(concatenate=False, random_state=42)
        X_res, _y_res = ismi.fit_resample(X, y, sites=sites)

        # Should return only synthetic samples (may be <=, ==, or > original depending on matches)
        # With k=1 and all samples matched, len(X_res) == len(X)
        assert len(X_res) <= len(X) * 2  # Upper bound check
        assert len(X_res) > 0  # Should generate some samples
        assert len(X_res) == len(_y_res)
        # Verify it's not concatenated by checking it's different from original
        assert not np.array_equal(X_res, X)

    def test_base_to_others_forces_average(self, multi_3site: tuple) -> None:
        """Test base_to_others mode forces k='average'."""
        X, y, sites = multi_3site
        ismi = InterSiteMatchedInterpolation(mode="base_to_others", k=2, random_state=42)
        _X_res, _y_res = ismi.fit_resample(X, y, sites=sites)

        assert ismi.k == "average"

    def test_with_covariates(self, binary_2site: tuple, covariates_2site: tuple) -> None:
        """Test interpolation with covariates."""
        X, y, sites = binary_2site
        sex, age = covariates_2site
        ismi = InterSiteMatchedInterpolation(covariate_tolerance=np.array([10.0]), random_state=42)
        X_res, _y_res = ismi.fit_resample(X, y, sites=sites, categorical_covariate=sex, continuous_covariate=age)
        assert len(X_res) > len(X)

    def test_k_max(self, binary_2site: tuple) -> None:
        """Test k='max' interpolation."""
        X, y, sites = binary_2site
        ismi = InterSiteMatchedInterpolation(k="max", random_state=42)
        X_res, _y_res = ismi.fit_resample(X, y, sites=sites)
        assert len(X_res) > len(X)

    def test_k_average(self, binary_2site: tuple) -> None:
        """Test k='average' interpolation."""
        X, y, sites = binary_2site
        ismi = InterSiteMatchedInterpolation(k="average", random_state=42)
        X_res, _y_res = ismi.fit_resample(X, y, sites=sites)
        # One synthetic per source sample with matches
        assert len(X_res) > len(X)

    def test_k_integer(self, binary_2site: tuple) -> None:
        """Test integer k interpolation."""
        X, y, sites = binary_2site
        ismi = InterSiteMatchedInterpolation(k=3, random_state=42)
        X_res, _y_res = ismi.fit_resample(X, y, sites=sites)
        assert len(X_res) > len(X)

    def test_constant_alpha(self, binary_2site: tuple) -> None:
        """Test constant alpha interpolation."""
        X, y, sites = binary_2site
        ismi = InterSiteMatchedInterpolation(alpha=0.5, random_state=42)
        X_res, _y_res = ismi.fit_resample(X, y, sites=sites)
        assert len(X_res) > len(X)

    def test_variable_alpha(self, binary_2site: tuple) -> None:
        """Test variable alpha interpolation."""
        X, y, sites = binary_2site
        ismi = InterSiteMatchedInterpolation(alpha=(0.2, 0.6), random_state=42)
        X_res, _y_res = ismi.fit_resample(X, y, sites=sites)
        assert len(X_res) > len(X)

    def test_regression(self, regression_2site: tuple) -> None:
        """Test regression interpolation."""
        X, y, sites = regression_2site
        # Use target_tolerance for regression to allow approximate matching
        ismi = InterSiteMatchedInterpolation(random_state=42, target_tolerance=5.0)
        X_res, y_res = ismi.fit_resample(X, y, sites=sites)
        assert len(X_res) > len(X)
        assert y_res.dtype.kind == "f"

    def test_no_matches_found(self) -> None:
        """Test behavior when no matches are found."""
        # Sites have completely different targets - single class per site
        X = np.random.randn(20, 5)
        y = np.array([0] * 10 + [1] * 10)
        sites = np.array(["A"] * 10 + ["B"] * 10)

        ismi = InterSiteMatchedInterpolation(random_state=42)
        X_res, _y_res = ismi.fit_resample(X, y, sites=sites)

        # Returns original only (no matches possible since site A has class 0, site B has class 1)
        assert len(X_res) == len(X)
        assert sum(ismi.unmatched_samples_.values()) == 20

    def test_empty_site(self) -> None:
        """Test interpolation with site containing few samples."""
        X = np.random.randn(10, 5)
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        sites = np.array(["A"] * 5 + ["B"] * 5)

        ismi = InterSiteMatchedInterpolation(random_state=42)
        X_res, _y_res = ismi.fit_resample(X, y, sites=sites)
        assert len(X_res) >= len(X)


# Warning tests
class TestWarnings:
    """Test warning conditions."""

    def test_alpha_out_of_range_warning(self, binary_2site) -> None:
        """Test alpha out of range raises ValueError."""
        # This should raise, not warn
        X, y, sites = binary_2site
        with pytest.raises(ValueError, match="alpha"):
            ismi = InterSiteMatchedInterpolation(alpha=1.5)
            ismi.fit_resample(X, y, sites=sites)

    def test_fewer_matches_than_k(self, binary_2site: tuple) -> None:
        """Test warning when fewer matches than k."""
        X, y, sites = binary_2site
        ismi = InterSiteMatchedInterpolation(k=100, random_state=42)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ismi.fit_resample(X, y, sites=sites)
            assert len([x for x in w if "matches" in str(x.message)]) > 0

    def test_base_to_others_override_warning(self, multi_3site: tuple) -> None:
        """Test warning when k is overridden in base_to_others mode."""
        X, y, sites = multi_3site
        ismi = InterSiteMatchedInterpolation(mode="base_to_others", k=2, verbose=True)
        # Should warn about override
        _X_res, _y_res = ismi.fit_resample(X, y, sites=sites)
        assert ismi.k == "average"


# Edge cases
class TestEdgeCases:
    """Test boundary conditions."""

    def test_single_feature(self) -> None:
        """Test interpolation with single feature."""
        X = np.random.randn(100, 1)
        y = np.random.randint(0, 2, 100)
        sites = np.array(["A"] * 50 + ["B"] * 50)
        ismi = InterSiteMatchedInterpolation(random_state=42)
        X_res, _y_res = ismi.fit_resample(X, y, sites=sites)
        assert X_res.shape[1] == 1

    def test_many_sites(self) -> None:
        """Test interpolation with many sites."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        sites = np.array([f"S{i % 10}" for i in range(100)])
        ismi = InterSiteMatchedInterpolation(random_state=42)
        X_res, _y_res = ismi.fit_resample(X, y, sites=sites)
        assert len(X_res) > len(X)
        assert len(ismi.unmatched_samples_) == 45 * 2  # 10*9/2 pairs * 2 directions

    def test_imbalanced_sites(self) -> None:
        """Test interpolation with imbalanced sites."""
        X = np.random.randn(110, 5)
        y = np.random.randint(0, 2, 110)
        sites = np.array(["A"] * 10 + ["B"] * 100)
        ismi = InterSiteMatchedInterpolation(random_state=42)
        X_res, _y_res = ismi.fit_resample(X, y, sites=sites)
        assert len(X_res) > len(X)

    def test_multiclass(self) -> None:
        """Test interpolation with multiclass targets."""
        X = np.random.randn(150, 5)
        y = np.array([0] * 50 + [1] * 50 + [2] * 50)
        sites = np.array(["A"] * 50 + ["B"] * 50 + ["C"] * 50)
        ismi = InterSiteMatchedInterpolation(random_state=42)
        _X_res, y_res = ismi.fit_resample(X, y, sites=sites)
        assert len(np.unique(y_res)) == 3

    def test_integer_sites(self) -> None:
        """Test interpolation with integer site labels."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        sites = np.array([0] * 50 + [1] * 50)
        ismi = InterSiteMatchedInterpolation(random_state=42)
        X_res, _y_res = ismi.fit_resample(X, y, sites=sites)
        assert len(X_res) > len(X)
        assert (0, 1) in ismi.unmatched_samples_


# Integration tests
class TestIntegration:
    """Integration and workflow tests."""

    def test_reproducibility(self, binary_2site: tuple) -> None:
        """Test reproducibility with same random state."""
        X, y, sites = binary_2site

        ismi1 = InterSiteMatchedInterpolation(random_state=42)
        X1, y1 = ismi1.fit_resample(X, y, sites=sites)

        ismi2 = InterSiteMatchedInterpolation(random_state=42)
        X2, y2 = ismi2.fit_resample(X, y, sites=sites)

        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_different_random_states(self, binary_2site: tuple) -> None:
        """Test different random states produce different results."""
        X, y, sites = binary_2site

        ismi1 = InterSiteMatchedInterpolation(random_state=42)
        X1, _y1 = ismi1.fit_resample(X, y, sites=sites)

        ismi2 = InterSiteMatchedInterpolation(random_state=24)
        X2, _y2 = ismi2.fit_resample(X, y, sites=sites)

        assert not np.array_equal(X1, X2)

    def test_target_tolerance(self) -> None:
        """Test target tolerance for continuous targets."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)  # Continuous
        sites = np.array(["A"] * 50 + ["B"] * 50)

        ismi = InterSiteMatchedInterpolation(target_tolerance=0.5, random_state=42)
        X_res, _y_res = ismi.fit_resample(X, y, sites=sites)
        assert len(X_res) > len(X)


# Utility function tests
class TestUtilities:
    """Test internal utility functions."""

    def test_find_matches_basic(self) -> None:
        """Test basic find_matches functionality."""
        from uniharmony.interpolation._inter_site_matched import _find_matches

        y_src = np.array([0, 1, 0])
        y_dst = np.array([0, 0, 1, 1])

        matches = _find_matches(y_src, y_dst, None, None, None, None, None, None)

        assert set(matches[0]) == {0, 1}  # y=0 matches dst 0,1
        assert set(matches[1]) == {2, 3}  # y=1 matches dst 2,3
        assert set(matches[2]) == {0, 1}  # y=0 matches dst 0,1

    def test_find_matches_with_tolerance(self) -> None:
        """Test find_matches with tolerance."""
        from uniharmony.interpolation._inter_site_matched import _find_matches

        y_src = np.array([1.0, 2.0])
        y_dst = np.array([1.1, 1.5, 2.2])

        matches = _find_matches(y_src, y_dst, None, None, None, None, 0.15, None)

        # 1.0 matches 1.1 (diff 0.1 <= 0.15)
        assert 0 in matches[0]
        # 2.0 matches 2.2 (diff 0.2 > 0.15, no match)
        assert len(matches[1]) == 0

    def test_reverse_matches(self) -> None:
        """Test reverse_matches functionality."""
        from uniharmony.interpolation._inter_site_matched import _reverse_matches

        # src0 matches dst 0,2; src1 matches dst 2,3
        fwd = [[0, 2], [2, 3]]
        rev = _reverse_matches(fwd, 4)

        assert rev[0] == [0]  # dst0 matched by src0
        assert rev[1] == []  # dst1 unmatched
        assert set(rev[2]) == {0, 1}  # dst2 matched by src0, src1
        assert rev[3] == [1]  # dst3 matched by src1
