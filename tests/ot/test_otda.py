"""Tests test suite for OptimalTransportDomainAdaptation."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from ot.da import SinkhornTransport
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from uniharmony.datasets import make_multisite_classification
from uniharmony.interpolation import IntraSiteInterpolation
from uniharmony.ot._otda import OptimalTransportDomainAdaptation


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_data():
    """Create simple 2D data with 2 sites for basic tests."""
    X, y, sites = make_multisite_classification()
    return X, y, sites


@pytest.fixture
def multi_site_data():
    """Create data with multiple sites."""
    X, y, sites = make_multisite_classification(n_sites=4)
    return X, y, sites


@pytest.fixture
def str_sites_data():
    """Create data with string site labels."""
    rng = np.random.RandomState(42)
    X = np.random.randn(100, 5)
    sites = np.array(["site_A"] * 50 + ["site_B"] * 50)
    y = rng.randint(0, 2, 100)
    return X, y, sites


# =============================================================================
# Test OptimalTransportDomainAdaptation Initialization
# =============================================================================
# @pytest.mark.parametrize(
#     "ot_method,reg,eta,expected",
#     [
#         pytest.param(
#             None,
#             None,
#             None,
#             None,
#             None
#         ),
#         # pytest.param(datetime(2001, 12, 11), datetime(2001, 12, 12), timedelta(-1), id="backward"),
#     ],
# )
# def test_init(ot_method, metric, reg, eta, expected):
#     """Test initialization."""
#     otda = OptimalTransportDomainAdaptation(ot_method, metric, reg, eta)
#     assert otda.ot_method == ot_method
#     assert otda.metric == metric
#     assert otda.reg == reg
#     assert otda.eta == eta


def test_default_init():
    """Test default initialization."""
    otda = OptimalTransportDomainAdaptation()
    assert otda.ot_method == "emd"
    assert otda.metric == "euclidean"
    assert otda.reg == 1.0
    assert otda.eta == 0.1


def test_init_with_string_method():
    """Test initialization with string method."""
    otda = OptimalTransportDomainAdaptation(ot_method="sinkhorn", reg=0.5)
    assert otda.ot_method == "sinkhorn"
    assert otda.reg == 0.5


def test_init_with_ot_instance():
    """Test initialization with pre-configured OT instance."""
    ot_solver = SinkhornTransport(reg_e=0.5, metric="sqeuclidean")
    otda = OptimalTransportDomainAdaptation(ot_method=ot_solver)
    assert otda.ot_method is ot_solver


def test_init_with_wrong_ot_instance(simple_data):
    """Test initialization with pre-configured OT instance."""
    ot_solver = IntraSiteInterpolation()
    otda = OptimalTransportDomainAdaptation(ot_method=ot_solver)
    X, y, sites = simple_data
    with pytest.raises(TypeError):
        otda.fit(X, sites, ref_site=1, y=y)
    assert otda.ot_method is ot_solver


def test_init_with_all_params():
    """Test initialization with all parameters."""
    otda = OptimalTransportDomainAdaptation(
        ot_method="emd_laplace",
        metric="cityblock",
        reg=0.5,
        eta=0.2,
        max_iter=50,
        cost_norm="median",
        limit_max=20,
    )
    assert otda.metric == "cityblock"
    assert otda.cost_norm == "median"


# =============================================================================
# Test OptimalTransportDomainAdaptation _validate_sites
# =============================================================================


def test_single_string_site(str_sites_data):
    """Test validation with single string reference site."""
    X, _, sites = str_sites_data
    otda = OptimalTransportDomainAdaptation()
    X_ref, X_harm, _, _ = otda._split_ref_harm_data(X, sites, "site_A")

    assert np.shape(X_ref) == (50, 5)
    assert np.shape(X_harm) == (50, 5)


def test_single_integer_site(simple_data):
    """Test validation with integer reference site."""
    X, _, sites = simple_data
    otda = OptimalTransportDomainAdaptation()
    X_ref, X_harm, _, _ = otda._split_ref_harm_data(X, sites, 0)

    assert np.shape(X_ref) == (500, 10)
    assert np.shape(X_harm) == (500, 10)


def test_list_of_sites(multi_site_data):
    """Test validation with list of reference sites."""
    X, _, sites = multi_site_data
    otda = OptimalTransportDomainAdaptation()
    X_ref, X_harm, _, _ = otda._split_ref_harm_data(X, sites, [1, 2])

    assert np.shape(X_ref) == (500, 10)  # A + B
    assert np.shape(X_harm) == (500, 10)  # C + D


def test_list_of_integer_sites(multi_site_data):
    """Test validation with list of integer reference sites."""
    X, _, sites = multi_site_data
    otda = OptimalTransportDomainAdaptation()
    _, _, _, _ = otda._split_ref_harm_data(X, sites, [0, 1])


def test_missing_site_error(simple_data):
    """Test error when reference site doesn't exist."""
    X, _, sites = simple_data
    otda = OptimalTransportDomainAdaptation()

    with pytest.raises(ValueError):
        otda._split_ref_harm_data(X, sites, "nonexistent_site")


def test_missing_site_in_list_error(multi_site_data):
    """Test error when one reference site in list doesn't exist."""
    X, _, sites = multi_site_data
    otda = OptimalTransportDomainAdaptation()

    with pytest.raises(ValueError):
        otda._split_ref_harm_data(X, sites, ["A", "Z"])  # Z doesn't exist


def test_no_reference_samples_error(simple_data):
    """Test error when no reference samples found."""
    X, _, sites = simple_data
    otda = OptimalTransportDomainAdaptation()

    with pytest.raises(ValueError):
        otda._split_ref_harm_data(X, sites, "Z")


# =============================================================================
# Test OptimalTransportDomainAdaptation Fit


def test_basic_fit(simple_data):
    """Test basic fit operation."""
    X, y, sites = simple_data
    otda = OptimalTransportDomainAdaptation()

    result = otda.fit(X, sites, ref_site=1, y=y)
    assert result is otda  # Returns self
    assert hasattr(otda, "ot_obj_")
    assert hasattr(otda, "coupling_")
    check_is_fitted(otda)


def test_fit_with_integer_sites(simple_data):
    """Test fit with integer site labels."""
    X, y, sites = simple_data
    otda = OptimalTransportDomainAdaptation()

    otda.fit(X, sites, ref_site=0, y=y)
    check_is_fitted(otda)


def test_fit_with_list_ref_site(multi_site_data):
    """Test fit with multiple reference sites."""
    X, y, sites = multi_site_data
    otda = OptimalTransportDomainAdaptation()
    otda.fit(X, sites, ref_site=[0, 1], y=y)
    check_is_fitted(otda)


def test_fit_without_labels(simple_data):
    """Test unsupervised fit without labels."""
    X, _, sites = simple_data
    otda = OptimalTransportDomainAdaptation()

    otda.fit(X, sites, ref_site=1, y=None)
    check_is_fitted(otda)


def test_fit_stores_attributes(simple_data):
    """Test that fit stores expected attributes."""
    X, y, sites = simple_data
    otda = OptimalTransportDomainAdaptation(ot_method="emd")

    otda.fit(X, sites, ref_site=1, y=y)

    assert otda.ref_site_ == 1
    assert otda.ot_obj_ is not None
    assert otda.coupling_ is not None
    assert otda.coupling_.shape == (500, 500)  # harm x ref


def test_fit_all_ot_methods(simple_data):
    """Test fit with all supported OT methods."""
    X, y, sites = simple_data

    methods = ["emd", "sinkhorn", "s", "sinkhorn_gl"]
    for method in methods:
        otda = OptimalTransportDomainAdaptation(ot_method=method, reg=0.1, eta=0.01)
        otda.fit(X, sites, ref_site=1, y=y)


def test_fit_with_preconfigured_ot(simple_data):
    """Test fit with pre-configured OT instance."""
    X, y, sites = simple_data
    ot_solver = SinkhornTransport(reg_e=0.5, max_iter=100)
    otda = OptimalTransportDomainAdaptation(ot_method=ot_solver)

    otda.fit(X, sites, ref_site=1, y=y)
    assert otda.ot_obj_ is ot_solver
    check_is_fitted(otda)


# =============================================================================
# Test OptimalTransportDomainAdaptation Transform
# =============================================================================
def test_transform_not_fitted_error(simple_data):
    """Test error when transforming before fitting."""
    X, _, _ = simple_data
    otda = OptimalTransportDomainAdaptation()

    with pytest.raises(NotFittedError, match="not fitted yet"):
        otda.transform(X)


def test_basic_transform(simple_data):
    """Test basic transform."""
    X, y, sites = simple_data
    otda = OptimalTransportDomainAdaptation()
    otda.fit(X, sites, ref_site=1, y=y)

    # Transform harmonization data
    X_transformed = otda.transform(X, sites=sites)

    assert X_transformed.shape == X.shape


def test_transform_batch_size(simple_data):
    """Test transform with different batch sizes."""
    X, y, sites = simple_data
    otda = OptimalTransportDomainAdaptation()
    otda.fit(X, sites, ref_site=1, y=y)

    # Test different batch sizes
    for batch_size in [16, 32, 64, 128]:
        X_trans = otda.transform(X=X, sites=sites, batch_size=batch_size)
        assert X_trans.shape == X.shape


# =============================================================================
# Test OptimalTransportDomainAdaptation Fit-Transform
# =============================================================================


def test_fit_transform(simple_data):
    """Test fit_transform in one call."""
    X, y, sites = simple_data
    otda = OptimalTransportDomainAdaptation()

    X_transformed = otda.fit_transform(X, sites, ref_site=1, y=y)

    check_is_fitted(otda)
    assert X_transformed.shape == X.shape


def test_fit_transform_equivalence(simple_data):
    """Test that fit_transform equals fit then transform."""
    X, y, sites = simple_data
    np.random.seed(42)
    otda1 = OptimalTransportDomainAdaptation(ot_method="emd")

    X_ft = otda1.fit_transform(X, sites, ref_site=1, y=y)

    np.random.seed(42)
    otda2 = OptimalTransportDomainAdaptation(ot_method="emd")
    otda2.fit(X, sites, ref_site=1, y=y)
    X_f_t = otda2.transform(X, sites=sites)
    assert_allclose(X_ft, X_f_t)


# =============================================================================
# Test OptimalTransportDomainAdaptation Inverse Transform
# =============================================================================
def test_inverse_transform_not_fitted_error(simple_data):
    """Test error when inverse transforming before fitting."""
    X, _, _ = simple_data
    otda = OptimalTransportDomainAdaptation()

    with pytest.raises(
        NotFittedError,
        match="not fitted yet",
    ):
        otda.inverse_transform(X)


def test_inverse_transform(simple_data):
    """Test basic inverse transform."""
    X, y, sites = simple_data
    otda = OptimalTransportDomainAdaptation()
    otda.fit(X, sites, ref_site=1, y=y)

    X_ref = X[sites == 1]
    X_inv = otda.inverse_transform(X_ref)

    assert X_inv.shape == X_ref.shape


def test_inverse_transform_with_sites(simple_data):
    """Test basic inverse transform."""
    X, y, sites = simple_data
    otda = OptimalTransportDomainAdaptation()
    otda.fit(X, sites, ref_site=1, y=y)

    X_inv = otda.inverse_transform(X, sites)

    assert X_inv.shape == X.shape


# =============================================================================
# Test OptimalTransportDomainAdaptation Properties
# =============================================================================


def test_coupling_property(simple_data):
    """Test coupling property access."""
    X, y, sites = simple_data
    otda = OptimalTransportDomainAdaptation()

    otda.fit(X, sites, ref_site=1, y=y)
    assert otda.coupling_ is not None


def test_cost_matrix_property(simple_data):
    """Test cost_matrix property access."""
    X, y, sites = simple_data
    otda = OptimalTransportDomainAdaptation()

    otda.fit(X, sites, ref_site=1, y=y)
    # Cost might be None depending on OT method
    if hasattr(otda, "cost_"):
        assert otda.cost_ is not None


# =============================================================================
# Integration Tests
# =============================================================================
def test_full_workflow_emd():
    """Test complete workflow with EMD."""
    X, y = make_classification(n_samples=200, n_features=10, n_classes=2)
    sites = np.array([1] * 100 + [2] * 100)

    otda = OptimalTransportDomainAdaptation(ot_method="emd")
    otda.fit(X, sites, ref_site=1, y=y)
    X_aligned = otda.transform(X[sites == 2])

    assert X_aligned.shape == (100, 10)


def test_full_workflow_sinkhorn(simple_data):
    """Test complete workflow with Sinkhorn."""
    X, y, sites = simple_data

    otda = OptimalTransportDomainAdaptation(ot_method="sinkhorn", reg=0.1, metric="sqeuclidean")
    otda.fit(X, sites, ref_site=1, y=y)
    X_aligned = otda.fit_transform(X, sites, ref_site=1, y=y)

    assert X_aligned.shape == (1000, 10)


def test_multi_site_harmonization(simple_data):
    """Test harmonizing multiple sites to one reference."""
    X, y, sites = simple_data

    otda = OptimalTransportDomainAdaptation(ot_method="sinkhorn", reg=0.1)
    otda.fit(X, sites, ref_site=1, y=y)

    # Transform each site separately
    X_transformed = otda.transform(X, sites=sites)
    X_site0 = X_transformed[sites == 0]
    X_site1 = X_transformed[sites == 0]

    assert X_site0.shape == (500, 10)
    assert X_site1.shape == (500, 10)


def test_consistency_across_runs(simple_data):
    """Test that same seed gives consistent results."""
    X, y, sites = simple_data
    otda1 = OptimalTransportDomainAdaptation(ot_method="emd")
    otda1.fit(X, sites, ref_site=1, y=y)
    trans1 = otda1.transform(X, sites=sites)

    otda2 = OptimalTransportDomainAdaptation(ot_method="emd")
    otda2.fit(X, sites, ref_site=1, y=y)
    trans2 = otda2.transform(X, sites=sites)

    assert_allclose(trans1, trans2)


def test_sklearn_tags():
    """Test initialization with pre-configured OT instance."""
    otda = OptimalTransportDomainAdaptation()
    otda.__sklearn_tags__()
