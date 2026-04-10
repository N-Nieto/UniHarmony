"""Tests test suite for OTDA and utilities."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from ot.da import EMDLaplaceTransport, EMDTransport, SinkhornL1l2Transport, SinkhornTransport
from sklearn.datasets import make_classification

from uniharmony.datasets import make_multisite_classification
from uniharmony.interpolation import IntraSiteInterpolation
from uniharmony.ot._otda import OTDA
from uniharmony.ot._utils import create_ot_object, data_consistency_check


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_data():
    """Create simple 2D data with 2 sites for basic tests."""
    X, y, sites = make_multisite_classification()
    return X, sites, y


@pytest.fixture
def multi_site_data():
    """Create data with multiple sites."""
    X, y, sites = make_multisite_classification(n_sites=4)
    return X, sites, y


# @pytest.fixture
# def str_sites_data():
#     """Create data with string site labels."""
#     np.random.seed(42)
#     X = np.random.randn(100, 5)
#     sites = np.array(["site_A"] * 50 + ["site_B"] * 50)
#     y = np.random.randint(0, 2, 100)
#     return X, sites, y


# @pytest.fixture
# def otda_instances():
#     """Return various OTDA configurations for testing."""
#     return [
#         OTDA(ot_method="emd"),
#         OTDA(ot_method="sinkhorn", reg=0.1),
#         OTDA(ot_method="sinkhorn_gl", reg=0.1, eta=0.01),
#         OTDA(ot_method="emd_laplace"),
#     ]


# =============================================================================
# Test create_ot_object
# =============================================================================


def test_create_emd():
    """Test EMD transport creation."""
    obj = create_ot_object("emd")
    assert isinstance(obj, EMDTransport)


def test_create_sinkhorn():
    """Test Sinkhorn transport creation."""
    obj = create_ot_object("sinkhorn", reg_e=0.5)
    assert isinstance(obj, SinkhornTransport)
    assert obj.reg_e == 0.5


def test_create_sinkhorn_gl():
    """Test Sinkhorn Group Lasso creation."""
    obj = create_ot_object("sinkhorn_gl", reg_e=0.1, reg_cl=0.01)
    assert isinstance(obj, SinkhornL1l2Transport)


def test_create_emd_laplace():
    """Test EMD Laplace creation."""
    obj = create_ot_object("emd_laplace")
    assert isinstance(obj, EMDLaplaceTransport)


def test_case_insensitive():
    """Test that method names are case insensitive."""
    obj1 = create_ot_object("EMD")
    obj2 = create_ot_object("EmD")
    obj3 = create_ot_object("emd")
    assert isinstance(obj1, EMDTransport)
    assert isinstance(obj2, EMDTransport)
    assert isinstance(obj3, EMDTransport)


def test_invalid_method():
    """Test error on invalid method."""
    with pytest.raises(ValueError, match="Unsupported OT method"):
        create_ot_object("invalid_method")


def test_kwargs_passing():
    """Test that kwargs are properly passed."""
    obj = create_ot_object("sinkhorn", reg_e=0.5, metric="sqeuclidean", norm="median", max_iter=100)
    assert obj.reg_e == 0.5
    assert obj.metric == "sqeuclidean"
    assert obj.norm == "median"
    assert obj.max_iter == 100


def test_valid_data_no_labels():
    """Test with valid data, no labels."""
    X_source = np.random.randn(50, 5)
    X_target = np.random.randn(30, 5)
    # Should not raise
    data_consistency_check(X_source, X_target)


def test_valid_data_with_labels():
    """Test with valid data and labels."""
    X_source = np.random.randn(50, 5)
    X_target = np.random.randn(30, 5)
    y_source = np.random.randint(0, 2, 50)
    y_target = np.random.randint(0, 2, 30)
    # Should not raise
    data_consistency_check(X_source, X_target, y_source, y_target)


def test_mismatched_source_samples():
    """Test error when X_source and y_source have different lengths."""
    X_source = np.random.randn(50, 5)
    X_target = np.random.randn(30, 5)
    y_source = np.random.randint(0, 2, 40)  # Wrong length
    with pytest.raises(RuntimeError, match="Mismatch in source samples"):
        data_consistency_check(X_source, X_target, y_source)


def test_mismatched_target_samples():
    """Test error when X_target and y_target have different lengths."""
    X_source = np.random.randn(50, 5)
    X_target = np.random.randn(30, 5)
    y_target = np.random.randint(0, 2, 20)  # Wrong length
    with pytest.raises(RuntimeError, match="Mismatch in target samples"):
        data_consistency_check(X_source, X_target, y_target=y_target)


def test_source_size_validation():
    """Test per-class sample count validation."""
    X_source = np.random.randn(100, 5)
    X_target = np.random.randn(50, 5)
    y_source = np.array([0] * 60 + [1] * 40)

    # Should not raise
    data_consistency_check(
        X_source,
        X_target,
        y_source,
    )


# =============================================================================
# Test OTDA Initialization
# =============================================================================
def test_default_init():
    """Test default initialization."""
    otda = OTDA()
    assert otda.ot_method == "emd"
    assert otda.metric == "euclidean"
    assert otda.reg == 1.0
    assert otda.eta == 0.1
    assert otda.cost_supervised is True


def test_init_with_string_method():
    """Test initialization with string method."""
    otda = OTDA(ot_method="sinkhorn", reg=0.5)
    assert otda.ot_method == "sinkhorn"
    assert otda.reg == 0.5


def test_init_with_ot_instance():
    """Test initialization with pre-configured OT instance."""
    ot_solver = SinkhornTransport(reg_e=0.5, metric="sqeuclidean")
    otda = OTDA(ot_method=ot_solver)
    assert otda.ot_method is ot_solver


def test_init_with_wrong_ot_instance(simple_data):
    """Test initialization with pre-configured OT instance."""
    ot_solver = IntraSiteInterpolation()
    otda = OTDA(ot_method=ot_solver)
    X, sites, y = simple_data
    with pytest.raises(TypeError):
        otda.fit(X, sites, ref_site=1, y=y)
    assert otda.ot_method is ot_solver


def test_init_with_all_params():
    """Test initialization with all parameters."""
    otda = OTDA(
        ot_method="emd_laplace",
        metric="cityblock",
        reg=0.5,
        eta=0.2,
        max_iter=50,
        cost_norm="median",
        limit_max=20,
        cost_supervised=False,
    )
    assert otda.metric == "cityblock"
    assert otda.cost_norm == "median"
    assert otda.cost_supervised is False


# =============================================================================
# Test OTDA _validate_sites
# =============================================================================

# def test_single_string_site(simple_data):
#     """Test validation with single string reference site."""
#     _, sites, _ = simple_data
#     otda = OTDA()
#     ref_mask, harm_mask = otda._validate_sites(sites, "site_A")

#     assert np.sum(ref_mask) == 50
#     assert np.sum(harm_mask) == 50
#     assert np.all(sites[ref_mask] == "site_A")
#     assert np.all(sites[harm_mask] == "site_B")


def test_single_integer_site(simple_data):
    """Test validation with integer reference site."""
    _, sites, _ = simple_data
    otda = OTDA()
    ref_mask, harm_mask = otda._validate_sites(sites, 0)

    assert np.sum(ref_mask) == 500
    assert np.sum(harm_mask) == 500
    assert np.all(sites[ref_mask] == 0)
    assert np.all(sites[harm_mask] == 1)


def test_list_of_sites(multi_site_data):
    """Test validation with list of reference sites."""
    _, sites, _ = multi_site_data
    otda = OTDA()
    ref_mask, harm_mask = otda._validate_sites(sites, [1, 2])

    assert np.sum(ref_mask) == 500  # A + B
    assert np.sum(harm_mask) == 500  # C + D


def test_list_of_integer_sites():
    """Test validation with list of integer reference sites."""
    sites = np.array([0, 0, 1, 1, 2, 2])
    otda = OTDA()
    ref_mask, harm_mask = otda._validate_sites(sites, [0, 1])

    assert np.sum(ref_mask) == 4  # 0s and 1s
    assert np.sum(harm_mask) == 2  # 2s


def test_mixed_type_list():
    """Test validation with mixed type list (should convert to strings)."""
    sites = np.array(["0", "1", "2", "3"])  # String sites
    otda = OTDA()
    ref_mask, harm_mask = otda._validate_sites(sites, [0, 1])  # Int ref

    assert np.sum(ref_mask) == 2
    assert np.sum(harm_mask) == 2


def test_missing_site_error(simple_data):
    """Test error when reference site doesn't exist."""
    _, sites, _ = simple_data
    otda = OTDA()

    with pytest.raises(ValueError):
        otda._validate_sites(sites, "nonexistent_site")


def test_missing_site_in_list_error(multi_site_data):
    """Test error when one reference site in list doesn't exist."""
    _, sites, _ = multi_site_data
    otda = OTDA()

    with pytest.raises(ValueError):
        otda._validate_sites(sites, ["A", "Z"])  # Z doesn't exist


def test_no_reference_samples_error():
    """Test error when no reference samples found."""
    sites = np.array(["A", "B", "C"])
    otda = OTDA()

    with pytest.raises(ValueError):
        otda._validate_sites(sites, "Z")


# =============================================================================
# Test OTDA Fit


def test_basic_fit(simple_data):
    """Test basic fit operation."""
    X, y, sites = make_multisite_classification(n_sites=3)
    otda = OTDA()

    result = otda.fit(X, sites, ref_site=1, y=y)
    assert result is otda  # Returns self
    assert otda._is_fitted is True
    assert hasattr(otda, "ot_obj_")
    assert hasattr(otda, "coupling_")


def test_fit_with_integer_sites(simple_data):
    """Test fit with integer site labels."""
    X, sites, y = simple_data
    otda = OTDA()

    otda.fit(X, sites, ref_site=0, y=y)
    assert otda._is_fitted is True


def test_fit_with_list_ref_site():
    """Test fit with multiple reference sites."""
    X, y, sites = make_multisite_classification(n_sites=3)
    otda = OTDA()

    otda.fit(X, sites, ref_site=[0, 1], y=y)
    assert otda._is_fitted is True


def test_fit_without_labels(simple_data):
    """Test unsupervised fit without labels."""
    X, _, sites = make_multisite_classification()
    otda = OTDA(cost_supervised=False)

    otda.fit(X, sites, ref_site=1, y=None)
    assert otda._is_fitted is True


def test_fit_cost_supervised_false_with_labels(simple_data):
    """Test that labels are ignored when cost_supervised=False."""
    X, y, sites = make_multisite_classification()
    otda = OTDA(cost_supervised=False)

    # Should use unsupervised fit even with labels provided
    otda.fit(X, sites, ref_site=1, y=y)
    assert otda._is_fitted is True


def test_fit_stores_attributes(simple_data):
    """Test that fit stores expected attributes."""
    X, y, sites = simple_data
    otda = OTDA(ot_method="emd")

    otda.fit(X, sites, ref_site=1, y=y)

    assert otda.ref_site_ == 1
    assert otda.ot_obj_ is not None
    assert otda.coupling_ is not None
    assert otda.coupling_.shape == (500, 500)  # harm x ref


def test_fit_all_ot_methods():
    """Test fit with all supported OT methods."""
    X, y, sites = make_multisite_classification()

    methods = ["emd", "sinkhorn", "s", "sinkhorn_gl"]
    for method in methods:
        otda = OTDA(ot_method=method, reg=0.1, eta=0.01)
        otda.fit(X, sites, ref_site=1, y=y)
        assert otda._is_fitted, f"Failed for method {method}"


def test_fit_with_preconfigured_ot(simple_data):
    """Test fit with pre-configured OT instance."""
    X, sites, y = simple_data
    ot_solver = SinkhornTransport(reg_e=0.5, max_iter=100)
    otda = OTDA(ot_method=ot_solver)

    otda.fit(X, sites, ref_site=1, y=y)
    assert otda._is_fitted is True
    assert otda.ot_obj_ is ot_solver


# =============================================================================
# Test OTDA Transform
# =============================================================================
def test_transform_not_fitted_error(simple_data):
    """Test error when transforming before fitting."""
    X, _, _ = simple_data
    otda = OTDA()

    with pytest.raises(RuntimeError, match="not fitted yet"):
        otda.transform(X)


def test_basic_transform(simple_data):
    """Test basic transform."""
    X, sites, y = simple_data
    otda = OTDA()
    otda.fit(X, sites, ref_site=1, y=y)

    # Transform harmonization data
    X_transformed = otda.transform(X, sites=sites)

    assert X_transformed.shape == X.shape


def test_transform_batch_size(simple_data):
    """Test transform with different batch sizes."""
    X, sites, y = simple_data
    otda = OTDA()
    otda.fit(X, sites, ref_site=1, y=y)

    # Test different batch sizes
    for batch_size in [16, 32, 64, 128]:
        X_trans = otda.transform(X=X, sites=sites, batch_size=batch_size)
        assert X_trans.shape == X.shape


# =============================================================================
# Test OTDA Fit-Transform
# =============================================================================


def test_fit_transform(simple_data):
    """Test fit_transform in one call."""
    X, sites, y = simple_data
    otda = OTDA()

    X_transformed = otda.fit_transform(X, sites, ref_site=1, y=y)

    assert otda._is_fitted is True
    assert X_transformed.shape == X.shape


def test_fit_transform_equivalence(simple_data):
    """Test that fit_transform equals fit then transform."""
    X, sites, y = simple_data
    np.random.seed(42)
    otda1 = OTDA(ot_method="emd")

    X_ft = otda1.fit_transform(X, sites, ref_site=1, y=y)

    np.random.seed(42)
    otda2 = OTDA(ot_method="emd")
    otda2.fit(X, sites, ref_site=1, y=y)
    X_f_t = otda2.transform(X, sites=sites)

    assert_allclose(X_ft, X_f_t)


# =============================================================================
# Test OTDA Inverse Transform
# =============================================================================


def test_inverse_transform_not_fitted_error(simple_data):
    """Test error when inverse transforming before fitting."""
    X, _, _ = simple_data
    otda = OTDA()

    with pytest.raises(RuntimeError, match="not fitted yet"):
        otda.inverse_transform(X)


def test_inverse_transform(simple_data):
    """Test basic inverse transform."""
    X, sites, y = simple_data
    otda = OTDA()
    otda.fit(X, sites, ref_site=1, y=y)

    X_ref = X[sites == 1]
    X_inv = otda.inverse_transform(X_ref)

    assert X_inv.shape == X_ref.shape


# =============================================================================
# Test OTDA Properties
# =============================================================================


def test_coupling_property(simple_data):
    """Test coupling property access."""
    X, sites, y = simple_data
    otda = OTDA()

    otda.fit(X, sites, ref_site=1, y=y)
    assert otda.coupling_ is not None


def test_cost_matrix_property(simple_data):
    """Test cost_matrix property access."""
    X, sites, y = simple_data
    otda = OTDA()

    otda.fit(X, sites, ref_site=1, y=y)
    # Cost might be None depending on OT method
    if hasattr(otda, "cost_"):
        assert otda.cost_ is not None


# =============================================================================
# Integration Tests
# =============================================================================
def test_full_workflow_emd():
    """Test complete workflow with EMD."""
    np.random.seed(42)
    X, y = make_classification(n_samples=200, n_features=10, n_classes=2)
    sites = np.array([1] * 100 + [2] * 100)

    otda = OTDA(ot_method="emd")
    otda.fit(X, sites, ref_site=1, y=y)
    X_aligned = otda.transform(X[sites == 2])

    assert X_aligned.shape == (100, 10)


def test_full_workflow_sinkhorn():
    """Test complete workflow with Sinkhorn."""
    np.random.seed(42)
    X, y, sites = make_multisite_classification()

    otda = OTDA(ot_method="sinkhorn", reg=0.1, metric="sqeuclidean")
    otda.fit(X, sites, ref_site=1, y=y)
    X_aligned = otda.fit_transform(X, sites, ref_site=1, y=y)

    assert X_aligned.shape == (1000, 10)


def test_multi_site_harmonization():
    """Test harmonizing multiple sites to one reference."""
    np.random.seed(42)
    X = np.random.randn(300, 5)
    sites = np.array([1] * 100 + [2] * 100 + [3] * 100)
    y = np.random.randint(0, 2, 300)

    otda = OTDA(ot_method="sinkhorn", reg=0.1)
    otda.fit(X, sites, ref_site=1, y=y)

    # Transform each site separately
    X_site1 = otda.transform(X[sites == 1])
    X_site2 = otda.transform(X[sites == 2])

    assert X_site1.shape == (100, 5)
    assert X_site2.shape == (100, 5)


def test_consistency_across_runs():
    """Test that same seed gives consistent results."""
    X, y, sites = make_multisite_classification()
    otda1 = OTDA(ot_method="emd")
    otda1.fit(X, sites, ref_site=1, y=y)
    trans1 = otda1.transform(X, sites=sites)

    otda2 = OTDA(ot_method="emd")
    otda2.fit(X, sites, ref_site=1, y=y)
    trans2 = otda2.transform(X, sites=sites)

    assert_allclose(trans1, trans2)


def test_sklearn_tags():
    """Test initialization with pre-configured OT instance."""
    otda = OTDA()
    otda.__sklearn_tags__()
