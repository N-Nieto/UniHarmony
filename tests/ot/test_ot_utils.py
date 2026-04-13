"""Tests suite for OptimalTransportDomainAdaptation utilities."""

import numpy as np
import pytest
from ot.da import EMDLaplaceTransport, EMDTransport, SinkhornL1l2Transport, SinkhornTransport

from uniharmony.ot._utils import create_ot_object, data_consistency_check


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
