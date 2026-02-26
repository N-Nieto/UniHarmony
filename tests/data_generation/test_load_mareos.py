"""Test suite for MAREoS dataset loading functions."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uniharmony.datasets import load_MAREoS
from uniharmony.datasets._load_mareos import (
    _ensure_mareos_data,
    _load_mareos_single_dataset,
)


def test_load_mareos_success():
    """Test basic functionality."""
    load_MAREoS()


def test_force_download(tmp_path):
    """Force download funtionality."""
    _ = load_MAREoS(force_download=True)
    _ = load_MAREoS(force_download=False)
    data_dir = Path(tmp_path) / "test_dir"
    _ = load_MAREoS(data_dir=data_dir, force_download=True)
    _ = load_MAREoS(data_dir=data_dir, force_download=False)


def test_verbose(tmp_path):
    """Force download funtionality."""
    data_dir = Path(tmp_path) / "test_dir"
    _ = load_MAREoS(data_dir=data_dir, force_download=True, verbose=True)


def test_invalid_load_params(tmp_path):
    """Force download funtionality."""
    data_dir = Path(tmp_path) / "test_dir"
    with pytest.raises(TypeError):
        _ = load_MAREoS(data_dir=1)  # type: ignore
    with pytest.raises(TypeError):
        _ = load_MAREoS(data_dir=data_dir, effect_examples=1)  # type: ignore
    with pytest.raises(ValueError):
        _ = load_MAREoS(data_dir=data_dir, effect_examples="wrong")  # type: ignore
    # Should be a list
    _ = load_MAREoS(data_dir=data_dir, effects="eos")  # type: ignore


def test_ensure_mareos_data_custom_dir(tmp_path):
    """Test using custom data directory."""
    data_dir = Path(tmp_path) / "test_dir"
    result_dir = _ensure_mareos_data(data_dir=data_dir)
    assert result_dir == data_dir / "MAREoS"
    assert data_dir.exists()


# ============================================================================
# Test _load_MAREoS_single_dataset
# ============================================================================
def test_load_mareos_single_dataset_validation():
    """Test parameter validation for single dataset loading."""
    # Test invalid effect
    data_dir = _ensure_mareos_data()
    with pytest.raises(FileNotFoundError):
        _load_mareos_single_dataset(
            data_dir,
            effect="invalid",
            effect_type="simple",
            effect_example="1",
        )

    # Test invalid effect_type
    with pytest.raises(FileNotFoundError):
        _load_mareos_single_dataset(
            data_dir,
            effect="eos",
            effect_type="invalid",
            effect_example="1",
        )

    # Test invalid effect_example
    with pytest.raises(FileNotFoundError):
        _load_mareos_single_dataset(
            data_dir,
            effect="invalid",
            effect_type="invalid",
            effect_example="1",
        )


def test_load_mareos_single_dataset_success():
    """Test successful loading of single dataset."""
    # Create mock data
    data_dir = _ensure_mareos_data()
    X, y, sites, covs, folds = _load_mareos_single_dataset(
        data_dir,
        effect="eos",
        effect_type="simple",
        effect_example="1",
    )
    # Verify outputs
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(sites, np.ndarray)
    assert isinstance(covs, np.ndarray)
    assert isinstance(folds, np.ndarray)

    assert X.shape == (1001, 14)
    assert y.shape == (1001,)
    assert sites.shape == (1001,)
    assert covs.shape == (1001, 2)
    assert folds.shape == (1001,)


def test_load_mareos_single_dataset_dataframe_funtionality():
    """Test load data as dataframe."""
    # Create mock data
    data_dir = _ensure_mareos_data()
    # Test with pandas output
    X_df, y_series, sites_series, covs_df, folds_series = (
        _load_mareos_single_dataset(
            data_dir=data_dir,
            effect="eos",
            effect_type="simple",
            effect_example="1",
            as_numpy=False,
        )
    )

    assert isinstance(X_df, pd.DataFrame)
    assert isinstance(y_series, pd.Series)
    assert isinstance(sites_series, pd.Series)
    assert isinstance(covs_df, pd.DataFrame)
    assert isinstance(folds_series, pd.Series)


def test_load_mareos_single_dataset_no_datadir():
    """No dir error handling handling."""
    # Create mock data
    with pytest.raises(TypeError):
        _, _, _, _, _ = _load_mareos_single_dataset(
            data_dir="no_dir",  # type: ignore
            effect="eos",
            effect_type="simple",
            effect_example="1",
        )
    with pytest.raises(RuntimeError):
        _, _, _, _, _ = _load_mareos_single_dataset(
            data_dir=Path("no_dir"),  # type: ignore
            effect="eos",
            effect_type="simple",
            effect_example="1",
        )


def test_load_mareos_single_dataset_verbose():
    """Test Verbose."""
    data_dir = _ensure_mareos_data()
    _, _, _, _, _ = _load_mareos_single_dataset(
        data_dir=data_dir,  # type: ignore
        effect="eos",
        effect_type="simple",
        effect_example="1",
        verbose=True,
    )

    _, _, _, _, _ = _load_mareos_single_dataset(
        data_dir=data_dir,  # type: ignore
        effect="eos",
        effect_type="simple",
        effect_example="1",
        verbose=False,
    )
