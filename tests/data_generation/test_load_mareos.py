"""Test suite for MAREoS dataset loading functions."""

import tempfile
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


def test_force_download():
    """Force download funtionality."""
    _ = load_MAREoS(force_download=True)
    _ = load_MAREoS(force_download=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "test_dir"
        _ = load_MAREoS(data_dir=data_dir, force_download=True)
        _ = load_MAREoS(data_dir=data_dir, force_download=False)


def test_verbose():
    """Force download funtionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "test_dir"
        _ = load_MAREoS(data_dir=data_dir, force_download=True, verbose=True)


def test_invalid_load_params():
    """Force download funtionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "test_dir"
        with pytest.raises(TypeError):
            _ = load_MAREoS(data_dir=1)  # type: ignore
        with pytest.raises(TypeError):
            _ = load_MAREoS(data_dir=data_dir, effect_examples=1)  # type: ignore
        with pytest.raises(ValueError):
            _ = load_MAREoS(data_dir=data_dir, effect_examples="wrong")  # type: ignore
        # Should be a list
        _ = load_MAREoS(data_dir=data_dir, effects="eos")  # type: ignore


def test_ensure_mareos_data_custom_dir():
    """Test using custom data directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "test_dir"
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


# def test__load_MAREoS_single_dataset_wrong_shape(mock_ensure, mock_read_csv):
#     """Test validation of data shape."""
#     # Create data with wrong number of samples
#     X_mock, y_mock = create_mock_mareos_data(n_samples=500)
#     mock_read_csv.side_effect = [X_mock, y_mock]
#     mock_ensure.return_value = Path("/tmp/test")

#     with pytest.raises(ValueError, match="Expected 1000 samples"):
#         _load_MAREoS_single_dataset()


# def test__load_MAREoS_single_dataset_with_custom_dir():
#     """Test loading with custom data directory."""
#     with tempfile.TemporaryDirectory() as tmpdir:
#         data_dir = Path(tmpdir)

#         # Create mock files
#         X_mock, y_mock = create_mock_mareos_data()
#         data_file = data_dir / "eos_simple1_data.csv"
#         response_file = data_dir / "eos_simple1_response.csv"

#         X_mock.to_csv(data_file)
#         y_mock.to_csv(response_file)

#         # Load with custom directory
#         X, _, _, _, _ = _load_MAREoS_single_dataset(
#             data_dir=data_dir,
#             as_numpy=False,
#         )

#         assert isinstance(X, pd.DataFrame)
#         assert X.shape[1] == 15  # Features only (metadata columns removed)


# ============================================================================
# Test load_MAREoS
# ============================================================================


# def test_load_mareos_validation():
#     """Test parameter validation for multi-dataset loading."""
#     # Test invalid effect in list
#     with pytest.raises(ValueError, match="effects contains invalid value"):
#         load_MAREoS(effects=["eos", "invalid"])

#     # Test invalid effect_type in list
#     with pytest.raises(
#         ValueError, match="effect_types contains invalid value"
#     ):
#         load_MAREoS(effect_types=["simple", "invalid"])

#     # Test invalid effect_example in list
#     with pytest.raises(
#         ValueError, match="effect_examples contains invalid value"
#     ):
#         load_MAREoS(effect_examples=["1", "3"])


# def test_load_mareos_pandas_output(mock_load):
#     """Test loading datasets as pandas objects."""
#     # Create pandas mock data
#     X_df = pd.DataFrame(np.random.randn(1000, 15))
#     y_series = pd.Series(np.random.randint(0, 2, 1000))
#     sites_series = pd.Series(np.random.randint(0, 8, 1000))
#     covs_df = pd.DataFrame(np.random.randn(1000, 2))
#     folds_series = pd.Series(np.random.randint(0, 10, 1000))

#     mock_load.return_value = (
#         X_df,
#         y_series,
#         sites_series,
#         covs_df,
#         folds_series,
#     )

#     datasets = load_MAREoS(as_numpy=False)

#     # Check all datasets are loaded with pandas objects
#     for dataset in datasets.values():
#         assert isinstance(dataset["X"], pd.DataFrame)
#         assert isinstance(dataset["y"], pd.Series)
#         assert isinstance(dataset["sites"], pd.Series)
#         assert isinstance(dataset["covs"], pd.DataFrame)
#         assert isinstance(dataset["folds"], pd.Series)


# def test_load_mareos_error_handling(mock_load):
#     """Test error handling when loading multiple datasets."""
#     # Make one dataset fail
#     mock_load.side_effect = [
#         (
#             np.random.randn(1000, 15),
#             np.random.randint(0, 2, 1000),
#             np.random.randint(0, 8, 1000),
#             np.random.randn(1000, 2),
#             np.random.randint(0, 10, 1000),
#         ),
#         FileNotFoundError("File not found"),
#         (
#             np.random.randn(1000, 15),
#             np.random.randint(0, 2, 1000),
#             np.random.randint(0, 8, 1000),
#             np.random.randn(1000, 2),
#             np.random.randint(0, 10, 1000),
#         ),
#     ]

#     with pytest.raises(RuntimeError, match="Failed to load dataset"):
#         load_MAREoS()
