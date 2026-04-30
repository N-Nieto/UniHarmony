"""Test module for load_onharmony function."""

from uniharmony.datasets import load_onharmony


def test_load_onharmony_success() -> None:
    """Test basic functionality."""
    load_onharmony(
        subjects="16981",
        sessions="NOT4GEP001",
        modalities="anat",
        data_types="T2w",
        extensions=".json",
        copy=False,
        force_download=True,
    )
    load_onharmony(
        subjects="16981",
        sessions="NOT4GEP001",
        modalities="anat",
        data_types="T1w",
        extensions=".json",
        copy=False,
    )
