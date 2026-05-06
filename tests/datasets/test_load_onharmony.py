"""Test module for load_onharmony function."""

import pytest

from uniharmony.datasets import load_ONharmony


@pytest.mark.parametrize(
    "subjects, sessions, modalities, suffixes, extensions, copy, force_download",
    [
        # Binary classification tests
        ("16981", "NOT4GEP001", "anat", "T2w", ".json", False, True),
        ("all", "NOT4GEP001", "anat", "T1w", ".json", False, True),
        ("16981", "all", "anat", "T1w", ".json", False, False),
        ("16981", "all", "all", "T1w", ".json", False, False),
    ],
)
def test_load_onharmony_success(subjects, sessions, modalities, suffixes, extensions, copy, force_download) -> None:
    """Test basic functionality."""
    load_ONharmony(
        subjects=subjects,
        sessions=sessions,
        modalities=modalities,
        suffixes=suffixes,
        extensions=extensions,
        copy=copy,
        force_download=force_download,
    )
