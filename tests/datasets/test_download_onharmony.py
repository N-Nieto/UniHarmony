"""Test module for load_onharmony function."""

import importlib.util

import pytest

from uniharmony.datasets import download_ONharmony


# Clones ON-Harmony with DataLad (needs the network, DataLad and git-annex)
pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(importlib.util.find_spec("datalad") is None, reason="DataLad not installed (uniharmony[datasets])"),
]


@pytest.mark.parametrize(
    "subjects, sessions, modalities, suffixes, extensions, copy, force_download, root_files, hidden",
    [
        ("16981", "NOT4GEP001", "anat", "T2w", ".json", False, True, "participants.tsv", True),
        ("all", "NOT4GEP001", "anat", "T1w", ".json", False, True, "all", False),
        ("16981", "all", "anat", "T1w", ".json", False, False, "all", False),
        ("16981", "all", "all", "T1w", ".json", False, False, "all", False),
    ],
)
def test_load_onharmony_success(
    subjects, sessions, modalities, suffixes, extensions, copy, force_download, root_files, hidden
) -> None:
    """Test basic functionality."""
    download_ONharmony(
        subjects=subjects,
        sessions=sessions,
        modalities=modalities,
        suffixes=suffixes,
        extensions=extensions,
        copy=copy,
        force_download=force_download,
        root_files=root_files,
        hidden=hidden,
    )
