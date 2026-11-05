"""DataLad is an optional dependency (``uniharmony[datasets]``)."""

import builtins

import pytest

from uniharmony.datasets import download_bids_dataset


def test_missing_datalad_raises_an_informative_error(monkeypatch, tmp_path) -> None:
    """Without DataLad, the download fails early with the install command, before creating any directory."""
    real_import = builtins.__import__

    def no_datalad(name, *args, **kwargs):
        if name == "datalad" or name.startswith("datalad."):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_datalad)
    target = tmp_path / "data"
    with pytest.raises(ImportError, match=r"uniharmony\[datasets\]"):
        download_bids_dataset(
            subjects="all",
            sessions="all",
            modalities="anat",
            tasks="all",
            runs="all",
            suffixes="T1w",
            extensions=".json",
            target_path=target,
            dataset_url="https://example.invalid/dataset.git",
            root_files="all",
        )
    assert not target.exists()


def test_import_without_datalad() -> None:
    """Importing uniharmony.datasets does not import DataLad."""
    import subprocess
    import sys

    code = (
        "import sys, uniharmony.datasets as d; d.make_multisite_classification; d.download_bids_dataset; "
        "assert 'datalad' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
