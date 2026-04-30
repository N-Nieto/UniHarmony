"""Datalad integration functions.

Provides functions to clone datalad datasets, get specific files,
list available files, and explore dataset structure.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urljoin

import structlog
from datalad import api as dl


logger = structlog.get_logger()

__all__ = ["_check_datalad_installed", "_ensure_hidden_dataset", "_list_available_files", "_make_visible_directory"]


def _check_datalad_installed() -> bool:
    try:
        subprocess.run(
            ["datalad", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except RuntimeError:
        return False


def _ensure_hidden_dataset(
    dataset_id: str,
    dataset_source: str,
    extension: str = ".git",
    force_download=False,
) -> Path:
    """Ensure that the hidden DataLad dataset exists in /tmp.

    If the dataset does not exist or is empty, it is cloned.
    If it already exists and contains data, it is reused.

    Returns
    -------
    hidden_dataset_path : Path
        Path to the hidden DataLad dataset.

    """
    # # Return existing path as string
    # return str(hidden_dataset_path)
    hidden_root = Path(tempfile.gettempdir()) / "datalad_cache"
    hidden_root.mkdir(parents=True, exist_ok=True)

    hidden_dataset_path = hidden_root / dataset_id

    # Dataset exists and is non-empty → reuse
    if hidden_dataset_path.exists() and any(hidden_dataset_path.iterdir()) and not force_download:
        logger.info(f"✓ Reusing cached DataLad dataset at {hidden_dataset_path}")
        return hidden_dataset_path

    # Otherwise clone
    if force_download:
        logger.info("Force download dataset")
        # If the dataset already exists, remove it to ensure a clean clone
        if hidden_dataset_path.exists():
            shutil.rmtree(hidden_dataset_path)
    logger.info("Cloning DataLad dataset into hidden cache...")

    source_url = urljoin(dataset_source, f"{dataset_id}{extension}")
    logger.debug(f"Source URL: {source_url}")

    dl.clone(
        source=source_url,
        path=hidden_dataset_path,
        result_renderer="disabled",
    )
    logger.debug(f"hidden_dataset_path created: {hidden_dataset_path}")

    return hidden_dataset_path


def _make_visible_directory(target_dir: str | Path, dataset_name: str) -> Path:
    """Prepare an empty visible directory for the datalad dataset.

    This function does NOT download any data and does NOT create any
    folder structure. Files and directories are created lazily when
    data is requested via get_data.
    """
    target_dir = Path(target_dir).resolve()
    dataset_path = target_dir / dataset_name

    dataset_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"✓ Visible dataset directory ready at {dataset_path}")

    return dataset_path


def _list_available_files(hidden_dataset_path: Path) -> list[Path]:
    """List all available files in the hidden DataLad dataset.

    This function is useful for debugging and exploration purposes. It
    returns a list of all files that are present in the hidden dataset,
    regardless of the filtering criteria.

    Parameters
    ----------
    hidden_dataset_path : Path
        Path to the hidden DataLad dataset.

    Returns
    -------
    list[Path]
        A list of paths to all available files in the hidden dataset.

    """
    return list(hidden_dataset_path.rglob("*.*"))
