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

__all__ = [
    "_list_available_files",
    "get_candidate_files",
    "get_files",
    "initialize_dl_dataset",
    "validate_arguments",
]


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
    dataset_source: str,
    dataset_id: str,
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


def get_candidate_files(
    hidden_dataset_path: Path,
    subjects: str | list[str],
    sessions: str | list[str],
    modalities: str | list[str],
    tasks: str | list[str],
    extensions: str | list[str],
):
    """Get candidate files following BIDS conventions."""
    candidate_files = []

    sub_dirs = (
        list(hidden_dataset_path.glob("sub-*")) if subjects == "all" else [hidden_dataset_path / f"sub-{s}" for s in subjects]
    )
    logger.debug(f"Processing directory: {hidden_dataset_path}")

    for sub_dir in sub_dirs:
        if not sub_dir.exists():
            continue

        ses_dirs = list(sub_dir.glob("ses-*")) if sessions == "all" else [sub_dir / f"ses-{s}" for s in sessions]
        logger.debug(f"Processing directory: {sub_dir}")

        for ses_dir in ses_dirs:
            if not ses_dir.exists():
                continue
            mod_dirs = list(ses_dir.glob("*")) if modalities == "all" else [ses_dir / mod for mod in modalities]
            logger.debug(f"Processing directory: {ses_dir}")

            for mod_dir in mod_dirs:
                if not mod_dir.exists():
                    continue
                logger.debug(f"Processing directory: {mod_dir}")

                # tasks_dir = list(mod_dir.glob("*")) if tasks == "all" else [mod_dir / task for task in tasks]
                for task in tasks:
                    for ex_dir in extensions:
                        logger.debug(f"Adding candidate file: {mod_dir.glob(f'*{task}{ex_dir}')}")

                        candidate_files.extend(mod_dir.glob(f"*{task}{ex_dir}"))

    if not candidate_files:
        raise ValueError(
            f"No matching files found for: \t"
            f"subjects: {subjects}\t"
            f"Sessions: {sessions}\t"
            f"Modalities: {modalities}\t"
            f"Tasks: {tasks}\t"
            f"Extensions: {extensions}"
        )
    return candidate_files


def validate_arguments(subjects, sessions, modalities, tasks, extensions):
    def _validate(arg):
        if isinstance(arg, (list)):
            return arg
        if isinstance(arg, (str)):
            if arg != "all":
                arg = [arg]
                return arg
            else:
                return arg
        raise ValueError(f"arg should be a string or a list of strings. Got {arg}, with type {type(arg)}")

    subjects = _validate(subjects)
    sessions = _validate(sessions)
    modalities = _validate(modalities)
    tasks = _validate(tasks)
    extensions = _validate(extensions)

    return subjects, sessions, modalities, tasks, extensions


def get_files(ds, candidate_files, hidden_dataset_path: Path, dataset_path: Path, copy: bool, cache: bool):
    for file in candidate_files:
        rel = file.relative_to(hidden_dataset_path)
        dest = dataset_path / rel

        # Skip if file already exists in destination
        if dest.exists():
            logger.info(f"Skipping, file already in destination: {rel}")
            continue

        logger.info(f"Getting: {rel}")

        # Materialize file
        ds.get(str(rel), on_failure="ignore", result_renderer="disabled")

        if copy:
            # Copy real file (dereference symlink)
            dest.parent.mkdir(parents=True, exist_ok=True)
            real_file = hidden_dataset_path / rel
            shutil.copyfile(real_file, dest, follow_symlinks=True)

        if not cache:
            # Drop content from hidden dataset
            ds.drop(
                str(rel),
                reckless="availability",
                on_failure="ignore",
                result_renderer="disabled",
            )
            logger.debug(f"Dropped {rel} from hidden dataset to save space.")
    if copy:
        logger.info("\n✓ Copied files downloaded.")
    else:
        logger.info(
            "\n✓ Files are available in the hidden dataset cache if cache=True."
            "Set copy=True to copy them to the target directory."
        )


def initialize_dl_dataset(target_path, dataset_name, dataset_source, dataset_id, force_download):
    if not _check_datalad_installed():
        raise RuntimeError("datalad not installed!")

    # ------------------------------------------------------------------
    # Prepare visible directory (empty)
    # ------------------------------------------------------------------
    dataset_path = _make_visible_directory(target_path, dataset_name)
    # ------------------------------------------------------------------
    # Ensure hidden DataLad dataset exists
    # ------------------------------------------------------------------
    hidden_dataset_path = _ensure_hidden_dataset(
        dataset_source=dataset_source, dataset_id=dataset_id, force_download=force_download
    )
    # ------------------------------------------------------------------
    # Initialize DataLad dataset
    # ------------------------------------------------------------------
    ds = dl.Dataset(hidden_dataset_path)

    # Ensure derivative subdataset metadata is installed
    ds.get(
        ".",
        recursive=True,
        get_data=False,
        on_failure="ignore",
        result_renderer="disabled",
    )
    return ds, hidden_dataset_path, dataset_path
