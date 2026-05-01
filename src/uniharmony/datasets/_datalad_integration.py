"""Datalad integration functions.

Provides functions to clone datalad datasets, get specific files,
list available files, and explore dataset structure.

Expected structlog configuration::

    import structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
"""

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
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
    """Check if the datalad command-line tool is available.

    Returns
    -------
    bool
        True if datalad is installed and callable, False otherwise.

    """
    try:
        subprocess.run(
            ["datalad", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except (RuntimeError, FileNotFoundError, subprocess.CalledProcessError):
        return False


def _ensure_hidden_dataset(
    dataset_source: str,
    dataset_id: str,
    extension: str = ".git",
    force_download: bool = False,
) -> Path:
    """Ensure that a hidden DataLad dataset exists in a temporary cache.

    If the dataset does not exist or is empty, it is cloned from the
    remote source. If it already exists and contains data, it is reused
    unless ``force_download`` is True.

    Parameters
    ----------
    dataset_source : str
        Base URL for the DataLad dataset repository (e.g.
        ``"https://github.com/OpenNeuroDatasets/"``).
    dataset_id : str
        Dataset identifier (e.g. ``"ds004712"``).
    extension : str, optional
        File extension appended to ``dataset_id`` when building the clone
        URL. Defaults to ``".git"``.
    force_download : bool, optional
        If True, remove any existing cached copy and re-clone. Defaults to
        False.

    Returns
    -------
    Path
        Absolute path to the cached DataLad dataset directory.

    Raises
    ------
    RuntimeError
        If the DataLad clone operation fails.

    """
    hidden_root = Path(tempfile.gettempdir()) / "datalad_cache"
    hidden_root.mkdir(parents=True, exist_ok=True)

    hidden_dataset_path = hidden_root / dataset_id

    # Dataset exists and is non-empty → reuse
    if hidden_dataset_path.exists() and any(hidden_dataset_path.iterdir()) and not force_download:
        logger.info(f"Reusing cached DataLad dataset at {hidden_dataset_path}")
        return hidden_dataset_path

    # Otherwise clone
    if force_download:
        logger.info("Force download dataset")
        if hidden_dataset_path.exists():
            shutil.rmtree(hidden_dataset_path)

    logger.info("Cloning DataLad dataset into hidden cache...")

    source_url = urljoin(dataset_source.rstrip("/") + "/", f"{dataset_id}{extension}")
    logger.debug(f"Source URL: {source_url}")

    # dl.clone() returns a Dataset object on success, raises on failure
    try:
        dl.clone(
            source=source_url,
            path=str(hidden_dataset_path),
            result_renderer="disabled",
        )
    except Exception as e:
        raise RuntimeError(f"Clone failed for {dataset_id}: {e}") from e

    # Verify the dataset was actually created
    if not hidden_dataset_path.exists() or not any(hidden_dataset_path.iterdir()):
        raise RuntimeError(f"Clone failed for {dataset_id}: directory is empty or missing")

    logger.debug(f"hidden_dataset_path created: {hidden_dataset_path}")
    return hidden_dataset_path


def _make_visible_directory(target_dir: str | Path, dataset_name: str) -> Path:
    """Prepare an empty visible directory for the datalad dataset.

    This function does NOT download any data and does NOT create any
    folder structure. Files and directories are created lazily when
    data is requested via ``get_data``.

    Parameters
    ----------
    target_dir : str or Path
        Parent directory where the visible dataset folder will be created.
    dataset_name : str
        Name of the dataset folder.

    Returns
    -------
    Path
        Absolute path to the newly created (or existing) visible directory.

    """
    target_dir = Path(target_dir).resolve()
    dataset_path = target_dir / dataset_name

    dataset_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Visible dataset directory ready at {dataset_path}")

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


def _resolve_child_dirs(
    parent_path: Path,
    values: str | list[str],
    path_template: str,
    glob_pattern: str,
) -> list[Path]:
    if values == "all":
        return list(parent_path.glob(glob_pattern))
    return [parent_path / path_template.format(value) for value in values]


def _resolve_modality_dirs(session_path: Path, modalities: str | list[str]) -> list[Path]:
    if modalities == "all":
        return [entry for entry in session_path.iterdir() if entry.is_dir()]
    return [session_path / modality for modality in modalities]


def _build_search_patterns(
    suffixes: str | list[str],
    extensions: str | list[str],
) -> list[str]:
    suffix_patterns = ["*"] if suffixes == "all" else [f"*{suffix}" for suffix in suffixes]
    return [f"{suffix_pattern}{extension}" for suffix_pattern in suffix_patterns for extension in extensions]


def get_candidate_files(
    hidden_dataset_path: Path,
    subjects: str | list[str],
    sessions: str | list[str],
    modalities: str | list[str],
    suffixes: str | list[str],
    extensions: str | list[str],
) -> list[Path]:
    """Get candidate files following BIDS conventions.

    Parameters
    ----------
    hidden_dataset_path : Path
        Root path of the hidden DataLad dataset.
    subjects : str or list[str]
        Subject identifiers to include, or ``"all"`` for every subject.
    sessions : str or list[str]
        Session identifiers to include, or ``"all"`` for every session.
    modalities : str or list[str]
        Modality folder names (e.g. ``"anat"``, ``"func"``) to include,
        or ``"all"`` for every modality.
    suffixes : str or list[str]
        BIDS suffixes to match in filenames (e.g. ``"bold"``,
        ``"T1w"``, ``"dwi"``), or ``"all"`` for every suffix.
    extensions : str or list[str]
        File extensions to match (e.g. ``".nii.gz"``, ``".json"``).

    Returns
    -------
    list[Path]
        List of matching file paths.

    Raises
    ------
    ValueError
        If no files match the provided criteria.

    """
    candidate_files: list[Path] = []

    for sub_dir in _resolve_child_dirs(hidden_dataset_path, subjects, "sub-{}", "sub-*"):
        if not sub_dir.exists():
            continue

        for ses_dir in _resolve_child_dirs(sub_dir, sessions, "ses-{}", "ses-*"):
            if not ses_dir.exists():
                continue

            for mod_dir in _resolve_modality_dirs(ses_dir, modalities):
                if not mod_dir.exists():
                    continue

                for pattern in _build_search_patterns(suffixes, extensions):
                    candidate_files.extend(mod_dir.glob(pattern))

    if not candidate_files:
        raise ValueError(
            f"No matching files found for: "
            f"subjects={subjects}, sessions={sessions}, "
            f"modalities={modalities}, suffixes={suffixes}, extensions={extensions}"
        )

    return candidate_files


def validate_arguments(
    subjects: str | list[str],
    sessions: str | list[str],
    modalities: str | list[str],
    suffixes: str | list[str],
    extensions: str | list[str],
) -> tuple[list[str] | str, ...]:
    """Normalize filtering arguments to lists or the string ``"all"``.

    Parameters
    ----------
    subjects, sessions, modalities, suffixes, extensions : str or list[str]
        Raw filtering arguments. The string ``"all"`` is kept as-is;
        any other string is wrapped in a single-element list.

    Returns
    -------
    tuple
        Normalized ``(subjects, sessions, modalities, suffixes, extensions)``.

    Raises
    ------
    ValueError
        If any argument is neither a string nor a list.

    """

    def _validate(arg: str | list[str]) -> str | list[str]:
        if isinstance(arg, list):
            return arg
        if isinstance(arg, str):
            return "all" if arg == "all" else [arg]
        raise ValueError(f"Argument should be a string or a list of strings. Got {arg!r} (type: {type(arg).__name__})")

    return (
        _validate(subjects),
        _validate(sessions),
        _validate(modalities),
        _validate(suffixes),
        _validate(extensions),
    )


@dataclass
class GetResult:
    """Result of a single file retrieval operation."""

    path: Path
    success: bool
    copied: bool
    dropped: bool
    error: str | None = None


def get_files(
    ds,
    candidate_files: list[Path],
    hidden_dataset_path: Path,
    dataset_path: Path,
    copy: bool,
    cache: bool,
) -> list[GetResult]:
    """Materialize candidate files and optionally copy them to a visible directory.

    Parameters
    ----------
    ds
        DataLad dataset instance (returned by ``dl.Dataset``).
    candidate_files : list[Path]
        List of file paths to materialize (relative to ``hidden_dataset_path``).
    hidden_dataset_path : Path
        Root path of the hidden DataLad cache.
    dataset_path : Path
        Root path of the visible output directory.
    copy : bool
        If True, copy the materialized files from the hidden cache to
        ``dataset_path``.
    cache : bool
        If False, drop the file content from the hidden cache after
        processing to save space.

    Returns
    -------
    list[GetResult]
        Results for each file operation.

    Raises
    ------
    ValueError
        If ``copy=False`` and ``cache=False`` (would materialize and
        immediately drop files).

    """
    # FIX: Guard against invalid copy/cache combination
    if not copy and not cache:
        logger.warning(
            "Combination copy=False and cache=False would materialize "
            "files and immediately drop them. Use copy=True or cache=True to retain files."
        )

    results: list[GetResult] = []

    for file in candidate_files:
        rel = file.relative_to(hidden_dataset_path)
        dest = dataset_path / rel

        # Skip if file already exists in destination
        if dest.exists():
            logger.info(f"Skipping, file already in destination: {rel}")
            results.append(GetResult(path=rel, success=True, copied=False, dropped=False))
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

        results.append(GetResult(path=rel, success=True, copied=copy, dropped=not cache))

    if copy:
        logger.info("Copied files downloaded.")
    else:
        logger.info(
            "Files are available in the hidden dataset cache if cache=True. Set copy=True to copy them to the target directory."
        )

    return results


def initialize_dl_dataset(
    target_path: str | Path,
    dataset_name: str,
    dataset_source: str,
    dataset_id: str,
    force_download: bool,
    install_subdatasets: bool = True,
):
    """Initialize the full DataLad dataset workflow.

    Creates a visible directory, ensures the hidden cache exists, and
    optionally installs subdataset metadata.

    Parameters
    ----------
    target_path : str or Path
        Parent directory for the visible dataset copy.
    dataset_name : str
        Name of the visible dataset folder.
    dataset_source : str
        Base URL for the remote DataLad repository.
    dataset_id : str
        Dataset identifier (e.g. ``"ds004712"``).
    force_download : bool
        If True, force a fresh clone of the hidden dataset.
    install_subdatasets : bool, optional
        If True, recursively install subdataset metadata. Defaults to True.
        Set to False for large datasets where only root-level files are needed.

    Returns
    -------
    tuple
        ``(ds, hidden_dataset_path, dataset_path)`` where ``ds`` is the
        DataLad dataset instance.

    Raises
    ------
    RuntimeError
        If datalad is not installed on the system.

    """
    if not _check_datalad_installed():
        raise RuntimeError("datalad not installed!")

    # Prepare visible directory (empty)
    dataset_path = _make_visible_directory(target_path, dataset_name)

    # Ensure hidden DataLad dataset exists
    hidden_dataset_path = _ensure_hidden_dataset(
        dataset_source=dataset_source,
        dataset_id=dataset_id,
        force_download=force_download,
    )

    # Initialize DataLad dataset
    ds = dl.Dataset(str(hidden_dataset_path))

    # Optionally install subdataset metadata
    if install_subdatasets:
        ds.get(
            ".",
            recursive=True,
            get_data=False,
            on_failure="ignore",
            result_renderer="disabled",
        )

    return ds, hidden_dataset_path, dataset_path
