"""Datalad integration functions.

Provides functions to clone datalad datasets, get specific files,
list available files, and explore dataset structure.

"""

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import structlog
from datalad import api as dl


logger = structlog.get_logger()

__all__ = [
    "clean_tmp_folder",
    "download_bids_dataset",
    "get_candidate_files",
    "get_files",
    "get_root_files",
    "initialize_dl_dataset",
    "list_available_files",
    "validate_arguments",
]


def download_bids_dataset(
    subjects: str | list[str],
    sessions: str | list[str],
    modalities: str | list[str],
    tasks: str | list[str],
    runs: str | list[str],
    target_path: str | Path,
    suffixes: str | list[str],
    extensions: str | list[str],
    dataset_source_URL: str,
    root_files: str | list[str],
    force_download: bool = False,
    copy: bool = True,
    hidden: bool = True,
    tmp_clean: bool = False,
    tmp_dir_name: str = "datalad_cache",
) -> None:
    """Download derivatives and root files from a BIDS-compatible dataset.

    This function transparently uses a hidden DataLad dataset (stored in
    a temporary location) to retrieve files from a repository. All DataLad
    operations happen in the background. The user-visible directory
    contains only regular files (no symbolic links, Git metadata, or
    DataLad traces).

    Only the requested files are downloaded. Each file is copied as a
    real file (not symbolic) into the visible dataset directory and immediately dropped
    from the hidden DataLad cache to minimize disk usage.

    Parameters
    ----------
    subjects : str or list[str]
        Subject identifiers to download.

    sessions : str or list[str]
        Session identifiers to download.

    modalities : str or list[str]
        Modalities to download ("all", "anat", "dwi", "fmap", "func", "swi").

    tasks : str or list[str]
        Tasks to download.

    runs : str or list[str]
        Runs to download.

    target_path : str or pathlib.Path
        Path to the visible dataset directory where files will be stored.

    suffixes : str or list[str]
        BIDS suffixes to match in filenames (e.g., 'T1w', 'T2w').

    extensions : str or list[str], default ".json"
        File extensions to download (e.g., '.json', '.nii.gz').

    dataset_source_URL : str
        Source URL to the BIDS-compatible dataset.

    root_files : str or list[str]
        Name of the file list of files to get from the dataset's root.

    force_download : bool, default False
        Whether to force re-download the dataset if it already exists in tmp.

    hidden : bool, default True
        Whether to use a hidden directory or not.
        If hidden=False, no hidden folder is made and the target directory acts as hidden.
        This will avoid getting the files in ``/tmp/{tmp_dir_name}`` and then copying them
        to the target directory.

    copy : bool, default True
        Whether to copy the downloaded files from the hidden cache to the target directory.
        Ignored when ``hidden=False`` (files are already in the target directory).

    tmp_clean : bool, default False
        Whether to drop the downloaded files from the hidden DataLad dataset after copying.
        If True, files are dropped immediately after copying to the target directory
        (if copy=True), to minimize disk usage. Ignored when ``hidden=False``.

    tmp_dir_name : str, default "datalad_cache"
        Name of the temporary directory to store the hidden dataset.

    Notes
    -----
    - The visible dataset directory will contain only regular files
      following the BIDS derivatives structure.
    - Repeated calls are safe and will only download missing files.

    """
    # ------------------------------------------------------------------
    #  Validate arguments
    # ------------------------------------------------------------------
    subjects, sessions, modalities, suffixes, extensions, root_files = validate_arguments(
        subjects, sessions, modalities, suffixes, extensions, root_files
    )

    # ------------------------------------------------------------------
    #  Prepare directories
    # ------------------------------------------------------------------
    dataset_path = _make_visible_directory(target_path)

    if hidden:
        hidden_dataset_path = _make_hidden_dataset(
            tmp_dir_name=tmp_dir_name,
            force_download=force_download,
        )
        logger.debug(f"Using hidden folder at {hidden_dataset_path}")
    else:
        hidden_dataset_path = dataset_path
        logger.debug(f"Using target folder as working directory at {hidden_dataset_path}")
        # When not using hidden mode, we must NOT drop files from the target
        if tmp_clean:
            logger.warning("tmp_clean=True is ignored when hidden=False (would delete target files)")
            tmp_clean = False

    # ------------------------------------------------------------------
    #  Initialize the DataLad dataset
    # ------------------------------------------------------------------
    logger.debug(f"Source URL: {dataset_source_URL}")
    ds = initialize_dl_dataset(hidden_dataset_path, dataset_source_URL)

    # ------------------------------------------------------------------
    # Collect candidate files
    # ------------------------------------------------------------------
    candidate_files = get_candidate_files(hidden_dataset_path, subjects, sessions, modalities, tasks, runs, suffixes, extensions)

    # ------------------------------------------------------------------
    # Download files
    # ------------------------------------------------------------------
    get_files(ds, candidate_files, hidden_dataset_path, dataset_path, copy, tmp_clean, hidden)

    get_root_files(ds, root_files, hidden_dataset_path, dataset_path, copy, tmp_clean, hidden)

    return


def validate_arguments(
    subjects: str | list[str],
    sessions: str | list[str],
    modalities: str | list[str],
    suffixes: str | list[str],
    extensions: str | list[str],
    root_files: str | list[str],
) -> tuple[list[str] | str, ...]:
    """Normalize filtering arguments to lists or the string ``"all"``.

    Parameters
    ----------
    subjects, sessions, modalities, suffixes, extensions, root_files : str or list[str]
        Raw filtering arguments. The string ``"all"`` is kept as-is;
        any other string is wrapped in a single-element list.

    Returns
    -------
    tuple
        Normalized ``(subjects, sessions, modalities, suffixes, extensions, root_files)``.

    Raises
    ------
    ValueError
        If any argument is neither a string nor a list.

    """
    logger.debug("Validating arguments")

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
        _validate(root_files),
    )


def initialize_dl_dataset(
    dataset_path: str | Path,
    source_url: str,
):
    """Initialize the full DataLad dataset workflow.

    Clones the remote repository into the given path and installs
    subdataset metadata.

    Parameters
    ----------
    dataset_path : str or Path
        Directory where the dataset will be cloned.
    source_url : str
        URL for the remote DataLad repository.

    Returns
    -------
    Dataset
        DataLad dataset instance.

    Raises
    ------
    RuntimeError
        If datalad is not installed on the system or cloning fails.

    """
    dataset_path = Path(dataset_path)

    # Only clone if directory is empty or doesn't exist
    if not dataset_path.exists() or not any(dataset_path.iterdir()):
        logger.info("Cloning DataLad dataset...")

        try:
            dl.clone(
                source=source_url,
                path=dataset_path,
                result_renderer="disabled",
            )
        except Exception as e:
            raise RuntimeError(f"Clone failed for {source_url}: {e}") from e

        # Verify the dataset was actually created
        if not dataset_path.exists() or not any(dataset_path.iterdir()):
            raise RuntimeError(f"Clone failed for {source_url}: directory is empty or missing")

    # Initialize DataLad dataset
    ds = dl.Dataset(dataset_path)

    return ds


def get_candidate_files(
    hidden_dataset_path: Path,
    subjects: str | list[str],
    sessions: str | list[str],
    modalities: str | list[str],
    tasks: str | list[str],
    runs: str | list[str],
    suffixes: str | list[str],
    extensions: str | list[str],
) -> list[Path]:
    """Get candidate files following BIDS conventions.

    Parameters
    ----------
    hidden_dataset_path : Path
        Root path of the DataLad dataset.
    subjects : str or list[str]
        Subject identifiers to include, or ``"all"`` for every subject.
    sessions : str or list[str]
        Session identifiers to include, or ``"all"`` for every session.
    modalities : str or list[str]
        Modality folder names (e.g. ``"anat"``, ``"func"``) to include,
        or ``"all"`` for every modality.
    tasks : str or list[str]
        tasks  (e.g. ``"eeg"``) to include,
        or ``"all"`` for every task.
    runs : str or list[str]
        runs  (e.g. ``"1"``) to include,
        or ``"all"`` for every run.
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

                for pattern in _build_search_patterns(tasks, runs, suffixes, extensions):
                    candidate_files.extend(mod_dir.glob(pattern))

    if not candidate_files:
        raise ValueError(
            f"No matching files found for: "
            f"subjects={subjects}, sessions={sessions}, "
            f"modalities={modalities}, suffixes={suffixes}, extensions={extensions}"
        )

    return candidate_files


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
    tmp_clean: bool,
    hidden: bool,
) -> list[GetResult]:
    """Materialize candidate files and optionally copy them to a visible directory.

    Parameters
    ----------
    ds
        DataLad dataset instance (returned by ``dl.Dataset``).
    candidate_files : list[Path]
        List of file paths to materialize (relative to ``hidden_dataset_path``).
    hidden_dataset_path : Path
        Root path of the DataLad dataset (hidden cache or target directory).
    dataset_path : Path
        Root path of the visible output directory.
    copy : bool
        If True and ``hidden=True``, copy the materialized files from the
        hidden cache to ``dataset_path``. Ignored when ``hidden=False``.
    tmp_clean : bool
        If True and ``hidden=True``, drop the file content from the hidden
        cache after processing to save space. Ignored when ``hidden=False``.
    hidden : bool
        Whether a hidden cache is being used. When False, files are
        materialized directly in ``dataset_path`` and neither copy nor
        drop operations are performed.

    Returns
    -------
    list[GetResult]
        Results for each file operation.

    """
    results: list[GetResult] = []

    # When hidden=False, source and destination are the same
    should_copy = copy and hidden
    should_drop = tmp_clean and hidden

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

        if should_copy:
            # Copy real file (dereference symlink)
            dest.parent.mkdir(parents=True, exist_ok=True)
            real_file = hidden_dataset_path / rel
            shutil.copyfile(real_file, dest, follow_symlinks=True)

        if should_drop:
            # Drop content from hidden dataset
            ds.drop(
                str(rel),
                reckless="availability",
                on_failure="ignore",
                result_renderer="disabled",
            )
            logger.debug(f"Dropped {rel} from hidden dataset to save space.")

        results.append(GetResult(path=rel, success=True, copied=should_copy, dropped=should_drop))

    if should_copy:
        logger.info("Copied derivative files to target directory.")
    else:
        logger.info("Derivative files are available in the dataset directory.")

    return results


def get_root_files(
    ds,
    root_files: str | list[str],
    hidden_dataset_path: Path,
    dataset_path: Path,
    copy: bool,
    tmp_clean: bool,
    hidden: bool,
) -> list[GetResult]:
    """Materialize root-level files and optionally copy them to a visible directory.

    Parameters
    ----------
    ds
        DataLad dataset instance (returned by ``dl.Dataset``).
    root_files : str or list[str]
        Files to materialize from the dataset root. Use ``"all"`` to
        include all root-level files, or a list of specific filenames
        (e.g. ``["dataset_description.json", "README"]``).
    hidden_dataset_path : Path
        Root path of the DataLad dataset (hidden cache or target directory).
    dataset_path : Path
        Root path of the visible output directory.
    copy : bool
        If True and ``hidden=True``, copy the materialized files from the
        hidden cache to ``dataset_path``. Ignored when ``hidden=False``.
    tmp_clean : bool
        If True and ``hidden=True``, drop the file content from the hidden
        cache after processing to save space. Ignored when ``hidden=False``.
    hidden : bool
        Whether a hidden cache is being used. When False, files are
        materialized directly in ``dataset_path`` and neither copy nor
        drop operations are performed.

    Returns
    -------
    list[GetResult]
        Results for each file operation.

    """
    results: list[GetResult] = []

    # When hidden=False, source and destination are the same
    should_copy = copy and hidden
    should_drop = tmp_clean and hidden

    # Resolve candidate files
    if root_files == "all":
        candidate_files = [f for f in hidden_dataset_path.glob("*") if f.is_file()]
    else:
        root_files = [root_files] if isinstance(root_files, str) else root_files
        candidate_files = [hidden_dataset_path / filename for filename in root_files]

    for file in candidate_files:
        rel = file.relative_to(hidden_dataset_path)
        dest = dataset_path / rel

        # Skip if file already exists in destination
        if dest.exists():
            logger.info(f"Skipping, file already in destination: {rel}")
            results.append(GetResult(path=rel, success=True, copied=False, dropped=False))
            continue

        # Skip if file does not exist in source (e.g. typo in filename)
        if not file.exists():
            logger.warning(f"Skipping, file not found: {rel}")
            results.append(GetResult(path=rel, success=False, copied=False, dropped=False))
            continue

        logger.info(f"Getting: {rel}")

        # Materialize file
        ds.get(str(rel), on_failure="ignore", result_renderer="disabled")

        if should_copy:
            # Copy real file (dereference symlink)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, dest, follow_symlinks=True)

        if should_drop:
            # Drop content from hidden dataset
            ds.drop(
                str(rel),
                reckless="availability",
                on_failure="ignore",
                result_renderer="disabled",
            )
            logger.debug(f"Dropped {rel} from hidden dataset to save space.")

        results.append(GetResult(path=rel, success=True, copied=should_copy, dropped=should_drop))

    if should_copy:
        logger.info("Copied root files to target directory.")
    else:
        logger.info("Root files are available in the dataset directory.")

    return results


def _make_hidden_dataset(
    tmp_dir_name: str = "datalad_cache",
    force_download: bool = False,
) -> Path:
    """Create or reuse a hidden DataLad dataset cache directory.

    Parameters
    ----------
    tmp_dir_name : str, optional
        Name of the temporary directory to store the hidden dataset.
        Defaults to ``"datalad_cache"``.
    force_download : bool, optional
        If True, remove any existing cached copy. Defaults to False.

    Returns
    -------
    Path
        Absolute path to the hidden dataset cache directory.

    """
    hidden_root = Path(tempfile.gettempdir()) / tmp_dir_name
    hidden_root.mkdir(parents=True, exist_ok=True)

    # Dataset exists and is non-empty → reuse
    if hidden_root.exists() and any(hidden_root.iterdir()) and not force_download:
        logger.info(f"Reusing cached DataLad dataset at {hidden_root}")
        return hidden_root

    # Otherwise clean and prepare for fresh clone
    if force_download and hidden_root.exists():
        logger.info("Force download: clearing existing cache")
        shutil.rmtree(hidden_root)
        hidden_root.mkdir(parents=True, exist_ok=True)

    logger.debug(f"Hidden dataset path prepared: {hidden_root}")
    return hidden_root


def _make_visible_directory(target_dir: str | Path) -> Path:
    """Prepare an empty visible directory for the dataset.

    This function does NOT download any data and does NOT create any
    folder structure. Files and directories are created lazily when
    data is requested.

    Parameters
    ----------
    target_dir : str or Path
        Parent directory where the visible dataset folder will be created.

    Returns
    -------
    Path
        Absolute path to the newly created (or existing) visible directory.

    """
    target_path = Path(target_dir).resolve()
    target_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Visible dataset directory ready at {target_path}")

    return target_path


def list_available_files(hidden_dataset_path: Path) -> list[Path]:
    """List all available files in the DataLad dataset.

    This function is useful for debugging and exploration purposes. It
    returns a list of all files that are present in the dataset,
    regardless of the filtering criteria.

    Parameters
    ----------
    hidden_dataset_path : Path
        Path to the DataLad dataset.

    Returns
    -------
    list[Path]
        A list of paths to all available files in the dataset.

    """
    return list(hidden_dataset_path.rglob("*.*"))


def _resolve_child_dirs(
    parent_path: Path,
    values: str | list[str],
    path_template: str,
    glob_pattern: str,
) -> list[Path]:
    """Resolve child directories based on filter values.

    Parameters
    ----------
    parent_path : Path
        Parent directory to search in.
    values : str or list[str]
        ``"all"`` or list of specific values.
    path_template : str
        Format string for constructing specific paths (e.g. ``"sub-{}"``).
    glob_pattern : str
        Glob pattern for discovering all children (e.g. ``"sub-*"``).

    Returns
    -------
    list[Path]
        List of resolved child directory paths.

    """
    if values == "all":
        return list(parent_path.glob(glob_pattern))
    return [parent_path / path_template.format(value) for value in values]


def _resolve_modality_dirs(session_path: Path, modalities: str | list[str]) -> list[Path]:
    """Resolve modality directories within a session.

    Parameters
    ----------
    session_path : Path
        Path to the session directory.
    modalities : str or list[str]
        ``"all"`` or list of modality folder names.

    Returns
    -------
    list[Path]
        List of modality directory paths.

    """
    if modalities == "all":
        return [entry for entry in session_path.iterdir() if entry.is_dir()]
    return [session_path / modality for modality in modalities]


def _build_search_patterns(
    tasks: str | list[str],
    runs: str | list[str],
    suffixes: str | list[str],
    extensions: str | list[str],
) -> list[str]:
    """Build glob search patterns from suffixes and extensions.

    Parameters
    ----------
    tasks : str or list[str]
        tasks  (e.g. ``"eeg"``) to include,
        or ``"all"`` for every task.
    runs : str or list[str]
        runs  (e.g. ``"01"``) to include,
        or ``"all"`` for every runs.
    suffixes : str or list[str]
        ``"all"`` or list of BIDS suffixes.
    extensions : str or list[str]
        List of file extensions.

    Returns
    -------
    list[str]
        List of glob patterns (e.g. ``"*T1w.nii.gz"``, ``"*.json"``).

    """
    tasks_patters = ["*"] if tasks == "all" else [f"*task-{task}" for task in tasks]
    runs_patters = ["*"] if runs == "all" else [f"*run-{run}" for run in runs]
    suffix_patterns = ["*"] if suffixes == "all" else [f"*{suffix}" for suffix in suffixes]
    return [
        f"{tasks_patter}{run_patter}{suffix_pattern}{extension}"
        for tasks_patter in tasks_patters
        for run_patter in runs_patters
        for suffix_pattern in suffix_patterns
        for extension in extensions
    ]


def clean_tmp_folder(tmp_dir_name: str = "datalad_cache") -> None:
    """Remove a temporary DataLad cache folder from /tmp.

    This function deletes the specified folder from the system temporary
    directory, including all its contents (files, subdirectories, and
    DataLad metadata).

    Parameters
    ----------
    tmp_dir_name : str, default "datalad_cache"
        Name of the temporary directory to remove.

    Raises
    ------
    FileNotFoundError
        If the temporary directory does not exist.
    PermissionError
        If the process lacks permission to delete the directory.

    Examples
    --------
    >>> clean_tmp_folder("datalad_cache")
    INFO     Removed temporary cache: /tmp/datalad_cache

    >>> clean_tmp_folder("my_custom_cache")
    INFO     Removed temporary cache: /tmp/my_custom_cache

    """
    tmp_path = Path(tempfile.gettempdir()) / tmp_dir_name

    if not tmp_path.exists():
        logger.warning(f"Temporary directory not found: {tmp_path}\nNothing to clean.")

    logger.info(f"Removing temporary files: {tmp_path}")
    shutil.rmtree(tmp_path)
    logger.info("Temporary cache removed successfully.")
