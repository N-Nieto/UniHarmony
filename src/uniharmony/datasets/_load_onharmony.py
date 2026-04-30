"""ON-Harmony dataset downloader using DataLad.

Provides functions to clone the dataset, get specific files,
list available files, and explore dataset structure.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import structlog
from datalad import api as dl


logger = structlog.get_logger()

__all__ = ["_get_onharmony_information_form_idps", "_list_available_files", "_list_available_possibilities", "load_onharmony"]


def load_onharmony(
    subjects: str | list[str],
    sessions: str | list[str],
    modalities: str | list[str],
    target_path: str | Path = ".",
    data_types: str | list[str] = "T1w",
    extensions: str | list[str] = ".json",
    dataset_source: str = "https://github.com/OpenNeuroDatasets/",
    force_download=False,
    copy=True,
) -> None:
    """Download derivatives from the ON-Harmony dataset and store them as files in a user-visible directory.

    This function transparently uses a hidden DataLad dataset (stored in
    a temporary location) to retrieve files from OpenNeuro. All DataLad
    operations happen in the background. The user-visible directory
    contains only regular files (no symbolic links, Git metadata, or
    DataLad traces).

    Only the requested files are downloaded. Each file is copied as a
    real file into the visible dataset directory and immediately dropped
    from the hidden DataLad cache to minimize disk usage.

    Parameters
    ----------
    subjects : str or list[str]
        Subject identifiers to download.

    sessions : str or list[str]
        Session identifiers to download.

    modalities : str or list[str]
        Modalities to download (e.g., 'T1w', 'FLAIR').

    target_path : str or pathlib.Path, default "."
        Path to the visible dataset directory where files will be stored.

    data_types : str or list[str], default "T1w"
        Data types to download (e.g., 'T1w', 'T2w').

    extensions : str or list[str], default ".json"
        File extensions to download (e.g., '.json', '.nii.gz').

    dataset_source : str, default "https://openneuro.org/datasets/ds004712/versions/2.0.1"
        Source URL or path to the ON-Harmony dataset.

    force_download : bool, default False
        Whether to force re-download the dataset if it already exists in cache.

    copy : bool, default True
        Whether to copy the downloaded files to the target directory to make it visible.

    Notes
    -----
    - The visible dataset directory will contain only regular files
      following the BIDS derivatives structure.
    - Repeated calls are safe and will only download missing files.

    """
    if not _check_datalad_installed():
        raise RuntimeError("datalad not installed!")
    # ------------------------------------------------------------------
    # Resolve visible and hidden dataset paths
    # ------------------------------------------------------------------
    # Prepare visible directory (empty)
    # ------------------------------------------------------------------
    dataset_path = _make_visible_directory(target_path, "onharmony")
    # ------------------------------------------------------------------
    # Ensure hidden DataLad dataset exists
    # ------------------------------------------------------------------
    hidden_dataset_path = _ensure_hidden_dataset(
        dataset_id="ds004712", dataset_source=dataset_source, force_download=force_download
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
    # ------------------------------------------------------------------
    #  Validate arguments
    # ------------------------------------------------------------------
    subjects, sessions, modalities, data_types, extensions = _validate_arguments(
        subjects, sessions, modalities, data_types, extensions
    )
    # ------------------------------------------------------------------
    # Collect candidate files first (for progress bar support)
    # ------------------------------------------------------------------
    candidate_files = _get_candidate_files(hidden_dataset_path, subjects, sessions, modalities, data_types, extensions)

    # ------------------------------------------------------------------
    # Create a list of files to download.
    # ------------------------------------------------------------------

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

        # Drop content from hidden dataset
        ds.drop(
            str(rel),
            reckless="availability",
            on_failure="ignore",
            result_renderer="disabled",
        )

    logger.info("\n✓ Copied files downloaded.")
    return


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
) -> str:
    """Ensure that the hidden DataLad dataset exists in /tmp.

    If the dataset does not exist or is empty, it is cloned.
    If it already exists and contains data, it is reused.

    Returns
    -------
    hidden_dataset_path : Path
        Path to the hidden DataLad dataset.

    """
    # # Build source URL as string (never use Path on URLs)
    # dataset_source = str(dataset_source).rstrip("/") + "/"
    # source_url = f"{dataset_source}{dataset_id}{extension}"
    # source_url = source_url.replace("https:/", "https://").replace("http:/", "http://")

    # # Local cache path
    # cache_dir = Path("/tmp/datalad_cache")
    # hidden_dataset_path = cache_dir / dataset_id

    # # Handle force re-download
    # if hidden_dataset_path.exists() and force_download:
    #     shutil.rmtree(hidden_dataset_path)

    # # Clone if not exists
    # if not hidden_dataset_path.exists():
    #     logger.info("Cloning DataLad dataset into hidden cache...")
    #     cache_dir.mkdir(parents=True, exist_ok=True)

    #     # dl.clone returns a Dataset object — we extract the path
    #     cloned_ds = dl.clone(
    #         source=source_url,
    #         path=str(hidden_dataset_path),
    #         result_renderer="disabled",
    #     )
    #     # IMPORTANT: Return the path string, not the Dataset object
    #     return str(cloned_ds.path)

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


def _get_candidate_files(
    hidden_dataset_path: Path,
    subjects: str | list[str] = "03286",
    sessions: str | list[str] = "NOT1ACH001",
    modalities: str | list[str] = "anat",
    data_types: str | list[str] = "T1w",
    extensions: str | list[str] = ".json",
):
    candidate_files = []

    subj_dirs = (
        list(hidden_dataset_path.glob("sub-*")) if subjects == "all" else [hidden_dataset_path / f"sub-{s}" for s in subjects]
    )
    logger.debug(f"Processing directory: {hidden_dataset_path}")

    for subj_dir in subj_dirs:
        if not subj_dir.exists():
            continue

        ses_dirs = list(subj_dir.glob("ses-*")) if sessions == "all" else [subj_dir / f"ses-{s}" for s in sessions]
        logger.debug(f"Processing directory: {subj_dir}")

        for ses_dir in ses_dirs:
            if not ses_dir.exists():
                continue
            mod_dirs = list(ses_dir.glob("*")) if modalities == "all" else [ses_dir / mod for mod in modalities]
            logger.debug(f"Processing directory: {ses_dir}")

            for mod_dir in mod_dirs:
                if not mod_dir.exists():
                    continue
                logger.debug(f"Processing directory: {mod_dir}")

                dt_dirs = list(mod_dir.glob("*")) if data_types == "all" else [mod_dir / dt for dt in data_types]
                logger.debug(f"dt_dirs: {dt_dirs}")
                for dt in data_types:
                    for ex_dir in extensions:
                        logger.debug(f"Adding candidate file: {mod_dir.glob(f'*{dt}{ex_dir}')}")

                        candidate_files.extend(mod_dir.glob(f"*{dt}{ex_dir}"))

    if not candidate_files:
        raise ValueError(
            f"No matching files found for: \t"
            f"subjects: {subjects}\t"
            f"Sessions: {sessions}\t"
            f"Modalities: {modalities}\t"
            f"Data Type: {data_types}\t"
            f"Extensions: {extensions}"
        )
    return candidate_files


def _validate_arguments(subjects, sessions, modalities, data_types, extensions):
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
    data_types = _validate(data_types)
    extensions = _validate(extensions)

    return subjects, sessions, modalities, data_types, extensions


####################################################################################################
#### This functions will be useful in the future
# TODO: fetch this data using pooch to avoid having files hardcoded in the codebase.
def _get_onharmony_information_form_idps() -> dict:
    """Get information about the ON-Harmony dataset from the IDPs.

    This function is a placeholder for future implementation. It should
    retrieve information about the dataset (e.g., available subjects,
    scanners, modalities, data types) from the IDPs and populate the
    corresponding lists.

    Returns
    -------
    dict
        A dictionary containing lists of subjects, scanners, modalities,
        and data types.

    """
    idps = pd.read_csv("onharmony_idps.csv")
    subjects_id = list(idps["subject"].unique())
    scanner_id = list(idps["scanner_code"].unique())
    scanner_name = list(idps["scanner_code"].unique())
    scanner_vendor = list(idps["vendor"].unique())

    ONHARMONY_MODALITY_LIST = ["anat", " dwi", "fmap", "func", "swi"]
    ONHARMONY_DATA_TYPE_LIST = []

    return {
        "subjects_id": subjects_id,
        "scanners_id": scanner_id,
        "scanner_name": scanner_name,
        "scanner_vendor": scanner_vendor,
        "modalities": ONHARMONY_MODALITY_LIST,
        "data_types": ONHARMONY_DATA_TYPE_LIST,
    }


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


def _list_available_possibilities() -> dict:
    """List all available combinations.

    Returns
    -------
    dict
        A dictionary with the possible combinations of modalities, data type and extension.

    """
    onharmony_posibilities = {
        "anat": {"T1w": [".nii.gz", ".json"], "T2w": [".nii.gz", ".json"], "mod-T1w_defacemask": [".nii.gz"]},
        "dwi": {"dir-AP_dwi": [".nii.gz", ".json", ".bval", ".bvec"], "dir-PA": [".nii.gz", ".json", ".bval", ".bvec"]},
        "fmap": {
            "dir-AP_epi": [".nii.gz", ".json"],
            "dir-PA_epi": [".nii.gz", ".json"],
        },
        "func": {
            "task-rest_bold": [".nii.gz", ".json"],
        },
        "swi": {
            "echo-1_part-mag_GRE": [".nii.gz", ".json"],
            "echo-1_part-phase_GRE": [".nii.gz", ".json"],
            "echo-2_part-mag_GRE": [".nii.gz", ".json"],
            "echo-2_part-phase_GRE": [".nii.gz", ".json"],
        },
    }
    return onharmony_posibilities
