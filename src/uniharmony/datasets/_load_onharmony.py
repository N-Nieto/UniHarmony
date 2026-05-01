"""ON-Harmony dataset downloader using DataLad."""

from pathlib import Path
from typing import Literal

import structlog

from uniharmony.datasets._datalad_integration import (
    get_candidate_files,
    get_files,
    initialize_dl_dataset,
    validate_arguments,
)


logger = structlog.get_logger()

__all__ = ["_list_available_possibilities", "load_onharmony"]

ONHARMONY_MODALITY_LIST = Literal["all", "anat", "dwi", "fmap", "func", "swi"]

ONHARMONY_SUBJECTS_LIST = Literal[
    "all",
    "03286",
    "03997",
    "10975",
    "12813",
    "13192",
    "13305",
    "14221",
    "14229",
    "14230",
    "14482",
    "15320",
    "16745",
    "16766",
    "16793",
    "16794",
    "16841",
    "16842",
    "16974",
    "16975",
    "16981",
]


def load_onharmony(
    subjects: str | list[str] | ONHARMONY_SUBJECTS_LIST,
    sessions: str | list[str],
    modalities: str | list[str] | ONHARMONY_MODALITY_LIST,
    target_path: str | Path = ".",
    dataset_name: str = "ONHarmony",
    suffixes: str | list[str] = "T1w",
    extensions: str | list[str] = ".json",
    dataset_source: str = "https://github.com/OpenNeuroDatasets/",
    dataset_id: str = "ds004712",
    force_download=False,
    copy=True,
    cache=False,
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
        Modalities to download ("all", "anat", " dwi", "fmap", "func", "swi").

    target_path : str or pathlib.Path, default "."
        Path to the visible dataset directory where files will be stored.

    dataset_name : str, default "ONHarmony"
        Name for the visible dataset.

    suffixes : str or list[str], default "T1w"
        BIDS suffixes to match in filenames (e.g., 'T1w', 'T2w').

    extensions : str or list[str], default ".json"
        File extensions to download (e.g., '.json', '.nii.gz').

    dataset_source : str, default "https://github.com/OpenNeuroDatasets/
        Source URL or path to the ON-Harmony dataset.

    dataset_id : str, default "ds004215"
        Identifier for the dataset to download (e.g., "ds004215" for ON-Harmony).

    force_download : bool, default False
        Whether to force re-download the dataset if it already exists in cache.

    copy : bool, default True
        Whether to copy the downloaded files to the target directory to make it visible.

    cache : bool, default False
        Whether to cache the downloaded files in the hidden DataLad dataset.
        If False, files are dropped immediately after copying to minimize disk usage.

    Notes
    -----
    - The visible dataset directory will contain only regular files
      following the BIDS derivatives structure.
    - Repeated calls are safe and will only download missing files.

    """
    # ------------------------------------------------------------------
    #  Validate arguments
    # ------------------------------------------------------------------
    subjects, sessions, modalities, suffixes, extensions = validate_arguments(
        subjects, sessions, modalities, suffixes, extensions
    )
    # ------------------------------------------------------------------
    #  Initialize the hidden DataLad dataset and the visible directory
    # ------------------------------------------------------------------
    ds, hidden_dataset_path, dataset_path = initialize_dl_dataset(
        target_path, dataset_name, dataset_source, dataset_id, force_download
    )

    # ------------------------------------------------------------------
    # Collect candidate files first (for progress bar support)
    # ------------------------------------------------------------------
    candidate_files = get_candidate_files(hidden_dataset_path, subjects, sessions, modalities, suffixes, extensions)

    # ------------------------------------------------------------------
    # Download the actual files. Make it visible if copy=True. Drop from cache if cache=False.
    # ------------------------------------------------------------------
    get_files(ds, candidate_files, hidden_dataset_path, dataset_path, copy, cache)

    return


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
