"""ON-Harmony dataset downloader using DataLad."""

from pathlib import Path
from typing import Literal

import structlog

from uniharmony.datasets import download_derivatives_bids_dataset


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
    suffixes: str | list[str] = "T1w",
    extensions: str | list[str] = ".json",
    target_path: str | Path = "./ON-Harmony",
    dataset_source: str = "https://github.com/OpenNeuroDatasets/ds004712.git",
    root_files: str | list[str] = "participants.tsv",
    force_download: bool = False,
    copy: bool = True,
    hidden: bool = True,
    tmp_clean: bool = False,
    tmp_dir_name: str = "datalad_cache",
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

    suffixes : str or list[str], default "T1w"
        BIDS suffixes to match in filenames (e.g., 'T1w', 'T2w').

    extensions : str or list[str], default ".json"
        File extensions to download (e.g., '.json', '.nii.gz').

    target_path : str or pathlib.Path, default "./ONHarmony"
        Path to the visible dataset directory where files will be stored.

    dataset_source : str, default "https://github.com/OpenNeuroDatasets/"
        Source URL or path to the ON-Harmony dataset.

    dataset_id : str, default "ds004215"
        Identifier for the dataset to download (e.g., "ds004215" for ON-Harmony).

    dataset_extension : str
        Web extension. For example .git.

    root_files: str | list[str].
        Name of the file list of files to get from the dataset's root.

    force_download : bool, default False
        Whether to force re-download the dataset if it already exists in cache.

    copy : bool, default True
        Whether to copy the downloaded files to the target directory to make it visible.

    hidden : bool, default True
        Whether to use a hidden directory or not.
        If hidden=False, no hidden folder is made and the target directory acts as hidden.
        This will avoid getting the files in ``/tmp/{tmp_dir_name}`` and then copying them
        to the target directory.

    tmp_clean : bool, default False
        Whether to drop the downloaded files from the hidden DataLad dataset after copying.
        If True, files are dropped immediately after copying to the target directory
        (if copy=True), to minimize disk usage. Ignored when ``hidden=False``.

    tmp_dir_name : str, default "datalad_cache"
        Name of the temporary directory to store the hidden dataset. Ignored when ``hidden=False``.

    Notes
    -----
     - The visible dataset directory will contain only regular files
       following the BIDS derivatives structure.
     - Repeated calls are safe and will only download missing files.

    """
    # Use the generic function to load a BIDS-compatible dataset.
    download_derivatives_bids_dataset(
        subjects=subjects,
        sessions=sessions,
        modalities=modalities,
        suffixes=suffixes,
        extensions=extensions,
        target_path=target_path,
        dataset_source_URL=dataset_source,
        root_files=root_files,
        force_download=force_download,
        copy=copy,
        tmp_clean=tmp_clean,
        tmp_dir_name=tmp_dir_name,
        hidden=hidden,
        tasks="all",  # this dataset does not have any task
        runs="all",  # this dataset does not have any run
    )

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
