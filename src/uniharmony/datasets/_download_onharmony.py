"""ON-Harmony dataset downloader using DataLad."""

from pathlib import Path
from typing import Literal

import structlog

from uniharmony.datasets import download_bids_dataset


__all__ = ["download_ONharmony", "list_available_possibilities_onharmony"]

logger = structlog.get_logger()


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

ONHARMONY_SESSIONS_LIST = Literal[
    "all",
    "NOT1ACH001",
    "NOT2ING001",
    "NOT3GEM001",
    "OXF1PRI001",
    "OXF2PRI001",
    "OXF3TRI001",
    "OXF1PRI002",
    "OXF1PRI003",
    "OXF1PRI004",
    "OXF1PRI005",
    "OXF1PRI006",
    "OXF2PRI002",
    "OXF2PRI003",
    "OXF2PRI004",
    "OXF2PRI005",
    "OXF2PRI006",
    "OXF3TRI002",
    "OXF3TRI003",
    "OXF3TRI004",
    "OXF3TRI005",
    "OXF3TRI006",
    "NOT1ACH002",
    "NOT1ACH003",
    "NOT1ACH004",
    "NOT1ACH005",
    "NOT1ACH006",
    "NOT4GEP001",
    "NOT2ING002",
    "NOT2ING003",
    "NOT2ING004",
    "NOT2ING005",
    "NOT2ING006",
    "NOT4GEP002",
    "NOT4GEP003",
    "NOT4GEP004",
    "NOT4GEP005",
    "NOT4GEP006",
]

ONHARMONY_MODALITY_LIST = Literal["all", "anat", "dwi", "fmap", "func", "swi"]

ONHARMONY_SUFFIXES_LIST = Literal[
    "all",
    "T1w",
    "T2w",
    "mod-T1w_defacemask",
    "dir-AP_dwi",
    "dir-PA",
    "dir-AP_epi",
    "dir-PA_epi",
    "task-rest_bold",
    "echo-1_part-mag_GRE",
    "echo-1_part-phase_GRE",
    "echo-2_part-mag_GRE",
    "echo-2_part-phase_GRE",
]
ONHARMONY_EXTENSIONS_LIST = Literal["all", ".nii.gz", ".json", ".bval", ".bvec"]

ROOT_FILES = Literal[
    "all",
    "participants.tsv",
    "all_idps.csv",
    "dataset_description.json",
    "data_dictionary.pdf",
    "participants.json",
    "README",
    "protocol_accelerations.csv",
    "radiographers.csv",
    "scan_delays.csv",
    "software_versions.csv",
    "CHANGES",
    ".bidsignore",
    ".gitattributes",
]


def download_ONharmony(  # noqa: N802
    subjects: str | list[str] | ONHARMONY_SUBJECTS_LIST,
    sessions: str | list[str] | ONHARMONY_SESSIONS_LIST,
    modalities: str | list[str] | ONHARMONY_MODALITY_LIST,
    suffixes: str | list[str] | ONHARMONY_SUFFIXES_LIST = "T1w",
    extensions: str | list[str] | ONHARMONY_EXTENSIONS_LIST = ".json",
    target_path: str | Path = "./ON-Harmony",
    root_files: str | list[str] | ROOT_FILES = "participants.tsv",
    force_download: bool = False,
    copy: bool = True,
    hidden: bool = True,
    tmp_clean: bool = False,
    tmp_dir_name: str = "datalad_cache",
) -> None:
    """Download derivatives from the ON-Harmony dataset and store them as files in a directory.

    This function is a particular case of the `download_bids_dataset` function. For details on how the function
    downloads the data using datalad, please refer to the `download_bids_dataset` documentation.

    Note that not all subjects have the same session.
    Additionally, not all the modalities have the same suffixes or extensions.
    Use `list_available_possibilities_onharmony` to retrieve the possible combinations.

    Please check https://openneuro.org/datasets/ds004712/versions/2.0.1 for further details on the dataset structure.

    Parameters
    ----------
    subjects : str or list
        Subject identifiers to download.

    sessions : str or list
        Session identifiers to download.

    modalities : str or list
        Modalities to download ("all", "anat", " dwi", "fmap", "func", "swi").

    suffixes : str or list, optional (default "T1w")
        BIDS suffixes to match in filenames (e.g., 'T1w', 'T2w').

    extensions : str or list, optional (default ".json")
        File extensions to download (e.g., '.json', '.nii.gz').

    target_path : str or pathlib.Path, optional (default "./ON-Harmony")
        Path to the visible dataset directory where files will be stored.

    dataset_url : str, optional (default "https://github.com/OpenNeuroDatasets/")
        Source URL or path to the ON-Harmony dataset.

    root_files: str or list, optional (default "participants.tsv")
        Name of the file list of files to get from the dataset's root.

    force_download : bool, optional (default False)
        Whether to force re-download the dataset if it already exists in cache.

    copy : bool, optional (default True)
        Whether to copy the downloaded files to the target directory to make it visible.

    hidden : bool, optional (default True)
        Whether to use a hidden directory or not.
        If hidden=False, no hidden folder is made and the target directory acts as hidden.
        This will avoid getting the files in ``/tmp/{tmp_dir_name}`` and then copying them
        to the target directory.

    tmp_clean : bool, optional (default False)
        Whether to drop the downloaded files from the hidden DataLad dataset after copying.
        If True, files are dropped immediately after copying to the target directory
        (if copy=True), to minimize disk usage. Ignored when ``hidden=False``.

    tmp_dir_name : str, optional (default "datalad_cache")
        Name of the temporary directory to store the hidden dataset. Ignored when ``hidden=False``.

    Notes
    -----
     - The visible dataset directory will contain only regular files
       following the BIDS derivatives structure.
     - Repeated calls are safe and will only download missing files.

    """
    download_bids_dataset(
        subjects=subjects,
        sessions=sessions,
        modalities=modalities,
        suffixes=suffixes,
        extensions=extensions,
        target_path=target_path,
        dataset_url="https://github.com/OpenNeuroDatasets/ds004712.git",  # Fix to OpenNeuro ON-HARMONY
        root_files=root_files,
        force_download=force_download,
        copy=copy,
        tmp_clean=tmp_clean,
        tmp_dir_name=tmp_dir_name,
        hidden=hidden,
        tasks="all",  # this dataset does not have any task
        runs="all",  # this dataset does not have any run
    )


def list_available_possibilities_onharmony() -> dict:
    """List all available combinations in the ON-Harmony dataset.

    Returns
    -------
    dict
        A dictionary with the possible combinations of modalities, data type and extension.

    """
    return {
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
