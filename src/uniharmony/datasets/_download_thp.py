"""ON-Harmony dataset downloader using DataLad."""

from pathlib import Path
from typing import Literal

import structlog

from uniharmony.datasets import download_bids_dataset


__all__ = ["download_THP"]

logger = structlog.get_logger()


SUBJECTS_LIST = Literal[
    "all",
    "THP0001",
    "THP0002",
    "THP0003",
    "THP0004",
    "THP0005",
    "THPBALL",
]

SESSIONS_LIST = Literal["all", "CCF1", "DART1", "IOWA1", "IOWA2", "IOWA3", "JHU1", "MGH1", "UCI1", "UMN1", "UW1"]

MODALITY_LIST = Literal["all", "anat", "dwi"]

RUNS_LIST = Literal["01", "02", "03", "04"]

SUFFIXES_LIST = Literal[
    "all",
    "T1w",
    "T2w",
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
EXTENSIONS_LIST = Literal["all", ".nii.gz", ".json", ".bval", ".bvec"]

ROOT_FILES = Literal[
    "dataset_description.json",
    "README",
    "CHANGES",
    ".gitattributes",
]


def download_THP(  # noqa: N802
    subjects: str | list[str] | SUBJECTS_LIST,
    sessions: str | list[str] | SESSIONS_LIST,
    modalities: str | list[str] | MODALITY_LIST,
    runs: str | list[str] | RUNS_LIST,
    suffixes: str | list[str] | SUFFIXES_LIST = "T1w",
    extensions: str | list[str] | EXTENSIONS_LIST = ".json",
    target_path: str | Path = "./THP",
    root_files: str | list[str] | ROOT_FILES = "dataset_description.tsv",
    force_download: bool = False,
    copy: bool = True,
    hidden: bool = True,
    tmp_clean: bool = False,
    tmp_dir_name: str = "datalad_cache",
) -> None:
    """Download derivatives from the Traveling Human Phantom Study dataset and store them as files in a directory.

    This function is a particular case of the `download_bids_dataset` function. For details on how the function
    downloads the data using datalad, please refer to the `download_bids_dataset` documentation.


    Please check https://openneuro.org/datasets/ds000206/versions/1.0.0 for further details on the dataset structure.

    Parameters
    ----------
    subjects : str or list
        Subject identifiers to download.

    sessions : str or list
        Session identifiers to download.

    modalities : str or list
        Modalities to download ("all", "anat", " dwi", "fmap", "func", "swi").

    runs : str or list
        Runs to download in the DWI data ("01", "02", "03", "04").

    suffixes : str or list, optional (default "T1w")
        BIDS suffixes to match in filenames (e.g., 'T1w', 'T2w').

    extensions : str or list, optional (default ".json")
        File extensions to download (e.g., '.json', '.nii.gz').

    target_path : str or pathlib.Path, optional (default "./THP")
        Path to the visible dataset directory where files will be stored.

    dataset_url : str, optional (default "https://github.com/OpenNeuroDatasets/")
        Source URL or path to the THP dataset.

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
        dataset_url="https://openneuro.org/datasets/ds000206/versions/1.0.0",  # Fix to OpenNeuro ON-HARMONY
        root_files=root_files,
        force_download=force_download,
        copy=copy,
        tmp_clean=tmp_clean,
        tmp_dir_name=tmp_dir_name,
        hidden=hidden,
        tasks="all",  # this dataset does not have any task
        runs=runs,  # this dataset does not have any run
    )
