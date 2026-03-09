"""Functions to load the MAREoS datasets.

The MAREoS (Methods Aiming to Remove Effect of Site) datasets provide
standardized benchmarks for evaluating data harmonization
methods with known ground truth.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pooch
import structlog
from pooch import HTTPDownloader, Unzip


__all__ = ["_ensure_mareos_data", "load_MAREoS"]

logger = structlog.get_logger()

# Constants
MAREOS_ZIP_URL = (
    "https://www.imardgroup.com/mareos-benchmark/public_datasets.zip"
)

VALID_EFFECTS = ["eos", "true"]
VALID_EFFECT_TYPES = ["simple", "interaction"]
VALID_EFFECT_EXAMPLES = ["1", "2"]

# All dataset file names (without extension)
DATASET_NAMES = [
    "eos_simple1",
    "eos_simple2",
    "eos_interaction1",
    "eos_interaction2",
    "true_simple1",
    "true_simple2",
    "true_interaction1",
    "true_interaction2",
]

# Initialize Pooch instance for the ZIP file
mareos_pooch = pooch.create(
    path=pooch.os_cache("uniharmony"),
    base_url="https://www.imardgroup.com/mareos-benchmark/",
    registry={
        "public_datasets.zip": None,
    },
    retry_if_failed=3,
)


def load_MAREoS(  # noqa: N802
    effects: list[str] | str | None = None,
    effect_types: list[str] | str | None = None,
    effect_examples: list[str] | str | None = None,
    as_numpy: bool = True,
    data_dir: Path | None = None,
    force_download: bool = False,
    verbose: bool = False,
) -> dict[str, dict[str, pd.DataFrame | np.ndarray]]:
    """Load multiple MAREoS datasets.

    Parameters
    ----------
    effects : list of str, str or None, optional (default None)
        List of effects to load. If None, loads all ["eos", "true"]
    effect_types : list of str, str or None, optional (default None)
        List of effect types to load.
        If None, loads all ["simple", "interaction"]
    effect_examples : list of str, str or None, optional (default None)
        List of examples to load.
        If None, loads all ["1", "2"].
    as_numpy : bool, optional (default True)
        If True, return ``numpy.ndarray``, else ``pandas.DataFrame``.
    data_dir : Path | None, optional (default None)
        Directory containing MAREoS data files. If None, downloads to cache.
    force_download : bool, optional (default False)
        Force to download again the dataset in case of corrupt files.
    verbose : bool, optional (default False)
        Control verbosity.

    Returns
    -------
    dict of str and dict
        Nested dictionary where keys are dataset names
        containing:
        - "X": Feature matrix
        - "y": Target labels
        - "sites": Site labels
        - "covs": Covariates
        - "folds": Cross-validation folds

    Raises
    ------
    ValueError
        If any parameter contains invalid values.

    Examples
    --------
    >>> datasets = load_MAREoS()
    >>> len(datasets)
    8
    >>> datasets = load_MAREoS(effects=["eos"], effect_types=["simple"])
    >>> len(datasets)
    2
    >>> list(datasets.keys())
    ['eos_simple1', 'eos_simple2']

    """
    effects, effect_types, effect_examples = _validate_mareos_parameters(
        effects, effect_types, effect_examples
    )

    # Ensure all requested data is available
    data_dir = _ensure_mareos_data(data_dir, force_download, verbose)

    # Load all requested datasets
    dataset_dict = {}

    for effect in effects:
        for e_type in effect_types:
            for e_example in effect_examples:
                dataset_name = f"{effect}_{e_type}{e_example}"

                X, y, sites, covariates, folds = _load_mareos_single_dataset(
                    data_dir=data_dir,
                    effect=effect,
                    effect_type=e_type,
                    effect_example=e_example,
                    as_numpy=as_numpy,
                    verbose=verbose,
                )

                dataset_dict[dataset_name] = {
                    "X": X,
                    "y": y,
                    "sites": sites,
                    "covs": covariates,
                    "folds": folds,
                }

    return dataset_dict


def _load_mareos_single_dataset(
    data_dir: Path,
    effect: str,
    effect_type: str,
    effect_example: str,
    as_numpy: bool = True,
    verbose: bool = False,
) -> tuple[pd.DataFrame | np.ndarray, ...]:
    """Load a single MAREoS dataset.

    This function is expected to be called only within "load_MAREoS"

    Parameters
    ----------
    data_dir : Path
        Directory containing MAREoS data files.
    effect : str
        Type of effect. Options: "true" (true biological signal) or
        "eos" (effect of site, spurious signal)
    effect_type : str
        Type of effect. Options: "simple" (linear effects) or
        "interaction" (non-linear interactions)
    effect_example : str
        Which of the two simulated datasets to load. Options: "1" or "2"
    as_numpy : bool, optional (default True)
        If True, return ``numpy.ndarray``, else ``pandas.DataFrame``.
    verbose : bool, optional (default False).
        Control verbosity.

    Returns
    -------
    X : pd.DataFrame or np.ndarray
        Feature matrix of shape (1000, 15)
    y : pd.DataFrame or np.ndarray
        Target labels of shape (1000, 1)
    sites : pd.Series or np.ndarray
        Site labels of shape (1000,)
    covariates : pd.DataFrame or np.ndarray
        Covariates (cov1, cov2) of shape (1000, 2)
    folds : pd.Series or np.ndarray
        Predefined cross-validation folds of shape (1000,)

    Raises
    ------
    ValueError
        If any parameter has an invalid value.
    RuntimeError
        If ``data_dir`` is not found.
    FileNotFoundError
        If the requested dataset files are not found.

    """
    # Ensure data is available
    if not isinstance(data_dir, Path):
        raise TypeError(f"data_dir must be a Path, got: {type(data_dir)}")

    if not data_dir.exists():
        raise RuntimeError(f"Data directory not found: {data_dir}")

    # Construct dataset name and file paths
    dataset_name = f"{effect}_{effect_type}{effect_example}"

    data_file = data_dir / "public_datasets" / f"{dataset_name}_data.csv"
    response_file = (
        data_dir / "public_datasets" / f"{dataset_name}_response.csv"
    )
    if verbose:
        logger.info(f"Getting data file: {data_file}")
    # Verify files were found
    if not data_file.exists():
        raise FileNotFoundError(
            f"Data file not found for dataset {dataset_name}. "
            f"Searched in: {data_dir}"
        )

    if not response_file.exists():
        raise FileNotFoundError(
            f"Response file not found for dataset {dataset_name}. "
            f"Searched in: {data_dir}"
        )

    X_df = pd.read_csv(data_file, index_col=0)
    y_df = pd.read_csv(response_file, index_col=0)

    sites = X_df["site"]
    covariates = X_df.loc[:, ["cov1", "cov2"]]
    folds = X_df["folds"]

    # Drop metadata columns from features
    X_df = X_df.drop(columns=["site", "cov1", "cov2", "folds"])

    # Convert to numpy if requested
    if as_numpy:
        X = X_df.to_numpy()
        y = y_df.to_numpy().flatten()  # Flatten to 1D array
        sites = sites.to_numpy()
        covariates = covariates.to_numpy()
        folds = folds.to_numpy()
    else:
        X = X_df
        y = y_df.iloc[:, 0]  # Get first column as Series

    return X, y, sites, covariates, folds  # type: ignore


def _ensure_mareos_data(
    data_dir: Path | str | None = None,
    force_download: bool = False,
    verbose: bool = False,
) -> Path:
    """Ensure MAREoS datasets are available locally, downloading if necessary.

    Downloads the entire ZIP file and extracts it to the cache directory.

    Parameters
    ----------
    data_dir : Path or str or None, optional (default None)
        Custom directory to store data. If None, uses default cache location.
    force_download : bool, optional (default False)
        Force re-download even if files exist.
    verbose : bool, optional (default False)
        Control verbosity.

    Returns
    -------
    Path
        Path to the directory containing the extracted MAREoS data files.

    Raises
    ------
    ConnectionError
        If unable to download or extract the ZIP file.

    Examples
    --------
    >>> data_path = ensure_mareos_data()
    >>> data_path = ensure_mareos_data(data_dir=Path("./data"))

    """
    # Determine target directory
    target_dir = _create_target_dir(data_dir)

    # Check if all files exist
    files_exist = all(
        (target_dir / "public_datasets" / f"{name}_data.csv").exists()
        and (target_dir / "public_datasets" / f"{name}_response.csv").exists()
        for name in DATASET_NAMES
    )

    if not force_download and files_exist:
        if verbose:
            logger.info(f"MAREoS datasets already exist at: {target_dir}")
        return target_dir

    # If files do not exist of force download
    # Download and extract ZIP file
    # Pooch's Unzip processor returns a list of extracted file paths
    extracted_files = mareos_pooch.fetch(
        "public_datasets.zip",
        downloader=HTTPDownloader(progressbar=True),  # type: ignore
        processor=Unzip(extract_dir=target_dir),
    )
    # Verify extraction
    if not extracted_files:
        raise RuntimeError("No files extracted from ZIP")
    check_dir = Path(extracted_files[0]).parent

    # Count CSV files
    csv_files = list(check_dir.glob("*.csv"))
    if not csv_files:
        # Also check in target_dir in case files are there
        csv_files = list(check_dir.glob("*.csv"))

    if not csv_files:
        raise RuntimeError(f"No CSV files found in {check_dir}")

    if verbose:
        logger.info(
            f"MAREoS datasets downloaded: {len(csv_files)} CSV files in "
            f"{target_dir}"
        )

    return target_dir


# ######## Internal helper functions
# Validate dataset's parameters
def _validate_mareos_parameters(
    effects: str | list[str] | None = None,
    effect_types: str | list[str] | None = None,
    effect_examples: str | list[str] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Validate and normalize MAREoS dataset selection parameters.

    Parameters
    ----------
    effects : str, list of str, or None, optional (default None)
        Type(s) of effect to load. Valid: "eos", "true".
    effect_types : str, list of str, or None, optional (default None)
        Type(s) of effect pattern. Valid: "simple", "interaction".
    effect_examples : str, list of str, or None, optional (default None)
        Example number(s). Valid: "1", "2".

    Returns
    -------
    tuple
        Normalized lists of validated parameters.

    """
    # Process all parameters
    effects_list = _process_effect_param(effects, VALID_EFFECTS, "effects")
    types_list = _process_effect_param(
        effect_types, VALID_EFFECT_TYPES, "effect_types"
    )
    examples_list = _process_effect_param(
        effect_examples, VALID_EFFECT_EXAMPLES, "effect_examples"
    )

    return effects_list, types_list, examples_list


def _create_target_dir(data_dir: Path | str | None) -> Path:
    """Create target directory.

    Parameters
    ----------
    data_dir : pathlib.Path or str or None
        Base data directory.

    Returns
    -------
    pathlib.Path
        Target directory.

    """
    if data_dir is None:
        # Use Pooch's cache directory
        target_dir = Path(mareos_pooch.path) / "MAREoS"
    else:
        # Create a target dir using the absolute path
        data_dir = Path(data_dir).absolute()
        target_dir = data_dir / "MAREoS"

    # Create directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


# Process each parameter using a helper function
def _process_effect_param(
    param: str | list[str] | None,
    default_values: list[str],
    param_name: str,
) -> list[str]:
    """Process and validate a single parameter.

    Parameters
    ----------
    param : str or list of str or None
        Effect parameter.
    default_values : list of str
        Default values for ``param``.
    param_name : str
        Parameter name.

    Returns
    -------
    list of str
        Validated parameter.

    Raises
    ------
    ValueError
        If ``param`` has invalid values.
    TypeError
        If ``param`` has invalid type.

    """
    # Set default if None
    if param is None:
        return default_values.copy()

    # Convert string to list
    if isinstance(param, str):
        param = [param]

    # Validate type
    if not isinstance(param, list):
        raise TypeError(f"{param_name} must be str or list, got {type(param)}")

    # Validate values
    invalid = [v for v in param if v not in default_values]
    if invalid:
        raise ValueError(
            f"{param_name} contains invalid value(s): {invalid}. Must be one "
            f"of {default_values}"
        )

    return param
