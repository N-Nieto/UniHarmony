"""Load structural extracted features."""

# Download the file using
from pathlib import Path

import pandas as pd
import pooch


__all__ = ["load_onharmony_structural_features"]


def load_onharmony_structural_features(
    url: str = "https://raw.githubusercontent.com/Jake-Turnbull/HarmonisationDiagnostics/main/tests/onharmony.csv",
    data_dir: str | Path | None = None,
    output_file: str = "onharmony.csv",
) -> pd.DataFrame:
    """Load ON-Harmony structural features CSV file, downloading if necessary.

    Downloads the ON-Harmony structural features dataset from GitHub and caches it
    locally. Uses pooch for file management with SHA-256 hash verification.

    Parameters
    ----------
    url : str (default: GitHub raw URL)
        URL to download the ON-Harmony structural features CSV file.

    data_dir : Path or str or None (default "None")
        Directory to store the downloaded file. If None, uses pooch's default

        cache location.
    output_file : str (default "onharmony.csv")
        Local filename to save the downloaded CSV file as.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the ON-Harmony structural features data.

    Raises
    ------
    ConnectionError
        If unable to download the file from the specified URL.
    ValueError
        If the downloaded file fails hash verification.

    Examples
    --------
    >>> df = load_onharmony_structural_features()

    """
    # Convert data_dir to string if it's a Path object
    path = str(data_dir) if data_dir is not None else None

    # Download the file using pooch
    try:
        file_path = pooch.retrieve(
            url=url,
            known_hash="sha256:0a947c76aa8109f54f43be20cb85ad2a25d148a51b975b4415889ffa05f6f63c",
            fname=output_file,
            path=path,
            progressbar=False,  # small file, no need for progressbar.
        )
    except Exception as e:
        raise ConnectionError(f"Failed to download ON-Harmony data from {url}: {e}") from e
    # Read and return the CSV file
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise ValueError(f"Failed to read CSV file from {file_path}: {e}") from e
