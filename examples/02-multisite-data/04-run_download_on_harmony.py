"""
Download ON-Harmony dataset
=======================
"""

# %%
# Imports
# -------

from uniharmony import verbosity
from uniharmony.datasets import clean_tmp, download_ONharmony
verbosity("debug")


# %%
# Load dataset
# ------------

download_ONharmony(
    subjects="16981",
    sessions="NOT4GEP001",
    modalities="anat",
    suffixes="T1w",
    extensions=".json",
    # If you aim to download the dataset in a "permanent" folder, please set copy=True
    copy=False,
)


# %%
# Clean up
# --------

# Later, clean up the temporary cache
clean_tmp("datalad_cache")
