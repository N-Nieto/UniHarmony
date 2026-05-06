"""
Load ON-Harmony dataset
=======================
"""

# %%
# Imports
# -------

from uniharmony import verbosity
from uniharmony.datasets import clean_tmp, load_ONharmony


verbosity("debug")


# %%
# Load dataset
# ------------

load_ONharmony(
    subjects="16981",
    sessions="NOT4GEP001",
    modalities="anat",
    suffixes="T1w",
    extensions=".json",
)


# %%
# Clean up
# --------

# Later, clean up the temporary cache
clean_tmp("datalad_cache")
