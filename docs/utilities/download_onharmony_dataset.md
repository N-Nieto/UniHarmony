# UniHarmony Helper function for downloading ON-Harmony dataset.

## Overview

This module provides a high-level interface for downloading the ON-Harmony dataset using datalad.

The function allows to download specific files from the OpenNeuro repository.

Note: Not all files exist for all sessions.


### Understanding File Availability

The ON-Harmony dataset (~58 GB) contains [^18^]:

- **20 participants** × **6 scanners** each
- **9 participants** with additional **5 within-scanner repeats**
- **5 modalities**: T1w, T2w, SWI, dMRI, rfMRI
- Defaced anatomical images with defacing masks


## Complete Examples

### Example 1: Download All files from a subject and a session

```python
from uniharmony.datasets import load_ONharmony

# Download everything for two participants
load_ONharmony(
    subjects="15320",
    sessions="NOT1ACH001",
    modalities="anat",
    suffixes="all",
    extensions="all",
    target_path="./ON-Harmony",
    dataset_source_URL="https://github.com/OpenNeuroDatasets/ds004712.git", # This is also the default
    root_files=[],  # Passing an empty list to not get any file
    hidden=True,
    copy=True,
    tmp_clean=False,  # Keep cache for reuse, this allows to recall the same function and not downloading the dataset again.
)

# Now if we want all the sessions for the same subject, as we did not clean the tmp directory, there is no need for clone the dataset again
# This will speed up the process, as we only need to "get" the files.

# Download everything for two participants
load_ONharmony(
    subjects="15320",
    sessions="all",
    modalities="anat",
    suffixes="all",
    extensions="all",
    target_path="./ON-Harmony",
    root_files=[],  # Passing an empty list to not get any file
    hidden=True,
    copy=True,
    tmp_clean=True,  # Now we clean the tmp
)
```

### Example 2: Direct Download (No Hidden Cache)

```python
from uniharmony.datasets import (
    load_ONharmony,
)

# Download directly to target (no cache)
load_ONharmony(
    subjects="15320",
    sessions="NOT1ACH001",
    modalities="anat",
    suffixes="all",
    extensions="all",
    target_path="./ON-Harmony",
    root_files=[],  # Passing an empty list to not get any file
    hidden=False,
    copy=True,          # Not use when hidden is False
    tmp_clean=False,    # Not use when hidden is False
)

```

### Example 4: Download Diffusion MRI with All Sidecars

```python
from uniharmony.datasets import load_ONharmony

# DWI requires .nii.gz, .json, .bval, and .bvec
load_ONharmony(
    subjects="all",
    sessions="all",
    modalities="dwi",
    tasks="all",
    runs="all",
    target_path="./on_harmony_dwi",
    suffixes="dwi",
    extensions=[".nii.gz", ".json", ".bval", ".bvec"],
    root_files=["dataset_description.json"],
    hidden=True,
    copy=True,
    tmp_clean=True,
)
```
---

## Advanced Usage

### Working with ON-Harmony Session Names

The ON-Harmony dataset uses session codes that encode scanner information [^18^]:

| Session Code | Scanner | Site |
|-------------|---------|------|
| `NOT1ACH001` | Philips Achieva | Nottingham |
| `NOT2ING001` | Philips Ingenia | Nottingham |
| `NOT3GEM001` | GE MR750 | Nottingham |
| `NOT4GEP001` | GE Premier | Nottingham |
| `OXF1PRI001` | Siemens Prisma (32ch) | Oxford |
| `OXF2PRI001` | Siemens Prisma (64ch) | Oxford |
| `OXF3TRI001` | Siemens Trio | Oxford |
| `OXF4GEP001` | GE Premier (21ch) | Oxford |

```python

# Download only Oxford Prisma scans
load_ONharmony(
    subjects="all",
    sessions=["OXF1PRI001", "OXF2PRI001"],
    modalities="anat",
    target_path="./oxford_prisma",
    suffixes="T1w",
    extensions=".nii.gz",
    root_files="all",
)
```

## Citation

If you use the ON-Harmony dataset, please cite:

```bibtex
@article{warrington2025multi,
  title={A multi-site, multi-modal travelling-heads resource for brain MRI harmonisation},
  author={Warrington, Shaun and Torchi, Andrea and Mougin, Olivier and Campbell, Jon and Ntata, Asante and Craig, Martin and Assimopoulos, Stephania and Alfaro-Almagro, Fidel and Miller, Karla L and Jenkinson, Mark and others},
  journal={Scientific data},
  volume={12},
  number={1},
  pages={609},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```
