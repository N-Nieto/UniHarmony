# Datalad Integration

``uniharmony`` provides a transparent way for downloading, managing, and interacting with BIDS-compatible neuroimaging datasets using DataLad. Designed primarily for the [ON-Harmony dataset (ds004712)](https://openneuro.org/datasets/ds004712) — a multi-site, multi-modal travelling-heads MRI harmonisation resource.

## Overview

We can use it for:

- **Cloning** DataLad datasets from available repositories
- **Selectively downloading** BIDS-compliant files (subjects, sessions, modalities, tasks, runs, suffixes, extensions)
- **Managing disk space** via hidden caches and automatic cleanup
- **Operating in two modes**: hidden cache (default) or direct target directory

### Key Features

| Feature | Description |
|---------|-------------|
| Hidden/Visible mode | Use a temporary cache (`hidden=True`) or download directly to target (`hidden=False`) |
| Selective downloads | Filter by subject, session, modality, task, run, suffix, and extension |
| Automatic cleanup | Drop files from cache after copying to save disk space (`tmp_clean=True`) |
| Symlink resolution | Convert DataLad annex symlinks to real files when using `copy=True` |
| Cache management | Clean temporary folders on demand |
| BIDS-aware | Follows Brain Imaging Data Structure conventions for file discovery |


## Core Concepts

### Hidden vs. Direct Mode

There are two operational modes:

#### Hidden Mode (`hidden=True`, default)

```
/tmp/datalad_cache/          ← Hidden DataLad repository (git annex, symlinks)
    └── ds004712/
        ├── .git/
        ├── .datalad/
        ├── sub-01/
        └── dataset_description.json

/data/my_dataset/            ← Clean target directory (real files only)
    └── sub-01/
        └── ses-NOT1ACH001/
            └── anat/
                └── sub-01_ses-NOT1ACH001_T1w.nii.gz
```

**Pros**: Target directory contains only data files (no `.git/`, `.datalad/`). Easy to clean cache. The hidden structure can be preserved and several "get" calls can be perform without the need of cloning the datalad dataset several times.

**Cons**: Files are copied from cache to target (time overhead for large files).

#### Direct Mode (`hidden=False`)

```
/data/my_dataset/            ← DataLad repository + target in one place
    ├── .git/
    ├── .datalad/
    ├── sub-01/
    └── dataset_description.json
```

**Pros**: No copy overhead. Files materialise directly where needed.

**Cons**: Directory contains DataLad metadata. Files are initially symlinks - must resolve them. The whole dataset structure is presented to the user, populated with symlinks.


## Usage

### Downloading Data

#### `download_bids_dataset()`

Download derivative files (processed data) from a BIDS dataset.

```python
from uniharmony.datasets import download_bids_dataset

download_bids_dataset(
    subjects=subjects,
    sessions=sessions,
    modalities=modalities,
    tasks="all",
    runs="all",
    suffixes=suffixes,
    extensions=extensions,
    target_path=target_path,
    dataset_url=dataset_url,
    root_files=root_files,
)
```

### Dataset Management

#### `clean_tmp()`

Remove the temporary DataLad cache directory.

```python
from uniharmony.datasets import clean_tmp

# Remove default cache
 clean_tmp()

# Remove custom-named cache
clean_tmp("my_custom_cache")
```

### Utility Functions

#### `list_available_files()`

List all files in a dataset (useful for exploration).

```python
from uniharmony.datasets import list_available_files
from pathlib import Path

files = list_available_files(Path("/tmp/datalad_cache/ds004712"))
print(f"Found {len(files)} files")
for f in files[:10]:
    print(f)
```

## Troubleshooting

### Disk space running out

Enable `tmp_clean=True` to drop files from cache after copying:

```python
download_bids_dataset(
    # ...
    tmp_clean=True,
)
```

Or clean the cache manually:

```python
from uniharmony.datasets import clean_tmp_folder

clean_tmp_folder()
```

### Clone fails with SSL error

Configure Git to use HTTPS instead of SSH:

```bash
git config --global url."https://".insteadOf "git://"
git config --global url."https://github.com/".insteadOf "git@github.com:"
```
