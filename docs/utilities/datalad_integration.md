# UniHarmony Datalad Integration

A Python module for downloading, managing, and interacting with BIDS-compatible neuroimaging datasets using DataLad. Designed primarily for the [ON-Harmony dataset (ds004712)](https://openneuro.org/datasets/ds004712) — a multi-site, multi-modal travelling-heads MRI harmonisation resource [^18^].

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Core Concepts](#core-concepts)
4. [Public API](#public-api)
   - [Downloading Data](#downloading-data)
   - [Dataset Management](#dataset-management)
   - [Utility Functions](#utility-functions)
5. [Troubleshooting](#troubleshooting)

---

## Overview

This module provides a high-level interface for:

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

---


## Core Concepts

### Hidden vs. Direct Mode

The module supports two operational modes:

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

**Pros**: No copy overhead. Files materialize directly where needed.

**Cons**: Directory contains DataLad metadata. Files are initially symlinks - must resolve them. The whole dataset structure is presented to the user, populated with symlinks.

---

## Public API

### Downloading Data

#### `download_bids_dataset()`

Download derivative files (processed data) from a BIDS dataset.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `subjects` | `str \| list[str]` | — | Subject IDs or `"all"` |
| `sessions` | `str \| list[str]` | — | Session IDs or `"all"` |
| `modalities` | `str \| list[str]` | — | `"anat"`, `"dwi"`, `"fmap"`, `"func"`, `"swi"` |
| `tasks` | `str \| list[str]` | — | Task names or `"all"` |
| `runs` | `str \| list[str]` | — | Run numbers or `"all"` |
| `target_path` | `str \| Path` | — | Output directory |
| `suffixes` | `str \| list[str]` | — | BIDS suffixes: `"T1w"`, `"bold"`, `"dwi"` |
| `extensions` | `str \| list[str]` | — | File extensions: `".nii.gz"`, `".json"`, `".bval"` |
| `dataset_source_URL` | `str` | — | Full Git URL to the dataset |
| `root_files` | `str \| list[str]` | — | Root-level files or `"all"` |
| `force_download` | `bool` | `False` | Re-clone even if cache exists |
| `copy` | `bool` | `True` | Copy files to target (hidden mode only) |
| `hidden` | `bool` | `True` | Use hidden cache |
| `tmp_clean` | `bool` | `False` | Drop files from cache after copy |
| `tmp_dir_name` | `str` | `"datalad_cache"` | Name of temp directory |


---

### Dataset Management

#### `clean_tmp_folder()`

Remove the temporary DataLad cache directory.

```python
from uniharmony.datasets import clean_tmp_folder

# Remove default cache
 clean_tmp_folder()

# Remove custom-named cache
clean_tmp_folder("my_custom_cache")
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tmp_dir_name` | `str` | `"datalad_cache"` | Name of the temp folder to remove |

**Raises:**
- `FileNotFoundError` — if the directory does not exist

---

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

---

# Troubleshooting
### Issue: Disk space running out

**Solution**: Enable `tmp_clean=True` to drop files from cache after copying:
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

### Issue: Clone fails with SSL error

**Solution**: Configure Git to use HTTPS instead of SSH:
```bash
git config --global url."https://".insteadOf "git://"
git config --global url."https://github.com/".insteadOf "git@github.com:"
```

---

## API Reference Summary

### Main Functions

| Function | Purpose |
|----------|---------|
| `download_bids_dataset()` | Download data from a BIDS-compatible dataset |
| `clean_tmp_folder()` | Remove temporary cache |
| `list_available_files()` | Explore dataset contents |

### Internal Functions (for advanced use)

| Function | Purpose |
|----------|---------|
| `initialize_dl_dataset()` | Clone and initialize a DataLad dataset |
| `get_candidate_files()` | Find files matching BIDS filters |
| `get_derivative_files()` | Materialize and copy derivative files |
| `get_raw_files()` | Materialize and copy raw files |
| `get_root_files()` | Materialize root-level files |
| `validate_arguments()` | Normalize filter arguments |

---
