# ▶️ Getting Started

## 💾 Installation

UniHarmony is written in Python and [published on PyPI](https://pypi.org/project/uniharmony/). It is recommended to install it in a *virtual environment*.

### Install with ``pip``

```console
pip install uniharmony
```

### Install with ``uv``

```console
uv add uniharmony
```

## 🪄 Usage

### Data loading and simulation

- [Simulate multisite classification problems](#simulate-multisite-data)
- [Download and load MAREoS dataset](#load-mareos-datasets)

### Harmonization

#### ComBat methods

- [NeuroComBat](#neurocombat-long)
- [PrettYharmonize](#prettyharmonize-long): leakage-free ComBat for ML when class balance differs across sites

#### Interpolation models

- [IntraSiteInterpolation (ISI)](#intrasite-long)

## 📣 Logging

UniHarmony only prints warnings and errors by default. Use `uniharmony.verbosity` to see more (or less), for the whole session or inside a block:

```python
import uniharmony

uniharmony.verbosity("info")  # main steps of each method; "debug" also shows intermediate values

with uniharmony.verbosity_context("error"):  # only errors inside this block
    ...
```

The level only applies to UniHarmony's messages. UniHarmony logs with [structlog](https://www.structlog.org) but never changes its global configuration, so an application that configures structlog (output format, its own level) keeps its settings.
