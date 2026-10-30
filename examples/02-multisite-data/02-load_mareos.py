"""
Load MAREoS dataset
===================

.. note::

   This example downloads the MAREoS datasets (about 3 MB), so it is not
   executed when building the documentation. Run it locally to see the outputs;
   the expected outputs are shown as comments.
"""

# %%
# Imports
# -------

from pathlib import Path

import pandas as pd

from uniharmony.datasets import load_MAREoS


# %%
# We can call the helper function to load all the datasets (about 3 MB).
# The files are stored in the cache, so they are only downloaded once.
datasets = load_MAREoS()


# %%
# Exploration
# -----------
# Let's explore now how the datasets look like.
print(datasets.keys())
# dict_keys(['eos_simple1', 'eos_simple2', 'eos_interaction1', 'eos_interaction2',
#            'true_simple1', 'true_simple2', 'true_interaction1', 'true_interaction2'])


# %%
# We have now all the datasets in a dictionary. There is a total of 8 datasets.

# Select one dataset and explore what is inside the dictionary
dataset = datasets["eos_simple1"]
print(dataset.keys())
# dict_keys(['X', 'y', 'sites', 'covs', 'folds'])


# %%
# Let's unpack what is inside the keys. This is the typical way you can use
# the dataset for further downstream analysis.
X = dataset["X"]
y = dataset["y"]

print(f"Load X with shape:{X.shape} and y:{y.shape}")
# Load X with shape:(1000, 15) and y:(1000,)


# %%
# Variations
# ----------

# You can use the helper function to only return a part of the datasets
datasets = load_MAREoS(effects="eos")
print(datasets.keys())
# dict_keys(['eos_simple1', 'eos_simple2', 'eos_interaction1', 'eos_interaction2'])


# %%
datasets = load_MAREoS(effects="eos", effect_types="simple")
print(datasets.keys())
# dict_keys(['eos_simple1', 'eos_simple2'])


# %%
datasets = load_MAREoS(effects="eos", effect_types="simple", effect_examples="1")
print(datasets.keys())
# dict_keys(['eos_simple1'])


# %%
# Returning the dataset as DataFrame allows to see the simulated brain areas.

datasets = load_MAREoS(effects="eos", effect_types="simple", effect_examples="1", as_numpy=False)
dataset = datasets["eos_simple1"]["X"]
# Show only 8 columns
pd.set_option("display.max_columns", 8)
dataset.head()


# %%
# Load the dataset in a folder of your choice
# -------------------------------------------
# To keep the CSV files in a folder of your choice (e.g., to inspect them),
# pass ``data_dir``. The files are only downloaded if they are not there yet.
data_dir = Path.home() / "uniharmony_data" / "mareos"
datasets = load_MAREoS(data_dir=data_dir)
