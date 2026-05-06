"""
Load MAREoS dataset
===================
"""

# %%
# Imports
# -------

from pathlib import Path

import pandas as pd

from uniharmony import load_MAREoS


# %%
# We can call the helper funtion to load all the dataset (aprox 3MB).
# The files will be stored in the cache, so we don't have to worry about them
datasets = load_MAREoS()


# %%
# Exploration
# -----------
# Let's explore now how the datasets looks like
print(datasets.keys())


# %%
# We have now all the datasets in a dictionary. There is a total of 8 datasets.

# Select one dataset and explore what is inside the dictionary
dataset = datasets["eos_simple1"]
print(dataset.keys())


# %%
# Let's unpack what is inside the keys. This is the typical way you can use
# the dataset for further downstream analysis.
X = dataset["X"]
y = dataset["y"]

print(f"Load X with shape:{X.shape} and y:{y.shape}")


# %%
# Variations
# ----------

# You can use the helper function to only return a part of the datasets
datasets = load_MAREoS(effects="eos")
print(datasets.keys())


# %%
datasets = load_MAREoS(effects="eos", effect_types="simple")
print(datasets.keys())


# %%
datasets = load_MAREoS(effects="eos", effect_types="simple", effect_examples="1")
print(datasets.keys())


# %%
# Returning the dataset as DataFrame allows to see the simulated areas
# You can chose to load the dataset as pandas.DataFrame, with has the simulated areas of the brain.

datasets = load_MAREoS(effects="eos", effect_types="simple", effect_examples="1", as_numpy=False)
dataset = datasets["eos_simple1"]["X"]
# Show only 5 columns
pd.set_option("display.max_columns", 8)
dataset.head()


# %%
# Load the dataset in a user determine folder
# We could also want to see the csv files in a folder, we could pass a directory for the function to save the data
# Let's pass a directory inside the repository.
# We will use a relative path from this example to look for appropiated path
data_dir = Path().resolve().parents[1] / "src" / "uniharmony" / "datasets" / "data"
datasets = load_MAREoS(data_dir=data_dir)
