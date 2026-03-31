# This tutorial shows how to download the MAREoS dataset from the web

### Let's start with the imports


```python
# Imports
from pathlib import Path

from uniharmony import load_MAREoS
```


```python
# We can call the helper funtion to load all the dataset (aprox 3MB).
# The files will be stored in the cache, so we don't have to worry about them
datasets = load_MAREoS()
```


```python
# Let's explore now how the datasets looks like
print(datasets.keys())
```

    dict_keys(['eos_simple1', 'eos_simple2', 'eos_interaction1', 'eos_interaction2', 'true_simple1', 'true_simple2', 'true_interaction1', 'true_interaction2'])


## We have now all the datasets in a dictionary. There is a total of 8 datasets.


```python
# Select one dataset and explore what is inside the dictionary
dataset = datasets["eos_simple1"]
print(dataset.keys())
```

    dict_keys(['X', 'y', 'sites', 'covs', 'folds'])



```python
# Let's unpack what is inside the keys. This is the typical way you can use
# the dataset for further downstream analysis.
X = dataset["X"]
y = dataset["y"]

print(f"Load X with shape:{X.shape} and y:{y.shape}")
```

    Load X with shape:(1001, 14) and y:(1001,)


## Load datasets by condition


```python
# You can use the helper function to only return a part of the datasets
datasets = load_MAREoS(effects="eos")
print(datasets.keys())
```

    dict_keys(['eos_simple1', 'eos_simple2', 'eos_interaction1', 'eos_interaction2'])



```python
datasets = load_MAREoS(effects="eos", effect_types="simple")
print(datasets.keys())
```

    dict_keys(['eos_simple1', 'eos_simple2'])



```python
datasets = load_MAREoS(effects="eos", effect_types="simple", effect_examples="1")
print(datasets.keys())
```

    dict_keys(['eos_simple1'])


### Returning the dataset as DataFrame allows to see the simulated areas
You can chose to load the dataset as pandas.DataFrame, with has the simulated areas of the brain.


```python
datasets = load_MAREoS(effects="eos", effect_types="simple", effect_examples="1", as_numpy=False)
dataset = datasets["eos_simple1"]["X"]
dataset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Lthal</th>
      <th>Rthal</th>
      <th>Lcaud</th>
      <th>Rcaud</th>
      <th>Lput</th>
      <th>Rput</th>
      <th>Lpal</th>
      <th>Rpal</th>
      <th>Lhippo</th>
      <th>Rhippo</th>
      <th>Lamyg</th>
      <th>Ramyg</th>
      <th>Laccumb</th>
      <th>Raccumb</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>8895.369099</td>
      <td>8383.870372</td>
      <td>3803.558492</td>
      <td>4357.165963</td>
      <td>7231.227420</td>
      <td>5647.496253</td>
      <td>1294.448052</td>
      <td>2270.489516</td>
      <td>3928.692453</td>
      <td>5421.703185</td>
      <td>1563.622497</td>
      <td>1854.229137</td>
      <td>698.637972</td>
      <td>701.906213</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8679.346875</td>
      <td>6654.136742</td>
      <td>3924.041654</td>
      <td>3745.063498</td>
      <td>5895.190311</td>
      <td>5164.702016</td>
      <td>1939.028843</td>
      <td>2017.027485</td>
      <td>3110.500978</td>
      <td>6202.638815</td>
      <td>1511.933005</td>
      <td>1020.152948</td>
      <td>709.090077</td>
      <td>534.448106</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9191.801201</td>
      <td>7159.776871</td>
      <td>3444.265568</td>
      <td>3158.455008</td>
      <td>4858.917213</td>
      <td>5392.683202</td>
      <td>2191.623004</td>
      <td>2415.533638</td>
      <td>4202.892743</td>
      <td>5147.131015</td>
      <td>1761.132128</td>
      <td>1114.841164</td>
      <td>785.199200</td>
      <td>717.806882</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7531.473405</td>
      <td>6694.021219</td>
      <td>4984.063517</td>
      <td>4689.035649</td>
      <td>5553.881746</td>
      <td>6494.219094</td>
      <td>2242.955773</td>
      <td>2381.085589</td>
      <td>3629.289874</td>
      <td>5748.866950</td>
      <td>1774.472741</td>
      <td>742.652391</td>
      <td>1104.007368</td>
      <td>769.837240</td>
    </tr>
    <tr>
      <th>5</th>
      <td>7070.478721</td>
      <td>5575.244389</td>
      <td>3285.175734</td>
      <td>2234.129050</td>
      <td>6526.943910</td>
      <td>6041.981350</td>
      <td>1601.192576</td>
      <td>1630.750753</td>
      <td>3944.610918</td>
      <td>6150.220123</td>
      <td>1722.570593</td>
      <td>1414.669078</td>
      <td>1000.597680</td>
      <td>440.375965</td>
    </tr>
  </tbody>
</table>
</div>



# Load the dataset in a user determine folder

## We could also want to see the csv files in a folder, we could pass a directory for the function to save the data


```python
# Let's pass a directory inside the repository.
# We will use a relative path from this example to look for appropiated path
data_dir = Path().resolve().parent / "src" / "uniharmony" / "datasets" / "data"
datasets = load_MAREoS(data_dir=data_dir)
```
