# 🧪 MultiSite Data Simulation

## What's This?
 This handy tool helps researchers generate synthetic datasets that mimic real-world multi-center studies. Whether you're testing machine learning algorithms, developing statistical methods, or teaching data science concepts, this simulator creates realistic data with built-in complexities that mirror actual research scenarios.


## 🔍 Behind the Scenes
Each sample's features combine three components that can be adapted:

Feature = Signal + Noise + Site Effect

### Flexible Configuration
You can choose how many sites you want, set the total number of samples, control how many features and control the class balance.

Work with binary classification (yes/no) or multi-class problems

### 🚀 Quick Start

``` python
# Import the simulator
from uniharmony import simulate_multi_site_data
# Generate data with 3 sites, 500 samples total
X, y, sites = simulate_multi_site_data(
    n_sites=3,
    n_samples=500,
    n_features=20,
    random_state=42  # For reproducibility
)
# X contains your features, y has labels, sites tells you which site each sample came from
```


# 📊 MAREoS Benchmark Datasets for Harmonization Method Evaluation
## Overview and Purpose
The MAREoS (Methods Aiming to Remove Effect of Site) datasets constitute a standardized benchmark suite specifically designed for rigorous evaluation and comparison of data harmonization methods. Developed by [Solanas et al. (2023)](https://www.sciencedirect.com/science/article/pii/S1053811922009211), these synthetic datasets provide controlled experimental conditions that enable systematic comparision of harmonization algorithm performance.

## Dataset Structure and Experimental Design
The benchmark implements three controlled variables:

### Effect:

- True Signal: Genuine biological signal uncorrelated with site identity.
- Effect of Site (EoS): Spurious signal arising from systematic site differences.

### Effect Type:

- Linear: Additive effects of features
- Interaction: Non-linear feature interactions

### Replication Examples:

Two independent examples for each condition combination

## Dataset Specifications
- Total datasets: 8 (2 signal types × 2 patterns × 2 replications)
- Samples per dataset: ∼ 1,000
- Features: 14 simulated baseline MRI data (cortical thickness, cortical surface area, or subcortical volumes).
- Sites: 8 distinct sources

Predefined evaluation: 10-fold cross-validation scheme encoded in dataset

## Scientific Rationale

​The authors proposed two types of effect, True and Effect of Site (EoS) effect. A Machine Learning model trained on the EoS effect will **fraudulently** give a balanced accuracy (bACC) performance of around 80%, as the target and the sites are correlated.
The harmonization methods should remove this relationship, thus an ML model should give a chance performance. In the datasets with True signal, the harmonization method should not remove the True signal while aiming to remove the Effect of Site (which in these cases are not present).

### 🚀 Quick Start
``` python
# Import the helper function
from uniharmony import load_MAREoS
# Load the 8 datasets as a dictionary. Use the dictionary `keys` to access each dataset
datasets = load_MAREoS()
# Each dataset contains X, y, sites, covs, folds
```

# Citation
If you are using this datasets in your research, please cite the original publication:

``` biblatex
@article{solanes2023removing,
  title={Removing the effects of the site in brain imaging machine-learning--Measurement and extendable benchmark},
  author={Solanes, Aleix and Gosling, Corentin J and Fortea, Lydia and Ortu{\~n}o, Mar{\'\i}a and Lopez-Soley, Elisabet and Llufriu, Sara and Madero, Santiago and Martinez-Heras, Eloy and Pomarol-Clotet, Edith and Solana, Elisabeth and others},
  journal={NeuroImage},
  volume={265},
  pages={119800},
  year={2023},
  publisher={Elsevier}
}
