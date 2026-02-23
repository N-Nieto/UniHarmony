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





