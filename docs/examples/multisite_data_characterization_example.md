# Characterize a multisite problem using UniHarmony

## The first step before applying any harmonization technique is to understand and characterize our data


```python
# Imports
from uniharmony._multisite_data_generation import simulate_multisite_data
from uniharmony.multisite_data_characterization import (
    get_site_data_statistics,
    print_statistics_summary,
)
```

## Let's use the multisite data generator to simulate some data


```python
print("Generating example data...")
X, y, sites = simulate_multisite_data(
    n_sites=3,
    n_samples=100,
    n_features=10,
    n_classes=3,
    random_state=42,
    verbose=True,
)

print("\n" + "=" * 60)
```

    Generating example data...
    Using balanced classes: [[0.3333333333333333, 0.3333333333333333, 0.3333333333333333], [0.3333333333333333, 0.3333333333333333, 0.3333333333333333], [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]]
    Generating 34 samples for site 0
    Generating 33 samples for site 1
    Generating 33 samples for site 2
    Generated 100 samples across 3 sites
    Class distribution: [34 33 33]
    Site distribution: [34 33 33]

    ============================================================


# Now lets compute some statistics


```python
print("Computing statistics...")
print("=" * 60)

# Compute statistics
stats = get_site_data_statistics(
    x=X,
    y=y,
    site_labels=sites,
    feature_names=[f"feat_{i}" for i in range(X.shape[1])],
    compute_comprehensive=True,
    verbose=True,
)

# Print summary
print_statistics_summary(stats)
```

    Computing statistics...
    ============================================================
    Computing statistics for 100 samples, 10 features, 3 sites, 3 classes
      Processing site 0...
      Processing site 1...
      Processing site 2...
      Processing class 0...
      Processing class 1...
      Processing class 2...
    ============================================================
    DATASET STATISTICS SUMMARY
    ============================================================

    OVERALL:
      Samples: 100
      Features: 10
      Sites: 3
      Classes: 3

    CLASS DISTRIBUTION:
      class_0: 34 samples (34.0%)
      class_1: 33 samples (33.0%)
      class_2: 33 samples (33.0%)

    SITE DISTRIBUTION:
      site_0: 34 samples (34.0%)
      site_1: 33 samples (33.0%)
      site_2: 33 samples (33.0%)

    SITE STATISTICS (summary):
      site_0:
        Samples: 34
        Class distribution: {'class_0': 12, 'class_1': 11, 'class_2': 11}
      site_1:
        Samples: 33
        Class distribution: {'class_0': 11, 'class_1': 11, 'class_2': 11}
      site_2:
        Samples: 33
        Class distribution: {'class_0': 11, 'class_1': 11, 'class_2': 11}

    FEATURE STATISTICS (first 5 features):
      feat_0: mean=-1.5404, std=3.4675
      feat_1: mean=-1.0764, std=1.6735
      feat_2: mean=2.2555, std=2.2806
      feat_3: mean=1.2875, std=4.7056
      feat_4: mean=1.1251, std=1.8842

    CORRELATIONS:
      Average Inter-Site Correlation: 0.0513
    ============================================================
