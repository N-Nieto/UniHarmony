```python
"""Basic example of ICI harmonization for multiclass classification."""

import numpy as np

from uniharmony.interpolation import ICIHarmonization


x = np.random.randn(300, 10)
y = np.array([0] * 180 + [1] * 80 + [2] * 40)
sites = np.array([0] * 150 + [1] * 150)
y = np.random.permutation(y)

ici = ICIHarmonization("adasyn")
x_r, y_r = ici.fit_resample(x, y, sites=sites)
s_r = ici.sites_resampled_

for site in np.unique(s_r):
    print(site, np.unique(y_r[s_r == site], return_counts=True))

for site in np.unique(s_r):
    print(site, np.unique(y[sites == site], return_counts=True))
```

    0 (array([0, 1, 2]), array([85, 84, 91]))
    1 (array([0, 1, 2]), array([95, 91, 92]))
    0 (array([0, 1, 2]), array([85, 38, 27]))
    1 (array([0, 1, 2]), array([95, 42, 13]))
