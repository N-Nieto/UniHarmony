"""
Traveling subjects
==================

In a traveling-subjects study, the same people are measured in every site.
Their true values do not change, so the differences between the measurements
of a subject are measurement noise and site effects only. This makes it the
most direct check of a harmonization: afterwards, a subject's measurements
should agree across sites about as well as repeated measurements in one site.
:func:`~uniharmony.datasets.make_traveling_subjects` simulates such a study.
"""

# %%
# Imports
# -------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from uniharmony import verbosity
from uniharmony.combat import NeuroComBat
from uniharmony.datasets import make_traveling_subjects


sns.set_theme(style="whitegrid")
verbosity("error")

# %%
# Data generation
# ---------------
#
# 20 subjects are measured once in each of 3 sites. Every measurement has a
# noise of SD 0.3 (in units of the SD between subjects), and each site has a
# location and a scale effect.

n_subjects = 20
X, sites, subjects, truth = make_traveling_subjects(
    n_subjects=n_subjects,
    n_sites=3,
    noise_strength=0.3,
    site_effect={"location": 2.0, "scale": 2.0},
    return_ground_truth=True,
    random_state=0,
)
print(f"{X.shape[0]} measurements of {n_subjects} subjects in 3 sites")

###############################################################################
# .. caution::
#
#    The harmonizer is fitted on the whole dataset to illustrate its effect.

# %%
# Each subject across the sites
# -----------------------------
#
# Each line joins the measurements of one subject. Without site effect, the
# lines are nearly flat (noise only); the site effect moves all subjects of a
# site together. NeuroComBat removes the shift of each site.

data = {
    "Without site effect (truth)": truth["site_free"],
    "Raw": X,
    "NeuroComBat": NeuroComBat().fit_transform(X, sites=sites),
}
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for ax, (name, values) in zip(axes, data.items(), strict=True):
    df = pd.DataFrame({"Site": sites, "Subject": subjects, "Feature 1": values[:, 0]})
    sns.lineplot(df, x="Site", y="Feature 1", units="Subject", estimator=None, alpha=0.6, ax=ax)
    ax.set_xticks([0, 1, 2])
    ax.set_title(name)
plt.tight_layout()
plt.show()

# %%
# How much do the measurements of a subject differ between sites?
# ---------------------------------------------------------------
#
# The SD of a subject's measurements across the sites, averaged over subjects
# and features, reflects the noise only without site effect (a little below
# 0.3, as an SD estimated from 3 values is biased low). The site effect
# doubles it; after NeuroComBat it is close to the noise again.


def between_site_sd(values):
    """SD of each subject's measurements across sites, averaged over subjects and features."""
    return np.mean([values[subjects == subject].std(axis=0, ddof=1).mean() for subject in range(n_subjects)])


print(pd.Series({name: between_site_sd(values) for name, values in data.items()}, name="Between-site SD").round(2).to_string())

###############################################################################
# .. admonition:: Take-home message
#
#    With the same subjects in every site, the site effect is the only
#    systematic difference between sites, so a harmonization can be checked
#    directly on each subject.
