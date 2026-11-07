"""Names, defaults and thresholds shared by the multi-site simulators.

Every simulator (``make_multisite_classification``,
``make_multisite_regression``, ``make_traveling_subjects``) and the
covariates take their accepted names and their defaults from this module, so
a new option or simulator only needs to be declared here once.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Signals: the generator of the base problem
# ---------------------------------------------------------------------------
# Classification: scikit-learn generators
CLASSIFICATION_SIGNAL_TYPES = Literal["linear", "circles", "moons", "blobs", "gaussian_quantiles"]
# Regression: y = s * f(X) + N(0, 1), with f standardized
REGRESSION_SIGNAL_TYPES = Literal["linear", "friedman1"]

DEFAULT_CLASSIFICATION_SIGNAL: dict[str, float] = {"linear": 1.0}
DEFAULT_REGRESSION_SIGNAL: dict[str, float] = {"linear": 2.0}  # R² = s² / (1 + s²) = 0.8

# ---------------------------------------------------------------------------
# Effects of site (EoS)
# ---------------------------------------------------------------------------
# Several types can be combined in one ``site_effect`` dict
SITE_EFFECT_TYPES = Literal["location", "scale", "variance", "nonlinear", "dropout", "combat", "class_interaction"]
# Types that need class labels
CLASS_SITE_EFFECT_TYPES = ("class_interaction",)
DEFAULT_SITE_EFFECT: dict[str, float] = {"location": 3.0}

# "dropout" takes {"value": ..., "prob": ...} and, optionally, "per"
DROPOUT_UNITS = Literal["feature", "value"]
DEFAULT_DROPOUT_UNIT = "feature"

# {type: strength}, a strength being a float or one float per site; "dropout" takes a dict
SiteEffect = Mapping[str, float | Sequence[float] | Mapping[str, Any]]

# ---------------------------------------------------------------------------
# Covariates
# ---------------------------------------------------------------------------
PRESET_COVARIATES = Literal["age", "sex", "tiv", "quality"]
EFFECT_SHAPES = Literal["linear", "quadratic", "sigmoid"]

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
# Samplers such as SMOTE need k_neighbors + 1 samples of a class in a site (6 by default). Only used for a warning.
FEW_SAMPLES = 6
# Narrowest target range of a site in make_multisite_regression (fraction of the population), so that the
# rejection sampling stays fast (about 1 / width draws per sample)
MIN_TARGET_RANGE = 0.01
