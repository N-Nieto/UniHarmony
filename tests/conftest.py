"""Shared pytest configuration and fixtures for the UniHarmony test suite.

All synthetic multisite data used across method tests is generated here, so a
change in the data generation process is picked up by every test that uses
these fixtures. pytest discovers this file automatically: fixtures are
requested by name as test arguments and need no import.

Usage
-----
>>> def test_something(multisite_data):
...     X, sites = multisite_data.X, multisite_data.sites
...     cat = multisite_data.get("sex", "education")  # (n_samples, 2)
...     cont = multisite_data.get("age")  # (n_samples, 1)

"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pytest


DEFAULT_SEED = 42

#: Covariates generated as categorical (integer-coded levels).
CATEGORICAL_COVARIATES = ("sex", "education")
#: Covariates generated as continuous.
CONTINUOUS_COVARIATES = ("age", "extra")


@dataclass(frozen=True)
class MultisiteData:
    """Synthetic multisite dataset.

    Attributes
    ----------
    X : ndarray, shape (n_samples, n_features)
        Features with site effects (location and scale) and covariate effects.
    sites : ndarray, shape (n_samples,)
        Site label of each sample.
    y : ndarray, shape (n_samples,)
        Binary target (0/1).
    covariates : dict of str to ndarray, shape (n_samples,)
        ``age`` (continuous, 20-80), ``sex`` (2 levels), ``education``
        (4 levels) and ``extra`` (continuous, standard normal).

    """

    X: npt.NDArray
    sites: npt.NDArray
    y: npt.NDArray
    covariates: dict[str, npt.NDArray] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        """Number of samples."""
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        """Number of features."""
        return self.X.shape[1]

    @property
    def n_sites(self) -> int:
        """Number of unique sites."""
        return len(np.unique(self.sites))

    def get(self, *names: str) -> npt.NDArray | None:
        """Return the requested covariates stacked as columns.

        Parameters
        ----------
        *names : str
            Covariate names, in the desired column order.

        Returns
        -------
        ndarray, shape (n_samples, len(names)) or None
            The stacked covariates, or ``None`` if no name is given (handy for
            parametrized tests where a covariate group may be empty).

        """
        if not names:
            return None
        return np.column_stack([self.covariates[name] for name in names])


def make_multisite_data(
    n_samples_per_site: Sequence[int] = (50, 50, 50),
    n_features: int = 8,
    site_labels: Sequence | None = None,
    seed: int = DEFAULT_SEED,
) -> MultisiteData:
    """Generate a synthetic multisite dataset.

    Each feature is the sum of:
      * a covariate effect (age, sex, education),
      * a site-specific location shift and scale factor,
      * Gaussian noise.

    Parameters
    ----------
    n_samples_per_site : sequence of int, optional (default (50, 50, 50))
        Number of samples in each site. Its length sets the number of sites.
    n_features : int, optional (default 8)
        Number of features.
    site_labels : sequence or None, optional (default None)
        Label of each site. If None, sites are labelled 1, 2, ..., n_sites.
    seed : int, optional (default 42)
        Seed of the random number generator.

    Returns
    -------
    MultisiteData
        The generated dataset.

    """
    rng = np.random.default_rng(seed)
    n_sites = len(n_samples_per_site)
    if site_labels is None:
        site_labels = list(range(1, n_sites + 1))
    if len(site_labels) != n_sites:
        raise ValueError("site_labels must have one label per site")

    sites = np.repeat(np.asarray(site_labels), n_samples_per_site)
    n_samples = len(sites)

    covariates = {
        "age": rng.uniform(20, 80, n_samples),
        "sex": rng.integers(0, 2, n_samples),
        "education": rng.integers(0, 4, n_samples),
        "extra": rng.normal(size=n_samples),
    }
    y = rng.integers(0, 2, n_samples)

    # Biological signal (to be preserved when covariates are given)
    signal = (
        0.03 * covariates["age"][:, np.newaxis]
        + 0.4 * covariates["sex"][:, np.newaxis]
        + 0.2 * covariates["education"][:, np.newaxis]
    )
    # Site effects (to be removed by harmonization)
    site_index = np.repeat(np.arange(n_sites), n_samples_per_site)
    location = rng.normal(0, 1.5, size=(n_sites, n_features))
    scale = rng.uniform(0.6, 1.6, size=(n_sites, n_features))
    noise = rng.normal(size=(n_samples, n_features))
    X = signal + location[site_index] + scale[site_index] * noise

    return MultisiteData(X=X, sites=sites, y=y, covariates=covariates)


def make_three_site_covariance_data(seed: int = DEFAULT_SEED) -> MultisiteData:
    """Generate three sites with strong mean, variance and covariance site effects.

    Designed for covariance harmonization (e.g., CovBat): site "B" has inflated
    variance and a positive correlation between features 0 and 1, site "C" has
    shrunk variance and a negative correlation between features 2 and 3. Age
    has a linear effect on every feature (biological signal to preserve).

    Parameters
    ----------
    seed : int, optional (default 42)
        Seed of the random number generator.

    Returns
    -------
    MultisiteData
        60 samples per site ("A", "B", "C"), 20 features, ``age`` covariate.

    """
    rng = np.random.default_rng(seed)
    n_per_site = 60
    n_features = 20
    site_labels = ("A", "B", "C")
    sites = np.repeat(site_labels, n_per_site)
    age = rng.normal(50, 10, len(sites))
    mean_shift = {"A": 0.0, "B": 2.5, "C": -1.5}

    X = []
    for site in site_labels:
        idx = sites == site
        if site == "A":
            cov = np.eye(n_features)
        elif site == "B":
            cov = np.eye(n_features) * 1.5
            cov[0, 1] = cov[1, 0] = 0.8
        else:
            cov = np.eye(n_features) * 0.7
            cov[2, 3] = cov[3, 2] = -0.6
        samples = rng.multivariate_normal(mean=np.full(n_features, mean_shift[site]), cov=cov, size=idx.sum())
        samples += age[idx][:, np.newaxis] * np.linspace(0.1, 0.5, n_features)
        X.append(samples)

    y = rng.integers(0, 2, len(sites))
    return MultisiteData(X=np.vstack(X), sites=sites, y=y, covariates={"age": age})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random generator (fresh for every test, so results do not depend on test order)."""
    return np.random.default_rng(DEFAULT_SEED)


@pytest.fixture
def make_multisite() -> Callable[..., MultisiteData]:
    """Return :func:`make_multisite_data` to build custom datasets in a test."""
    return make_multisite_data


@pytest.fixture
def multisite_data() -> MultisiteData:
    """Three balanced sites (50 samples each), 8 features, integer site labels."""
    return make_multisite_data()


@pytest.fixture
def multisite_data_imbalanced() -> MultisiteData:
    """Three sites of different size (80, 40, 20 samples)."""
    return make_multisite_data(n_samples_per_site=(80, 40, 20))


@pytest.fixture
def multisite_data_two_sites() -> MultisiteData:
    """Two balanced sites (60 samples each)."""
    return make_multisite_data(n_samples_per_site=(60, 60))


@pytest.fixture
def multisite_data_string_sites() -> MultisiteData:
    """Three balanced sites with string labels ("A", "B", "C")."""
    return make_multisite_data(site_labels=("A", "B", "C"))


@pytest.fixture(params=["multisite_data", "multisite_data_imbalanced", "multisite_data_string_sites"])
def any_multisite_data(request: pytest.FixtureRequest) -> MultisiteData:
    """Run a test once per dataset variant (balanced, imbalanced, string sites)."""
    return request.getfixturevalue(request.param)


@pytest.fixture
def three_site_data() -> MultisiteData:
    """Three sites ("A", "B", "C") with strong mean, variance and covariance site effects."""
    return make_three_site_covariance_data()
