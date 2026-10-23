"""Fixtures for the IQM-based harmonization tests.

The data follow the simulation of the BARTharm R code
(``simulate_data()`` in https://github.com/NeuroSML/BARTharm, Simulation framework 1 of the paper).
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest


@dataclass(frozen=True)
class IQMData:
    """Simulated data with scanner effects driven by image quality metrics.

    Attributes
    ----------
    X : ndarray, shape (n_samples, 1)
        Observed outcome: biological signal + noise + scanner effect.
    X_clean : ndarray, shape (n_samples, 1)
        Outcome without scanner effect (the harmonization target).
    iqms : ndarray, shape (n_samples, 4)
        Noise level, resolution, SNR and CNR.
    biological_covariates : ndarray, shape (n_samples, 5)
        Age, sex, two noisy biological features and a group indicator.
    sites : ndarray, shape (n_samples,)
        Scanner of each sample (10 scanners).

    """

    X: npt.NDArray
    X_clean: npt.NDArray
    iqms: npt.NDArray
    biological_covariates: npt.NDArray
    sites: npt.NDArray


def simulate_iqm_data(
    n_subjects: int = 1000,
    linear_tau: bool = True,
    linear_mu: bool = True,
    seed: int = 0,
) -> IQMData:
    """Port of ``simulate_data()`` from the BARTharm R code."""
    rng = np.random.default_rng(seed)
    # Scanner properties: 3 low, 4 medium, 3 high resolution scanners
    scanner = rng.integers(1, 11, n_subjects)
    resolution = np.where(scanner <= 3, 1, np.where(scanner <= 7, 2, 3))
    noise_level = np.where(scanner <= 3, 3, np.where(scanner <= 7, 2, 1))

    age = rng.integers(20, 61, n_subjects).astype(float)
    sex = rng.integers(0, 2, n_subjects).astype(float)
    group = rng.integers(0, 2, n_subjects).astype(float)
    bio_1 = rng.uniform(0, 25, n_subjects)
    bio_2 = rng.uniform(0, 30, n_subjects)

    outcome = rng.normal(size=n_subjects)
    if linear_tau:
        outcome = outcome + np.where(sex == 0, -25, 25)
        outcome = outcome - 4 * bio_1 + 6 * bio_2 + 0.1 * bio_1 * sex + 0.1 * bio_1 * bio_2
        outcome = outcome + 0.01 * bio_2 * age
    else:
        # As in the R code, the random effects are overwritten in the nonlinear case
        outcome = np.where(sex == 0, -50.0, 50.0)
        outcome = outcome - 0.1 * age**2 + 5 * np.sin(age)
        outcome = outcome - 2 * bio_1**2 + 3 * np.log(bio_2 + 1) + 0.5 * bio_1 * bio_2**2 / 2
        outcome = outcome + 0.1 * bio_1 * sex + 0.01 * bio_1 * bio_2
        outcome = outcome + 0.01 * bio_2 * age

    if linear_mu:
        observed = outcome - 0.2 * resolution + 0.15 * noise_level + 0.05 * noise_level * resolution
    else:
        observed = outcome - 5 * resolution**2 + 2 * np.exp(-noise_level) + 0.5 * resolution * noise_level**2

    bio_1_observed = bio_1 + rng.normal(0, noise_level * 0.25)
    bio_2_observed = bio_2 + rng.normal(0, noise_level * 0.1)

    snr_low, snr_high = (
        np.select([resolution == 3, resolution == 2], [50, 30], 10),
        np.select([resolution == 3, resolution == 2], [65, 49], 29),
    )
    snr = rng.uniform(snr_low, snr_high) - (3 - noise_level) * 5
    cnr_low, cnr_high = (
        np.select([resolution == 3, resolution == 2], [5, 15], 30),
        np.select([resolution == 3, resolution == 2], [14, 29], 40),
    )
    cnr = rng.uniform(cnr_low, cnr_high) + (3 - noise_level) * 2

    observed = observed + 1.75 * snr - 1.55 * cnr - 0.5 * snr * cnr

    return IQMData(
        X=observed[:, np.newaxis],
        X_clean=outcome[:, np.newaxis],
        iqms=np.column_stack([noise_level, resolution, snr, cnr]).astype(float),
        biological_covariates=np.column_stack([age, sex, bio_1_observed, bio_2_observed, group]),
        sites=scanner,
    )


@pytest.fixture
def iqm_data() -> IQMData:
    """Simulate data with linear biological and scanner effects."""
    return simulate_iqm_data(n_subjects=300, seed=0)


@pytest.fixture
def iqm_data_nonlinear() -> IQMData:
    """Simulate data with nonlinear biological and scanner effects."""
    return simulate_iqm_data(n_subjects=300, linear_tau=False, linear_mu=False, seed=1)
