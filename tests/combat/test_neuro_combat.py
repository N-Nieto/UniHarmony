"""Tests for NeuroComBat transformer."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils.estimator_checks import parametrize_with_checks

from uniharmony.combat import NeuroComBat


def _ex_failed_checks(_) -> dict[str, str]:
    return {
        "check_transformers_unfitted": "checked inside",
        "check_n_features_in_after_fitting": "not needed",
        "check_estimators_nan_inf": "checked inside",
        "check_fit_score_takes_y": "sites instead of y",
        "check_estimators_dtypes": "sites instead of y",
        "check_dtype_object": "sites instead of y",
        "check_estimators_pickle": "sites instead of y",
        "check_f_contiguous_array_estimator": "sites instead of y",
        "check_transformer_data_not_an_array": "sites instead of y",
        "check_transformer_preserve_dtypes": "sites instead of y",
        "check_transformer_general": "sites instead of y",
        "check_methods_sample_order_invariance": "sites instead of y",
        "check_methods_subset_invariance": "sites instead of y",
        "check_dict_unchanged": "sites instead of y",
        "check_fit_idempotent": "sites instead of y",
        "check_n_features_in": "not needed",
        "check_fit2d_predict1d": "sites instead of y",
        "check_fit2d_1sample": "custom message",
        "check_requires_y_none": "target cannot be None",
    }


@parametrize_with_checks(
    [
        NeuroComBat(empirical_bayes=True, parametric_adjustments=True, mean_only=True),
        NeuroComBat(empirical_bayes=True, parametric_adjustments=True, mean_only=False),
        NeuroComBat(empirical_bayes=False, parametric_adjustments=True, mean_only=True),
        NeuroComBat(empirical_bayes=False, parametric_adjustments=True, mean_only=False),
    ],
    expected_failed_checks=_ex_failed_checks,
)
def test_neuro_combat_compat_sklearn(estimator: object, check: callable) -> None:
    """Test NeuroComBat compatibility with sklearn.

    Parameters
    ----------
    estimator : object
        Instance of NeuroComat.
    check : callable
        sklearn fixture.

    """
    check(estimator)


def test_neuro_combat_ops_original() -> None:
    """Test operation of NeuroComBat with original."""
    data = np.genfromtxt(Path(__file__).parent / "test_data.csv", delimiter=",", skip_header=1)
    batches = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2]).reshape(-1, 1)
    genders = np.array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2]).reshape(-1, 1)
    data_combat = NeuroComBat().fit_transform(
        data.T,
        batches,
        categorical_covariates=genders,
    )
    assert data_combat.shape == data.T.shape


def test_neuro_combat_ops_impl() -> None:
    """Test operation of NeuroComBat with reference sklearn implementaion."""
    data = np.load(Path(__file__).parent / "bladder-expr.npy")
    covars = pd.read_csv(Path(__file__).parent / "bladder-pheno.txt", delimiter="\t")
    data_combat = NeuroComBat().fit_transform(
        data,
        covars[["batch"]].to_numpy(),
        categorical_covariates=covars[["cancer"]].to_numpy(),
        continuous_covariates=covars[["age"]].to_numpy(),
    )
    assert data_combat.shape == data.shape
