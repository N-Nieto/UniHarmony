"""Tests for ComBatGAM transformer."""

from collections.abc import Callable

import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from uniharmony.combat import ComBatGAM
from uniharmony.datasets import make_multisite_classification


def _ex_failed_checks(_) -> dict[str, str]:
    return {
        "check_estimators_overwrite_params": "missing smooth covariates",
        "check_estimators_fit_returns_self": "missing smooth covariates",
        "check_readonly_memmap_input": "missing smooth covariates",
        "check_positive_only_tag_during_fit": "missing smooth covariates",
        "check_complex_data": "missing smooth covariates",
        "check_pipeline_consistency": "missing smooth covariates",
        "check_estimator_sparse_tag": "missing smooth covariates",
        "check_estimator_sparse_array": "missing smooth covariates",
        "check_estimator_sparse_matrix": "missing smooth covariates",
        "check_estimators_empty_data_messages": "missing smooth covariates",
        "check_dont_overwrite_parameters": "missing smooth covariates",
        "check_fit_score_takes_y": "sites instead of y; missing smooth covariates",
        "check_n_features_in_after_fitting": "not needed",
        "check_estimators_dtypes": "sites instead of y; missing smooth covariates",
        "check_dtype_object": "sites instead of y; missing smooth covariates",
        "check_estimators_pickle": "sites instead of y; missing smooth covariates",
        "check_f_contiguous_array_estimator": "sites instead of y; missing smooth covariates",
        "check_transformer_data_not_an_array": "sites instead of y; missing smooth covariates",
        "check_transformer_preserve_dtypes": "sites instead of y; missing smooth covariates",
        "check_transformer_general": "sites instead of y; missing smooth covariates",
        "check_transformers_unfitted": "checked inside",
        "check_methods_sample_order_invariance": "sites instead of y; missing smooth covariates",
        "check_methods_subset_invariance": "sites instead of y; missing smooth covariates",
        "check_dict_unchanged": "sites instead of y; missing smooth covariates",
        "check_fit_idempotent": "sites instead of y; missing smooth covariates",
        "check_n_features_in": "not needed",
        "check_fit2d_1sample": "missing smooth covariates",
        "check_fit2d_1feature": "missing smooth covariates",
        "check_fit2d_predict1d": "sites instead of y; missing smooth covariates",
        "check_requires_y_none": "target cannot be None",
        "check_fit_check_is_fitted": "missing smooth covariates",
        "check_fit1d": "missing smooth covariates",
    }


@parametrize_with_checks(
    [
        ComBatGAM(empirical_bayes=True, parametric_adjustments=True, mean_only=True),
        ComBatGAM(empirical_bayes=True, parametric_adjustments=True, mean_only=False),
        ComBatGAM(empirical_bayes=False, parametric_adjustments=True, mean_only=True),
        ComBatGAM(empirical_bayes=False, parametric_adjustments=True, mean_only=False),
    ],
    expected_failed_checks=_ex_failed_checks,
)
def test_combat_gam_compat_sklearn(estimator: object, check: Callable) -> None:
    """Test ComBatGAM compatibility with sklearn.

    Parameters
    ----------
    estimator : object
        Instance of ComBatGAM.
    check : callable
        sklearn fixture.

    """
    check(estimator)


@pytest.mark.parametrize(
    "empirical_bayes, parametric_adjustments, mean_only",
    [
        (True, True, True),
        (True, True, False),
        (True, False, True),
        (True, False, False),
        (False, True, True),
        (False, True, False),
        (False, False, True),
        (False, False, False),
    ],
)
def test_combat_gam(empirical_bayes: bool, parametric_adjustments: bool, mean_only: bool) -> None:
    """Test ComBatGAM.

    Parameters
    ----------
    empirical_bayes : bool
        Parametrized value of empirical_bayes.
    parametric_adjustments : bool
        Parametrized value of parametric_adjustments.
    mean_only : bool
        Parametrized value of mean_only.


    """
    X, y, sites = make_multisite_classification(n_samples=100)

    combat_gam = ComBatGAM(empirical_bayes=empirical_bayes, parametric_adjustments=parametric_adjustments, mean_only=mean_only)
    X_corrected = combat_gam.fit_transform(X, sites, smooth_covariates=y.reshape(-1, 1))
    assert X_corrected.shape == X.shape
    assert not np.allclose(X, X_corrected)  # Should be different from original
