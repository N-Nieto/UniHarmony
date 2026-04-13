"""Tests for ComBatGAM transformer."""

from collections.abc import Callable

from sklearn.utils.estimator_checks import parametrize_with_checks

from uniharmony.combat import ComBatGAM


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
