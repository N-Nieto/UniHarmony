"""Tests for ComBatGAM transformer."""

from collections.abc import Callable

from sklearn.utils.estimator_checks import parametrize_with_checks

from uniharmony.combat import ComBatGAM


def _ex_failed_checks(_) -> dict[str, str]:
    return {
        "check_fit_score_takes_y": "sites instead of y",
        "check_n_features_in_after_fitting": "not needed",
        "check_estimators_dtypes": "sites instead of y",
        "check_dtype_object": "sites instead of y",
        "check_estimators_pickle": "sites instead of y",
        "check_f_contiguous_array_estimator": "sites instead of y",
        "check_transformer_data_not_an_array": "sites instead of y",
        "check_transformer_preserve_dtypes": "sites instead of y",
        "check_transformer_general": "sites instead of y",
        "check_transformers_unfitted": "checked inside",
        "check_methods_sample_order_invariance": "sites instead of y",
        "check_methods_subset_invariance": "sites instead of y",
        "check_dict_unchanged": "sites instead of y",
        "check_fit_idempotent": "sites instead of y",
        "check_n_features_in": "not needed",
        "check_fit2d_predict1d": "sites instead of y",
        "check_requires_y_none": "target cannot be None",
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
