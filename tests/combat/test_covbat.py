"""Tests for CovBat transformer."""

from collections.abc import Callable

from sklearn.utils.estimator_checks import parametrize_with_checks

from uniharmony.combat import CovBat


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
        CovBat(),
    ],
    expected_failed_checks=_ex_failed_checks,
)
def test_neuro_combat_compat_sklearn(estimator: object, check: Callable) -> None:
    """Test NeuroComBat compatibility with sklearn.

    Parameters
    ----------
    estimator : object
        Instance of NeuroComBat.
    check : callable
        sklearn fixture.

    """
    check(estimator)
