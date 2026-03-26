"""Tests for NeuroComBat transformer."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.estimator_checks import parametrize_with_checks

from uniharmony.combat import NeuroComBat
from uniharmony.datasets import load_MAREoS


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
        NeuroComBat(empirical_bayes=True, parametric_adjustments=False, mean_only=False),
        NeuroComBat(empirical_bayes=True, parametric_adjustments=False, mean_only=True),
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


def test_neuro_combat_reproducibility() -> None:
    """Test reproducibility of NeuroComBat."""
    data = np.genfromtxt(Path(__file__).parent / "test_data.csv", delimiter=",", skip_header=1)
    batches = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2]).reshape(-1, 1)
    genders = np.array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2]).reshape(-1, 1)
    data_combat_v1 = NeuroComBat().fit_transform(
        data.T,
        batches,
        categorical_covariates=genders,
    )
    data_combat_v2 = NeuroComBat().fit_transform(
        data.T,
        batches,
        categorical_covariates=genders,
    )
    np.testing.assert_array_equal(data_combat_v1, data_combat_v2, strict=True)


def test_neuro_combat_reproducibility_categorical() -> None:
    """Test reproducibility of NeuroComBat with categoricals."""
    data = np.genfromtxt(Path(__file__).parent / "test_data.csv", delimiter=",", skip_header=1)
    batches = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2]).reshape(-1, 1)
    genders = np.array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2]).reshape(-1, 1)
    data_combat_v1 = NeuroComBat().fit_transform(
        data.T,
        batches,
        categorical_covariates=genders,
    )
    data_combat_v2 = NeuroComBat().fit_transform(
        data.T,
        batches,
    )
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(data_combat_v1, data_combat_v2, strict=True)


def test_neuro_combat_no_parametrics() -> None:
    """Test reproducibility of NeuroComBat with categoricals."""
    data = np.genfromtxt(Path(__file__).parent / "test_data.csv", delimiter=",", skip_header=1)
    batches = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2]).reshape(-1, 1)
    genders = np.array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2]).reshape(-1, 1)
    _ = NeuroComBat(empirical_bayes=True, parametric_adjustments=False, mean_only=False).fit_transform(
        data.T,
        batches,
        categorical_covariates=genders,
    )


def test_neuro_combat_site_with_nan() -> None:
    """Test Site with nan."""
    data = np.genfromtxt(Path(__file__).parent / "test_data.csv", delimiter=",", skip_header=1)
    batches = np.array([1, 1, 1, np.nan, 1, 2, 2, 2, 2, 2], dtype=object).reshape(-1, 1)
    genders = np.array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2]).reshape(-1, 1)
    with pytest.raises(ValueError):
        _ = NeuroComBat(empirical_bayes=True, parametric_adjustments=False, mean_only=False).fit_transform(
            data.T,
            batches,
            categorical_covariates=genders,
        )


def test_neuro_combat_covars_with_nan() -> None:
    """Test covars with nan."""
    data = np.genfromtxt(Path(__file__).parent / "test_data.csv", delimiter=",", skip_header=1)
    batches = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2], dtype=object).reshape(-1, 1)
    genders = np.array([1, 2, 1, 2, np.nan, 2, 1, 2, 1, 2], dtype=object).reshape(-1, 1)
    with pytest.raises(ValueError):
        _ = NeuroComBat(empirical_bayes=True, parametric_adjustments=False, mean_only=False).fit_transform(
            data.T,
            batches,
            categorical_covariates=genders,
        )


def test_neuro_combat_sites_as_str() -> None:
    """Test sites as str."""
    data = np.genfromtxt(Path(__file__).parent / "test_data.csv", delimiter=",", skip_header=1)
    batches = ["1", "1", "1", "1", "1", "2", "2", "2", "2", "2"]
    genders = np.array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2]).reshape(-1, 1)

    _ = NeuroComBat(empirical_bayes=True, parametric_adjustments=False, mean_only=False).fit_transform(
        data.T,
        batches,
        categorical_covariates=genders,
    )


def test_neuro_combat_unseen_site() -> None:
    """Test unseen sites."""
    data = np.genfromtxt(Path(__file__).parent / "test_data.csv", delimiter=",", skip_header=1)
    batches = ["1", "1", "1", "1", "1", "2", "2", "2", "2", "2"]
    genders = np.array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2]).reshape(-1, 1)
    neurocombat = NeuroComBat(empirical_bayes=True, parametric_adjustments=False, mean_only=False)
    _ = neurocombat.fit(
        data.T,
        batches,
        categorical_covariates=genders,
    )
    batches_unseen = ["3", "3", "3", "3", "3", "4", "4", "5", "5", "5"]
    with pytest.raises(ValueError):
        _ = neurocombat.transform(data.T, batches_unseen, categorical_covariates=genders)


def test_neuro_combat_unseen_site_same_nsites() -> None:
    """Test sites  unseen sites with matching number of sites."""
    data = np.genfromtxt(Path(__file__).parent / "test_data.csv", delimiter=",", skip_header=1)
    batches = ["1", "1", "1", "1", "1", "2", "2", "2", "2", "2"]
    genders = np.array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2]).reshape(-1, 1)
    neurocombat = NeuroComBat(empirical_bayes=True, parametric_adjustments=False, mean_only=False)
    _ = neurocombat.fit(
        data.T,
        batches,
        categorical_covariates=genders,
    )
    batches_unseen = ["3", "3", "3", "3", "3", "4", "4", "4", "4", "4"]
    with pytest.raises(ValueError):
        _ = neurocombat.transform(data.T, batches_unseen, categorical_covariates=genders)


def test_neuro_combat_max_iter() -> None:
    """Test sites max iter."""
    data = np.genfromtxt(Path(__file__).parent / "test_data.csv", delimiter=",", skip_header=1)
    batches = ["1", "1", "1", "1", "1", "2", "2", "2", "2", "2"]
    genders = np.array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2]).reshape(-1, 1)
    neurocombat = NeuroComBat(empirical_bayes=True, parametric_adjustments=True, mean_only=False)
    _ = neurocombat.fit(data.T, batches, categorical_covariates=genders, max_iter=1)


def test_neuro_combat_performance_mareos() -> None:
    """Test performance of NeuroComBat with MAREoS dataset."""
    # Load the MAREoS dataset, made for benchmarking harmonization methods.
    datasets = load_MAREoS()

    # Define the different effects, effect types, and examples to iterate over
    effects = ["true", "eos"]
    effect_types = ["simple", "interaction"]
    effect_examples = ["1", "2"]

    random_state = 23
    baseline_bacc = []
    neuro_combat_bacc = []
    clf = LogisticRegression()
    # Define the harmonization model to use (NeuroComBat in this case)
    harm_model = NeuroComBat()
    for effect in effects:
        for e_types in effect_types:
            if e_types == "interaction":
                clf = RandomForestClassifier(n_estimators=10, random_state=random_state)
            elif e_types == "simple":
                clf = LogisticRegression(random_state=random_state)
            for e_example in effect_examples:
                example = effect + "_" + e_types + e_example
                data = datasets[example]
                folds = data["folds"]
                folds = pd.Series(folds)

                for fold in folds.unique():
                    # Train Data
                    X = data["X"].copy()
                    y = data["y"].copy()
                    sites = data["sites"].copy()

                    # Train data
                    X_train = X[data["folds"] != fold]
                    site_train = sites[data["folds"] != fold]
                    y_train = y[data["folds"] != fold]

                    # Test data
                    X_test = X[data["folds"] == fold]
                    site_test = sites[data["folds"] == fold]
                    y_test = y[data["folds"] == fold]

                    # Unharmonized baseline model
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X=X_test)
                    bacc_baseline = balanced_accuracy_score(y_true=y_test, y_pred=y_pred)
                    baseline_bacc.append(bacc_baseline)
                    # neuroComBat (do not include target as covariate - avoiding data leakage)
                    X_train_harm = harm_model.fit_transform(X=X_train, sites=site_train)
                    # Fit the model with the harmonized train
                    clf.fit(X_train_harm, y_train)
                    # harmonize the test data
                    X_test_harm = harm_model.transform(X=X_test, sites=site_test)
                    y_pred = clf.predict(X=X_test_harm)
                    bacc_neurocombat = balanced_accuracy_score(y_true=y_test, y_pred=y_pred)
                    neuro_combat_bacc.append(bacc_neurocombat)

                # Analyze the results for the current effect
                if effect == "true":
                    # For true effects, we expect the performance to be the same as the baseline, as no EoS are present.
                    assert np.isclose(np.array(baseline_bacc).mean(), np.array(neuro_combat_bacc).mean(), atol=0.2)
                    # The baseline performance should be around 80% bacc. If not, the model failed.
                    assert np.isclose(np.array(baseline_bacc).mean(), 0.8, atol=0.2)

                elif effect == "eos":
                    # For EOS effects, we expect the harmonization performance to be chance, if it is able to remove the EOS.
                    assert np.isclose(0.5, np.array(neuro_combat_bacc).mean(), atol=0.2)
                    # The baseline performance should still be high using EoS information, around 80% bacc.
                    assert np.isclose(np.array(baseline_bacc).mean(), 0.8, atol=0.2)
