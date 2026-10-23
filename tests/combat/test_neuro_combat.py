"""NeuroComBat-specific tests.

Behaviour shared by all ComBat variants (API, site validation, covariates,
harmonization effect, sklearn compatibility) is tested in ``test_combat_common.py``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from uniharmony.combat import NeuroComBat
from uniharmony.datasets import load_MAREoS


# ---------------------------------------------------------------------------
# Original neuroCombat example data and MAREoS benchmark
# ---------------------------------------------------------------------------


def test_neuro_combat_ops_original() -> None:
    """Test operation of NeuroComBat with original."""
    data = np.genfromtxt(Path(__file__).parent / "test_data.csv", delimiter=",", skip_header=1)
    batches = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    genders = np.array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2])
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


def test_neuro_combat_several_covariates() -> None:
    """Several continuous and categorical covariates (one column each) are kept, and their effects preserved."""
    rng = np.random.default_rng(0)
    n, p = 300, 20
    sites = np.repeat([0, 1, 2], n // 3)
    age = rng.uniform(20, 80, n)
    sex = rng.integers(0, 2, n)
    group = rng.integers(0, 3, n)
    X = (
        0.05 * np.outer(age, np.ones(p))
        + 0.5 * np.outer(sex, np.ones(p))
        + np.array([0.0, 2.0, -1.0])[sites][:, None]
        + rng.standard_normal((n, p))
    )
    cont = np.column_stack([age, age**2 / 100])
    cat = np.column_stack([sex, group])
    combat = NeuroComBat().fit(X, sites, categorical_covariates=cat, continuous_covariates=cont)
    Xh = combat.transform(X, sites, categorical_covariates=cat, continuous_covariates=cont)
    assert Xh.shape == X.shape
    # site means removed, age and sex effects preserved
    assert np.ptp([Xh[sites == k].mean() for k in range(3)]) < 0.3
    assert np.corrcoef(Xh.mean(1), age)[0, 1] > 0.8
    # a single covariate passed as a 1-D array still works
    Xh1 = NeuroComBat().fit(X, sites, continuous_covariates=age).transform(X, sites, continuous_covariates=age)
    assert Xh1.shape == X.shape
