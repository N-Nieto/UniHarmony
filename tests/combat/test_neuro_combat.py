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


# ---------------------------------------------------------------------------
# Design matrix and agreement with the reference neuroCombat implementation
# Data comes from the shared fixtures in tests/conftest.py
# ---------------------------------------------------------------------------

REFERENCE_COVARIATE_COMBINATIONS = [
    (("sex", "education"), ("age",)),
    (("sex",), ("age", "education")),
    (("sex", "education"), ("age", "extra")),
]


def test_neuro_combat_design_matrix_one_block_per_covariate(multisite_data) -> None:
    """Each covariate column gets its own block in the design matrix."""
    d = multisite_data
    model = NeuroComBat()
    design = model.fit_design_matrix(
        sites=d.sites,
        categorical_covariates=d.get("sex", "education"),
        continuous_covariates=d.get("age"),
    )
    # 3 sites + 1 sex dummy (drop-first) + 3 education dummies (drop-first) + 1 age column
    assert design.shape == (d.n_samples, 3 + 1 + 3 + 1)
    assert len(model._categorical_encoders) == 2
    np.testing.assert_array_equal(design[:, -1], d.covariates["age"])


@pytest.mark.parametrize(("categorical", "continuous"), REFERENCE_COVARIATE_COMBINATIONS)
def test_neuro_combat_matches_reference_with_multiple_covariates(multisite_data, categorical: tuple, continuous: tuple) -> None:
    """Results match the reference neuroCombat implementation."""
    neuro_combat = pytest.importorskip("neuroCombat")
    d = multisite_data
    covars = pd.DataFrame({"batch": d.sites, **{name: d.covariates[name] for name in categorical + continuous}})
    expected = neuro_combat.neuroCombat(
        dat=d.X.T,
        covars=covars,
        batch_col="batch",
        categorical_cols=list(categorical),
        continuous_cols=list(continuous),
    )["data"].T
    result = NeuroComBat().fit_transform(
        d.X, d.sites, categorical_covariates=d.get(*categorical), continuous_covariates=d.get(*continuous)
    )
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)
