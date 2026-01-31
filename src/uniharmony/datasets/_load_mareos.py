"""Functions to load the MAREoS dataset."""

__all__ = ["load_MAREoS", "load_MAREoS_single_dataset"]
from pathlib import Path

import pandas as pd


def load_MAREoS_single_dataset(  # noqa: N802
    effect: str = "eos",
    effect_type: str = "simple",
    effect_example: str = "1",
    as_numpy: bool = True,
):
    """Docstring for load_MAREoS.

    :param effects: type of effect. Available options: true or eos
    :param effect_types: Type of effect.
                        Availabe options "simple", "interaction"
    :param effect_examples: Which of the two simulated datasets to load.
    """
    dataset_name = effect + "_" + effect_type + effect_example
    data_dir = Path(__file__).resolve().parent
    # TODO: download only the necesary files.
    X = pd.read_csv(
        data_dir / "data" / "MAREoS" / (dataset_name + "_data.csv"),
        index_col=0,
    )
    y = pd.read_csv(
        data_dir / "data" / "MAREoS" / (dataset_name + "_response.csv"),
        index_col=0,
    )
    sites = X["site"]
    covs = X.loc[:, ["cov1", "cov2"]]
    folds = X["folds"]

    X.drop(columns=["site", "cov1", "cov2", "folds"], inplace=True)
    if as_numpy:
        X = X.to_numpy()
        y = y.to_numpy()
        sites = sites.to_numpy()
        covs = covs.to_numpy()
        folds = folds.to_numpy()

    return X, y, sites, covs, folds


def load_MAREoS(  # noqa: N802
    effects=None,
    effect_types=None,
    effect_examples=None,
    as_numpy: bool = True,
):
    if effects is None:
        effects = ["eos", "true"]
    if effect_types is None:
        effect_types = ["simple", "interaction"]
    if effect_examples is None:
        effect_examples = ["1", "2"]

    dataset = {}
    for effect in effects:
        for e_type in effect_types:
            for e_example in effect_examples:
                dataset_name = effect + "_" + e_type + e_example

                X, y, sites, covs, folds = load_MAREoS_single_dataset(
                    effect, e_type, e_example, as_numpy
                )
                dataset[dataset_name] = {
                    "X": X,
                    "y": y,
                    "sites": sites,
                    "covs": covs,
                    "folds": folds,
                }

    return dataset
