"""Tests for the registry of ComBat-based models."""

import inspect

import pytest

import uniharmony.combat
from uniharmony.combat import ComBatGAM, NeuroComBat
from uniharmony.combat._available_models import COMBAT_MODELS, get_covariate_arguments


def test_every_combat_model_is_registered() -> None:
    """Every model exported by uniharmony.combat is in the registry."""
    exported = {getattr(uniharmony.combat, name) for name in uniharmony.combat.__all__}
    assert exported == set(COMBAT_MODELS)


@pytest.mark.parametrize("model", list(COMBAT_MODELS))
def test_registered_arguments_exist(model) -> None:
    """The registered covariate arguments are accepted by fit and transform, the bounds option by fit."""
    arguments = COMBAT_MODELS[model]
    fit_params = inspect.signature(model.fit).parameters
    transform_params = inspect.signature(model.transform).parameters
    for name in arguments.names:
        assert name in fit_params
        assert name in transform_params
    assert set(arguments.required) <= set(arguments.names)
    if arguments.smooth_bounds is not None:
        assert arguments.smooth_bounds in fit_params


def test_get_covariate_arguments() -> None:
    """Subclasses inherit the entry of their registered parent; other objects are rejected."""

    class MyComBat(NeuroComBat):
        pass

    assert get_covariate_arguments(MyComBat()) == COMBAT_MODELS[NeuroComBat]
    assert get_covariate_arguments(ComBatGAM()).smooth == "smooth_covariates"
    with pytest.raises(TypeError, match="ComBat-based model"):
        get_covariate_arguments(object())
