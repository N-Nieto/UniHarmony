"""Tests for ``uniharmony.verbosity`` and ``uniharmony.verbosity_context``."""

import importlib
import pkgutil

import numpy as np
import pytest
import structlog
from structlog._config import BoundLoggerLazyProxy

import uniharmony
from uniharmony import verbosity, verbosity_context
from uniharmony.combat import NeuroComBat


@pytest.fixture(autouse=True)
def restore_logging_config():
    """Leave the global structlog configuration as it was found."""
    wrapper_class = structlog.get_config()["wrapper_class"]
    yield
    structlog.configure(wrapper_class=wrapper_class)


def _module_loggers():
    """All module-level ``logger`` objects of the package."""
    for module_info in pkgutil.walk_packages(uniharmony.__path__, prefix="uniharmony."):
        module = importlib.import_module(module_info.name)
        if hasattr(module, "logger"):
            yield module_info.name, module.logger


@pytest.mark.parametrize("name,logger", list(_module_loggers()))
def test_module_loggers_are_lazy(name, logger):
    """Loggers must stay lazy, otherwise they freeze the level that was active at import time."""
    assert isinstance(logger, BoundLoggerLazyProxy), f"{name}.logger was bound at import time"


@pytest.mark.parametrize("name,logger", list(_module_loggers()))
def test_module_loggers_have_src(name, logger):
    """Every module logger labels its lines with ``src``, e.g. ``structlog.get_logger(src="NeuroComBat")``."""
    assert logger._initial_values.get("src"), f"{name}.logger has no src label"


def test_verbosity_after_import_silences_harmonizers(capsys):
    """Setting the level after importing a harmonizer still applies to its logs."""
    rng = np.random.default_rng(0)
    X, sites = rng.normal(size=(40, 3)), np.repeat([0, 1], 20)

    verbosity("debug")
    NeuroComBat().fit(X, sites)
    assert "[debug" in capsys.readouterr().out

    verbosity("error")
    NeuroComBat().fit(X, sites)
    assert capsys.readouterr().out == ""


def test_verbosity_context_restores_level(capsys):
    """The previous level is active again after the context."""
    logger = structlog.get_logger()
    verbosity("info")
    with verbosity_context("error"):
        logger.info("hidden")
    logger.info("shown")
    assert capsys.readouterr().out.count("[info") == 1


def test_invalid_level_raises():
    """An unknown level gives a clear error."""
    with pytest.raises(ValueError, match="min_level"):
        verbosity("loud")
