"""Tests for UniHarmony's logging: ``verbosity``, ``verbosity_context`` and the module loggers."""

import importlib
import pkgutil

import numpy as np
import pytest
import structlog
from structlog._config import BoundLoggerLazyProxy
from structlog.testing import capture_logs

import uniharmony
from uniharmony import verbosity, verbosity_context
from uniharmony._verbose import _State, _UniHarmonyBoundLogger, get_logger
from uniharmony.combat import NeuroComBat


@pytest.fixture(autouse=True)
def restore_logging_config():
    """Leave UniHarmony's level and structlog's global configuration as they were found."""
    level = _State.level
    config = structlog.get_config()
    yield
    _State.level = level
    structlog.configure(**config)


@pytest.fixture
def data():
    """Two sites, enough samples for a quiet fit."""
    rng = np.random.default_rng(0)
    return rng.normal(size=(80, 3)), np.repeat([0, 1], 40)


def _module_loggers():
    """All module-level ``logger`` objects of the package."""
    for module_info in pkgutil.walk_packages(uniharmony.__path__, prefix="uniharmony."):
        module = importlib.import_module(module_info.name)
        if hasattr(module, "logger"):
            yield module_info.name, module.logger


@pytest.mark.parametrize("name,logger", list(_module_loggers()))
def test_module_loggers_are_lazy(name, logger):
    """Loggers must stay lazy, otherwise they freeze the structlog configuration of import time."""
    assert isinstance(logger, BoundLoggerLazyProxy), f"{name}.logger was bound at import time"


@pytest.mark.parametrize("name,logger", list(_module_loggers()))
def test_module_loggers_have_src(name, logger):
    """Every module logger labels its lines with ``src``, e.g. ``get_logger(src="NeuroComBat")``."""
    assert logger._initial_values.get("src"), f"{name}.logger has no src label"


@pytest.mark.parametrize("name,logger", list(_module_loggers()))
def test_module_loggers_follow_uniharmony_level(name, logger):
    """Every module logger is created with ``uniharmony._verbose.get_logger``."""
    assert logger._wrapper_class is _UniHarmonyBoundLogger, f"{name}.logger does not use uniharmony._verbose.get_logger"


def test_default_level_is_warning(capsys, data):
    """By default only warnings and errors are printed: a normal fit is silent."""
    _State.level = 30  # the import-time default, in case another test changed it
    NeuroComBat().fit(*data).transform(*data)
    assert capsys.readouterr().out == ""
    get_logger(src="test").warning("shown")
    assert "[warning" in capsys.readouterr().out


def test_transform_of_a_subset_of_sites_is_silent(capsys, data):
    """Transforming samples of only some fitted sites is normal use, not a warning."""
    X, sites = data
    harmonizer = NeuroComBat().fit(X, sites)
    capsys.readouterr()
    harmonizer.transform(X[sites == 0], sites[sites == 0])
    assert capsys.readouterr().out == ""


def test_verbosity_after_import_applies_to_harmonizers(capsys, data):
    """Setting the level after importing a harmonizer still applies to its logs."""
    verbosity("debug")
    NeuroComBat().fit(*data)
    assert "[debug" in capsys.readouterr().out

    verbosity("error")
    NeuroComBat().fit(*data)
    assert capsys.readouterr().out == ""


def test_verbosity_does_not_change_structlog_configuration():
    """UniHarmony's level is its own: structlog's global configuration (the application's) is untouched."""
    config = structlog.get_config()
    verbosity("debug")
    with verbosity_context("error"):
        pass
    assert structlog.get_config() == config


def test_application_level_is_independent(capsys):
    """An application that configures structlog keeps its own level; UniHarmony keeps its own."""
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger("error"))
    verbosity("info")
    structlog.get_logger().info("application info")
    get_logger(src="test").info("uniharmony info")
    out = capsys.readouterr().out
    assert "application info" not in out
    assert "uniharmony info" in out


def test_capture_logs_sees_uniharmony_messages():
    """``structlog.testing.capture_logs`` captures UniHarmony's messages at the active level."""
    verbosity("info")
    with capture_logs() as logs:
        get_logger(src="test").info("captured")
        get_logger(src="test").debug("filtered out")
    assert [entry["event"] for entry in logs] == ["captured"]
    assert logs[0]["src"] == "test"


def test_verbosity_context_restores_level(capsys):
    """The previous level is active again after the context."""
    logger = get_logger(src="test")
    verbosity("info")
    with verbosity_context("error"):
        logger.info("hidden")
    logger.info("shown")
    assert capsys.readouterr().out.count("[info") == 1


def test_exception_is_logged_at_error_level(capsys):
    """``logger.exception`` counts as an error for the level filter."""
    verbosity("error")
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        get_logger(src="test").exception("failed")
    assert "failed" in capsys.readouterr().out


def test_invalid_level_raises():
    """An unknown level gives a clear error."""
    with pytest.raises(ValueError, match="min_level"):
        verbosity("loud")  # pyright: ignore[reportArgumentType]  # invalid on purpose
