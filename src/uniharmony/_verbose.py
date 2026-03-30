"""Provide function to configure logging."""

from contextlib import contextmanager

import structlog


__all__ = ["verbosity", "verbosity_context"]


def verbosity(min_level="info") -> None:
    """Set verbosity level of logger.

    Parameters
    ----------
    min_level : {"critical", "error", "warning", "info", "debug"}
        Minimum level to log.

    Raises
    ------
    ValueError
        If `min_level` value is invalid.

    """
    try:
        structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(min_level))
    except KeyError as e:
        lvls = ["critical", "error", "warning", "info", "debug"]
        raise ValueError(f"`min_level` needs to be one of: {lvls}") from e


@contextmanager
def verbosity_context(min_level="info") -> None:
    """Context manager to control the logger verbosity.

    Parameters
    ----------
    min_level : {"critical", "error", "warning", "info", "debug"}
        Minimum level to log.

    Yields
    ------
    None

    """
    wrapper_cls = structlog.get_config()["wrapper_class"]
    verbosity(min_level)
    try:
        yield
    finally:
        structlog.configure(wrapper_class=wrapper_cls)
