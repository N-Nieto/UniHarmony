"""Provide UniHarmony's logging: module loggers and the verbosity controls.

UniHarmony logs with structlog but never changes structlog's global
configuration, which belongs to the application. Its loggers carry their own
level filter, set with :func:`verbosity` (default ``"warning"``), so that

* a plain ``import uniharmony`` and ``fit`` print nothing unless something is
  wrong (warnings and errors);
* the UniHarmony level is independent of the level other libraries or the
  application configure for structlog;
* the rest of the application's structlog configuration (processors,
  renderer, output) still applies to UniHarmony's messages, and
  ``structlog.testing.capture_logs`` captures them.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Literal

import structlog
from structlog.typing import FilteringBoundLogger


__all__ = ["get_logger", "verbosity", "verbosity_context"]

MIN_LEVEL = Literal["critical", "error", "warning", "info", "debug"]

_LEVELS = {"debug": 10, "info": 20, "warning": 30, "error": 40, "critical": 50}
_DEFAULT_LEVEL = "warning"


class _State:
    """Mutable holder of the active UniHarmony level (a number from ``_LEVELS``)."""

    level: int = _LEVELS[_DEFAULT_LEVEL]


# structlog's bound logger with every level enabled (created at runtime, hence typed as Any)
_UnfilteredBoundLogger: type[Any] = structlog.make_filtering_bound_logger("debug")


class _UniHarmonyBoundLogger(_UnfilteredBoundLogger):
    """Bound logger whose level filter reads UniHarmony's level at every call.

    structlog's filtering loggers fix their level when the class is created.
    Here every method is enabled and the filter is applied when a message is
    sent, so :func:`verbosity` takes effect immediately, also for loggers
    created at import time.
    """

    def _proxy_to_logger(self, method_name: str, event: Any = None, **event_kw: Any) -> Any:
        # "exception" is logged at error level; unknown method names are treated as info
        level = _LEVELS.get("error" if method_name == "exception" else method_name, _LEVELS["info"])
        if level < _State.level:
            return None
        return super()._proxy_to_logger(method_name, event, **event_kw)

    def is_enabled_for(self, level: int) -> bool:
        """Whether a message at ``level`` (a :mod:`logging` level number) would be logged."""
        return level >= _State.level

    def get_effective_level(self) -> int:
        """Return the active UniHarmony level as a :mod:`logging` level number."""
        return _State.level


def get_logger(**initial_values: Any) -> FilteringBoundLogger:
    """Return a UniHarmony module logger.

    The logger is lazy (bound at the first message, so it follows the
    application's structlog configuration at that moment) and filtered by
    UniHarmony's level.

    Parameters
    ----------
    **initial_values
        Values bound to every message, e.g. ``src="NeuroComBat"``.

    Returns
    -------
    structlog logger
        The module logger.

    """
    return structlog.wrap_logger(None, wrapper_class=_UniHarmonyBoundLogger, **initial_values)


def verbosity(min_level: MIN_LEVEL = "warning") -> None:
    """Set the minimum level of the messages UniHarmony logs.

    Only UniHarmony's messages are affected; structlog's global configuration
    is left unchanged.

    Parameters
    ----------
    min_level : {"critical", "error", "warning", "info", "debug"}, optional (default "warning")
        Minimum level to log. ``"info"`` reports the main steps of each
        method, ``"debug"`` also intermediate values.

    Raises
    ------
    ValueError
        If `min_level` value is invalid.

    """
    try:
        _State.level = _LEVELS[min_level]
    except KeyError as e:
        raise ValueError(f"`min_level` needs to be one of: {list(_LEVELS)}") from e


@contextmanager
def verbosity_context(min_level: MIN_LEVEL = "warning") -> Iterator[None]:
    """Context manager to set UniHarmony's verbosity temporarily.

    Parameters
    ----------
    min_level : {"critical", "error", "warning", "info", "debug"}, optional (default "warning")
        Minimum level to log inside the context.

    Yields
    ------
    None

    """
    previous = _State.level
    verbosity(min_level)
    try:
        yield
    finally:
        _State.level = previous
