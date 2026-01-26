# stdlib
import logging
import os
from typing import Any, Callable, NoReturn, TextIO, Union

# third party
from loguru import logger

LOG_FORMAT = "[{time}][{process.id}][{level}] {message}"
DEFAULT_SINK = "decaf_{time}.log"


def remove() -> None:
    """Remove all configured loggers."""
    logger.remove()


def add(
    sink: Union[None, str, os.PathLike, TextIO, logging.Handler] = None,
    level: str = "ERROR",
) -> None:
    """Add a logging sink."""
    sink = DEFAULT_SINK if sink is None else sink
    try:
        logger.add(
            sink=sink,
            format=LOG_FORMAT,
            enqueue=True,
            colorize=False,
            diagnose=True,
            backtrace=True,
            rotation="10 MB",
            retention="1 day",
            level=level,
        )
    except BaseException:
        logger.add(
            sink=sink,
            format=LOG_FORMAT,
            colorize=False,
            diagnose=True,
            backtrace=True,
            level=level,
        )


def traceback_and_raise(e: Any, verbose: bool = False) -> NoReturn:
    """Log an exception and re-raise."""
    try:
        if verbose:
            logger.opt(lazy=True).exception(e)
        else:
            logger.opt(lazy=True).critical(e)
    except BaseException:
        pass
    if not isinstance(e, Exception):
        e = Exception(e)
    raise e


def _log(level: str) -> Callable:
    def log_and_print(*args: Any, **kwargs: Any) -> None:
        try:
            method = getattr(logger.opt(lazy=True), level, None)
            if method:
                method(*args, **kwargs)
            else:
                logger.debug(*args, **kwargs)
        except BaseException:
            pass

    return log_and_print


traceback = _log("exception")
critical = _log("critical")
error = _log("error")
warning = _log("warning")
info = _log("info")
debug = _log("debug")
trace = _log("trace")
