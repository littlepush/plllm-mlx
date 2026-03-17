"""
Logging configuration for plllm-mlx service.

This module provides logging configuration and setup utilities for
the plllm-mlx service, including custom handlers and formatters.
"""

import logging
import sys
from typing import Callable

from .config import LoggingConfig


def setup_logging(
    level: str = "info",
    config: LoggingConfig | None = None,
    log_callback: Callable[[str], None] | None = None,
) -> logging.Logger:
    """
    Configure and return the root logger for plllm-mlx.

    Args:
        level: Logging level (debug, info, warning, error, critical).
        config: Optional LoggingConfig instance for detailed configuration.
        log_callback: Optional callback function for custom log handling.

    Returns:
        Configured logger instance for plllm-mlx.
    """
    if config is None:
        config = LoggingConfig()

    level = level.lower()
    log_level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger("plllm_mlx")
    logger.setLevel(log_level)
    logger.handlers.clear()

    formatter = logging.Formatter(config.format)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if config.file:
        file_handler = logging.FileHandler(config.file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if log_callback is not None:
        callback_handler = CallbackHandler(log_callback)
        callback_handler.setLevel(log_level)
        callback_handler.setFormatter(formatter)
        logger.addHandler(callback_handler)

    logger.propagate = False

    return logger


class CallbackHandler(logging.Handler):
    """Custom logging handler that calls a callback function for each log record."""

    def __init__(self, callback: Callable[[str], None]) -> None:
        super().__init__()
        self.callback = callback

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.callback(msg)
        except Exception:
            self.handleError(record)


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger instance for a specific module."""
    root_logger = logging.getLogger("plllm_mlx")

    if not root_logger.handlers:
        setup_logging()

    if name:
        return logging.getLogger(f"plllm_mlx.{name}")

    return root_logger


def set_log_level(level: str) -> None:
    """Set the logging level for the plllm-mlx root logger."""
    level = level.lower()
    valid_levels = {"debug", "info", "warning", "error", "critical"}

    if level not in valid_levels:
        raise ValueError(f"Invalid log level: {level}. Must be one of {valid_levels}")

    logger = logging.getLogger("plllm_mlx")
    log_level = getattr(logging, level.upper())
    logger.setLevel(log_level)
