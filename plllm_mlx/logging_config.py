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

    This function sets up logging with console output and optional file output
    and custom callback for external log handling.

    Args:
        level: Logging level (debug, info, warning, error, critical).
        config: Optional LoggingConfig instance for detailed configuration.
        log_callback: Optional callback function for custom log handling.
            The callback receives formatted log messages as strings.

    Returns:
        Configured logger instance for plllm-mlx.

    Example:
        >>> logger = setup_logging(level="debug")
        >>> logger.info("Service started")

        >>> def my_callback(msg: str) -> None:
        ...     print(f"[CALLBACK] {msg}")
        >>> logger = setup_logging(log_callback=my_callback)
    """
    # Use config if provided, otherwise use defaults
    if config is None:
        config = LoggingConfig()

    # Normalize log level
    level = level.lower()
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Get or create the plllm-mlx logger
    logger = logging.getLogger("plllm_mlx")
    logger.setLevel(log_level)

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(config.format)

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Add file handler if specified
    if config.file:
        file_handler = logging.FileHandler(config.file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Add callback handler if specified
    if log_callback is not None:
        callback_handler = CallbackHandler(log_callback)
        callback_handler.setLevel(log_level)
        callback_handler.setFormatter(formatter)
        logger.addHandler(callback_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


class CallbackHandler(logging.Handler):
    """
    Custom logging handler that calls a callback function for each log record.

    This handler is useful for integrating with external logging systems
    or for custom log processing.

    Attributes:
        callback: Callable that receives formatted log messages.
    """

    def __init__(self, callback: Callable[[str], None]) -> None:
        """
        Initialize the callback handler.

        Args:
            callback: Function to call with formatted log messages.
        """
        super().__init__()
        self.callback = callback

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record by calling the callback.

        Args:
            record: The log record to emit.
        """
        try:
            msg = self.format(record)
            self.callback(msg)
        except Exception:
            # Silently ignore errors in callback to prevent logging loops
            self.handleError(record)


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    If the plllm-mlx root logger has not been configured yet, this function
    will configure it with default settings.

    Args:
        name: Optional module name. If provided, returns a child logger.
            If None, returns the plllm-mlx root logger.

    Returns:
        Logger instance for the specified module or the root logger.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Module initialized")

        >>> root_logger = get_logger()
        >>> root_logger.info("Service started")
    """
    root_logger = logging.getLogger("plllm_mlx")

    # Configure with defaults if not already configured
    if not root_logger.handlers:
        setup_logging()

    if name:
        # Create child logger
        return logging.getLogger(f"plllm_mlx.{name}")

    return root_logger


def set_log_level(level: str) -> None:
    """
    Set the logging level for the plllm-mlx root logger.

    This function allows dynamic adjustment of the logging level
    during runtime.

    Args:
        level: Logging level (debug, info, warning, error, critical).

    Raises:
        ValueError: If the level string is not a valid logging level.

    Example:
        >>> set_log_level("debug")  # Enable debug logging
        >>> set_log_level("warning")  # Reduce verbosity
    """
    level = level.lower()
    valid_levels = {"debug", "info", "warning", "error", "critical"}

    if level not in valid_levels:
        raise ValueError(f"Invalid log level: {level}. Must be one of {valid_levels}")

    logger = logging.getLogger("plllm_mlx")
    log_level = getattr(logging, level.upper())
    logger.setLevel(log_level)

    # Update all handlers
    for handler in logger.handlers:
        handler.setLevel(log_level)
