"""
Package initialization for plllm-mlx.

This module provides the main entry point for creating and configuring
the plllm-mlx FastAPI application.

plllm-mlx is a standalone MLX-based LLM inference service with OpenAI compatible API,
designed specifically for Apple Silicon.

Example:
    >>> from plllm_mlx import create_app
    >>> app = create_app(config=my_config)
    >>> # Or with config file:
    >>> app = create_app(config_file="config.yaml")

    >>> # Run with uvicorn:
    >>> import uvicorn
    >>> uvicorn.run(app, host="0.0.0.0", port=8000)
"""

from __future__ import annotations

from typing import Callable, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import PlConfig
from .logging_config import setup_logging


def create_app(
    config: Optional[PlConfig] = None,
    config_file: Optional[str] = None,
    host: str = "0.0.0.0",
    port: int = 8080,
    log_level: str = "INFO",
    log_callback: Optional[Callable[[str], None]] = None,
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    This is the main entry point for creating the plllm-mlx FastAPI application.
    It sets up the app with proper configuration, middleware, and routers.

    Args:
        config: PlConfig instance with configuration settings. If provided,
            takes precedence over config_file.
        config_file: Path to YAML configuration file. Used only if config is None.
        host: Server host address (default: "0.0.0.0").
        port: Server port number (default: 8080).
        log_level: Logging level - debug, info, warning, error, critical
            (default: "INFO").
        log_callback: Optional callback function for custom log handling.
            The callback receives formatted log messages as strings.

    Returns:
        Configured FastAPI application instance.

    Raises:
        FileNotFoundError: If config_file is provided but does not exist.
        ConfigurationError: If configuration is invalid.

    Example:
        >>> # Create with default configuration
        >>> app = create_app()

        >>> # Create with configuration object
        >>> from plllm_mlx.config import PlConfig, ModelConfig
        >>> config = PlConfig(model=ModelConfig(name="Qwen/Qwen2.5-7B-Instruct"))
        >>> app = create_app(config=config)

        >>> # Create with configuration file
        >>> app = create_app(config_file="config.yaml")

        >>> # Create with log callback for external logging
        >>> def my_logger(msg: str) -> None:
        ...     print(f"[LOG] {msg}")
        >>> app = create_app(log_callback=my_logger)
    """
    # Setup logging first
    logger = setup_logging(
        level=log_level,
        config=config.logging if config else None,
        log_callback=log_callback,
    )

    # Load configuration
    if config is None:
        if config_file is not None:
            config = PlConfig.from_yaml(config_file)
        else:
            config = PlConfig()

    # Merge with CLI overrides
    config = config.merge_with_overrides(
        host=host,
        port=port,
        log_level=log_level.lower(),
    )

    # Create FastAPI application
    app = FastAPI(
        title="plllm-mlx",
        description="Standalone MLX-based LLM inference service with OpenAI compatible API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Store configuration in app state
    app.state.config = config
    app.state.logger = logger

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    # Note: Routers will be added as they are implemented
    # from .routers import chat, models, completions
    # app.include_router(chat.router, prefix="/v1", tags=["chat"])
    # app.include_router(models.router, prefix="/v1", tags=["models"])
    # app.include_router(completions.router, prefix="/v1", tags=["completions"])

    # Health check endpoint
    @app.get("/health", tags=["health"])
    async def health_check() -> dict:
        """
        Health check endpoint.

        Returns:
            Status dictionary indicating service health.
        """
        return {"status": "healthy", "service": "plllm-mlx"}

    # Root endpoint
    @app.get("/", tags=["root"])
    async def root() -> dict:
        """
        Root endpoint with service information.

        Returns:
            Service information dictionary.
        """
        return {
            "service": "plllm-mlx",
            "version": "1.0.0",
            "description": "Standalone MLX-based LLM inference service",
            "docs": "/docs",
            "health": "/health",
        }

    logger.info(
        f"FastAPI application created - "
        f"host={config.server.host}, port={config.server.port}"
    )

    return app


__all__: list[str] = [
    "create_app",
    "PlConfig",
]
