"""
Command-line interface for plllm-mlx service.

This module provides the command-line entry point for running the plllm-mlx service.
It parses command-line arguments, loads configuration, and starts the FastAPI server.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from .config import PlConfig
from .logging_config import setup_logging


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed argument namespace with host, port, log_level, and config options.
    """
    parser = argparse.ArgumentParser(
        prog="plllm-mlx",
        description="Standalone MLX-based LLM inference service with OpenAI compatible API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  plllm-mlx --config config.yaml
  plllm-mlx --host 0.0.0.0 --port 8000
  plllm-mlx --log-level debug

For more information, visit: https://github.com/littlepush/plllm-mlx
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=os.environ.get("PLLLM_MLX_CONFIG", "config.yaml"),
        help="Path to configuration file (default: config.yaml, env: PLLLM_MLX_CONFIG)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.environ.get("PLLLM_MLX_HOST", "0.0.0.0"),
        help="Server host address (default: 0.0.0.0, env: PLLLM_MLX_HOST)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PLLLM_MLX_PORT", "8000")),
        help="Server port (default: 8000, env: PLLLM_MLX_PORT)",
    )
    parser.add_argument(
        "--log-level",
        "-l",
        type=str,
        choices=["debug", "info", "warning", "error", "critical"],
        default=os.environ.get("PLLLM_MLX_LOG_LEVEL", "info"),
        help="Log level (default: info, env: PLLLM_MLX_LOG_LEVEL)",
    )

    return parser.parse_args()


def load_config(config_path: str) -> Optional[PlConfig]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        PlConfig instance if file exists, None otherwise.
    """
    path = Path(config_path)
    if not path.exists():
        logging.warning(f"Config file not found: {config_path}, using defaults")
        return None

    try:
        return PlConfig.from_yaml(path)
    except Exception as e:
        logging.warning(f"Failed to load config file: {e}, using defaults")
        return None


def main() -> int:
    """
    Main entry point for the plllm-mlx CLI.

    This function parses command-line arguments, loads configuration,
    and starts the FastAPI server.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    args = parse_args()
    logger = setup_logging(level=args.log_level)

    logger.info("Starting plllm-mlx service...")
    logger.info(f"Configuration file: {args.config}")
    logger.info(f"Server: {args.host}:{args.port}")
    logger.info(f"Log level: {args.log_level}")

    # Load configuration
    config = load_config(args.config)
    if config is None:
        config = PlConfig()

    # Merge command-line arguments
    config = config.merge_with_overrides(
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )

    try:
        # Import server module (deferred import for faster startup)
        from plllm_mlx import create_app
        import uvicorn

        # Create FastAPI application
        app = create_app(
            config=config,
            host=args.host,
            port=args.port,
            log_level=args.log_level,
        )

        # Start the server
        uvicorn.run(
            app,
            host=config.server.host,
            port=config.server.port,
            log_level=config.server.log_level,
        )
        return 0

    except ImportError as e:
        logger.error(f"Failed to import required module: {e}")
        logger.error("Make sure plllm_mlx is properly installed")
        return 1

    except KeyboardInterrupt:
        logger.info("Shutting down...")
        return 0

    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
