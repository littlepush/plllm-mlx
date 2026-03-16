#!/usr/bin/env python3
"""
plllm-mlx - Standalone MLX-based LLM inference service

This is the main entry point for the plllm-mlx service.
It can be run directly or installed as a uv tool.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
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
        choices=["debug", "info", "warning", "error"],
        default=os.environ.get("PLLLM_MLX_LOG_LEVEL", "info"),
        help="Log level (default: info, env: PLLLM_MLX_LOG_LEVEL)",
    )

    return parser.parse_args()


def setup_logging(level: str) -> logging.Logger:
    """Configure and return the root logger."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("plllm_mlx")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        logging.warning(f"Config file not found: {config_path}, using defaults")
        return {}

    with open(path, "r") as f:
        config = yaml.safe_load(f) or {}

    return config


def merge_config(args: argparse.Namespace, config: dict) -> dict:
    """Merge command line arguments with config file settings."""
    # Command line arguments take precedence
    merged = {
        "server": {
            "host": args.host,
            "port": args.port,
            "log_level": args.log_level,
        },
        "model": config.get("model", {}),
        "cache": config.get("cache", {}),
        "logging": config.get("logging", {}),
    }

    # Override with config file values if not specified on command line
    if "server" in config:
        merged["server"].update(config["server"])

    return merged


def main() -> int:
    """Main entry point for the plllm-mlx service."""
    args = parse_args()
    logger = setup_logging(args.log_level)

    logger.info("Starting plllm-mlx service...")
    logger.info(f"Configuration file: {args.config}")
    logger.info(f"Server: {args.host}:{args.port}")
    logger.info(f"Log level: {args.log_level}")

    # Load and merge configuration
    config = load_config(args.config)
    merged_config = merge_config(args, config)

    try:
        # Import and create app
        from plllm_mlx import create_app
        import uvicorn

        # Create FastAPI application
        app = create_app(
            config_file=args.config if Path(args.config).exists() else None,
            host=args.host,
            port=args.port,
            log_level=args.log_level,
        )

        # Start the server
        logger.info(f"Starting server on {args.host}:{args.port}")
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level=args.log_level,
        )
        return 0

    except ImportError as e:
        logger.error(f"Failed to import plllm_mlx: {e}")
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
