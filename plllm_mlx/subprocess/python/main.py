#!/usr/bin/env python3
"""
Entry point for Python subprocess.

This is the main entry point for the Python model subprocess.
It starts a FastAPI server listening on a Unix domain socket.

Usage:
    python -m plllm_mlx.subprocess.python.main --socket /path/to/socket.sock
"""

from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser(description="Python model subprocess")
    parser.add_argument(
        "--socket",
        "-s",
        required=True,
        help="Unix domain socket path",
    )
    parser.add_argument(
        "--model",
        "-m",
        help="Model name to load on startup",
    )
    args = parser.parse_args()

    from plllm_mlx.subprocess.python.server import run_server

    run_server(socket_path=args.socket, model_name=args.model)


if __name__ == "__main__":
    main()
