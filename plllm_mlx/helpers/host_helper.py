"""
Host utilities for system information.

This module provides utilities for getting system host information.
"""

from __future__ import annotations

import socket


def get_hostname() -> str:
    """
    Get the hostname of the current machine.

    Removes the '.local' suffix if present.

    Returns:
        The hostname string.
    """
    return socket.gethostname().removesuffix(".local")
