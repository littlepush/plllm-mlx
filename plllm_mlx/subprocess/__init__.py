"""
Subprocess service architecture for model isolation.

This module provides:
- PlSubprocessManager: Manages subprocess lifecycle (main process)
- PlSubprocessHandle: HTTP over UDS client (main process)
- PlModelProxy: Model proxy for routers (main process)
- python/: Python subprocess implementation
"""

from .client import PlSubprocessHandle
from .manager import PlSubprocessManager, get_subprocess_manager
from .proxy import PlModelProxy

__all__ = [
    "PlSubprocessHandle",
    "PlSubprocessManager",
    "PlModelProxy",
    "get_subprocess_manager",
]
