"""
Package initialization for plllm-mlx models submodule.

This module provides:
- Local model file management (PlLocalModelManager)
- Model type detection (PlModelDetector)
- Model proxy for subprocess communication (via PlModelProxy)

For subprocess model loading, see plllm_mlx.subprocess.python
"""

from __future__ import annotations

from .local_models import PlLocalModelManager, PlLocalModelInfo, get_local_model_manager
from .model_detector import PlModelDetector

__all__: list[str] = [
    "PlLocalModelManager",
    "PlLocalModelInfo",
    "get_local_model_manager",
    "PlModelDetector",
]
