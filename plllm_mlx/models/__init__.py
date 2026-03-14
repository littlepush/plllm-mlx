"""
Package initialization for plllm-mlx models submodule.

This module provides model loading and inference functionality for the plllm-mlx service.
It contains model loaders for MLX-based models with OpenAI-compatible interfaces.
"""

from __future__ import annotations

# Base classes (no external dependencies)
from .base_step_processor import PlStepProcessor
from .model_loader import PlModelLoader

# Model management (no external dependencies)
from .model_detector import PlModelDetector
from .local_models import PlLocalModelManager, PlLocalModelInfo, get_local_model_manager
from .process_manager import PlProcessManager, PlModelSubprocess

# KV Cache (requires psutil)
from .kv_cache import PlMessageBasedKVCache, PlKVCacheMessage

# Step processors (import to trigger registration)
try:
    from .qwen3_thinking_step_processor import Qwen3ThinkingStepProcessor
    from .openai_step_processor import PlOpenAIStepProcessor
except Exception as e:
    import sys

    print(f"[ERROR] Failed to import step processors: {e}", file=sys.stderr)
    import traceback

    traceback.print_exc()

# Model loaders (require MLX - lazy import or optional)
# These are imported automatically when model_loader.py is loaded
# The loader classes register themselves via PlModelLoader.registerModelLoader()

__all__: list[str] = [
    # Base classes
    "PlStepProcessor",
    "PlModelLoader",
    # Model loaders (available via PlModelLoader.__LOADER_MAP__ if MLX is installed)
    # "PlMlxModel",  # Use PlModelLoader.__LOADER_MAP__["mlx"] instead
    # "PlMlxVlmModel",  # Use PlModelLoader.__LOADER_MAP__["mlxvlm"] instead
    # Model management
    "PlModelDetector",
    "PlLocalModelManager",
    "PlLocalModelInfo",
    "get_local_model_manager",
    "PlProcessManager",
    "PlModelSubprocess",
    # KV Cache
    "PlMessageBasedKVCache",
    "PlKVCacheMessage",
]
