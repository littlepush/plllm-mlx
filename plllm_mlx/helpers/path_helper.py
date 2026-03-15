"""
Path helper for HuggingFace cache directory.

This module provides utilities for finding model paths in HuggingFace cache.
"""

from __future__ import annotations

import os
from pathlib import Path


def get_hf_cache_dir() -> str:
    """
    Get HuggingFace cache directory with priority order.

    Priority:
    1. HF_HUB_CACHE environment variable
    2. HF_HOME environment variable + "/hub"
    3. HUGGING_FACE_PATH environment variable (legacy)
    4. Default: ~/.cache/huggingface/hub

    Returns:
        Path to HuggingFace cache directory.
    """
    if os.environ.get("HF_HUB_CACHE"):
        return os.environ["HF_HUB_CACHE"]
    if os.environ.get("HF_HOME"):
        return os.path.join(os.environ["HF_HOME"], "hub")
    if os.environ.get("HUGGING_FACE_PATH"):
        return os.environ["HUGGING_FACE_PATH"]
    return f"{Path.home()}/.cache/huggingface/hub"


def get_model_cache_path(model_name: str) -> Path | None:
    """
    Get the cache path for a model.

    Args:
        model_name: HuggingFace model name (e.g., "Qwen/Qwen3-8B")

    Returns:
        Path to model cache directory, or None if not found.
    """
    cache_dir = Path(get_hf_cache_dir())
    model_id_clean = model_name.replace("/", "--")
    model_path = cache_dir / f"models--{model_id_clean}"

    if not model_path.exists():
        return None

    return model_path


def get_model_snapshot_path(model_name: str) -> Path | None:
    """
    Get the snapshot path for a model (the actual model files).

    Args:
        model_name: HuggingFace model name (e.g., "Qwen/Qwen3-8B")

    Returns:
        Path to model snapshot directory, or None if not found.
    """
    model_path = get_model_cache_path(model_name)
    if model_path is None:
        return None

    snapshots_dir = model_path / "snapshots"
    if not snapshots_dir.exists():
        return None

    snapshots = list(snapshots_dir.iterdir())
    if not snapshots:
        return None

    return snapshots[0]


HF_HUB_CACHE = get_hf_cache_dir()
