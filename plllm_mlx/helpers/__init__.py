"""
Helpers module for plllm-mlx.

This module provides utility classes and functions for:
- OpenAI format response building (chat_helper)
- Chunk data structure (chunk_helper)
- Tool call parsing (toolcall_helper)
- Token usage statistics (step_info)
- Message chain caching (chain_cache)
- Class discovery utilities (clz_helper)
- Host utilities (host_helper)
- Response utilities (response_helper)
- Path utilities (path_helper)
"""

from __future__ import annotations

from .chain_cache import PlChain, PlChainCache
from .chat_helper import PlChatCompletionHelper
from .chunk_helper import PlChunk, PlChunkDataType
from .clz_helper import PlFindSpecifialSubclass, PlRootPath, PlUnpackPath
from .host_helper import get_hostname
from .path_helper import (
    get_hf_cache_dir,
    get_model_cache_path,
    get_model_snapshot_path,
    HF_HUB_CACHE,
)
from .response_helper import get_finish_reason, PlMlxGetFinishReason
from .step_info import PlStepHelper, PlStepUsage
from .toolcall_helper import PlCommonToolcallParser

__all__: list[str] = [
    # Chain cache
    "PlChain",
    "PlChainCache",
    # Chat completion
    "PlChatCompletionHelper",
    # Chunk data
    "PlChunk",
    "PlChunkDataType",
    # Class utilities (with Pl prefix for compatibility)
    "PlRootPath",
    "PlFindSpecifialSubclass",
    "PlUnpackPath",
    # Host utilities
    "get_hostname",
    # Path utilities
    "get_hf_cache_dir",
    "get_model_cache_path",
    "get_model_snapshot_path",
    "HF_HUB_CACHE",
    # Response utilities
    "get_finish_reason",
    "PlMlxGetFinishReason",
    # Step usage
    "PlStepHelper",
    "PlStepUsage",
    # Tool call parsing
    "PlCommonToolcallParser",
]
