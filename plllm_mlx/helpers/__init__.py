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
"""

from __future__ import annotations

from .chain_cache import PlChain, PlChainCache
from .chat_helper import PlChatCompletionHelper
from .chunk_helper import PlChunk, PlChunkDataType
from .clz_helper import PlFindSpecifialSubclass, PlRootPath, PlUnpackPath
from .host_helper import get_hostname
from .response_helper import get_finish_reason
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
    # Response utilities
    "get_finish_reason",
    # Step usage
    "PlStepHelper",
    "PlStepUsage",
    # Tool call parsing
    "PlCommonToolcallParser",
]
