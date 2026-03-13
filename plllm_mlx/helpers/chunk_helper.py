"""
Chunk data structure for streaming responses.

This module defines the data types used for representing
chunks in streaming chat completions, including reasoning,
content, and tool call chunks.
"""

from enum import Enum
from typing import Optional, Union, Any
from pydantic import BaseModel

from .step_info import PlStepUsage


class PlChunkDataType(Enum):
    """
    Enumeration of chunk data types.

    Attributes:
        NONE: No specific data type.
        REASONING: Reasoning/thinking content.
        CONTENT: Regular text content.
        TOOLCALL: Tool/function call.
    """

    NONE = 0
    REASONING = 1
    CONTENT = 2
    TOOLCALL = 3


class PlChunk(BaseModel):
    """
    A single chunk in a streaming response.

    This model represents one chunk of data in a streaming chat
    completion response, containing the data type, content,
    finish reason, and usage statistics.

    Attributes:
        data_type: The type of data in this chunk.
        data: The actual content (string for text, dict for tool calls).
        finish_reason: Reason for completion if this is the final chunk.
        step: Token usage statistics for this step.

    Example:
        >>> chunk = PlChunk(
        ...     data_type=PlChunkDataType.CONTENT,
        ...     data="Hello, world!",
        ...     step=PlStepUsage(completion_tokens=3)
        ... )
    """

    data_type: PlChunkDataType = PlChunkDataType.NONE
    """The type of data contained in this chunk."""

    data: Optional[Union[str, dict]] = None
    """The chunk content (string for text, dict for tool calls)."""

    finish_reason: Optional[str] = None
    """Reason for completion (e.g., 'stop', 'tool_calls')."""

    step: Optional[PlStepUsage] = None
    """Token usage statistics for this chunk."""
