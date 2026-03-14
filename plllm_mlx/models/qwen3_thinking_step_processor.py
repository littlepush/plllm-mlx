"""
Qwen3 thinking step processor for handling thinking mode generation.

This module provides a step processor that handles Qwen3's thinking mode,
which generates reasoning content before the actual response.
"""

from __future__ import annotations

import time
from typing import Any, List, Optional

from plllm_mlx.helpers import PlChunk, PlChunkDataType
from plllm_mlx.logging_config import get_logger
from plllm_mlx.models.base_step_processor import PlStepProcessor

logger = get_logger(__name__)


# Helper functions for MLX generation response
def PlMlxGetFinishReason(gr):
    """Get finish reason from MLX generation response."""
    if hasattr(gr, 'finish_reason'):
        return gr.finish_reason
    return None


class Qwen3ThinkingStepProcessor(PlStepProcessor):
    """
    Step processor for Qwen3 models with thinking mode support.
    
    This processor handles the special thinking mode where the model generates
    reasoning content before producing the actual response.
    """

    def __init__(self):
        super().__init__()
        self.is_in_thinking = False
        self.toolcall_buffer = []
        self.first_token_time = None
        self.stop_reason = ""
        self.is_stop_by_length = False
        self.full_content = ""

    @staticmethod
    def step_clz_name():
        return "qwen3think"

    def step(self, generate_response: Optional[Any] = None) -> Optional[PlChunk]:
        gr = generate_response
        self.total_tokens += 1
        if gr is not None:
            self.full_content += gr.text

        if self.first_token_time is None:
            self.first_token_time = time.time()

        # Just stop the generation
        if (gr is not None) and (PlMlxGetFinishReason(gr) is not None):
            self.is_stop_by_length = (
                True if PlMlxGetFinishReason(gr) == "length" else False
            )
            self.stop_reason = (
                PlMlxGetFinishReason(gr) if PlMlxGetFinishReason(gr) else "stop"
            )
            self.stop()
            return None

        # No content
        if gr is None:
            return None

        text = gr.text

        # Detect thinking mode start/end
        if not self.is_in_thinking:
            # Check if entering thinking mode
            if "hallucination_start" in text:
                self.is_in_thinking = True
                return None
            
            # Regular content
            return PlChunk(data=text, data_type=PlChunkDataType.CONTENT)
        else:
            # In thinking mode
            if "hallucination_end" in text:
                self.is_in_thinking = False
                return None
            
            # Still in thinking mode
            return PlChunk(data=text, data_type=PlChunkDataType.REASONING)

    def tool_calls(self) -> List[PlChunk]:
        """Return tool calls from buffer."""
        result = []
        self.stop_reason = "tool_calls"
        return result

    def finish(self) -> PlChunk:
        """Finish processing and return the final chunk."""
        finish_chunk = PlChunk(finish_reason=self.stop_reason)
        return finish_chunk


# Register the step processor
PlStepProcessor.registerStepProcessor("qwen3think", Qwen3ThinkingStepProcessor)
