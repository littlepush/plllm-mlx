"""
Default step processor for basic LLM generation.

This module provides a simple step processor that handles standard text generation
without special processing like tool calls or reasoning channels.
"""

from __future__ import annotations

from typing import Any, List, Optional

from plllm_mlx.helpers import PlChunk, PlChunkDataType, PlStepUsage
from plllm_mlx.logging_config import get_logger
from plllm_mlx.models.base_step_processor import PlStepProcessor

logger = get_logger(__name__)


class PlDefaultStepProcessor(PlStepProcessor):
    """
    Default step processor for basic LLM generation.

    This processor handles standard text generation without special processing.
    It simply yields content chunks and handles finish reasons.
    """

    def __init__(self) -> None:
        """Initialize the default step processor."""
        super().__init__()
        self._finish_reason: Optional[str] = None

    @staticmethod
    def step_clz_name() -> str:
        """
        Return the name of this step processor class.

        Returns:
            The unique name identifier for this processor.
        """
        return "default"

    def step(self, generation_result: Any) -> Optional[PlChunk]:
        """
        Process a single generation step.

        Args:
            generation_result: The generation result from the LLM.
                Expected to have attributes: text, finish_reason, prompt_tokens,
                generation_tokens, prompt_tps, generation_tps.

        Returns:
            A PlChunk with the generated text, or None if no output.
        """
        gr = generation_result
        self.total_tokens += 1

        try:
            # Check for finish reason
            if gr.finish_reason is not None:
                self._finish_reason = gr.finish_reason
                self.is_running = False
                return None

            # Build usage info
            usage = PlStepUsage()
            usage.prompt_tokens = getattr(gr, "prompt_tokens", 0)
            usage.completion_tokens = getattr(gr, "generation_tokens", 0)
            usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
            usage.prompt_tps = getattr(gr, "prompt_tps", 0.0)
            usage.generation_tps = getattr(gr, "generation_tps", 0.0)

            # Create content chunk
            chunk = PlChunk(
                data=gr.text,
                data_type=PlChunkDataType.CONTENT,
                step=usage,
            )
            return chunk

        except Exception as e:
            logger.error(f"Step processing error: {e}")
            self.is_running = False
            return None

    def tool_calls(self) -> List[PlChunk]:
        """
        Get any tool calls detected during processing.

        The default processor does not detect tool calls.

        Returns:
            Empty list as default processor doesn't handle tool calls.
        """
        return []

    def finish(self) -> PlChunk:
        """
        Finish processing and return the final chunk.

        Returns:
            A PlChunk with the finish reason.
        """
        finish_reason = self._finish_reason or "stop"
        return PlChunk(finish_reason=finish_reason)


# Register the default step processor
PlStepProcessor.registerStepProcessor("default", PlDefaultStepProcessor)
PlStepProcessor.registerStepProcessor(
    "base", PlDefaultStepProcessor
)  # Alias for compatibility

logger.debug("Registered default step processor")
