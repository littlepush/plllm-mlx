"""
Token usage statistics for chat completion steps.

This module provides data structures and helpers for tracking
token usage during LLM inference, including prompt tokens,
completion tokens, and performance metrics.
"""

import time
from pydantic import BaseModel


class PlStepUsage(BaseModel):
    """
    Usage statistics for a single chat completion step.

    This model tracks token counts and performance metrics for
    prompt processing and token generation.

    Attributes:
        prompt_tokens: Number of tokens in the input prompt.
        completion_tokens: Number of tokens in the completion.
        total_tokens: Total tokens (prompt + completion).
        prompt_tps: Tokens per second during prompt processing.
        generation_tps: Tokens per second during content generation.
        prompt_process: Time in milliseconds to process the prompt.
        first_token: Time in seconds from prompt start to first token.
    """

    prompt_tokens: int = 0
    """Number of tokens in the input prompt."""

    completion_tokens: int = 0
    """Number of tokens in the completion output."""

    total_tokens: int = 0
    """Total tokens (prompt_tokens + completion_tokens)."""

    prompt_tps: float = 0
    """Tokens per second during prompt processing (thinking + generation)."""

    generation_tps: float = 0
    """Tokens per second during content generation only."""

    prompt_process: float = 0
    """Time in milliseconds to process the prompt."""

    first_token: float = 0
    """Time in seconds from prompt processing begin to first token generated."""


class PlStepHelper:
    """
    Helper class for building step usage statistics.

    This class tracks timing and usage metrics throughout the
    chat completion lifecycle, from prompt processing to token generation.

    Example:
        >>> helper = PlStepHelper()
        >>> helper.begin_process_prompt()
        >>> # ... process prompt ...
        >>> helper.end_process_prompt()
        >>> helper.update_step(PlStepUsage(prompt_tokens=100, completion_tokens=50))
        >>> usage = helper.build_usage()
    """

    def __init__(self) -> None:
        """Initialize the step helper with default values."""
        self._begin_prompt_time: float | None = None
        self._prompt_processed: bool = False
        self._first_token_generated: bool = False
        self._last_step_usage = PlStepUsage()

    def begin_process_prompt(self) -> None:
        """
        Mark the beginning of prompt processing.

        Records the current time for later calculation of prompt
        processing duration. Only takes effect on the first call.
        """
        if self._prompt_processed:
            return
        self._begin_prompt_time = time.time() * 1000

    def end_process_prompt(self) -> None:
        """
        Mark the end of prompt processing.

        Calculates and stores the prompt processing duration.
        Only takes effect on the first call after begin_process_prompt().
        """
        if self._prompt_processed:
            return
        now_time = time.time() * 1000
        self._prompt_processed = True
        self._last_step_usage.prompt_process = now_time - (
            self._begin_prompt_time or now_time
        )

    def update_step(self, step: PlStepUsage) -> None:
        """
        Update the step usage statistics.

        Also calculates the time to first token on the first update.

        Args:
            step: PlStepUsage instance with current token counts and metrics.
        """
        if not self._first_token_generated:
            self._first_token_generated = True
            if self._begin_prompt_time is not None:
                elapsed = time.time() * 1000 - self._begin_prompt_time
                prompt_time = self._last_step_usage.prompt_process
                self._last_step_usage.first_token = (elapsed - prompt_time) / 1000
        self._last_step_usage.prompt_tokens = step.prompt_tokens
        self._last_step_usage.completion_tokens = step.completion_tokens
        self._last_step_usage.total_tokens = step.total_tokens
        self._last_step_usage.prompt_tps = step.prompt_tps
        self._last_step_usage.generation_tps = step.generation_tps

    def build_usage(self) -> dict:
        """
        Build the usage statistics as a dictionary.

        Returns:
            Dictionary containing all usage metrics, suitable for
            JSON serialization in OpenAI API responses.
        """
        return self._last_step_usage.model_dump(mode="json")
