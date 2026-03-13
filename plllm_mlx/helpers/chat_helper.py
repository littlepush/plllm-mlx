"""
OpenAI format response builder for chat completions.

This module provides a helper class for building OpenAI-compatible
streaming chat completion responses, including support for reasoning,
content, and tool calls.
"""

from typing import Optional
import time
import uuid
import json

from .step_info import PlStepUsage, PlStepHelper


class PlChatCompletionHelper:
    """
    Helper class for building OpenAI-compatible chat completion chunks.

    This class manages the state and formatting of streaming chat
    completion responses, tracking the conversation through reasoning,
    content generation, and tool calls.

    Attributes:
        _model_name: The model name to include in responses.
        _begin_time: Timestamp when the completion started.
        _chat_id: Unique identifier for this chat completion.
        _step_helper: Helper for tracking token usage.
        _is_finished: Whether the completion has finished.
        _finish_reason: Reason for completion (e.g., 'stop', 'tool_calls').
        _last_delta: The last delta content to include in the chunk.
        _tool_calls: Accumulated tool calls.
        _include_usage: Whether to include usage statistics in final chunk.
        _is_first_chunk: Whether this is the first chunk (includes role).

    Example:
        >>> helper = PlChatCompletionHelper("gpt-4", include_usage=True)
        >>> helper.prompt_processed()
        >>> helper.update_content_step("Hello")
        >>> chunk = helper.build_yield_chunk()
        >>> helper.finish_step("stop")
        >>> final_chunk = helper.build_yield_chunk()
    """

    def __init__(self, model_name: str, include_usage: bool = False) -> None:
        """
        Initialize the chat completion helper.

        Args:
            model_name: The model name to include in responses.
            include_usage: Whether to include usage statistics in the final chunk.
        """
        self._model_name = model_name
        self._begin_time = time.time()
        self._chat_id = str(uuid.uuid4()).replace("-", "")
        self._step_helper = PlStepHelper()
        self._is_finished = False
        self._finish_reason: Optional[str] = None
        self._last_delta: dict = {}
        self._tool_calls: list = []
        self._include_usage = include_usage
        self._is_first_chunk = True

        self._step_helper.begin_process_prompt()

    def prompt_processed(self) -> None:
        """Mark that prompt processing has completed."""
        self._step_helper.end_process_prompt()

    def update_reason_step(
        self, reasoning: str, step: Optional[PlStepUsage] = None
    ) -> None:
        """
        Update with a reasoning/thinking chunk.

        The first chunk includes the 'role' field, subsequent chunks
        only include the reasoning content.

        Args:
            reasoning: The reasoning/thinking content.
            step: Optional token usage statistics.
        """
        if self._is_finished:
            return
        if self._is_first_chunk:
            self._last_delta = {"reasoning": reasoning, "role": "assistant"}
            self._is_first_chunk = False
        else:
            self._last_delta = {"reasoning": reasoning}
        if step is not None:
            self._step_helper.update_step(step)

    def update_content_step(
        self, content: str, step: Optional[PlStepUsage] = None
    ) -> None:
        """
        Update with a content chunk.

        The first chunk includes the 'role' field, subsequent chunks
        only include the content.

        Args:
            content: The text content.
            step: Optional token usage statistics.
        """
        if self._is_finished:
            return
        if self._is_first_chunk:
            self._last_delta = {"content": content, "role": "assistant"}
            self._is_first_chunk = False
        else:
            self._last_delta = {"content": content}
        if step is not None:
            self._step_helper.update_step(step)

    def update_tool_step(
        self, tool_call: dict, step: Optional[PlStepUsage] = None
    ) -> None:
        """
        Update with a tool call chunk.

        Args:
            tool_call: Dictionary containing 'name' and 'arguments' for the tool.
            step: Optional token usage statistics.
        """
        if self._is_finished:
            return
        self._last_delta = {
            "tool_calls": [
                {
                    "index": 0,
                    "id": f"toolcall-{str(uuid.uuid4()).replace('-', '')}",
                    "type": "function_call",
                    "function": tool_call,
                }
            ]
        }
        if self._is_first_chunk:
            self._last_delta["role"] = "assistant"
            self._is_first_chunk = False
        if step is not None:
            self._step_helper.update_step(step)

    def finish_step(self, finish_reason: str) -> None:
        """
        Mark the completion as finished.

        Args:
            finish_reason: Reason for completion ('stop', 'tool_calls', etc.).
        """
        if self._is_finished:
            return
        self._is_finished = True
        self._finish_reason = finish_reason
        # Only clear _last_delta if finish_reason is NOT tool_calls
        # For tool_calls, we need to keep the delta with tool_calls data
        if finish_reason != "tool_calls":
            self._last_delta = {}

    def build_chunk(self) -> dict:
        """
        Build the chunk as a dictionary.

        Returns:
            OpenAI-compatible chat completion chunk dictionary.
        """
        chunk = {
            "id": f"chatcmpl-{self._chat_id}",
            "object": "chat.completion.chunk",
            "created": int(self._begin_time),
            "model": self._model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": self._last_delta,
                    "finish_reason": self._finish_reason,
                }
            ],
        }
        # Only include usage in final chunk if include_usage is True
        if self._is_finished and self._include_usage:
            chunk["usage"] = self._step_helper.build_usage()
        return chunk

    def build_text(self) -> dict:
        """
        Build the text completion chunk (non-chat mode).

        Returns:
            OpenAI-compatible text completion chunk dictionary.
        """
        return {
            "id": f"cmpl-{self._chat_id}",
            "object": "text_completion.chunk",
            "created": int(self._begin_time),
            "model": self._model_name,
            "choices": [
                {
                    "index": 0,
                    "text": self._last_delta.get(
                        "reasoning", self._last_delta.get("content", "")
                    ),
                    "finish_reason": self._finish_reason,
                }
            ],
            "usage": self._step_helper.build_usage(),
        }

    def build_yield_chunk(self, direct_json: bool = False) -> str | dict:
        """
        Build the chunk for SSE streaming.

        Args:
            direct_json: If True, return dict; otherwise, return SSE-formatted string.

        Returns:
            Either a dictionary or SSE-formatted string with 'data: ' prefix.
        """
        if direct_json:
            return self.build_chunk()
        else:
            return f"data: {json.dumps(self.build_chunk())}\n\n"

    def build_yield_text(self, direct_json: bool = False) -> str | dict:
        """
        Build the text completion chunk for SSE streaming.

        Args:
            direct_json: If True, return dict; otherwise, return SSE-formatted string.

        Returns:
            Either a dictionary or SSE-formatted string with 'data: ' prefix.
        """
        if direct_json:
            return self.build_text()
        else:
            return f"data: {json.dumps(self.build_text())}\n\n"
