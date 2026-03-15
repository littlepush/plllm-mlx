"""
Base step processor with dynamic special token support.

This processor automatically handles:
- Thinking/reasoning mode (if special_tokens.has_thinking())
- Tool calls (if special_tokens.has_tool_call())
- Regular content generation
"""

from plllm_mlx.logging_config import get_logger
from plllm_mlx.helpers import (
    PlChunk,
    PlChunkDataType,
    PlStepUsage,
    PlCommonToolcallParser,
    PlMlxGetFinishReason,
)
from plllm_mlx.models.step_processor import PlStepProcessor
from typing import Any, Optional, List, TYPE_CHECKING
import time

if TYPE_CHECKING:
    from plllm_mlx.models.special_tokens import SpecialTokens

logger = get_logger(__name__)


class PlBaseStepProcessor(PlStepProcessor):
    """
    Base step processor with dynamic special token support.

    Supports:
    - Thinking/reasoning mode (when special_tokens.has_thinking() is True)
    - Tool call processing (when special_tokens.has_tool_call() is True)
    - Regular content generation
    """

    def __init__(self, special_tokens: Optional["SpecialTokens"] = None):
        super().__init__(special_tokens)
        self._is_in_thinking = False
        self._toolcall_buffer: List[str] = []
        self._first_token_time: Optional[float] = None
        self._stop_reason: str = ""
        self._is_stop_by_length: bool = False
        self._full_content: str = ""
        self._in_tool_call: bool = False

    @staticmethod
    def step_clz_name() -> str:
        return "base"

    def step(self, generate_response: Optional[Any] = None) -> Optional[PlChunk]:
        gr = generate_response
        self.total_tokens += 1
        if gr is not None:
            self._full_content += gr.text

        if self._first_token_time is None:
            self._first_token_time = time.time()

        if (gr is not None) and (PlMlxGetFinishReason(gr) is not None):
            self._is_stop_by_length = PlMlxGetFinishReason(gr) == "length"
            self._stop_reason = PlMlxGetFinishReason(gr) or "stop"
            self.stop()
            return None

        step_text = self.unprocessed_text + (gr.text if gr is not None else "")
        self.unprocessed_text = ""

        if not step_text:
            return None

        tokens = self.special_tokens

        if tokens.has_thinking():
            result = self._handle_thinking(step_text, tokens)
            if result is not None:
                return result
            step_text = self.unprocessed_text
            self.unprocessed_text = ""
            if not step_text:
                return None

        if tokens.has_tool_call():
            result = self._handle_tool_call(step_text, tokens)
            if result is not None:
                return result
            if self._in_tool_call:
                return None

        return self._build_chunk(step_text, gr, PlChunkDataType.CONTENT)

    def _handle_thinking(self, text: str, tokens: "SpecialTokens") -> Optional[PlChunk]:
        """Handle thinking mode transitions. Returns chunk if ready, None if buffering."""
        think_start = tokens.think_start_token
        think_end = tokens.think_end_token

        if not self._is_in_thinking:
            if think_start and think_start in text:
                idx = text.find(think_start)
                before = text[:idx]
                self._is_in_thinking = True
                after = text[idx + len(think_start) :]

                if before:
                    chunk = self._build_chunk(before, None, PlChunkDataType.CONTENT)
                    self.unprocessed_text = after
                    return chunk

                if after:
                    if think_end and think_end in after:
                        end_idx = after.find(think_end)
                        thinking_content = after[:end_idx]
                        self._is_in_thinking = False
                        self.unprocessed_text = after[end_idx + len(think_end) :]
                        if thinking_content:
                            return self._build_chunk(
                                thinking_content, None, PlChunkDataType.REASONING
                            )
                    else:
                        self.unprocessed_text = after
                        if after:
                            return self._build_chunk(
                                after, None, PlChunkDataType.REASONING
                            )
                return None

            self.unprocessed_text = text
            return None

        else:
            if think_end and think_end in text:
                idx = text.find(think_end)
                thinking_content = text[:idx]
                self._is_in_thinking = False
                self.unprocessed_text = text[idx + len(think_end) :]

                if thinking_content:
                    return self._build_chunk(
                        thinking_content, None, PlChunkDataType.REASONING
                    )
                return None
            else:
                return self._build_chunk(text, None, PlChunkDataType.REASONING)

    def _handle_tool_call(
        self, text: str, tokens: "SpecialTokens"
    ) -> Optional[PlChunk]:
        """Handle tool call mode. Returns content chunk if any, None if in tool call."""
        tc_start = tokens.tool_call_start_token
        tc_end = tokens.tool_call_end_token

        if not self._in_tool_call:
            if tc_start and tc_start in text:
                idx = text.find(tc_start)
                before = text[:idx]
                self._in_tool_call = True
                after = text[idx + len(tc_start) :]

                self._toolcall_buffer = [tc_start]

                if before:
                    self.unprocessed_text = after
                    return self._build_chunk(before, None, PlChunkDataType.CONTENT)

                if after:
                    if tc_end and tc_end in after:
                        end_idx = after.find(tc_end)
                        self._toolcall_buffer.append(after[:end_idx])
                        self._toolcall_buffer.append(tc_end)
                        self._in_tool_call = False
                        self.stop()
                    else:
                        self._toolcall_buffer.append(after)
                return None

            return None

        else:
            if tc_end and tc_end in text:
                idx = text.find(tc_end)
                self._toolcall_buffer.append(text[:idx])
                self._toolcall_buffer.append(tc_end)
                self._in_tool_call = False
                self.stop()
            else:
                self._toolcall_buffer.append(text)
            return None

    def _build_chunk(self, text: str, gr: Any, data_type: PlChunkDataType) -> PlChunk:
        """Build a chunk with step usage info."""
        sr = PlStepUsage()
        if gr is not None:
            sr.prompt_tokens = gr.prompt_tokens
            sr.completion_tokens = gr.generation_tokens
            sr.total_tokens = sr.prompt_tokens + sr.completion_tokens
            sr.prompt_tps = gr.prompt_tps
            sr.generation_tps = gr.generation_tps

        if sr.generation_tps is None:
            if self.total_tokens == 1:
                sr.generation_tps = 1.0
            elif self._first_token_time:
                sr.generation_tps = self.total_tokens / (
                    time.time() - self._first_token_time
                )

        chunk = PlChunk(step=sr)
        chunk.data = text
        chunk.data_type = data_type
        return chunk

    def tool_calls(self) -> List[PlChunk]:
        """Return tool calls if any."""
        result = []
        if len(self._toolcall_buffer) > 0:
            content = (
                self._toolcall_buffer[1:-1] if len(self._toolcall_buffer) > 2 else []
            )
            tool_call_chunk = PlCommonToolcallParser(content)
            if tool_call_chunk is not None:
                result.append(tool_call_chunk)
                self._stop_reason = "tool_calls"
            else:
                logger.warning(
                    "[BaseStepProcessor] Failed to parse tool call from buffer"
                )
        return result

    def finish(self) -> PlChunk:
        """Return finish chunk."""
        logger.debug(f"full content: {self._full_content}")
        return PlChunk(finish_reason=self._stop_reason)
