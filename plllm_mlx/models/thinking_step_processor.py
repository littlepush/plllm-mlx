"""
Thinking step processor with dynamic special token support.

This processor handles:
- Message boundary tokens (begin_tokens, end_tokens) - filtered out
- Thinking/reasoning mode (detects think_start_token and think_end_token)
- Tool calls (if special_tokens.has_tool_call())
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


class PlThinkingStepProcessor(PlStepProcessor):
    """
    Thinking step processor with dynamic special token support.

    Qwen3 models have a unique thinking mechanism:
    - The thinking process starts from the first token (no explicit think_start tag needed)
    - Thinking continues until think_end tag is encountered
    - After think_end, regular content generation begins
    """

    def __init__(self, special_tokens: Optional["SpecialTokens"] = None):
        super().__init__(special_tokens)
        self.is_in_thinking = False
        self.toolcall_buffer = []
        self.first_token_time = None
        self.stop_reason = ""
        self.is_stop_by_length = False
        self.full_content = ""

    @staticmethod
    def step_clz_name() -> str:
        return "thinking"

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

        # Handle thinking mode
        step_text_to_process = self.unprocessed_text + (
            gr.text if gr is not None else ""
        )
        self.unprocessed_text = ""

        if step_text_to_process == "":
            return None

        tokens = self.special_tokens

        if not self.is_in_thinking:
            # Check for begin token to detect start of thinking
            has_begin_token = any(
                bt in step_text_to_process for bt in tokens.begin_tokens if bt
            )
            if has_begin_token:
                # Check if think_start_token is present
                if (
                    tokens.think_start_token
                    and tokens.think_start_token in step_text_to_process
                ):
                    self.is_in_thinking = True
                    step_text_to_process = step_text_to_process[
                        step_text_to_process.find(tokens.think_start_token)
                        + len(tokens.think_start_token) :
                    ]
                else:
                    # Not enough tokens
                    self.unprocessed_text = step_text_to_process
                    return None

        if self.is_in_thinking:
            if (
                tokens.think_end_token
                and tokens.think_end_token in step_text_to_process
            ):
                text_before_end_think = step_text_to_process[
                    : step_text_to_process.find(tokens.think_end_token)
                ]
                if text_before_end_think != "":
                    # Unprocessed text include think_end_token
                    self.unprocessed_text = step_text_to_process[
                        step_text_to_process.find(tokens.think_end_token) :
                    ]
                    # Process the thinking text before think_end_token
                    step_text_to_process = text_before_end_think
                else:
                    self.is_in_thinking = False
                    step_text_to_process = step_text_to_process[
                        step_text_to_process.find(tokens.think_end_token)
                        + len(tokens.think_end_token) :
                    ]
                    if step_text_to_process == "":
                        return None

        if not self.is_in_thinking:
            # Begin of tool call
            if (
                tokens.tool_call_start_token
                and tokens.tool_call_start_token in step_text_to_process
            ):
                if not step_text_to_process.startswith(tokens.tool_call_start_token):
                    pos_tool_call = step_text_to_process.find(
                        tokens.tool_call_start_token
                    )
                    self.unprocessed_text = step_text_to_process[pos_tool_call:]
                    step_text_to_process = step_text_to_process[:pos_tool_call]
                else:
                    self.toolcall_buffer.append(tokens.tool_call_start_token)
                    step_text_to_process = step_text_to_process[
                        step_text_to_process.find(tokens.tool_call_start_token)
                        + len(tokens.tool_call_start_token) :
                    ]
                    if step_text_to_process == "":
                        return None
            # If already in tool call mode
            if len(self.toolcall_buffer) > 0:
                # End of tool call, all generation should stop
                append_end_tool_call = False
                if (
                    tokens.tool_call_end_token
                    and tokens.tool_call_end_token in step_text_to_process
                ):
                    pos_end = step_text_to_process.find(tokens.tool_call_end_token)
                    step_text_to_process = step_text_to_process[:pos_end]
                    append_end_tool_call = True
                if step_text_to_process != "":
                    self.toolcall_buffer.append(step_text_to_process)
                if append_end_tool_call:
                    self.toolcall_buffer.append(tokens.tool_call_end_token)
                    self.stop()
                return None

        sr = PlStepUsage()
        sr.prompt_tokens = gr.prompt_tokens if gr is not None else 0
        sr.completion_tokens = gr.generation_tokens if gr is not None else 0
        sr.total_tokens = sr.prompt_tokens + sr.completion_tokens
        sr.prompt_tps = gr.prompt_tps if gr is not None else 0
        sr.generation_tps = gr.generation_tps if gr is not None else 0
        if sr.generation_tps is None:
            sr.generation_tps = self.total_tokens / (
                time.time() - self.first_token_time
            )

        chunk = PlChunk(step=sr)
        chunk.data = step_text_to_process
        chunk.data_type = (
            PlChunkDataType.REASONING
            if self.is_in_thinking
            else PlChunkDataType.CONTENT
        )
        return chunk

    def tool_calls(self) -> List[PlChunk]:
        """Return tool calls from regular content mode (after thinking ends)"""
        result = []
        if len(self.toolcall_buffer) > 0:
            tool_call_chunk = PlCommonToolcallParser(self.toolcall_buffer[1:-1])
            if tool_call_chunk is not None:
                result.append(tool_call_chunk)
                self.stop_reason = "tool_calls"
            else:
                logger.warning("[StepProcessor] Failed to parse tool call from buffer")

        return result

    def finish(self) -> PlChunk:
        """Return finish chunk."""
        finish_chunk = PlChunk(finish_reason=self.stop_reason)
        return finish_chunk
