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
    reasoning content enclosed in special tokens before producing the actual response.
    """


    Qwen3 models have a unique thinking mechanism:
    - The thinking process starts from the first token (no explicit <think> tag needed)
    - Thinking continues until </think> tag is encountered
    - After </think>, regular content generation begins
    """

    def __init__(self):
        super().__init__()
        self.is_in_thinking = False  # Test for `add_generation_prompt=False` in loader, if False, start with thinking mode
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

        # Handle thinking mode
        step_text_to_process = self.unprocessed_text + (
            gr.text if gr is not None else ""
        )
        self.unprocessed_text = ""

        if step_text_to_process == "":
            return None

        if not self.is_in_thinking:
            if "<|im_start|>" in step_text_to_process:
                # Remove the start thinking tag and everything before it
                if "<think>" in step_text_to_process:
                    self.is_in_thinking = True
                    step_text_to_process = step_text_to_process[
                        step_text_to_process.find("<think>") + len("<think>") :
                    ]
                else:
                    # Not enough tokens
                    self.unprocessed_text = step_text_to_process
                    return None

        if self.is_in_thinking:
            if "</think>" in step_text_to_process:
                text_before_end_think = step_text_to_process[
                    : step_text_to_process.find("</think>")
                ]
                if text_before_end_think != "":
                    # Unprocessed text include </think>
                    self.unprocessed_text = step_text_to_process[
                        step_text_to_process.find("</think>") :
                    ]
                    # Process the thinking text before </think>
                    step_text_to_process = text_before_end_think
                else:
                    self.is_in_thinking = False
                    step_text_to_process = step_text_to_process[
                        step_text_to_process.find("</think>") + len("</think>") :
                    ]
                    if step_text_to_process == "":
                        return None

        if not self.is_in_thinking:
            # Begin of tool call
            if "<tool_call>" in step_text_to_process:
                if not step_text_to_process.startswith("<tool_call>"):
                    pos_tool_call = step_text_to_process.find("<tool_call>")
                    self.unprocessed_text = step_text_to_process[pos_tool_call:]
                    step_text_to_process = step_text_to_process[:pos_tool_call]
                else:
                    self.toolcall_buffer.append("<tool_call>")
                    step_text_to_process = step_text_to_process[
                        step_text_to_process.find("<tool_call>") + len("<tool_call>") :
                    ]
                    if step_text_to_process == "":
                        return None
            # If already in tool call mode
            if len(self.toolcall_buffer) > 0:
                # End of tool call, all generation should stop
                append_end_tool_call = False
                if "</tool_call>" in step_text_to_process:
                    pos_end = step_text_to_process.find("</tool_call>")
                    # self.unprocessed_text = step_text_to_process[pos_end + len("</tool_call>"):].lstrip()
                    step_text_to_process = step_text_to_process[:pos_end]
                    append_end_tool_call = True
                if step_text_to_process != "":
                    self.toolcall_buffer.append(step_text_to_process)
                if append_end_tool_call:
                    self.toolcall_buffer.append("</tool_call>")
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
            if not tool_call_chunk is None:
                result.append(tool_call_chunk)
                self.stop_reason = "tool_calls"
            else:
                logger.warning(f"[StepProcessor] Failed to parse tool call from buffer")

        return result

    def finish(self) -> PlChunk:
        # logger.debug(f"full content: {self.full_content}")
        finish_chunk = PlChunk(finish_reason=self.stop_reason)
        return finish_chunk

# Register the step processor
PlStepProcessor.registerStepProcessor("qwen3think", Qwen3ThinkingStepProcessor)

