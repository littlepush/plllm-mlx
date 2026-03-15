"""
GPT-OSS step processor with dynamic special token support.

GPT-OSS models use a unique channel-based mechanism:
- [LT]|start|[GT] begins a message, followed by role
- [LT]|channel|[GT] starts channel name
- [LT]|end|[GT] ends the current channel
- "analysis" channel: thinking/reasoning content
- "final" channel: regular content
- "commentary to=xxx" channel: tool call
"""

from plllm_mlx.logging_config import get_logger
from plllm_mlx.helpers import (
    PlChunk,
    PlChunkDataType,
    PlStepUsage,
)
from plllm_mlx.models.step_processor import PlStepProcessor
from typing import Any, Optional, List, TYPE_CHECKING
import re
import json

if TYPE_CHECKING:
    from plllm_mlx.models.special_tokens import SpecialTokens

logger = get_logger(__name__)


class PlGptOssStepProcessor(PlStepProcessor):
    """
    GPT-OSS step processor with dynamic special token support.

    Handles the channel-based output format used by GPT-OSS models.
    """

    def __init__(self, special_tokens: Optional["SpecialTokens"] = None):
        super().__init__(special_tokens)
        self._channel_buffer: dict = {}
        self._current_channel_name: Optional[str] = None
        self._current_role: str = "assistant"
        self._current_key_in_channel: Optional[str] = None
        self._is_waiting_for_role: bool = False
        self._is_waiting_for_channel_name: bool = False
        self._is_stop_by_length: bool = False
        self._full_content: str = ""
        self._stop_reason: str = ""

        self._start_token: Optional[str] = None
        self._end_token: Optional[str] = None
        self._channel_token: Optional[str] = None
        self._init_tokens()

    def _init_tokens(self) -> None:
        """Initialize tokens from special_tokens."""
        tokens = self.special_tokens

        for t in tokens.begin_tokens:
            if "|start|" in t:
                self._start_token = t
                break

        for t in tokens.end_tokens:
            if "|end|" in t or "|return|" in t:
                self._end_token = t
                break

        self._channel_token = "[LT]|channel|[GT]"

    @staticmethod
    def step_clz_name() -> str:
        return "gptoss"

    def step(self, generate_response: Any) -> Optional[PlChunk]:
        gr = generate_response
        self.total_tokens += 1

        try:
            self._full_content += gr.text

            if hasattr(gr, "finish_reason") and gr.finish_reason is not None:
                self._is_stop_by_length = gr.finish_reason == "length"
                self.stop()
                return None

            text = gr.text

            if self._end_token and text == self._end_token:
                if self._current_channel_name == "final":
                    self.stop()
                    return None
                self._current_channel_name = None
                self._current_key_in_channel = None
                return None

            if self._start_token and text == self._start_token:
                self._is_waiting_for_role = True
                return None

            if self._is_waiting_for_role and not text.startswith("[LT]|"):
                self._current_role = text
                self._is_waiting_for_role = False
                return None

            if text == self._channel_token:
                self._is_waiting_for_channel_name = True
                self._current_channel_name = ""
                return None

            if self._is_waiting_for_channel_name:
                if text.startswith("[LT]|") and text.endswith("|[GT]"):
                    self._is_waiting_for_channel_name = False
                    self._current_channel_name = self._current_channel_name.strip()
                    self._current_key_in_channel = re.sub(
                        r"[LT]\|([^|]+)\|[GT]", r"\1", text
                    )
                    self._channel_buffer[self._current_channel_name] = {
                        "role": self._current_role,
                        "channel": self._current_channel_name,
                        self._current_key_in_channel: "",
                    }
                else:
                    self._current_channel_name += text
                return None

            strip_token = text.strip()
            if strip_token.startswith("[LT]|") and strip_token.endswith("|[GT]"):
                self._current_key_in_channel = re.sub(
                    r"[LT]\|([^|]+)\|[GT]", r"\1", text
                )
                if self._current_channel_name in self._channel_buffer:
                    self._channel_buffer[self._current_channel_name][
                        self._current_key_in_channel
                    ] = ""

            sr = PlStepUsage()
            sr.prompt_tokens = getattr(gr, "prompt_tokens", 0)
            sr.completion_tokens = getattr(gr, "generation_tokens", 0)
            sr.total_tokens = sr.prompt_tokens + sr.completion_tokens
            sr.prompt_tps = getattr(gr, "prompt_tps", 0)
            sr.generation_tps = getattr(gr, "generation_tps", 0)
            chunk = PlChunk(step=sr)

            if self._current_channel_name == "analysis":
                if self._current_key_in_channel == "message":
                    chunk.data = text
                    chunk.data_type = PlChunkDataType.REASONING
                    return chunk
                else:
                    logger.debug(
                        f"Token in analysis channel but not in message key: {self._current_channel_name}.{self._current_key_in_channel}"
                    )
                return None

            if self._current_channel_name == "final":
                if self._current_key_in_channel == "message":
                    chunk.data = text
                    chunk.data_type = PlChunkDataType.CONTENT
                    return chunk
                else:
                    logger.debug(
                        f"Token in final channel but not in message key: {self._current_channel_name}.{self._current_key_in_channel}"
                    )
                return None

            if (
                self._current_channel_name is None
                or self._current_key_in_channel is None
            ):
                return None

            self._channel_buffer[self._current_channel_name][
                self._current_key_in_channel
            ] += text
            return None

        except Exception as e:
            logger.debug(f"Step error: {str(e)}")
            logger.debug(f"full content: {self._full_content}")
            logger.debug(f"new token: {text if 'text' in dir() else 'N/A'}")
            self.stop()
            return None

    def tool_calls(self) -> List[PlChunk]:
        """Return tool calls from commentary channels."""
        self._stop_reason = "length" if self._is_stop_by_length else "stop"
        logger.debug(
            f"channel_buffer: {json.dumps(self._channel_buffer, ensure_ascii=False)}"
        )

        result = []
        for channel, data in self._channel_buffer.items():
            if channel.startswith("commentary to="):
                tool_call = {
                    "name": channel[channel.find("=") + 1 :],
                    "parameters": data.get("message", ""),
                }
                chunk = PlChunk(data=tool_call, data_type=PlChunkDataType.TOOLCALL)
                logger.debug(f"Detected tool call: {tool_call}")
                result.append(chunk)
                self._stop_reason = "tool_calls"

        return result

    def finish(self) -> PlChunk:
        """Return finish chunk."""
        logger.debug(f"full content: {self._full_content}")
        return PlChunk(finish_reason=self._stop_reason)
