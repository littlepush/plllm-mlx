"""
Auto-detection of model special tokens.

This module provides utilities for automatically detecting special tokens
from a tokenizer's added_tokens_decoder, including:
- Begin/End tokens for message boundaries (used by KVCache)
- Think tokens for reasoning models (e.g., Qwen3)
- Tool call tokens for function calling
- Vision tokens for VLM models
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from plllm_mlx.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SpecialTokens:
    """
    Container for model special tokens.

    Attributes:
        begin_tokens: Tokens that mark the start of a message (e.g., '<|im_start|>')
        end_tokens: Tokens that mark the end of a message (e.g., '<|im_end|>')
        think_start_token: Token that starts thinking/reasoning block (e.g., '思索开始')
        think_end_token: Token that ends thinking/reasoning block (e.g., '思索结束')
        tool_call_start_token: Token that starts a tool call block
        tool_call_end_token: Token that ends a tool call block
        vision_start_token: Token that marks vision content start (for VLM)
        vision_end_token: Token that marks vision content end (for VLM)
        channel_token: Token for channel switching (GPT-OSS specific, e.g., '<|channel|>')
        message_token: Token for message key (GPT-OSS specific, e.g., '<|message|>')
    """

    begin_tokens: List[str] = field(default_factory=list)
    end_tokens: List[str] = field(default_factory=list)
    think_start_token: Optional[str] = None
    think_end_token: Optional[str] = None
    tool_call_start_token: Optional[str] = None
    tool_call_end_token: Optional[str] = None
    vision_start_token: Optional[str] = None
    vision_end_token: Optional[str] = None
    channel_token: Optional[str] = None
    message_token: Optional[str] = None

    def has_thinking(self) -> bool:
        """Check if model supports thinking/reasoning mode."""
        return self.think_start_token is not None and self.think_end_token is not None

    def has_tool_call(self) -> bool:
        """Check if model supports tool calling."""
        return self.tool_call_start_token is not None

    def has_vision(self) -> bool:
        """Check if model is a vision-language model."""
        return self.vision_start_token is not None


_DEFAULT_BEGIN_TOKENS = ["<|start|>", "<|im_start|>"]
_DEFAULT_END_TOKENS = ["<|end|>", "<|im_end|>"]
_DEFAULT_VISION_BEGIN_TOKENS = ["<|vision_start|>"]
_DEFAULT_VISION_END_TOKENS = ["<|vision_end|>"]


def detect_special_tokens(tokenizer: Any) -> SpecialTokens:
    """
    Auto-detect special tokens from tokenizer's added_tokens_decoder.

    This function examines the tokenizer's added_tokens_decoder to identify
    special tokens for message boundaries, thinking, tool calls, and vision.

    Args:
        tokenizer: A tokenizer object with added_tokens_decoder attribute
                   (e.g., MLX TokenizerWrapper or HuggingFace tokenizer)

    Returns:
        SpecialTokens object with detected tokens

    Example:
        >>> from mlx_lm import load
        >>> model, tokenizer, config = load('Qwen/Qwen3-8B', return_config=True)
        >>> tokens = detect_special_tokens(tokenizer)
        >>> print(tokens.begin_tokens)  # ['<|im_start|>']
        >>> print(tokens.has_thinking())  # True
    """
    tokens = SpecialTokens()

    if not hasattr(tokenizer, "added_tokens_decoder"):
        logger.warning(
            "[SpecialTokens] Tokenizer has no added_tokens_decoder, using defaults"
        )
        tokens.begin_tokens = _DEFAULT_BEGIN_TOKENS.copy()
        tokens.end_tokens = _DEFAULT_END_TOKENS.copy()
        tokens.vision_start_token = _DEFAULT_VISION_BEGIN_TOKENS[0]
        tokens.vision_end_token = _DEFAULT_VISION_END_TOKENS[0]
        return tokens

    added_tokens = tokenizer.added_tokens_decoder

    for token_id, token_info in added_tokens.items():
        content = (
            str(token_info.content)
            if hasattr(token_info, "content")
            else str(token_info)
        )

        _detect_begin_token(tokens, content)
        _detect_end_token(tokens, content)
        _detect_think_token(tokens, content)
        _detect_tool_call_token(tokens, content)
        _detect_vision_token(tokens, content)
        _detect_channel_token(tokens, content)
        _detect_message_token(tokens, content)

    if not tokens.begin_tokens:
        tokens.begin_tokens = _DEFAULT_BEGIN_TOKENS.copy()
        logger.debug(
            f"[SpecialTokens] No begin_tokens found, using defaults: {tokens.begin_tokens}"
        )

    if not tokens.end_tokens:
        tokens.end_tokens = _DEFAULT_END_TOKENS.copy()
        logger.debug(
            f"[SpecialTokens] No end_tokens found, using defaults: {tokens.end_tokens}"
        )

    _log_detected_tokens(tokens)

    return tokens


def _detect_begin_token(tokens: SpecialTokens, content: str) -> None:
    """Detect begin/message start tokens."""
    if any(x in content for x in ["im_start", "|start|"]):
        tokens.begin_tokens.insert(0, content)
    elif content.lower().endswith("start>"):
        tokens.begin_tokens.append(content)


def _detect_end_token(tokens: SpecialTokens, content: str) -> None:
    """Detect end/message end tokens."""
    if any(x in content for x in ["im_end", "|end|", "|return|"]):
        tokens.end_tokens.insert(0, content)
    elif content.lower().endswith("end>"):
        tokens.end_tokens.append(content)


def _detect_think_token(tokens: SpecialTokens, content: str) -> None:
    """Detect thinking/reasoning tokens."""
    content_lower = content.lower()

    if content == "思索开始":
        tokens.think_start_token = content
    elif content == "思索结束":
        tokens.think_end_token = content
    elif content_lower == "think" or (
        content_lower.endswith("think>") and "/" not in content_lower
    ):
        tokens.think_start_token = content
    elif content_lower == "/think" or content_lower.endswith("/think>"):
        tokens.think_end_token = content


def _detect_tool_call_token(tokens: SpecialTokens, content: str) -> None:
    """Detect tool call tokens."""
    content_lower = content.lower()

    if "tool_call" in content_lower:
        if "/" in content_lower:
            tokens.tool_call_end_token = content
        else:
            tokens.tool_call_start_token = content


def _detect_vision_token(tokens: SpecialTokens, content: str) -> None:
    """Detect vision tokens for VLM models."""
    content_lower = content.lower()

    if "vision_start" in content_lower or "vision_start>" in content_lower:
        tokens.vision_start_token = content
    elif "vision_end" in content_lower or "vision_end>" in content_lower:
        tokens.vision_end_token = content


def _detect_channel_token(tokens: SpecialTokens, content: str) -> None:
    """Detect channel token for GPT-OSS models."""
    content_lower = content.lower()

    if content_lower == "|channel|" or content_lower == "<|channel|>":
        tokens.channel_token = content


def _detect_message_token(tokens: SpecialTokens, content: str) -> None:
    """Detect message token for GPT-OSS models."""
    content_lower = content.lower()

    if content_lower == "|message|" or content_lower == "<|message|>":
        tokens.message_token = content


def _log_detected_tokens(tokens: SpecialTokens) -> None:
    """Log detected special tokens."""
    parts = [
        f"begin={tokens.begin_tokens}",
        f"end={tokens.end_tokens}",
    ]

    if tokens.think_start_token:
        parts.append(f"think={tokens.think_start_token}/{tokens.think_end_token}")

    if tokens.tool_call_start_token:
        parts.append(
            f"tool_call={tokens.tool_call_start_token}/{tokens.tool_call_end_token}"
        )

    if tokens.vision_start_token:
        parts.append(f"vision={tokens.vision_start_token}/{tokens.vision_end_token}")

    if tokens.channel_token:
        parts.append(f"channel={tokens.channel_token}")

    if tokens.message_token:
        parts.append(f"message={tokens.message_token}")

    logger.info(f"[SpecialTokens] Detected: {', '.join(parts)}")
