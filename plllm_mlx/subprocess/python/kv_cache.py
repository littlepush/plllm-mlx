"""
Message-based KV Cache for MLX model inference.

This module provides a message-based prefix KV cache implementation for
efficient prompt reuse and incremental prefill during LLM inference.

Core concepts:
- Message ID (msg_id): MD5 hash of full message content (including begin/end tokens)
- Message chain: Ordered list of message IDs [msg_id_1, msg_id_2, ...]
- Cache matching: Find longest matching message chain
- Incremental prefill: Only process unmatched messages

Cache matching strategy:
- Skip first round: Don't cache when < 3 messages (new conversation, avoid false matches)
- Role order validation: 2 messages only allow (system+user), 3 messages allow specific patterns
- Chain matching: Find longest matching message chain
- Full match: Return cache when all message IDs match; for retry, use cache of first N-1 messages
- Incremental prefill: Only process unmatched messages

Cache upgrade mechanism:
- Temp cache (temp_cache_item): Stores intermediate state before assistant response
- When user sends next message, upgrade temp cache to official cache

Memory management:
- PLLLM_MEMORY_THRESHOLD (default 0.9): Trigger eviction when memory usage exceeds this threshold
- PLLLM_MEMORY_LOWBOUND_THRESHOLD (default 0.7): Target memory level after eviction
- PLLLM_CACHE_MIN_ENTRIES (default 3): Minimum cache entries to keep
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, List, Optional, Tuple

import psutil
from pydantic import BaseModel

from plllm_mlx.helpers.chain_cache import PlChain, PlChainCache
from plllm_mlx.logging_config import get_logger

logger = get_logger(__name__)


class PlKVCacheMessage(BaseModel):
    """Data model for a message in the KV cache."""

    msg_id: str
    role: str
    vision_count: int
    full_content: str


class PlMessageBasedKVCache:
    """
    Message-based Prefix KV Cache for MLX model inference.

    Uses MD5 hash of message content (including tokens) as unique identifier.

    Core concepts:
    - Message identifier (msg_id): MD5 hash of full message content (including begin/end tokens)
    - Message chain: Ordered list of message IDs [msg_id_1, msg_id_2, ...]
    - Request processing: Split prompt by message boundaries, generate msg_ids, query cache
    - Cache hit: If partial or full match, only prefill incremental portion

    Cache matching strategy:
    - Skip first round: Don't cache when < 3 messages (new conversation, avoid false matches)
    - Role order validation: 2 messages only allow (system+user), 3 messages allow specific patterns
    - Chain matching: Find longest matching message chain
    - Full match: Return cache when all message IDs match; for retry, use cache of first N-1 messages
    - Incremental prefill: Only process unmatched messages

    Cache upgrade mechanism:
    - Temp cache (temp_cache_item): Stores intermediate state before assistant response
    - When user sends next message, upgrade temp cache to official cache

    Memory management:
    - PLLLM_MEMORY_THRESHOLD (default 0.9): Trigger eviction when memory usage exceeds this threshold
    - PLLLM_MEMORY_LOWBOUND_THRESHOLD (default 0.7): Target memory level after eviction
    - PLLLM_CACHE_MIN_ENTRIES (default 3): Minimum cache entries to keep
    """

    _DEFAULT_MEMORY_THRESHOLD = 0.9
    _DEFAULT_MEMORY_LOWBOUND = 0.7
    _DEFAULT_MIN_ENTRIES = 3

    def __init__(
        self,
        begin_tokens: List[str] | None = None,
        end_tokens: List[str] | None = None,
        vision_begin_tokens: List[str] | None = None,
        vision_end_tokens: List[str] | None = None,
    ):
        """
        Initialize the Message-based KV Cache.

        Args:
            begin_tokens: List of begin tokens (e.g., ['<|start|>', '<|im_start|>'])
            end_tokens: List of end tokens (e.g., ['<|end|>', '<|im_end|>'])
            vision_begin_tokens: List of vision begin tokens for VLM models
            vision_end_tokens: List of vision end tokens for VLM models
        """
        # Default tokens if not provided
        self._begin_tokens = begin_tokens or ["<|start|>", "<|im_start|>"]
        self._end_tokens = end_tokens or ["<|end|>", "<|im_end|>"]
        self._vision_begin_tokens = vision_begin_tokens or ["<|vision_start|>"]
        self._vision_end_tokens = vision_end_tokens or ["<|vision_end|>"]

        # Memory thresholds
        self._memory_threshold = float(
            os.environ.get("PLLLM_MEMORY_THRESHOLD", self._DEFAULT_MEMORY_THRESHOLD)
        )
        self._memory_lowbound = float(
            os.environ.get(
                "PLLLM_MEMORY_LOWBOUND_THRESHOLD", self._DEFAULT_MEMORY_LOWBOUND
            )
        )
        self._min_entries = int(
            os.environ.get("PLLLM_CACHE_MIN_ENTRIES", self._DEFAULT_MIN_ENTRIES)
        )

        # Chain cache for managing cache chains
        self._chain_cache = PlChainCache()

        # Estimated memory per layer (16MB is a conservative estimate)
        self._num_layers = 40
        self._memory_per_layer = 16 * 1024 * 1024  # 16MB per layer

        logger.info(
            f"MessageBasedKVCache initialized: threshold={self._memory_threshold}, "
            f"lowbound={self._memory_lowbound}, min_entries={self._min_entries}, "
            f"begin_tokens={self._begin_tokens}, end_tokens={self._end_tokens}"
        )

    def set_num_layers(self, num_layers: int) -> None:
        """Set number of layers (for memory estimation)."""
        self._num_layers = num_layers

    @property
    def begin_tokens(self) -> List[str]:
        return self._begin_tokens

    @begin_tokens.setter
    def begin_tokens(self, tokens: List[str]) -> None:
        self._begin_tokens = tokens

    @property
    def end_tokens(self) -> List[str]:
        return self._end_tokens

    @end_tokens.setter
    def end_tokens(self, tokens: List[str]) -> None:
        self._end_tokens = tokens

    def clear(self) -> None:
        """Clear all cached data."""
        self._chain_cache.clear()
        logger.info("MessageBasedKVCache cleared")

    def _generate_msg_id(self, full_message: str) -> str:
        """Generate a unique ID for a message using the full message content (including begin/end tokens)."""
        # Handle None or empty content
        if full_message is None:
            # This happens when assistant content is null (thinking removed)
            # Use a placeholder that still includes the role
            full_message = ""

        if not isinstance(full_message, str):
            logger.warning(
                f"[PlMessageBasedKVCache] _generate_msg_id received non-string: {type(full_message)}"
            )
            full_message = str(full_message) if full_message else ""

        return hashlib.md5(full_message.encode("utf-8")).hexdigest()

    def _find_valid_token_pair(self, prompt: str) -> Tuple[str, str]:
        """
        Find the valid begin/end token pair used by the model.

        Uses startswith for begin token and endswith for end token to avoid
        false matches from content.

        Returns:
            Tuple of (begin_token, end_token) that matches the prompt
        """
        # First, find the first message's begin token using startswith
        valid_begin = None
        for begin_token in self._begin_tokens:
            if prompt.startswith(begin_token):
                valid_begin = begin_token
                break

        if not valid_begin:
            # Fallback to first token
            valid_begin = self._begin_tokens[0] if self._begin_tokens else None

        # Find if prompt ends with any end token
        valid_end = None
        for end_token in self._end_tokens:
            if prompt.rstrip().endswith(end_token):
                valid_end = end_token
                break

        if not valid_end:
            # Find the end token that appears somewhere in the prompt (more flexible)
            for end_token in self._end_tokens:
                if end_token in prompt:
                    valid_end = end_token
                    break

        if not valid_end:
            valid_end = self._end_tokens[0] if self._end_tokens else None

        return (valid_begin, valid_end)  # type: ignore

    def _find_all_substring(self, origin_str: str, sub_str: str) -> List[int]:
        """Find all occurrences of substring in string."""
        start = 0
        result = []
        while True:
            start = origin_str.find(sub_str, start)
            if start == -1:
                break
            result.append(start)
            start += len(sub_str)
        return result

    def _find_valid_vision_token_pair(
        self, prompt: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Find valid vision token pair for VLM models."""
        for begin_token in self._vision_begin_tokens:
            begin_matches = self._find_all_substring(prompt, begin_token)
            logger.debug(
                f"test vision begin token: {begin_token}, matches: {len(begin_matches)}"
            )
            if len(begin_matches) > 0:
                for end_token in self._vision_end_tokens:
                    end_matches = self._find_all_substring(prompt, end_token)
                    logger.debug(f"test vision end token: {end_token}")
                    if len(end_matches) != len(begin_matches):
                        continue
                    end_is_after_begin = True
                    for index in range(0, len(begin_matches)):
                        if not (begin_matches[index] < end_matches[index]):
                            end_is_after_begin = False
                            break
                        if index != (len(begin_matches) - 1):  # not last
                            if not (end_matches[index] < begin_matches[index + 1]):
                                end_is_after_begin = False
                                break
                    if end_is_after_begin:
                        return (begin_token, end_token)
        return (None, None)

    def _estimate_cache_memory(self, cache: Any) -> int:
        """Estimate memory usage of a KV cache in bytes."""
        if not cache:
            return 0
        try:
            offset = max(c.offset for c in cache)  # type: ignore
            return self._num_layers * 16 * 1024 * max(1, offset)
        except AttributeError:
            pass
        try:
            total_tokens = 0
            for c in cache:
                state = c.state  # type: ignore
                if state is not None and state[0] is not None:
                    shape = state[0].shape if hasattr(state[0], "shape") else None  # type: ignore
                    if shape and len(shape) >= 3:
                        total_tokens = max(total_tokens, shape[2])  # type: ignore
            return self._num_layers * 16 * 1024 * max(1, total_tokens)
        except Exception:
            pass
        return self._num_layers * self._memory_per_layer

    def split_prompt_by_messages(self, prompt: str) -> List[PlKVCacheMessage]:
        """
        Split the full prompt into individual messages.

        Format: <begin_token>role\ncontent<end_token><begin_token>role\ncontent<end_token>...
        role is the first word after begin_token (system, user, assistant, tool)

        Returns:
            List of PlKVCacheMessage objects for each message
        """
        messages = []

        # Ensure prompt is a string
        if not isinstance(prompt, str):
            prompt = str(prompt) if prompt else ""

        # Find the valid token pair
        begin_token, end_token = self._find_valid_token_pair(prompt)
        if not begin_token or not end_token:
            messages.append(
                PlKVCacheMessage(
                    msg_id=self._generate_msg_id(prompt),
                    role="user",  # Treat as user by default if we can't find tokens
                    vision_count=0,
                    full_content=prompt,
                )
            )
            return messages

        # Parse each message: <begin_token>role\ncontent<end_token>
        # Return full message strings (including begin/end tokens) for msg_id generation
        current_pos = 0

        def _parse_message(full_msg: str) -> PlKVCacheMessage:
            # Extract role and content from the full message
            # full_msg format: <begin_token>role\ncontent<end_token>
            first_line = full_msg[: full_msg.find("\n")]
            role_content = first_line[len(begin_token) :].strip()
            vbegin, vend = self._find_valid_vision_token_pair(full_msg)
            logger.debug(f"found validate vision tokens: {vbegin}, {vend}")
            vcount = 0
            if vbegin is not None and vend is not None:
                # We do have vision in prompt
                vcount = len(self._find_all_substring(full_msg, vbegin))
            logger.debug(f"found vision count: {vcount}")
            return PlKVCacheMessage(
                msg_id=self._generate_msg_id(full_msg),
                role=role_content,
                vision_count=vcount,
                full_content=full_msg,
            )

        while current_pos < len(prompt):
            # Find next begin token
            begin_idx = prompt.find(begin_token, current_pos)
            if begin_idx < 0:
                break

            # Find end token after this begin token (for this message)
            content_start = begin_idx + len(begin_token)
            end_idx = prompt.find(end_token, content_start)

            if end_idx < 0:
                # No end token found, take rest of prompt
                full_msg = prompt[begin_idx:]
                messages.append(_parse_message(full_msg))
                break

            # Extract full message including begin/end tokens
            full_msg = prompt[begin_idx : end_idx + len(end_token)]
            messages.append(_parse_message(full_msg))

            # Move to next message
            current_pos = end_idx + len(end_token)

        # If we couldn't split properly, treat the whole thing as one message
        if not messages:
            messages.append(
                PlKVCacheMessage(
                    msg_id=self._generate_msg_id(prompt),
                    role="user",  # Treat as user by default if we can't parse
                    vision_count=0,
                    full_content=prompt,
                )
            )

        return messages

    def _evict_if_needed(self, estimated_new_cache_memory: int) -> None:
        """Evict LRU entries while memory usage is high."""
        if len(self._chain_cache) <= self._min_entries:
            return

        mem = psutil.virtual_memory()
        total_memory = mem.total
        threshold_bytes = self._memory_threshold * total_memory
        lowbound_bytes = self._memory_lowbound * total_memory

        logger.debug(
            f"[PlMessageBasedKVCache] Memory: current={mem.used / 1024 / 1024:.1f}MB, "
            f"threshold={threshold_bytes / 1024 / 1024:.1f}MB ({self._memory_threshold})"
        )

        if mem.used + estimated_new_cache_memory < threshold_bytes * 0.95:
            # If we're under 95% of the threshold even after adding the new cache, we can skip eviction
            return

        while True:
            # Remove oldest cache until we're under the threshold or we have only min_entries left
            self._chain_cache.remove_oldest_cache()
            mem = psutil.virtual_memory()
            if mem.used < lowbound_bytes or len(self._chain_cache) <= self._min_entries:
                break

    def get_kv_cache(
        self,
        message_splits: List[PlKVCacheMessage],
    ) -> Optional[PlChain]:
        """
        Get KV cache based on message list.

        Processing flow:
        1. Validate message count and role order (skip non-compliant cases)
        2. Find longest matching message chain from chain_cache
        3. Handle temp cache upgrade scenario (when user sends new message)
        4. Handle full match scenario (retry request)

        Returns:
            PlChain object containing matched chain's KV cache and message chain IDs
            - Full match: Return complete cache
            - Partial match: Return cache of first N messages, prefill incremental portion
        """
        # Debug: Log cache lookup info
        msg_roles = [m.role for m in message_splits]
        logger.debug(
            f"[KVCache Debug] Lookup: {len(message_splits)} messages, roles={msg_roles}"
        )
        logger.debug(
            f"[KVCache Debug] Msg IDs: {[m.msg_id[:8] for m in message_splits]}"
        )

        if len(self._chain_cache) == 0:
            logger.info("[PlMessageBasedKVCache] No cached entries")
            return None

        # Only one message, no need to lookup cache as it's unlikely to match
        if len(message_splits) == 1:
            logger.info(
                "[PlMessageBasedKVCache] Only 1 message, skipping cache lookup to avoid false match"
            )
            return None

        if len(message_splits) == 2:
            # check the msg role order
            # only (sys + user) is allowed.
            if not (
                message_splits[0].role == "system" and message_splits[1].role == "user"
            ):
                logger.info(
                    "[PlMessageBasedKVCache] Only 2 messages but role order is not system->user, skipping cache lookup"
                )
                return None

        if len(message_splits) == 3:
            # check the msg role order
            # only allowed:
            # * (sys + assistant + user)
            # * (user + assistant + user)
            # * (sys + user + user)
            if not (
                (
                    message_splits[0].role == "system"
                    and message_splits[1].role == "assistant"
                    and message_splits[2].role == "user"
                )
                or (
                    message_splits[0].role == "user"
                    and message_splits[1].role == "assistant"
                    and message_splits[2].role == "user"
                )
                or (
                    message_splits[0].role == "system"
                    and message_splits[1].role == "user"
                    and message_splits[2].role == "user"
                )
            ):
                logger.info(
                    "[PlMessageBasedKVCache] Only 3 messages but role order is not valid, skipping cache lookup"
                )
                return None

        def _search(
            msg_ids: List[str], allow_full_match: bool = False
        ) -> Optional[PlChain]:
            if len(msg_ids) == 0:
                return None
            logger.debug(f"[KVCache Debug] Searching for chain: {len(msg_ids)} msg_ids")
            cached_chain = self._chain_cache.search_max_chain(msg_ids)
            if cached_chain is None:
                logger.debug("[KVCache Debug] No match found")
                return None
            logger.debug(
                f"[KVCache Debug] Found match: {len(cached_chain.node_ids)} msgs, "
                f"has_cache={cached_chain.cache_item is not None}, "
                f"has_temp={cached_chain.temp_cache_item is not None}"
            )
            # Check full match first (before temp cache handling)
            if len(cached_chain.node_ids) == len(msg_ids):
                # Full match scenario
                if cached_chain.cache_item is not None:
                    # Has real cache, can use directly
                    logger.info(
                        f"[PlMessageBasedKVCache] Cache HIT: full match for {len(msg_ids)} messages"
                    )
                    if allow_full_match:
                        return cached_chain
                    # full match, maybe the user start a `retry` request in history.
                    # We need at least one new message to generate new response
                    # so use the cache for the previous messages and ignore the last one.
                    return _search(msg_ids[:-1], allow_full_match=True)
                else:
                    # Only has temp_cache (retry scenario)
                    # Need to use shorter chain to regenerate
                    logger.info(
                        "[PlMessageBasedKVCache] Full match but only temp_cache, searching shorter chain for retry"
                    )
                    return _search(cached_chain.node_ids[:-1], allow_full_match=True)

            # Now cached_chain is shorter than msg_ids
            # Check if this is a temp_cache that can be upgraded
            if (
                cached_chain.cache_item is None
                and cached_chain.temp_cache_item is not None
            ):
                # only has temp cache item, so this the cache is a response temp cache
                if len(cached_chain.node_ids) + 2 == len(msg_ids):
                    # the cached_chain is id[..., last user message] + cache[..., last user message, last assistant cache]
                    # and current input message is: [..., last user message, last assistant cache, current user message]
                    # the key is two steps below the cached chain, but the cache is only one step below
                    logger.info(
                        "[PlMessageBasedKVCache] find last round cache, upgrade"
                    )
                    extend_chain = PlChain(
                        cached_chain.node_ids + [msg_ids[len(cached_chain.node_ids)]]
                    )
                    cached_chain.upgrade_cache(extend_chain)
                    del self._chain_cache[cached_chain.chain_id]
                    self._chain_cache[extend_chain.chain_id] = extend_chain
                    return extend_chain
                else:
                    # match a temp cache, which is not we want, but this matching means the shortest chain
                    # is a temp cache, so if we want to find a real cache, we need a shorter chain
                    return _search(cached_chain.node_ids[:-1], allow_full_match=True)
            # Now the cached_chain is a shorter chain than the incoming msg_ids
            # and not a temp cache, which means this is the max chain
            return cached_chain

        matched_chain = _search([m.msg_id for m in message_splits])
        if matched_chain is not None:
            logger.info(
                f"[PlMessageBasedKVCache] Cache HIT: matched {len(matched_chain.node_ids)}/{len(message_splits)} messages"
            )
            return matched_chain.duplicate()
        return None

    def add_kv_cache(
        self, chain_ids: List[str], cache: Any, is_resp_cache: bool = False
    ) -> Optional[PlChain]:
        """
        Add a new cache entry.

        Args:
            chain_ids: List of message IDs forming the chain
            cache: The KV cache data
            is_resp_cache: If True, store as temp cache (for response caching)

        Returns:
            The newly created PlChain, or None if cache is None
        """
        if cache is None:
            logger.warning("[PlMessageBasedKVCache] add_kv_cache: cache is None")
            return None

        # Debug: Log cache addition
        logger.debug(
            f"[KVCache Debug] Adding cache: {len(chain_ids)} msg_ids, is_resp_cache={is_resp_cache}"
        )

        new_cache_memory = self._estimate_cache_memory(cache)
        self._evict_if_needed(new_cache_memory)

        # Add to chain cache
        new_chain = PlChain(
            chain_ids,
            cache_item=(None if is_resp_cache else cache),
            temp_cache_item=(cache if is_resp_cache else None),
        )
        self._chain_cache[new_chain.chain_id] = new_chain

        logger.info(
            f"[PlMessageBasedKVCache] Added cache: chain_ids={chain_ids[:3]}..., "
            f"est_memory={new_cache_memory / 1024 / 1024:.1f}MB, "
            f"total_entries={len(self._chain_cache)}"
        )
        return new_chain

    def upgrade_chain(
        self, old_chain: PlChain, addition_ids: List[str]
    ) -> Optional[PlChain]:
        """
        Upgrade a chain by adding more message IDs.

        Args:
            old_chain: The existing chain to upgrade
            addition_ids: Additional message IDs to add

        Returns:
            The upgraded chain, or None if old_chain has no temp cache
        """
        if old_chain.temp_cache_item is None:
            logger.error(
                f"[PlMessageBasedKVCache] upgrade_chain: old_chain({old_chain.chain_id}) has no temp cache item"
            )
            return None
        new_chain = PlChain(old_chain.node_ids + addition_ids)
        old_chain.upgrade_cache(new_chain)
        del self._chain_cache[old_chain.chain_id]
        self._chain_cache[new_chain.chain_id] = new_chain
        return new_chain

    def get_cache_info(self) -> dict:
        """Get diagnostic info about the cache."""
        mem = psutil.virtual_memory()
        return {
            "cache_count": len(self._chain_cache),
            "total_memory_mb": mem.total / 1024 / 1024,
            "memory_usage_percent": mem.used / mem.total * 100,
            "thresholds": {
                "high": self._memory_threshold,
                "lowbound": self._memory_lowbound,
            },
            "min_entries": self._min_entries,
        }
