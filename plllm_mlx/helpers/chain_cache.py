"""
Chain-based Cache Implementation for KV Cache Management.

PlChain: Represents a message chain and its KV cache
- node_ids: List of message IDs in order
- cache_item: Official KV cache (can be upgraded after assistant response)
- temp_cache_item: Temporary cache (stores intermediate state before assistant response)

PlChainCache: LRU cache based on OrderedDict
- Uses message chain ID (MD5 of node_ids) as key
- Automatically moves accessed entries to the end (implements LRU)
- Provides search_max_chain method to find longest matching chain
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from copy import deepcopy
from typing import Any, List, Optional

from plllm_mlx.logging_config import get_logger

logger = get_logger(__name__)


class PlChain:
    """
    Message chain object representing a sequence of messages and their KV cache.

    Attributes:
        node_ids: List of message IDs in conversation order
        cache_item: Official KV cache data
        temp_cache_item: Temporary cache (for cache upgrade mechanism)
        chain_id: Unique identifier based on node_ids (MD5 hash)
    """

    def __init__(
        self,
        node_ids: List[str],
        cache_item: Optional[Any] = None,
        temp_cache_item: Optional[Any] = None,
    ):
        self._node_ids = deepcopy(node_ids)
        self._unique_id = ""
        if len(self._node_ids) > 0:
            self._unique_id = hashlib.md5(
                "_".join(node_ids).encode("utf-8")
            ).hexdigest()
        self._cache_item = None
        self._temp_cache_item = None
        if cache_item is not None:
            self._cache_item = deepcopy(cache_item)
        if temp_cache_item is not None:
            self._temp_cache_item = deepcopy(temp_cache_item)

    @property
    def chain_id(self) -> str:
        return self._unique_id

    @property
    def has_cache(self) -> bool:
        return (self._cache_item is not None) or (self._temp_cache_item is not None)

    @property
    def cache_item(self) -> Optional[Any]:
        return self._cache_item

    @property
    def temp_cache_item(self) -> Optional[Any]:
        return self._temp_cache_item

    @property
    def node_ids(self) -> List[str]:
        return self._node_ids

    @cache_item.setter
    def cache_item(self, cache_item: Optional[Any]):
        if cache_item is not None:
            self._cache_item = deepcopy(cache_item)
        else:
            self._cache_item = None

    @temp_cache_item.setter
    def temp_cache_item(self, temp_cache_item: Optional[Any]):
        if temp_cache_item is not None:
            self._temp_cache_item = deepcopy(temp_cache_item)
        else:
            self._temp_cache_item = None

    def swap_cache(self, another_chain: PlChain):
        """Swap cache items between this chain and another chain."""
        self._cache_item, another_chain._cache_item = (
            another_chain._cache_item,
            self._cache_item,
        )

    def swap_temp_cache(self, another_chain: PlChain):
        """Swap temp cache items between this chain and another chain."""
        self._temp_cache_item, another_chain._temp_cache_item = (
            another_chain._temp_cache_item,
            self._temp_cache_item,
        )

    def upgrade_cache(self, another_chain: PlChain):
        """Upgrade another chain's cache with this chain's temp cache."""
        another_chain._cache_item = self._temp_cache_item
        self._temp_cache_item = None

    def duplicate(self) -> PlChain:
        """Create a duplicate of this chain."""
        return PlChain(self._node_ids, self._cache_item, self._temp_cache_item)


class PlChainCache(OrderedDict):
    """
    LRU chain cache based on OrderedDict.

    Features:
    - Inherits OrderedDict to maintain insertion order
    - __getitem__ automatically moves entry to end (LRU)
    - __setitem__ updates order if key exists
    - search_max_chain recursively finds longest matching chain

    Usage:
    - Used as underlying storage for PlMessageBasedKVCache
    - Index PlChain objects by chain_id (MD5 of node_ids)
    """

    def __init__(self):
        super().__init__()

    def __getitem__(self, key):
        value = super().__getitem__(key)
        if value is not None:
            self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)

    def search_max_chain(self, node_ids: List[str]) -> Optional[PlChain]:
        """Search for the longest matching chain."""
        temp_chain = PlChain(node_ids)
        match_chain = self.get(temp_chain.chain_id, None)
        if match_chain is None:
            return self.search_max_chain(node_ids[:-1]) if len(node_ids) > 1 else None
        else:
            return match_chain

    def remove_oldest_cache(self):
        """Remove the oldest (least recently used) cache entry."""
        self.popitem(last=False)
