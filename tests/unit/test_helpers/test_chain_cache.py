"""Tests for chain cache module."""


from plllm_mlx.helpers import PlChain, PlChainCache


class TestPlChain:
    """Tests for PlChain class."""

    def test_init_empty(self):
        """Test initialization with empty node_ids."""
        chain = PlChain([])
        assert chain.node_ids == []
        assert chain.chain_id == ""
        assert chain.cache_item is None
        assert chain.temp_cache_item is None
        assert chain.has_cache is False

    def test_init_with_node_ids(self):
        """Test initialization with node_ids."""
        chain = PlChain(["node1", "node2", "node3"])
        assert len(chain.node_ids) == 3
        assert chain.chain_id != ""

    def test_chain_id_consistency(self):
        """Test that chain_id is consistent for same node_ids."""
        chain1 = PlChain(["a", "b", "c"])
        chain2 = PlChain(["a", "b", "c"])
        assert chain1.chain_id == chain2.chain_id

    def test_chain_id_uniqueness(self):
        """Test that chain_id is unique for different node_ids."""
        chain1 = PlChain(["a", "b", "c"])
        chain2 = PlChain(["a", "b", "d"])
        assert chain1.chain_id != chain2.chain_id

    def test_init_with_cache_item(self):
        """Test initialization with cache_item."""
        cache_data = {"key": "value"}
        chain = PlChain(["node1"], cache_item=cache_data)
        assert chain.cache_item == cache_data
        assert chain.has_cache is True

    def test_init_with_temp_cache_item(self):
        """Test initialization with temp_cache_item."""
        cache_data = {"temp": "data"}
        chain = PlChain(["node1"], temp_cache_item=cache_data)
        assert chain.temp_cache_item == cache_data
        assert chain.has_cache is True

    def test_cache_item_setter(self):
        """Test cache_item setter."""
        chain = PlChain(["node1"])
        chain.cache_item = {"new": "data"}
        assert chain.cache_item == {"new": "data"}

    def test_cache_item_setter_none(self):
        """Test cache_item setter with None."""
        chain = PlChain(["node1"], cache_item={"old": "data"})
        chain.cache_item = None
        assert chain.cache_item is None

    def test_temp_cache_item_setter(self):
        """Test temp_cache_item setter."""
        chain = PlChain(["node1"])
        chain.temp_cache_item = {"temp": "data"}
        assert chain.temp_cache_item == {"temp": "data"}

    def test_swap_cache(self):
        """Test swap_cache method."""
        chain1 = PlChain(["a"], cache_item={"chain1": "data"})
        chain2 = PlChain(["b"], cache_item={"chain2": "data"})
        chain1.swap_cache(chain2)
        assert chain1.cache_item == {"chain2": "data"}
        assert chain2.cache_item == {"chain1": "data"}

    def test_swap_temp_cache(self):
        """Test swap_temp_cache method."""
        chain1 = PlChain(["a"], temp_cache_item={"temp1": "data"})
        chain2 = PlChain(["b"], temp_cache_item={"temp2": "data"})
        chain1.swap_temp_cache(chain2)
        assert chain1.temp_cache_item == {"temp2": "data"}
        assert chain2.temp_cache_item == {"temp1": "data"}

    def test_upgrade_cache(self):
        """Test upgrade_cache method."""
        chain1 = PlChain(["a"], temp_cache_item={"upgraded": "data"})
        chain2 = PlChain(["b"])
        chain1.upgrade_cache(chain2)
        assert chain2.cache_item == {"upgraded": "data"}
        assert chain1.temp_cache_item is None

    def test_duplicate(self):
        """Test duplicate method."""
        original = PlChain(["a", "b"], cache_item={"key": "value"})
        duplicate = original.duplicate()
        assert duplicate.node_ids == original.node_ids
        assert duplicate.chain_id == original.chain_id
        assert duplicate.cache_item == original.cache_item
        assert duplicate is not original

    def test_node_ids_immutability(self):
        """Test that modifying original node_ids doesn't affect chain."""
        node_ids = ["a", "b", "c"]
        chain = PlChain(node_ids)
        node_ids.append("d")
        assert len(chain.node_ids) == 3


class TestPlChainCache:
    """Tests for PlChainCache class."""

    def test_init(self):
        """Test initialization."""
        cache = PlChainCache()
        assert len(cache) == 0

    def test_setitem_getitem(self):
        """Test __setitem__ and __getitem__."""
        cache = PlChainCache()
        chain = PlChain(["a", "b"])
        cache[chain.chain_id] = chain
        assert cache[chain.chain_id] == chain

    def test_getitem_moves_to_end(self):
        """Test that __getitem__ moves item to end (LRU)."""
        cache = PlChainCache()
        chain1 = PlChain(["a"])
        chain2 = PlChain(["b"])
        chain3 = PlChain(["c"])
        cache[chain1.chain_id] = chain1
        cache[chain2.chain_id] = chain2
        cache[chain3.chain_id] = chain3
        _ = cache[chain1.chain_id]
        keys = list(cache.keys())
        assert keys[-1] == chain1.chain_id

    def test_setitem_updates_order(self):
        """Test that __setitem__ updates order for existing key."""
        cache = PlChainCache()
        chain1 = PlChain(["a"])
        chain2 = PlChain(["b"])
        cache[chain1.chain_id] = chain1
        cache[chain2.chain_id] = chain2
        cache[chain1.chain_id] = chain1
        keys = list(cache.keys())
        assert keys[-1] == chain1.chain_id

    def test_search_max_chain_exact_match(self):
        """Test search_max_chain with exact match."""
        cache = PlChainCache()
        chain = PlChain(["a", "b", "c"], cache_item={"data": "value"})
        cache[chain.chain_id] = chain
        result = cache.search_max_chain(["a", "b", "c"])
        assert result == chain

    def test_search_max_chain_partial_match(self):
        """Test search_max_chain with partial match."""
        cache = PlChainCache()
        chain = PlChain(["a", "b"], cache_item={"data": "value"})
        cache[chain.chain_id] = chain
        result = cache.search_max_chain(["a", "b", "c"])
        assert result == chain

    def test_search_max_chain_no_match(self):
        """Test search_max_chain with no match."""
        cache = PlChainCache()
        result = cache.search_max_chain(["a", "b", "c"])
        assert result is None

    def test_search_max_chain_single_element(self):
        """Test search_max_chain with single element."""
        cache = PlChainCache()
        result = cache.search_max_chain(["a"])
        assert result is None

    def test_remove_oldest_cache(self):
        """Test remove_oldest_cache method."""
        cache = PlChainCache()
        chain1 = PlChain(["a"])
        chain2 = PlChain(["b"])
        cache[chain1.chain_id] = chain1
        cache[chain2.chain_id] = chain2
        cache.remove_oldest_cache()
        assert chain1.chain_id not in cache
        assert chain2.chain_id in cache

    def test_multiple_operations(self):
        """Test multiple cache operations."""
        cache = PlChainCache()
        chain1 = PlChain(["a", "b"])
        chain2 = PlChain(["c", "d"])
        chain3 = PlChain(["a", "b", "c"])
        cache[chain1.chain_id] = chain1
        cache[chain2.chain_id] = chain2
        assert len(cache) == 2
        result = cache.search_max_chain(["a", "b", "c", "d"])
        assert result == chain1
        cache[chain3.chain_id] = chain3
        assert len(cache) == 3
