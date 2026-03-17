"""Tests for local models module."""

import json

from plllm_mlx.models.local_models import (
    PlLocalModelInfo,
    PlLocalModelManager,
    _convert_config_str_to_dict,
    _convert_dict_to_config_str,
)


class TestPlLocalModelInfo:
    """Tests for PlLocalModelInfo model."""

    def test_default_values(self):
        """Test default values."""
        info = PlLocalModelInfo(name="test/model")
        assert info.name == "test/model"
        assert info.id == ""
        assert info.model_loader == "mlx"
        assert info.step_processor == "base"
        assert info.config == ""

    def test_custom_values(self):
        """Test custom values."""
        info = PlLocalModelInfo(
            name="test/model",
            id="gpt-4",
            model_loader="mlxvlm",
            step_processor="thinking",
            config='{"temperature": 0.7}',
        )
        assert info.name == "test/model"
        assert info.id == "gpt-4"
        assert info.model_loader == "mlxvlm"
        assert info.step_processor == "thinking"
        assert info.config == '{"temperature": 0.7}'

    def test_get_effective_id_with_id(self):
        """Test get_effective_id when id is set."""
        info = PlLocalModelInfo(name="test/model", id="gpt-4")
        assert info.get_effective_id() == "gpt-4"

    def test_get_effective_id_without_id(self):
        """Test get_effective_id when id is empty."""
        info = PlLocalModelInfo(name="test/model")
        assert info.get_effective_id() == "test/model"

    def test_model_dump(self):
        """Test model serialization."""
        info = PlLocalModelInfo(name="test/model", model_loader="mlx")
        data = info.model_dump()
        assert data["name"] == "test/model"
        assert data["model_loader"] == "mlx"


class TestConfigConversion:
    """Tests for config conversion functions."""

    def test_convert_config_str_to_dict_valid(self):
        """Test converting valid JSON string to dict."""
        result = _convert_config_str_to_dict('{"key": "value"}')
        assert result == {"key": "value"}

    def test_convert_config_str_to_dict_invalid(self):
        """Test converting invalid JSON string."""
        result = _convert_config_str_to_dict("not json")
        assert result == {}

    def test_convert_config_str_to_dict_empty(self):
        """Test converting empty string."""
        result = _convert_config_str_to_dict("")
        assert result == {}

    def test_convert_config_str_to_dict_not_dict(self):
        """Test converting JSON that's not a dict."""
        result = _convert_config_str_to_dict('["a", "b"]')
        assert result == {}

    def test_convert_dict_to_config_str_valid(self):
        """Test converting dict to JSON string."""
        result = _convert_dict_to_config_str({"key": "value"})
        assert json.loads(result) == {"key": "value"}

    def test_convert_dict_to_config_str_empty(self):
        """Test converting empty dict."""
        result = _convert_dict_to_config_str({})
        assert result == "{}"

    def test_roundtrip(self):
        """Test roundtrip conversion."""
        original = {"temperature": 0.7, "max_tokens": 2048}
        config_str = _convert_dict_to_config_str(original)
        result = _convert_config_str_to_dict(config_str)
        assert result == original


class TestPlLocalModelManager:
    """Tests for PlLocalModelManager class."""

    def test_init_empty_cache(self, tmp_path, monkeypatch):
        """Test initialization with empty cache."""
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
        manager = PlLocalModelManager()
        assert len(manager._model_proxies) == 0
        assert len(manager._models_on_disk) == 0

    def test_list_models_on_disk_empty(self, tmp_path, monkeypatch):
        """Test list_models_on_disk with empty cache."""
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
        manager = PlLocalModelManager()
        result = manager.list_models_on_disk()
        assert result == []

    def test_list_model_info_empty(self, tmp_path, monkeypatch):
        """Test list_model_info with empty cache."""
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
        manager = PlLocalModelManager()
        result = manager.list_model_info()
        assert result == []

    def test_find_model_not_found(self, tmp_path, monkeypatch):
        """Test find_model when model not found."""
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
        manager = PlLocalModelManager()
        result = manager.find_model("nonexistent/model")
        assert result is None

    def test_add_model_id(self, tmp_path, monkeypatch):
        """Test add_model_id."""
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
        manager = PlLocalModelManager()
        manager._model_proxies["test/model"] = mock_proxy()
        result = manager.add_model_id("gpt-4", "test/model")
        assert result is True
        assert manager._id_to_name["gpt-4"] == "test/model"

    def test_add_model_id_model_not_found(self, tmp_path, monkeypatch):
        """Test add_model_id when model not found."""
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
        manager = PlLocalModelManager()
        result = manager.add_model_id("gpt-4", "nonexistent/model")
        assert result is False

    def test_remove_model_id(self, tmp_path, monkeypatch):
        """Test remove_model_id."""
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
        manager = PlLocalModelManager()
        manager._model_proxies["test/model"] = mock_proxy()
        manager.add_model_id("gpt-4", "test/model")
        result = manager.remove_model_id("gpt-4")
        assert result is True
        assert "gpt-4" not in manager._id_to_name

    def test_remove_model_id_not_found(self, tmp_path, monkeypatch):
        """Test remove_model_id when ID not found."""
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
        manager = PlLocalModelManager()
        result = manager.remove_model_id("nonexistent")
        assert result is False

    def test_list_model_ids(self, tmp_path, monkeypatch):
        """Test list_model_ids."""
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
        manager = PlLocalModelManager()
        manager._model_proxies["test/model1"] = mock_proxy()
        manager._model_proxies["test/model2"] = mock_proxy()
        manager.add_model_id("gpt-4", "test/model1")
        manager.add_model_id("gpt-3.5", "test/model2")
        result = manager.list_model_ids()
        assert result["gpt-4"] == "test/model1"
        assert result["gpt-3.5"] == "test/model2"


def mock_proxy():
    """Create a mock model proxy."""
    from unittest.mock import MagicMock

    proxy = MagicMock()
    proxy.model_name = "test/model"
    proxy.loader = "mlx"
    proxy.step_processor = "base"
    proxy.is_loaded = False
    proxy.get_config.return_value = {}
    return proxy
