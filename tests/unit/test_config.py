"""Tests for configuration module."""

from pathlib import Path

import pytest
import yaml

from plllm_mlx.config import (
    CacheConfig,
    LoggingConfig,
    ModelConfig,
    PlConfig,
    ServerConfig,
)


class TestServerConfig:
    """Tests for ServerConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ServerConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.log_level == "info"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ServerConfig(host="127.0.0.1", port=9000, log_level="debug")
        assert config.host == "127.0.0.1"
        assert config.port == 9000
        assert config.log_level == "debug"

    def test_log_level_validation_lowercase(self):
        """Test log level validation normalizes to lowercase."""
        config = ServerConfig(log_level="DEBUG")
        assert config.log_level == "debug"

    def test_log_level_validation_invalid(self):
        """Test log level validation rejects invalid values."""
        with pytest.raises(ValueError, match="Invalid log level"):
            ServerConfig(log_level="invalid")

    def test_port_validation_range(self):
        """Test port validation."""
        with pytest.raises(ValueError):
            ServerConfig(port=0)
        with pytest.raises(ValueError):
            ServerConfig(port=70000)
        config = ServerConfig(port=1)
        assert config.port == 1
        config = ServerConfig(port=65535)
        assert config.port == 65535


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ModelConfig()
        assert config.name == "Qwen/Qwen2.5-7B-Instruct"
        assert config.max_tokens == 4096
        assert config.temperature == 0.7
        assert config.top_p == 1.0
        assert config.top_k == 50
        assert config.repetition_penalty == 1.0
        assert config.trust_remote_code is True
        assert config.use_fast_tokenizer is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ModelConfig(
            name="custom/model",
            max_tokens=2048,
            temperature=0.5,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.1,
        )
        assert config.name == "custom/model"
        assert config.max_tokens == 2048
        assert config.temperature == 0.5
        assert config.top_p == 0.9
        assert config.top_k == 40
        assert config.repetition_penalty == 1.1

    def test_temperature_validation(self):
        """Test temperature validation."""
        with pytest.raises(ValueError):
            ModelConfig(temperature=-0.1)
        with pytest.raises(ValueError):
            ModelConfig(temperature=2.1)
        config = ModelConfig(temperature=0.0)
        assert config.temperature == 0.0
        config = ModelConfig(temperature=2.0)
        assert config.temperature == 2.0

    def test_top_p_validation(self):
        """Test top_p validation."""
        with pytest.raises(ValueError):
            ModelConfig(top_p=-0.1)
        with pytest.raises(ValueError):
            ModelConfig(top_p=1.1)

    def test_max_tokens_validation(self):
        """Test max_tokens validation."""
        with pytest.raises(ValueError):
            ModelConfig(max_tokens=0)
        with pytest.raises(ValueError):
            ModelConfig(max_tokens=-1)


class TestCacheConfig:
    """Tests for CacheConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CacheConfig()
        assert config.enable_prefix_cache is True
        assert config.max_memory_ratio == 0.9
        assert config.min_entries == 3

    def test_custom_values(self):
        """Test custom configuration values."""
        config = CacheConfig(
            enable_prefix_cache=False,
            max_memory_ratio=0.8,
            min_entries=5,
        )
        assert config.enable_prefix_cache is False
        assert config.max_memory_ratio == 0.8
        assert config.min_entries == 5

    def test_max_memory_ratio_validation(self):
        """Test max_memory_ratio validation."""
        with pytest.raises(ValueError):
            CacheConfig(max_memory_ratio=-0.1)
        with pytest.raises(ValueError):
            CacheConfig(max_memory_ratio=1.1)

    def test_min_entries_validation(self):
        """Test min_entries validation."""
        with pytest.raises(ValueError):
            CacheConfig(min_entries=0)


class TestLoggingConfig:
    """Tests for LoggingConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LoggingConfig()
        assert config.level == "info"
        assert config.file is None

    def test_custom_values(self):
        """Test custom configuration values."""
        config = LoggingConfig(level="debug", file="/var/log/app.log")
        assert config.level == "debug"
        assert config.file == "/var/log/app.log"

    def test_level_validation(self):
        """Test level validation."""
        with pytest.raises(ValueError):
            LoggingConfig(level="invalid")


class TestPlConfig:
    """Tests for PlConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PlConfig()
        assert isinstance(config.server, ServerConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.cache, CacheConfig)
        assert isinstance(config.logging, LoggingConfig)

    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "server": {"host": "127.0.0.1", "port": 9000},
            "model": {"name": "test/model", "temperature": 0.5},
        }
        config = PlConfig.from_dict(data)
        assert config.server.host == "127.0.0.1"
        assert config.server.port == 9000
        assert config.model.name == "test/model"
        assert config.model.temperature == 0.5

    def test_from_yaml(self, tmp_path: Path):
        """Test loading config from YAML file."""
        config_data = {
            "server": {"host": "127.0.0.1", "port": 9000, "log_level": "debug"},
            "model": {"name": "test/model", "max_tokens": 2048},
            "cache": {"enable_prefix_cache": False},
            "logging": {"level": "warning"},
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = PlConfig.from_yaml(config_file)
        assert config.server.host == "127.0.0.1"
        assert config.server.port == 9000
        assert config.server.log_level == "debug"
        assert config.model.name == "test/model"
        assert config.model.max_tokens == 2048
        assert config.cache.enable_prefix_cache is False
        assert config.logging.level == "warning"

    def test_from_yaml_file_not_found(self):
        """Test loading config from non-existent file."""
        with pytest.raises(FileNotFoundError):
            PlConfig.from_yaml("/nonexistent/path/config.yaml")

    def test_from_yaml_empty_file(self, tmp_path: Path):
        """Test loading config from empty YAML file."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")

        config = PlConfig.from_yaml(config_file)
        assert isinstance(config, PlConfig)
        assert config.server.port == 8000

    def test_merge_with_overrides(self):
        """Test merging configuration with overrides."""
        config = PlConfig()
        merged = config.merge_with_overrides(
            host="192.168.1.1",
            port=8080,
            log_level="warning",
        )
        assert merged.server.host == "192.168.1.1"
        assert merged.server.port == 8080
        assert merged.server.log_level == "warning"

    def test_merge_with_overrides_partial(self):
        """Test merging configuration with partial overrides."""
        config = PlConfig()
        config.server.port = 9000
        merged = config.merge_with_overrides(log_level="debug")
        assert merged.server.port == 9000
        assert merged.server.log_level == "debug"

    def test_merge_with_overrides_none(self):
        """Test merging configuration with None overrides."""
        config = PlConfig()
        config.server.port = 9000
        merged = config.merge_with_overrides()
        assert merged.server.port == 9000

    def test_model_dump(self):
        """Test serializing config to dictionary."""
        config = PlConfig(
            server=ServerConfig(port=9000),
            model=ModelConfig(name="test/model"),
        )
        data = config.model_dump()
        assert data["server"]["port"] == 9000
        assert data["model"]["name"] == "test/model"
