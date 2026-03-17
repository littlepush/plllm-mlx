"""
Configuration management for plllm-mlx service.

This module provides Pydantic-based configuration classes for managing
all aspects of the plllm-mlx service, including server settings, model
parameters, cache configuration, and logging.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class ServerConfig(BaseModel):
    """
    Server configuration settings.

    Attributes:
        host: Server host address.
        port: Server port number.
        log_level: Logging level (debug, info, warning, error).
    """

    host: str = Field(default="0.0.0.0", description="Server host address")
    port: int = Field(default=8000, ge=1, le=65535, description="Server port number")
    log_level: str = Field(
        default="info",
        description="Logging level",
    )

    @field_validator("log_level", mode="before")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate and normalize log level."""
        v = v.lower()
        valid_levels = {"debug", "info", "warning", "error", "critical"}
        if v not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v


class ModelConfig(BaseModel):
    """
    Model configuration settings.

    Attributes:
        name: Model name or path (HuggingFace model ID or local path).
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.
        top_p: Top-p (nucleus) sampling parameter.
        top_k: Top-k sampling parameter.
        repetition_penalty: Repetition penalty factor.
        trust_remote_code: Whether to trust remote code in model files.
        use_fast_tokenizer: Whether to use fast tokenizer.
    """

    name: str = Field(
        default="Qwen/Qwen2.5-7B-Instruct",
        description="Model name or path",
    )
    max_tokens: int = Field(
        default=4096,
        ge=1,
        description="Maximum tokens to generate",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    top_p: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Top-p (nucleus) sampling",
    )
    top_k: int = Field(
        default=50,
        ge=0,
        description="Top-k sampling",
    )
    repetition_penalty: float = Field(
        default=1.0,
        ge=1.0,
        le=2.0,
        description="Repetition penalty",
    )
    trust_remote_code: bool = Field(
        default=True,
        description="Trust remote code in model files",
    )
    use_fast_tokenizer: bool = Field(
        default=True,
        description="Use fast tokenizer",
    )


class CacheConfig(BaseModel):
    """
    KV cache configuration settings.

    Attributes:
        enable_prefix_cache: Enable prefix KV cache for multi-turn conversations.
        max_memory_ratio: Maximum memory usage ratio (0.0 - 1.0).
        min_entries: Minimum number of cache entries to keep during eviction.
    """

    enable_prefix_cache: bool = Field(
        default=True,
        description="Enable prefix KV cache",
    )
    max_memory_ratio: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Maximum memory usage ratio",
    )
    min_entries: int = Field(
        default=3,
        ge=1,
        description="Minimum cache entries to keep",
    )


class LoggingConfig(BaseModel):
    """
    Logging configuration settings.

    Attributes:
        level: Logging level (debug, info, warning, error, critical).
        format: Log message format string.
        file: Optional log file path.
    """

    level: str = Field(
        default="info",
        description="Logging level (debug, info, warning, error, critical)",
    )
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string",
    )
    file: str | None = Field(
        default=None,
        description="Optional log file path",
    )

    @field_validator("level", mode="before")
    @classmethod
    def validate_level(cls, v: str) -> str:
        """Validate and normalize log level."""
        v = v.lower()
        valid_levels = {"debug", "info", "warning", "error", "critical"}
        if v not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v


class PlConfig(BaseModel):
    """
    Main configuration class for plllm-mlx service.

    This class aggregates all configuration sections and provides
    methods for loading configuration from files and merging
    with command-line arguments.

    Attributes:
        server: Server configuration.
        model: Model configuration.
        cache: Cache configuration.
        logging: Logging configuration.
    """

    server: ServerConfig = Field(default_factory=ServerConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PlConfig":
        """
        Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            PlConfig instance with loaded settings.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            yaml.YAMLError: If the file is not valid YAML.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls.model_validate(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlConfig":
        """
        Create configuration from a dictionary.

        Args:
            data: Dictionary containing configuration values.

        Returns:
            PlConfig instance with the provided settings.
        """
        return cls.model_validate(data)

    def merge_with_overrides(
        self,
        host: str | None = None,
        port: int | None = None,
        log_level: str | None = None,
    ) -> "PlConfig":
        """
        Create a new configuration with overridden values.

        This method is useful for merging command-line arguments with
        file-based configuration, where CLI arguments take precedence.

        Args:
            host: Override server host.
            port: Override server port.
            log_level: Override log level.

        Returns:
            New PlConfig instance with merged settings.
        """
        data = self.model_dump()

        if host is not None:
            data["server"]["host"] = host
        if port is not None:
            data["server"]["port"] = port
        if log_level is not None:
            data["server"]["log_level"] = log_level

        return PlConfig.model_validate(data)
