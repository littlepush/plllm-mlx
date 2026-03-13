"""
Local model management without external dependencies.

This module provides local model management functionality without Redis
or other external storage. Model configurations are stored in memory only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel

from plllm_mlx.helpers import get_hostname
from plllm_mlx.logging_config import get_logger

if TYPE_CHECKING:
    from .base_step_processor import PlStepProcessor
    from .model_loader import PlModelLoader

logger = get_logger(__name__)


class PlLocalModelInfo(BaseModel):
    """Model information for local models."""

    model_name: str
    model_loader: str = "mlx"
    step_processor: str = "base"
    config: str = ""


def _convert_config_str_to_dict(config_str: str) -> dict:
    """Convert config string to dictionary."""
    try:
        config = json.loads(config_str)
        if isinstance(config, dict):
            return config
    except json.JSONDecodeError:
        pass
    return {}


def _convert_dict_to_config_str(config: dict) -> str:
    """Convert dictionary to config string."""
    try:
        return json.dumps(config)
    except (TypeError, ValueError):
        pass
    return ""


class PlLocalModelManager:
    """
    Manager for local models.

    This class manages local models without external storage (no Redis).
    Model configurations are stored in memory only.

    Attributes:
        _models_in_memory: Dictionary of loaded model instances.
        _models_on_disk: List of models found on disk.
        _model_configs: In-memory storage for model configurations.
    """

    def __init__(self) -> None:
        """Initialize the local model manager."""
        self._models_in_memory: Dict[str, "PlModelLoader"] = {}
        self._models_on_disk: List[str] = []
        self._model_configs: Dict[str, PlLocalModelInfo] = {}
        # Load local models on initialization
        self._load_local_models()

    def _load_local_models(self) -> None:
        """Load local models from HuggingFace cache."""
        hg_path = os.environ.get(
            "HUGGING_FACE_PATH", f"{Path.home()}/.cache/huggingface/hub"
        )
        cache_dir = Path(hg_path)
        if not cache_dir.exists():
            logger.info(f"HuggingFace cache directory not found: {hg_path}")
            return

        models_folders = [p.name for p in cache_dir.iterdir() if p.is_dir()]
        self._models_on_disk = []

        for folder in models_folders:
            parts = folder.split("--")
            if len(parts) >= 3 and parts[0] == "models":
                model_name = f"{parts[1]}/{parts[2]}"
                self._models_on_disk.append(model_name)

        logger.info(f"Found {len(self._models_on_disk)} models on disk")

        for model_name in self._models_on_disk:
            if model_name in self._models_in_memory:
                continue

            # Get or create model info
            model_info = self._model_configs.get(model_name)
            if model_info is None:
                model_info = PlLocalModelInfo(model_name=model_name)
                self._model_configs[model_name] = model_info

            # Create the model but don't load into memory
            from .model_loader import PlModelLoader

            local_model = PlModelLoader.createModel(
                model_info.model_loader, model_name, model_info.step_processor
            )
            if local_model is None:
                logger.error(f"Failed to create local model loader for: {model_name}")
                continue

            local_model.set_config(_convert_config_str_to_dict(model_info.config))
            self._models_in_memory[model_name] = local_model

    def reload_local_models(self) -> None:
        """Reload local models from disk."""
        self._load_local_models()

    async def update_step_processor(
        self, model_name: str, step_processor_name: str
    ) -> bool:
        """
        Update the step processor for a model.

        Args:
            model_name: The model name.
            step_processor_name: The new step processor name.

        Returns:
            True if successful, False otherwise.
        """
        from .base_step_processor import PlStepProcessor

        model_info = self._model_configs.get(model_name)
        if model_info is None:
            model_info = PlLocalModelInfo(model_name=model_name)

        # No need to update if same
        if model_info.step_processor == step_processor_name:
            return True

        step_clz = PlStepProcessor.findStepProcessor(step_processor_name)
        if step_clz is None:
            logger.error(
                f"Failed to update step processor, no such processor: {step_processor_name}"
            )
            return False

        model_info.step_processor = step_processor_name
        self._model_configs[model_name] = model_info

        # Update in memory
        if model_name in self._models_in_memory:
            model = self._models_in_memory[model_name]
            model.update_step_processor(step_processor_name)

        return True

    async def update_model_config(self, model_name: str, key: str, value: Any) -> bool:
        """
        Update a model configuration value.

        Args:
            model_name: The model name.
            key: The configuration key.
            value: The configuration value.

        Returns:
            True if successful.
        """
        model_info = self._model_configs.get(model_name)
        if model_info is None:
            model_info = PlLocalModelInfo(model_name=model_name)

        config = _convert_config_str_to_dict(model_info.config)
        config[key] = value
        model_info.config = _convert_dict_to_config_str(config)
        self._model_configs[model_name] = model_info

        if model_name in self._models_in_memory:
            model = self._models_in_memory[model_name]
            model.set_config({key: value})

        return True

    async def delete_model_config(self, model_name: str, key: str) -> bool:
        """
        Delete a model configuration key.

        Args:
            model_name: The model name.
            key: The configuration key to delete.

        Returns:
            True if successful.
        """
        model_info = self._model_configs.get(model_name)
        if model_info is None:
            return True

        config = _convert_config_str_to_dict(model_info.config)
        if key in config:
            del config[key]
            model_info.config = _convert_dict_to_config_str(config)
            self._model_configs[model_name] = model_info

            if model_name in self._models_in_memory:
                model = self._models_in_memory[model_name]
                model.set_config(config)

        return True

    async def update_model_loader(
        self, model_name: str, model_loader_name: str
    ) -> bool:
        """
        Update the model loader for a model.

        Args:
            model_name: The model name.
            model_loader_name: The new loader name.

        Returns:
            True if successful, False otherwise.
        """
        from .model_loader import PlModelLoader

        model_info = self._model_configs.get(model_name)
        if model_info is None:
            model_info = PlLocalModelInfo(model_name=model_name)

        if model_info.model_loader == model_loader_name:
            return True

        loader_list = PlModelLoader.listModelLoaders()
        if model_loader_name not in loader_list:
            logger.error(
                f"Failed to update model loader, no such loader: {model_loader_name}"
            )
            return False

        model_info.model_loader = model_loader_name
        self._model_configs[model_name] = model_info

        if model_name not in self._models_in_memory:
            return True

        model_is_loaded = self._models_in_memory[model_name].is_loaded
        if model_is_loaded:
            await self._models_in_memory[model_name].unload_model()

        local_model = PlModelLoader.createModel(
            model_loader_name, model_name, model_info.step_processor
        )
        if local_model is None:
            logger.error(f"Failed to create model with loader: {model_loader_name}")
            return False

        self._models_in_memory[model_name] = local_model

        if model_is_loaded:
            await local_model.load_model()
            local_model.set_config(_convert_config_str_to_dict(model_info.config))

        return True

    def find_model(self, model_name: str) -> Optional["PlModelLoader"]:
        """
        Find a model by name.

        Args:
            model_name: The model name.

        Returns:
            The model loader instance, or None if not found.
        """
        return self._models_in_memory.get(model_name)

    def list_model_info(self) -> List[Dict[str, Any]]:
        """
        List information for all models.

        Returns:
            List of model information dictionaries.
        """
        result = []
        for model in self._models_in_memory.values():
            info = {
                "model_name": model.model_name,
                "model_loader": type(model).model_loader_name(),
                "step_processor": model.step_processor_clz.step_clz_name(),
                "config": model.get_config(),
            }
            result.append(info)
        return result

    def list_models_on_disk(self) -> List[str]:
        """
        List all models found on disk.

        Returns:
            List of model names.
        """
        return self._models_on_disk.copy()


# Singleton instance
_local_model_manager: Optional[PlLocalModelManager] = None


def get_local_model_manager() -> PlLocalModelManager:
    """Get the singleton local model manager instance."""
    global _local_model_manager
    if _local_model_manager is None:
        _local_model_manager = PlLocalModelManager()
    return _local_model_manager
