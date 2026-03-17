"""
Local model management without external dependencies.

This module provides local model management functionality.
All models are loaded in separate subprocesses via PlModelProxy.

Main process responsibilities:
- Manage model configurations
- Start/stop model subprocesses
- Provide model proxy objects for inference
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel

from plllm_mlx.helpers import get_hf_cache_dir
from plllm_mlx.logging_config import get_logger

if TYPE_CHECKING:
    from plllm_mlx.subprocess.proxy import PlModelProxy

logger = get_logger(__name__)


class PlLocalModelInfo(BaseModel):
    """
    Model information for local models.

    Attributes:
        id: Model unique identifier (e.g., "gpt-4"). If empty, use name as id.
        name: Real model name (e.g., "mlx-community/Qwen2.5-7B-Instruct-8bit").
        model_loader: Model loader type (e.g., "mlx", "mlxvlm").
        step_processor: Step processor type (e.g., "base", "qwen3think").
        config: Model configuration in JSON string format.
    """

    id: str = ""
    name: str
    model_loader: str = "mlx"
    step_processor: str = "base"
    config: str = ""

    def get_effective_id(self) -> str:
        """
        Get effective ID for the model.

        Returns:
            The id if not empty, otherwise the name.
        """
        return self.id if self.id else self.name


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

    This class manages local models. All models are loaded in separate
    subprocesses via PlModelProxy.

    Attributes:
        _model_proxies: Dictionary of model proxy instances (key: model name).
        _models_on_disk: List of models found on disk.
        _model_configs: In-memory storage for model configurations (key: model name).
        _id_to_name: Mapping from model ID to model name.
    """

    def __init__(self) -> None:
        """Initialize the local model manager."""
        self._model_proxies: Dict[str, "PlModelProxy"] = {}
        self._models_on_disk: List[str] = []
        self._model_configs: Dict[str, PlLocalModelInfo] = {}
        self._id_to_name: Dict[str, str] = {}
        self._load_local_models()

    def _load_local_models(self) -> None:
        """Load local models from HuggingFace cache."""
        from plllm_mlx.subprocess.proxy import PlModelProxy

        cache_dir = Path(get_hf_cache_dir())
        if not cache_dir.exists():
            logger.info(f"HuggingFace cache directory not found: {cache_dir}")
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
            if model_name in self._model_proxies:
                continue

            model_info = self._model_configs.get(model_name)
            if model_info is None:
                model_info = PlLocalModelInfo(name=model_name)
                self._model_configs[model_name] = model_info

            proxy = PlModelProxy(
                model_name=model_info.name,
                loader=model_info.model_loader,
                step_processor=model_info.step_processor,
            )
            proxy.set_config(_convert_config_str_to_dict(model_info.config))
            self._model_proxies[model_info.name] = proxy

            effective_id = model_info.get_effective_id()
            self._id_to_name[effective_id] = model_info.name

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
        from plllm_mlx.subprocess.python.step_processor import PlStepProcessor

        model_info = self._model_configs.get(model_name)
        if model_info is None:
            model_info = PlLocalModelInfo(name=model_name)

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

        if model_name in self._model_proxies:
            proxy = self._model_proxies[model_name]
            await proxy.update_step_processor(step_processor_name)

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
            model_info = PlLocalModelInfo(name=model_name)

        config = _convert_config_str_to_dict(model_info.config)
        config[key] = value
        model_info.config = _convert_dict_to_config_str(config)
        self._model_configs[model_name] = model_info

        if model_name in self._model_proxies:
            self._model_proxies[model_name].set_config({key: value})

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

            if model_name in self._model_proxies:
                self._model_proxies[model_name].set_config(config)

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
        from plllm_mlx.subprocess.python.loader import PlModelLoader

        model_info = self._model_configs.get(model_name)
        if model_info is None:
            model_info = PlLocalModelInfo(name=model_name)

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

        if model_name not in self._model_proxies:
            return True

        old_proxy = self._model_proxies[model_name]
        was_loaded = old_proxy.is_loaded

        if was_loaded:
            await old_proxy.unload_model()

        new_proxy = PlModelProxy(
            model_name=model_name,
            loader=model_loader_name,
            step_processor=model_info.step_processor,
        )
        new_proxy.set_config(_convert_config_str_to_dict(model_info.config))
        self._model_proxies[model_name] = new_proxy

        if was_loaded:
            await new_proxy.load_model()

        return True

    def find_model(self, model_id_or_name: str) -> Optional["PlModelProxy"]:
        """
        Find a model by id or name.

        Args:
            model_id_or_name: The model ID or real model name.

        Returns:
            The model proxy instance, or None if not found.
        """
        real_name = self._id_to_name.get(model_id_or_name)
        if real_name is not None:
            return self._model_proxies.get(real_name)
        return self._model_proxies.get(model_id_or_name)

    def add_model_id(self, model_id: str, model_name: str) -> bool:
        """
        Add a model ID mapping.

        Args:
            model_id: The model ID (e.g., "gpt-4").
            model_name: The real model name.

        Returns:
            True if successful, False if model not found.
        """
        if model_name not in self._model_proxies:
            logger.warning(
                f"Cannot add id '{model_id}': model '{model_name}' not found"
            )
            return False

        self._id_to_name[model_id] = model_name

        model_info = self._model_configs.get(model_name)
        if model_info:
            model_info.id = model_id

        logger.info(f"Added model id '{model_id}' -> '{model_name}'")
        return True

    def remove_model_id(self, model_id: str) -> bool:
        """
        Remove a model ID mapping.

        Args:
            model_id: The model ID to remove.

        Returns:
            True if successful, False if ID not found.
        """
        if model_id in self._id_to_name:
            model_name = self._id_to_name[model_id]
            del self._id_to_name[model_id]

            model_info = self._model_configs.get(model_name)
            if model_info:
                model_info.id = ""

            logger.info(f"Removed model id '{model_id}'")
            return True
        return False

    def list_model_ids(self) -> Dict[str, str]:
        """
        List all model ID mappings.

        Returns:
            Dictionary mapping model ID to model name.
        """
        return self._id_to_name.copy()

    def list_model_info(self) -> List[Dict[str, Any]]:
        """
        List information for all models.

        Returns:
            List of model information dictionaries.
        """
        result = []
        for proxy in self._model_proxies.values():
            info = {
                "model_name": proxy.model_name,
                "model_loader": proxy.loader,
                "step_processor": proxy.step_processor,
                "is_loaded": proxy.is_loaded,
                "config": proxy.get_config(),
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


_local_model_manager: Optional[PlLocalModelManager] = None


def get_local_model_manager() -> PlLocalModelManager:
    """Get the singleton local model manager instance."""
    global _local_model_manager
    if _local_model_manager is None:
        _local_model_manager = PlLocalModelManager()
    return _local_model_manager
