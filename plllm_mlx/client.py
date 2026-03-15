"""
HTTP client for plllm-mlx service.

This module provides a client to interact with the plllm-mlx service API.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

import httpx

from plllm_mlx.daemon import get_service_port, is_service_running


class PlClient:
    """Client for plllm-mlx HTTP API."""

    def __init__(self, url: Optional[str] = None):
        """
        Initialize client.

        Args:
            url: Service URL. If None, will auto-discover.
        """
        if url is None:
            url = self._discover_service_url()
        self.base_url = url
        self.client = httpx.Client(timeout=30.0)

    def _discover_service_url(self) -> str:
        """Auto-discover service URL."""
        if not is_service_running():
            print("Error: Service not running")
            print("Start with: plllm-mlx serve")
            sys.exit(1)

        port = get_service_port()
        if port is None:
            port = 8000

        return f"http://localhost:{port}"

    def _check_service(self) -> bool:
        """Check if service is running."""
        try:
            response = self.client.get(f"{self.base_url}/health", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    def health_check(self) -> Dict[str, Any]:
        """Check service health."""
        response = self.client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def list_models(self, loaded_only: bool = False) -> List[Dict[str, Any]]:
        """
        List models.

        Args:
            loaded_only: If True, only return loaded models.

        Returns:
            List of model information.
        """
        response = self.client.get(f"{self.base_url}/v1/model/list")
        response.raise_for_status()
        models = response.json().get("data", [])

        if loaded_only:
            models = [m for m in models if m.get("is_loaded")]

        return models

    def search_models(self, keyword: str = "", limit: int = 50) -> List[Dict[str, Any]]:
        """
        Search models on HuggingFace.

        Args:
            keyword: Search keyword.
            limit: Maximum results.

        Returns:
            List of search results.
        """
        response = self.client.get(
            f"{self.base_url}/v1/model/search", params={"keyword": keyword}
        )
        response.raise_for_status()
        return response.json().get("data", [])[:limit]

    def load_model(
        self,
        model_name: str,
        loader: Optional[str] = None,
        step_processor: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load a model.

        Args:
            model_name: Model name.
            loader: Model loader type (mlx/mlxvlm). If None, auto-detect.
            step_processor: Step processor type (base/qwen3think/openai). If None, auto-detect.

        Returns:
            Response data.
        """
        payload: Dict[str, Any] = {"model_name": model_name}
        if loader is not None:
            payload["loader"] = loader
        if step_processor is not None:
            payload["step_processor"] = step_processor

        response = self.client.post(f"{self.base_url}/v1/model/load", json=payload)
        response.raise_for_status()
        return response.json()

    def unload_model(self, model_name: str) -> Dict[str, Any]:
        """
        Unload a model.

        Args:
            model_name: Model name.

        Returns:
            Response data.
        """
        response = self.client.post(
            f"{self.base_url}/v1/model/unload", json={"model_name": model_name}
        )
        response.raise_for_status()
        return response.json()

    def download_model(
        self,
        model_id: str,
        loader: Optional[str] = None,
        step_processor: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Download a model.

        Args:
            model_id: HuggingFace model ID.
            loader: Model loader type (mlx/mlxvlm). If None, auto-detect.
            step_processor: Step processor type (base/qwen3think/openai). If None, auto-detect.

        Returns:
            Response with task_id.
        """
        payload: Dict[str, Any] = {"model_id": model_id}
        if loader is not None:
            payload["model_loader"] = loader
        if step_processor is not None:
            payload["step_processor"] = step_processor

        response = self.client.post(
            f"{self.base_url}/v1/model/download",
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    def get_download_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get download task status.

        Args:
            task_id: Task ID.

        Returns:
            Task status.
        """
        response = self.client.get(
            f"{self.base_url}/v1/model/download/status/{task_id}"
        )
        response.raise_for_status()
        return response.json()

    def delete_model(self, model_name: str) -> Dict[str, Any]:
        """
        Delete a model.

        Args:
            model_name: Model name.

        Returns:
            Response data.
        """
        response = self.client.post(
            f"{self.base_url}/v1/model/delete", json={"model_name": model_name}
        )
        response.raise_for_status()
        return response.json()

    def update_config(self, model_name: str, key: str, value: Any) -> Dict[str, Any]:
        """
        Update model config.

        Args:
            model_name: Model name.
            key: Config key.
            value: Config value.

        Returns:
            Response data.
        """
        response = self.client.post(
            f"{self.base_url}/v1/model/update/config",
            json={"model_name": model_name, "key": key, "value": value},
        )
        response.raise_for_status()
        return response.json()

    def close(self):
        """Close client."""
        self.client.close()
