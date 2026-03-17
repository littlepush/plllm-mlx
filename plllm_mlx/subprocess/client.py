"""
Subprocess handle - HTTP over UDS client.

This module provides the client-side interface for communicating with
model subprocesses via Unix Domain Sockets.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Optional

import httpx

from plllm_mlx.logging_config import get_logger

if TYPE_CHECKING:
    import asyncio.subprocess

logger = get_logger(__name__)


class PlSubprocessHandle:
    """
    Subprocess handle - HTTP over UDS client.

    This class manages the connection to a model subprocess and provides
    methods for health checks, model loading, and inference.
    """

    def __init__(
        self,
        socket_path: Path,
        process: Optional["asyncio.subprocess.Process"] = None,
    ):
        """
        Initialize the subprocess handle.

        Args:
            socket_path: Path to the Unix domain socket
            process: Optional subprocess process object (if started by main process)
        """
        self._socket_path = socket_path
        self._process = process
        self._client: Optional[httpx.AsyncClient] = None
        self._model_name: str = ""

    @property
    def socket_path(self) -> Path:
        return self._socket_path

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_connected(self) -> bool:
        return self._client is not None

    async def connect(self, timeout: float = 5.0) -> bool:
        """
        Connect to the subprocess.

        Args:
            timeout: Connection timeout in seconds

        Returns:
            True if connected successfully, False otherwise
        """
        if self._client is not None:
            return True

        if not self._socket_path.exists():
            logger.warning(f"Socket not found: {self._socket_path}")
            return False

        try:
            self._client = httpx.AsyncClient(
                transport=httpx.AsyncHTTPTransport(uds=str(self._socket_path)),
                base_url="http://localhost",
                timeout=httpx.Timeout(timeout, connect=timeout),
            )

            # Verify connection with health check
            healthy = await self.health_check(timeout=timeout)
            if not healthy:
                await self._close_client()
                return False

            logger.debug(f"Connected to subprocess: {self._socket_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to subprocess: {e}")
            await self._close_client()
            return False

    async def _close_client(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None

    async def health_check(self, timeout: float = 1.0) -> bool:
        """
        Check if the subprocess is healthy.

        Args:
            timeout: Request timeout in seconds

        Returns:
            True if healthy, False otherwise
        """
        if self._client is None:
            return False

        try:
            resp = await self._client.get("/health", timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("status") == "healthy"
        except Exception:
            pass
        return False

    async def status(self) -> Optional[Dict[str, Any]]:
        """
        Get subprocess status.

        Returns:
            Status dictionary or None if failed
        """
        if self._client is None:
            return None

        try:
            resp = await self._client.get("/status")
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.error(f"Failed to get status: {e}")
        return None

    async def load_model(
        self,
        model_name: str,
        loader: str = "mlx",
        step_processor: str = "base",
        config: Optional[Dict[str, Any]] = None,
        timeout: float = 300.0,
    ) -> bool:
        """
        Load a model in the subprocess.

        Args:
            model_name: Model name or path
            loader: Loader type (mlx, mlxvlm)
            step_processor: Step processor type
            config: Model configuration
            timeout: Load timeout in seconds

        Returns:
            True if loaded successfully
        """
        if self._client is None:
            raise RuntimeError("Not connected to subprocess")

        try:
            resp = await self._client.post(
                "/load",
                json={
                    "model_name": model_name,
                    "loader": loader,
                    "step_processor": step_processor,
                    "config": config or {},
                },
                timeout=timeout,
            )

            if resp.status_code == 200:
                data = resp.json()
                if data.get("success"):
                    self._model_name = model_name
                    logger.info(f"Model loaded: {model_name}")
                    return True
                else:
                    logger.error(f"Failed to load model: {data.get('error')}")
            else:
                logger.error(f"Load request failed: {resp.status_code}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

        return False

    async def unload_model(self, timeout: float = 60.0) -> bool:
        """
        Unload the model.

        Args:
            timeout: Unload timeout in seconds

        Returns:
            True if unloaded successfully
        """
        if self._client is None:
            return True

        try:
            resp = await self._client.post("/unload", timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("success"):
                    self._model_name = ""
                    logger.info("Model unloaded")
                    return True
        except Exception as e:
            logger.error(f"Failed to unload model: {e}")

        return False

    async def get_config(self) -> Optional[Dict[str, Any]]:
        """
        Get current model configuration.

        Returns:
            Configuration dictionary or None
        """
        if self._client is None:
            return None

        try:
            resp = await self._client.get("/config")
            if resp.status_code == 200:
                return resp.json().get("config", {})
        except Exception as e:
            logger.error(f"Failed to get config: {e}")
        return None

    async def update_config(self, config: Dict[str, Any]) -> bool:
        """
        Update model configuration.

        Args:
            config: Configuration updates

        Returns:
            True if successful
        """
        if self._client is None:
            return False

        try:
            resp = await self._client.put("/config", json=config)
            if resp.status_code == 200:
                return resp.json().get("success", False)
        except Exception as e:
            logger.error(f"Failed to update config: {e}")
        return False

    async def infer(
        self,
        messages: list,
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Run inference.

        Args:
            messages: Chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling
            **kwargs: Additional parameters

        Yields:
            SSE data lines
        """
        if self._client is None:
            raise RuntimeError("Not connected to subprocess")

        body = {
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if top_k is not None:
            body["top_k"] = top_k
        # Filter out non-serializable kwargs (like cancel_event)
        for key, value in kwargs.items():
            if key in ("cancel_event", "include_usage"):
                continue
            if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                body[key] = value

        async with self._client.stream("POST", "/infer", json=body) as resp:
            async for line in resp.aiter_lines():
                if line:
                    # Ensure proper SSE format with \n\n suffix
                    yield line + "\n\n"

    async def cleanup(self) -> None:
        """
        Clean up resources and terminate subprocess if owned.
        """
        await self._close_client()

        # If we own the process, terminate it
        if self._process is not None and self._process.returncode is None:
            logger.info(f"Terminating subprocess: {self._socket_path}")
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()

        # Clean up socket file if it exists
        if self._socket_path.exists():
            try:
                self._socket_path.unlink()
            except Exception:
                pass

    @staticmethod
    def socket_path_for_model(model_name: str) -> Path:
        """
        Get the socket path for a model name.

        Args:
            model_name: Model name

        Returns:
            Socket path
        """
        model_hash = hashlib.md5(model_name.encode()).hexdigest()[:8]
        return Path.home() / ".plllm-mlx" / "subprocess" / f"{model_hash}.sock"
