"""
Subprocess manager for the main process.

This module manages model subprocesses - discovery, startup, monitoring,
and lifecycle management.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from plllm_mlx.logging_config import get_logger

from .client import PlSubprocessHandle

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class PlSubprocessManager:
    """
    Subprocess manager - runs in the main process.

    This class manages model subprocesses:
    - Discovery: Find existing subprocesses via socket files
    - Startup: Start new subprocesses via external command
    - Monitoring: Health check polling (1 second interval)
    - Lifecycle: Cleanup and restart failed subprocesses
    """

    _instance: Optional["PlSubprocessManager"] = None

    def __new__(cls) -> "PlSubprocessManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._subprocess_dir = Path.home() / ".plllm-mlx" / "subprocess"
            cls._instance._subprocesses: Dict[str, PlSubprocessHandle] = {}
            cls._instance._health_check_interval = 1.0
            cls._instance._health_check_task: Optional[asyncio.Task] = None
            cls._instance._started = False
        return cls._instance

    @classmethod
    def get_instance(cls) -> "PlSubprocessManager":
        """Get the singleton instance."""
        return cls()

    @property
    def subprocess_dir(self) -> Path:
        return self._subprocess_dir

    def socket_path(self, model_name: str) -> Path:
        """
        Get the socket path for a model.

        Args:
            model_name: Model name

        Returns:
            Socket path
        """
        model_hash = hashlib.md5(model_name.encode()).hexdigest()[:8]
        return self._subprocess_dir / f"{model_hash}.sock"

    async def start_health_check_loop(self) -> None:
        """Start the health check polling loop."""
        if self._health_check_task is not None:
            return

        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Health check loop started")

    async def stop_health_check_loop(self) -> None:
        """Stop the health check loop."""
        if self._health_check_task is not None:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
            logger.info("Health check loop stopped")

    async def _health_check_loop(self) -> None:
        """Health check polling loop - runs every second."""
        while True:
            try:
                await asyncio.sleep(self._health_check_interval)
                await self._check_all_subprocesses()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")

    async def _check_all_subprocesses(self) -> None:
        """Check health of all managed subprocesses."""
        for model_name, handle in list(self._subprocesses.items()):
            try:
                healthy = await handle.health_check(timeout=0.5)
                if not healthy:
                    logger.warning(f"Subprocess unhealthy: {model_name}")
                    # Try to restart
                    await self._restart_subprocess(model_name)
            except Exception as e:
                logger.error(f"Health check failed for {model_name}: {e}")
                await self._handle_dead_subprocess(model_name)

    async def _restart_subprocess(self, model_name: str) -> None:
        """Restart a failed subprocess."""
        handle = self._subprocesses.get(model_name)
        if handle is None:
            return

        # Clean up old handle
        await handle.cleanup()
        del self._subprocesses[model_name]

        # Remove zombie socket
        socket_path = self.socket_path(model_name)
        if socket_path.exists():
            socket_path.unlink(missing_ok=True)

        logger.info(f"Subprocess {model_name} marked for restart")

    async def _handle_dead_subprocess(self, model_name: str) -> None:
        """Handle a dead subprocess."""
        handle = self._subprocesses.pop(model_name, None)
        if handle:
            await handle.cleanup()

        # Clean up zombie socket
        socket_path = self.socket_path(model_name)
        if socket_path.exists():
            socket_path.unlink(missing_ok=True)

        logger.info(f"Cleaned up dead subprocess: {model_name}")

    async def discover(self) -> List[str]:
        """
        Discover all running subprocesses.

        Returns:
            List of model names for running subprocesses
        """
        if not self._subprocess_dir.exists():
            return []

        discovered = []
        for sock_file in self._subprocess_dir.glob("*.sock"):
            try:
                handle = PlSubprocessHandle(sock_file)
                if await handle.connect():
                    status = await handle.status()
                    if status:
                        model_name = status.get("model_name", "")
                        if model_name:
                            self._subprocesses[model_name] = handle
                            discovered.append(model_name)
                            continue
                await handle.cleanup()
            except Exception as e:
                logger.debug(f"Failed to discover {sock_file}: {e}")

        return discovered

    async def get_or_create(
        self,
        model_name: str,
        loader: str = "mlx",
        step_processor: str = "base",
        config: Optional[Dict[str, Any]] = None,
    ) -> PlSubprocessHandle:
        """
        Get an existing subprocess or create a new one.

        Args:
            model_name: Model name
            loader: Loader type
            step_processor: Step processor type
            config: Model configuration

        Returns:
            Subprocess handle
        """
        # Check if we already have this model
        if model_name in self._subprocesses:
            handle = self._subprocesses[model_name]
            if await handle.health_check():
                return handle
            # Unhealthy, clean up
            await self._handle_dead_subprocess(model_name)

        socket_path = self.socket_path(model_name)

        # Try to connect to existing subprocess
        if socket_path.exists():
            handle = await self._try_connect(socket_path)
            if handle:
                self._subprocesses[model_name] = handle
                return handle
            # Zombie socket, clean up
            socket_path.unlink(missing_ok=True)

        # Start new subprocess
        handle = await self._start_subprocess(model_name, socket_path)
        if handle:
            # Load the model
            success = await handle.load_model(
                model_name, loader, step_processor, config
            )
            if not success:
                logger.error(f"Failed to load model {model_name} in subprocess")
                await handle.cleanup()
                raise RuntimeError(f"Failed to load model {model_name}")
            self._subprocesses[model_name] = handle
            return handle

        raise RuntimeError(f"Failed to create subprocess for {model_name}")

    async def _try_connect(self, socket_path: Path) -> Optional[PlSubprocessHandle]:
        """Try to connect to an existing subprocess."""
        handle = PlSubprocessHandle(socket_path)
        try:
            if await handle.connect():
                return handle
        except Exception:
            pass
        await handle.cleanup()
        return None

    async def _start_subprocess(
        self, model_name: str, socket_path: Path
    ) -> Optional[PlSubprocessHandle]:
        """
        Start a new subprocess.

        Uses the same Python interpreter as the main process to run
        the subprocess entry point (subprocess/python/main.py).

        Args:
            model_name: Model name
            socket_path: Socket path

        Returns:
            Subprocess handle or None if failed
        """
        import sys

        from plllm_mlx.helpers import PlRootPath

        # Ensure directory exists
        socket_path.parent.mkdir(parents=True, exist_ok=True)

        # Find the Python subprocess entry point
        subprocess_main = Path(PlRootPath()) / "subprocess" / "python" / "main.py"
        if not subprocess_main.exists():
            logger.error(f"Subprocess entry point not found: {subprocess_main}")
            return None

        # Build command: use the same Python interpreter
        cmd = [
            sys.executable,
            str(subprocess_main),
            "--socket",
            str(socket_path),
        ]

        logger.info(f"Starting subprocess: {' '.join(cmd)}")

        try:
            # Start subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Wait for socket to be created
            await self._wait_for_socket(socket_path, timeout=30.0)

            # Create handle and connect
            handle = PlSubprocessHandle(socket_path, process)
            if await handle.connect():
                return handle

            # Failed to connect
            process.terminate()
            await process.wait()
            return None

        except Exception as e:
            logger.error(f"Failed to start subprocess: {e}")
            return None

    async def _wait_for_socket(self, socket_path: Path, timeout: float = 30.0) -> None:
        """
        Wait for socket file to be created.

        Args:
            socket_path: Socket path
            timeout: Timeout in seconds

        Raises:
            TimeoutError: If socket not created within timeout
        """
        start = time.time()
        while time.time() - start < timeout:
            if socket_path.exists():
                # Wait a bit for the server to bind
                await asyncio.sleep(0.1)
                return
            await asyncio.sleep(0.1)
        raise TimeoutError(f"Socket not created: {socket_path}")

    async def shutdown(self, model_name: str) -> bool:
        """
        Shutdown a subprocess.

        Args:
            model_name: Model name

        Returns:
            True if successful
        """
        handle = self._subprocesses.pop(model_name, None)
        if handle:
            await handle.cleanup()
            logger.info(f"Subprocess shutdown: {model_name}")
            return True
        return False

    async def shutdown_all(self) -> None:
        """Shutdown all subprocesses."""
        for model_name in list(self._subprocesses.keys()):
            await self.shutdown(model_name)
        await self.stop_health_check_loop()

    def list_subprocesses(self) -> List[str]:
        """List all managed subprocess model names."""
        return list(self._subprocesses.keys())


def get_subprocess_manager() -> PlSubprocessManager:
    """Get the singleton subprocess manager."""
    return PlSubprocessManager.get_instance()
