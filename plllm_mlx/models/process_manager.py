"""
Process manager for model subprocess isolation.

This module provides process isolation for LLM inference, allowing models to run
in separate processes to avoid blocking the main event loop and to isolate
memory usage.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import uuid
from multiprocessing import Process, Queue
from typing import TYPE_CHECKING, Dict, Optional

from plllm_mlx.logging_config import get_logger

if TYPE_CHECKING:
    from .base_step_processor import PlStepProcessor

logger = get_logger(__name__)


class PlModelSubprocess:
    """
    Model subprocess that runs in a separate process.

    Each model runs in its own subprocess to avoid blocking the main event loop
    and to isolate memory usage.

    Attributes:
        _model_id: The model identifier.
        _loader_name: The name of the model loader.
        _step_processor_name: The name of the step processor.
        _model_config: Model configuration dictionary.
        _process: The subprocess handle.
        _request_queue: Queue for sending requests to subprocess.
        _response_queue: Queue for receiving responses from subprocess.
        _is_running: Whether the subprocess is running.
        _current_forward_task: Current forward task being processed.
    """

    def __init__(
        self,
        model_id: str,
        loader_name: str,
        step_processor_name: str = "base",
        model_config: dict | None = None,
    ) -> None:
        """
        Initialize the model subprocess.

        Args:
            model_id: The model identifier.
            loader_name: The name of the model loader.
            step_processor_name: The name of the step processor.
            model_config: Model configuration dictionary.
        """
        self._model_id = model_id
        self._loader_name = loader_name
        self._step_processor_name = step_processor_name
        self._model_config = model_config or {}
        self._process: Optional[Process] = None
        self._request_queue: Optional[Queue] = None
        self._response_queue: Optional[Queue] = None
        self._is_running = False
        self._current_forward_task: Optional[asyncio.Task] = None

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def loader_name(self) -> str:
        return self._loader_name

    @property
    def step_processor_name(self) -> str:
        return self._step_processor_name

    @property
    def is_running(self) -> bool:
        return (
            self._is_running and self._process is not None and self._process.is_alive()
        )

    def start(self) -> None:
        """Start the subprocess."""
        if self._is_running:
            logger.warning(f"Subprocess {self._model_id} already running")
            return

        self._request_queue = mp.Queue()
        self._response_queue = mp.Queue()

        self._process = Process(
            target=_run_subprocess_entry,
            args=(
                self._model_id,
                self._loader_name,
                self._step_processor_name,
                self._model_config,
                self._request_queue,
                self._response_queue,
            ),
            daemon=True,
        )
        self._process.start()
        self._is_running = True
        logger.info(
            f"Started subprocess for model {self._model_id} (pid: {self._process.pid}), "
            f"step_processor: {self._step_processor_name}"
        )

    def stop(self) -> None:
        """Stop the subprocess."""
        if not self._is_running:
            return

        # Send stop signal
        if self._request_queue is not None:
            try:
                self._request_queue.put(None)
            except Exception as e:
                logger.warning(f"Failed to send stop signal: {e}")

        # Wait for process to finish
        if self._process is not None and self._process.is_alive():
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2)

        self._is_running = False
        logger.info(f"Stopped subprocess for model {self._model_id}")

        # Cleanup
        self._request_queue = None
        self._response_queue = None
        self._process = None

    async def infer(
        self, request: dict, cancel_event: asyncio.Event | None = None
    ) -> asyncio.Queue:
        """
        Submit inference request to subprocess.

        Args:
            request: The request body.
            cancel_event: Optional event for cancellation.

        Returns:
            An asyncio.Queue to receive chunks from subprocess.
        """
        if not self.is_running:
            raise RuntimeError(f"Subprocess {self._model_id} is not running")

        # Create asyncio queue for chunks
        chunk_queue: asyncio.Queue = asyncio.Queue()

        # Generate unique request ID
        request_id = str(uuid.uuid4())

        # Prepare request
        full_request = {"request_id": request_id, "body": request}

        # Put request in subprocess queue
        self._request_queue.put(full_request)  # type: ignore

        # Cancel any previous forward task
        if (
            self._current_forward_task is not None
            and not self._current_forward_task.done()
        ):
            self._current_forward_task.cancel()
            logger.debug(f"Cancelled previous forward task for model {self._model_id}")

        # Start background task to forward chunks
        self._current_forward_task = asyncio.create_task(
            self._forward_chunks(request_id, chunk_queue, cancel_event)
        )

        return chunk_queue

    async def _forward_chunks(
        self,
        request_id: str,
        chunk_queue: asyncio.Queue,
        cancel_event: asyncio.Event | None = None,
    ) -> None:
        """Forward chunks from multiprocessing Queue to asyncio Queue."""
        try:
            while True:
                # Check for cancellation
                if cancel_event is not None and cancel_event.is_set():
                    logger.info(f"Request {request_id} cancelled by client")
                    await chunk_queue.put({"type": "cancelled"})
                    break

                # Use non-blocking get with async sleep
                try:
                    msg = self._response_queue.get_nowait()  # type: ignore
                except Exception:
                    await asyncio.sleep(0.01)
                    continue

                if msg is None:
                    break

                # Check if this message belongs to our request
                msg_request_id = msg.get("request_id")
                if msg_request_id != request_id:
                    logger.debug(
                        f"Skipping chunk for different request: {msg_request_id} != {request_id}"
                    )
                    continue

                msg_type = msg.get("type")
                if msg_type == "chunk":
                    await chunk_queue.put(msg.get("data"))
                elif msg_type == "done":
                    await chunk_queue.put({"type": "done"})
                    break
                elif msg_type == "error":
                    await chunk_queue.put(
                        {"type": "error", "message": msg.get("message")}
                    )
                    break
        except Exception as e:
            logger.error(f"Error forwarding chunks: {e}")
            await chunk_queue.put({"type": "error", "message": str(e)})


def _run_subprocess_entry(
    model_id: str,
    loader_name: str,
    step_processor_name: str,
    model_config: dict,
    request_queue: Queue,
    response_queue: Queue,
) -> None:
    """
    Entry point for subprocess (runs in separate process).

    Args:
        model_id: The model identifier.
        loader_name: The name of the model loader.
        step_processor_name: The name of the step processor.
        model_config: Model configuration dictionary.
        request_queue: Queue for receiving requests.
        response_queue: Queue for sending responses.
    """
    import socket

    from plllm_mlx.logging_config import get_logger, setup_logging

    # Initialize logger for subprocess
    hostname = socket.gethostname().removesuffix(".local").lower()
    debug_mode = os.environ.get("DEBUG", "0") == "1"
    setup_logging(level="debug" if debug_mode else "info")
    logger = get_logger(__name__)
    logger.info(
        f"Subprocess logger initialized for model: {model_id} on host: {hostname}"
    )

    from .model_subprocess import run_subprocess

    run_subprocess(
        model_id,
        loader_name,
        step_processor_name,
        model_config,
        request_queue,
        response_queue,
    )


class PlProcessManager:
    """
    Manager for model subprocess lifecycle.

    Each model runs in its own subprocess to avoid blocking the main event loop.

    This is a singleton class.
    """

    _instance: Optional["PlProcessManager"] = None
    _enabled: bool = False

    def __new__(cls) -> "PlProcessManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._processes: Dict[str, PlModelSubprocess] = {}
            cls._instance._lock = asyncio.Lock()
        return cls._instance

    @classmethod
    def enable(cls) -> None:
        """Enable process isolation mode."""
        cls._enabled = True
        logger.info("Process isolation enabled")

    @classmethod
    def disable(cls) -> None:
        """Disable process isolation mode."""
        cls._enabled = False
        logger.info("Process isolation disabled")

    @classmethod
    def is_enabled(cls) -> bool:
        """Check if process isolation is enabled."""
        return cls._enabled

    @classmethod
    def get_instance(cls) -> "PlProcessManager":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def get_or_create_subprocess(
        self,
        model_id: str,
        loader_name: str,
        step_processor_name: str = "base",
        model_config: dict | None = None,
    ) -> PlModelSubprocess:
        """
        Get or create a subprocess for the model.

        Args:
            model_id: The model identifier.
            loader_name: The name of the model loader.
            step_processor_name: The name of the step processor.
            model_config: Model configuration dictionary.

        Returns:
            The subprocess instance.
        """
        key = f"{loader_name}:{model_id}"

        async with self._lock:
            subprocess = self._processes.get(key)
            if subprocess is None:
                subprocess = PlModelSubprocess(
                    model_id, loader_name, step_processor_name, model_config
                )
                subprocess.start()
                self._processes[key] = subprocess
                logger.info(f"Created subprocess for {key}")
            return subprocess

    async def submit_request(
        self,
        model_id: str,
        loader_name: str,
        step_processor_name: str,
        model_config: dict,
        request: dict,
        cancel_event: asyncio.Event | None = None,
    ) -> asyncio.Queue:
        """
        Submit an inference request to the model subprocess.

        Args:
            model_id: The model identifier.
            loader_name: The name of the model loader.
            step_processor_name: The name of the step processor.
            model_config: Model configuration dictionary.
            request: The request body.
            cancel_event: Optional event for cancellation.

        Returns:
            An asyncio.Queue to receive chunks.
        """
        subprocess = await self.get_or_create_subprocess(
            model_id, loader_name, step_processor_name, model_config
        )
        return await subprocess.infer(request, cancel_event)

    async def unload_model(self, model_id: str, loader_name: str) -> None:
        """
        Stop and cleanup the subprocess for a model.

        Args:
            model_id: The model identifier.
            loader_name: The name of the model loader.
        """
        key = f"{loader_name}:{model_id}"

        async with self._lock:
            subprocess = self._processes.pop(key, None)
            if subprocess is not None:
                subprocess.stop()
                logger.info(f"Unloaded model {model_id}")

    async def shutdown(self) -> None:
        """Shutdown all subprocesses."""
        async with self._lock:
            for key, subprocess in self._processes.items():
                subprocess.stop()
            self._processes.clear()
            logger.info("All model subprocesses shutdown")
