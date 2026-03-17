"""
Subprocess FastAPI server for model inference.

This module provides an isolated HTTP server that runs in a separate process,
communicating with the main process via Unix Domain Socket.

Thread Model:
- API thread (FastAPI/uvicorn): Handles HTTP requests
- Inference thread: Handles model loading/inference
- Shared state protected by threading.Lock
"""

from __future__ import annotations

import asyncio
import atexit
import json
import os
import signal
import threading
import time
import uuid
from pathlib import Path
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from plllm_mlx.logging_config import get_logger

if TYPE_CHECKING:
    from plllm_mlx.subprocess.python.loader import PlModelLoader

logger = get_logger(__name__)

# Global state
_loader: Optional["PlModelLoader"] = None
_model_name: str = ""
_loader_name: str = ""
_step_processor_name: str = ""
_start_time: float = time.time()
_state_lock = threading.Lock()
_inferencing: bool = False

# Thread communication
_infer_queue: Queue = Queue()
_result_queues: Dict[str, Queue] = {}


def _run_async(coro):
    """Run an async coroutine in a new event loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class LoadRequest(BaseModel):
    model_name: str
    loader: str = "mlx"
    step_processor: str = "base"
    config: Dict[str, Any] = {}


class InferRequest(BaseModel):
    messages: list
    stream: bool = True
    max_tokens: int = 1024
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    model: Optional[str] = None


app = FastAPI(title="plllm-mlx subprocess")


def _infer_worker():
    """Inference thread - handles all model operations."""
    global _loader, _inferencing

    while True:
        try:
            task = _infer_queue.get(timeout=1.0)
        except Empty:
            continue

        if task is None:
            break

        request_id, endpoint, payload = task
        result_queue = _result_queues.get(request_id)

        if result_queue is None:
            continue

        with _state_lock:
            _inferencing = True

        try:
            if endpoint == "load":
                _handle_load(payload, result_queue)
            elif endpoint == "unload":
                _handle_unload(result_queue)
            elif endpoint == "infer":
                _handle_infer(payload, result_queue)
        except Exception as e:
            logger.error(f"Error handling {endpoint}: {e}")
            result_queue.put({"error": str(e)})
        finally:
            with _state_lock:
                _inferencing = False


def _handle_load(payload: dict, result_queue: Queue):
    """Handle model load request."""
    global _loader, _model_name, _loader_name, _step_processor_name

    from plllm_mlx.subprocess.python.loader import PlModelLoader

    model_name = payload.get("model_name", "")
    loader_name = payload.get("loader", "mlx")
    step_processor_name = payload.get("step_processor", "base")
    config = payload.get("config", {})

    try:
        if _loader is not None:
            if _loader.is_loaded:
                _run_async(_loader.unload_model())
            _loader = None

        _loader = PlModelLoader.createModel(
            loader_name, model_name, step_processor_name
        )
        if _loader is None:
            result_queue.put({"error": f"Failed to create loader: {loader_name}"})
            return

        _loader.set_config(config)
        _run_async(_loader.load_model())

        _model_name = model_name
        _loader_name = loader_name
        _step_processor_name = step_processor_name

        result_queue.put({"success": True, "model_name": model_name})
        logger.info(f"Model loaded: {model_name}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        result_queue.put({"error": str(e)})


def _handle_unload(result_queue: Queue):
    """Handle model unload request."""
    global _loader, _model_name

    try:
        if _loader is not None and _loader.is_loaded:
            _run_async(_loader.unload_model())
        _loader = None
        _model_name = ""
        result_queue.put({"success": True})
        logger.info("Model unloaded")
    except Exception as e:
        logger.error(f"Failed to unload model: {e}")
        result_queue.put({"error": str(e)})


def _handle_infer(payload: dict, result_queue: Queue):
    """Handle inference request."""
    global _loader

    if _loader is None or not _loader.is_loaded:
        result_queue.put({"error": "Model not loaded"})
        return

    try:

        async def run_infer():
            body = {
                "messages": payload.get("messages", []),
                "stream": True,
                "max_tokens": payload.get("max_tokens", 1024),
            }
            if payload.get("temperature") is not None:
                body["temperature"] = payload["temperature"]
            if payload.get("top_p") is not None:
                body["top_p"] = payload["top_p"]
            if payload.get("top_k") is not None:
                body["top_k"] = payload["top_k"]

            async for chunk in _loader.chat_completions_stream(body):
                result_queue.put(chunk)
            result_queue.put(None)

        _run_async(run_infer())
    except Exception as e:
        logger.error(f"Inference error: {e}")
        result_queue.put({"error": str(e)})


# Start inference thread
_infer_thread = threading.Thread(target=_infer_worker, daemon=True, name="infer-worker")
_infer_thread.start()


@app.get("/health")
async def health():
    """Health check - fast, non-blocking."""
    with _state_lock:
        return {
            "status": "healthy",
            "model_loaded": _loader is not None and _loader.is_loaded,
            "model_name": _model_name,
            "inferencing": _inferencing,
        }


@app.get("/status")
async def status():
    """Get full status."""
    with _state_lock:
        config = {}
        if _loader is not None:
            config = _loader.get_config()

        return {
            "model_name": _model_name,
            "loader": _loader_name,
            "step_processor": _step_processor_name,
            "is_loaded": _loader is not None and _loader.is_loaded,
            "config": config,
            "pid": os.getpid(),
            "uptime_seconds": int(time.time() - _start_time),
            "inferencing": _inferencing,
        }


@app.post("/load")
async def load_model(request: LoadRequest):
    """Load model."""
    request_id = str(uuid.uuid4())
    result_queue = Queue()
    _result_queues[request_id] = result_queue

    _infer_queue.put((request_id, "load", request.model_dump()))

    try:
        result = result_queue.get(timeout=300)
        del _result_queues[request_id]
        if "error" in result:
            raise HTTPException(400, result["error"])
        return result
    except Exception as e:
        del _result_queues[request_id]
        raise HTTPException(500, str(e))


@app.post("/unload")
async def unload_model():
    """Unload model."""
    request_id = str(uuid.uuid4())
    result_queue = Queue()
    _result_queues[request_id] = result_queue

    _infer_queue.put((request_id, "unload", {}))

    try:
        result = result_queue.get(timeout=60)
        del _result_queues[request_id]
        if "error" in result:
            raise HTTPException(400, result["error"])
        return result
    except Exception as e:
        del _result_queues[request_id]
        raise HTTPException(500, str(e))


@app.get("/config")
async def get_config():
    """Get current config."""
    with _state_lock:
        if _loader is None:
            return {"config": {}}
        return {"config": _loader.get_config()}


@app.put("/config")
async def update_config(config: Dict[str, Any]):
    """Update config - routes to loader.set_config()."""
    with _state_lock:
        if _loader is None:
            raise HTTPException(400, "Model not loaded")
        _loader.set_config(config)
        return {"success": True, "config": _loader.get_config()}


@app.post("/infer")
async def infer(request: InferRequest):
    """Inference request - streaming response."""
    if _loader is None or not _loader.is_loaded:
        raise HTTPException(400, "Model not loaded")

    request_id = str(uuid.uuid4())
    result_queue = Queue()
    _result_queues[request_id] = result_queue

    _infer_queue.put((request_id, "infer", request.model_dump()))

    async def stream_results():
        try:
            while True:
                try:
                    result = result_queue.get_nowait()
                    if result is None:
                        break
                    if "error" in result:
                        yield f"data: {json.dumps(result)}\n\n"
                        break
                    # Ensure proper SSE format with \n\n suffix
                    if not result.endswith("\n\n"):
                        result = result.rstrip() + "\n\n"
                    yield result
                except Empty:
                    await asyncio.sleep(0.01)
        finally:
            del _result_queues[request_id]

    return StreamingResponse(stream_results(), media_type="text/event-stream")


def run_server(socket_path: str, model_name: Optional[str] = None):
    """
    Run the subprocess server.

    Args:
        socket_path: Path to Unix domain socket
        model_name: Optional model name to load on startup
    """
    socket_file = Path(socket_path)
    socket_file.parent.mkdir(parents=True, exist_ok=True)

    # Clean up any existing socket
    if socket_file.exists():
        socket_file.unlink()

    def cleanup():
        if socket_file.exists():
            socket_file.unlink()
        logger.info("Subprocess server stopped")

    atexit.register(cleanup)
    signal.signal(signal.SIGTERM, lambda s, f: cleanup())
    signal.signal(signal.SIGINT, lambda s, f: cleanup())

    logger.info(f"Starting subprocess server on {socket_path}")

    config = uvicorn.Config(
        app,
        uds=str(socket_file),
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    server.run()
