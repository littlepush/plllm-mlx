"""
Model manager API router for plllm-mlx.

This module provides endpoints for downloading and managing models from HuggingFace.
"""

from __future__ import annotations

import asyncio
import shutil
import uuid
from pathlib import Path

from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from plllm_mlx.helpers import get_hf_cache_dir, get_model_cache_path
from plllm_mlx.logging_config import get_logger
from plllm_mlx.models.local_models import get_local_model_manager

logger = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["Model Manager"])

localModelMgr = get_local_model_manager()

_download_tasks = {}


class SearchModelResponse(BaseModel):
    id: str
    model_id: str
    downloads: int
    likes: int
    tags: list[str]


class DownloadRequest(BaseModel):
    model_id: str
    model_loader: Optional[str] = None
    step_processor: Optional[str] = None


class DownloadResponse(BaseModel):
    task_id: str
    status: str
    message: str


class DeleteModelRequest(BaseModel):
    model_name: str


def _hf_matches_mlx(model_id: str) -> bool:
    """
    Check if a HuggingFace model supports MLX loading.
    We'll check if there's an 'mlx' or 'apple' folder in the model repo.
    """
    try:
        # Try to download a config file to verify the model exists
        # Check for MLX-specific files
        cache_model_dir = get_model_cache_path(model_id)
        if cache_model_dir is not None:
            snapshots_dir = cache_model_dir / "snapshots"
            if snapshots_dir.exists():
                for snap in snapshots_dir.iterdir():
                    if snap.is_dir():
                        mlx_files = list(snap.glob("*.mlx")) + list(
                            snap.glob("mlx/**/*")
                        )
                        if mlx_files:
                            return True
        return True
    except Exception:
        return True


@router.get("/model/search")
async def search_models(keyword: str = ""):
    """
    Search for models on HuggingFace that are compatible with MLX loader.
    Returns a list of matching models with their metadata.
    """
    try:
        from huggingface_hub import HfApi, list_models

        results = []
        search_keyword = keyword if keyword else "mlx"

        # Search for models with mlx or apple tags
        # Using the HF API to search models
        api = HfApi()

        # Search models (limit to 50 results)
        models = list(list_models(search=search_keyword, limit=50, full=True))

        for model in models:
            model_id = model.id
            # Filter for likely MLX-compatible models (those with mlx in name or tags)
            if model.tags:
                tags_str = " ".join([str(t) for t in model.tags])
                is_mlx_compatible = (
                    "mlx" in tags_str.lower() or "apple" in tags_str.lower()
                )
            else:
                is_mlx_compatible = "mlx" in model_id.lower()

            if is_mlx_compatible or keyword:
                results.append(
                    {
                        "id": model_id,
                        "model_id": model_id,
                        "downloads": getattr(model, "downloads", 0) or 0,
                        "likes": getattr(model, "likes", 0) or 0,
                        "tags": [str(t) for t in model.tags] if model.tags else [],
                        "author": getattr(model, "author", ""),
                        "pipeline_tag": getattr(model, "pipeline_tag", ""),
                    }
                )

        # If no keyword, also search for "apple mlx" models
        if not keyword:
            apple_models = list(list_models(search="apple", limit=30, full=True))
            for model in apple_models:
                model_id = model.id
                if model_id not in [r["id"] for r in results]:
                    results.append(
                        {
                            "id": model_id,
                            "model_id": model_id,
                            "downloads": getattr(model, "downloads", 0) or 0,
                            "likes": getattr(model, "likes", 0) or 0,
                            "tags": [str(t) for t in model.tags] if model.tags else [],
                            "author": getattr(model, "author", ""),
                            "pipeline_tag": getattr(model, "pipeline_tag", ""),
                        }
                    )

        return JSONResponse({"data": results, "total": len(results)})

    except Exception as e:
        logger.error(f"Failed to search models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


async def _download_model_async(
    task_id: str,
    model_id: str,
    model_loader: Optional[str] = None,
    step_processor: Optional[str] = None,
):
    """
    Background task to download a model from HuggingFace.
    Uses huggingface_hub to download files without git-lfs dependency.
    """
    from plllm_mlx.models.model_detector import PlModelDetector
    from huggingface_hub import snapshot_download, repo_info
    import threading

    try:
        _download_tasks[task_id]["status"] = "downloading"
        _download_tasks[task_id]["message"] = f"Starting download for {model_id}..."

        cache_dir = Path(get_hf_cache_dir())
        cache_dir.mkdir(parents=True, exist_ok=True)

        model_id_clean = model_id.replace("/", "--")
        target_dir = cache_dir / f"models--{model_id_clean}"

        if target_dir.exists():
            _download_tasks[task_id]["status"] = "completed"
            _download_tasks[task_id]["message"] = "Model already downloaded"
            _download_tasks[task_id]["model_name"] = model_id
            return

        target_dir.mkdir(parents=True, exist_ok=True)

        _download_tasks[task_id]["message"] = f"Fetching model info for {model_id}..."

        loop = asyncio.get_event_loop()

        total_files = 0
        try:
            info = await loop.run_in_executor(None, lambda: repo_info(model_id))
            if hasattr(info, "siblings") and info.siblings:
                total_files = len(info.siblings)
                _download_tasks[task_id]["total_files"] = total_files
        except Exception:
            pass

        _download_tasks[task_id]["message"] = f"Downloading {model_id}..."

        stop_monitor = threading.Event()

        async def monitor_progress():
            while not stop_monitor.is_set():
                await asyncio.sleep(0.5)
                try:
                    downloaded_files = 0
                    downloaded_bytes = 0
                    for f in target_dir.rglob("*"):
                        if f.is_file():
                            downloaded_files += 1
                            downloaded_bytes += f.stat().st_size

                    _download_tasks[task_id]["downloaded_files"] = downloaded_files
                    _download_tasks[task_id]["downloaded_bytes"] = downloaded_bytes
                    _download_tasks[task_id]["downloaded_mb"] = round(
                        downloaded_bytes / (1024 * 1024), 2
                    )

                    size_mb = downloaded_bytes / (1024 * 1024)
                    _download_tasks[task_id]["message"] = (
                        f"Downloading: {downloaded_files} files, {size_mb:.1f}MB"
                    )
                except Exception:
                    pass

        monitor_task = asyncio.create_task(monitor_progress())

        try:
            await loop.run_in_executor(
                None,
                lambda: snapshot_download(
                    repo_id=model_id,
                    local_dir=str(target_dir),
                    max_workers=4,
                ),
            )
        finally:
            stop_monitor.set()
            try:
                await monitor_task
            except Exception:
                pass

        _download_tasks[task_id]["message"] = "Download completed, loading model..."

        localModelMgr.reload_local_models()

        if model_loader is None or step_processor is None:
            _download_tasks[task_id]["message"] = (
                "Auto-detecting model configuration..."
            )
            detected = PlModelDetector.detect_from_local(model_id)
            if model_loader is None and detected.get("loader"):
                model_loader = detected["loader"]
            if step_processor is None and detected.get("step_processor"):
                step_processor = detected["step_processor"]

        if model_loader is not None:
            await localModelMgr.update_model_loader(model_id, model_loader)
        if step_processor is not None:
            await localModelMgr.update_step_processor(model_id, step_processor)

        _download_tasks[task_id]["model_name"] = model_id
        _download_tasks[task_id]["status"] = "completed"
        _download_tasks[task_id]["message"] = (
            f"Model {model_id} downloaded successfully (loader={model_loader or 'default'}, stpp={step_processor or 'default'})"
        )

        logger.info(f"Model {model_id} downloaded successfully")

    except Exception as e:
        _download_tasks[task_id]["status"] = "failed"
        _download_tasks[task_id]["message"] = f"Download failed: {str(e)}"
        logger.error(f"Model download failed: {str(e)}")


@router.post("/model/download")
async def download_model(req: DownloadRequest):
    """
    Download a model from HuggingFace in the background.
    Returns a task_id that can be used to check the download status.
    """
    model_id = req.model_id
    model_loader = req.model_loader
    step_processor = req.step_processor

    if not model_id or "/" not in model_id:
        model_id = f"mlx-community/{model_id}"

    task_id = str(uuid.uuid4())

    _download_tasks[task_id] = {
        "task_id": task_id,
        "model_id": model_id,
        "model_loader": model_loader,
        "step_processor": step_processor,
        "status": "pending",
        "message": "Task created",
        "model_name": None,
        "downloaded_files": 0,
        "total_files": 0,
        "current_file": "",
        "downloaded_bytes": 0,
    }

    asyncio.create_task(
        _download_model_async(task_id, model_id, model_loader, step_processor)
    )

    return JSONResponse(
        {
            "task_id": task_id,
            "status": "pending",
            "message": f"Download task created for {model_id}",
        }
    )


@router.get("/model/download/status/{task_id}")
async def get_download_status(task_id: str):
    """
    Get the status of a model download task.
    """
    if task_id not in _download_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = _download_tasks[task_id]

    response = {
        "task_id": task["task_id"],
        "model_id": task["model_id"],
        "status": task["status"],
        "message": task["message"],
        "model_name": task.get("model_name"),
    }

    if task["status"] == "downloading":
        response["progress"] = {
            "downloaded_files": task.get("downloaded_files", 0),
            "total_files": task.get("total_files", 0),
            "downloaded_bytes": task.get("downloaded_bytes", 0),
            "downloaded_mb": round(task.get("downloaded_bytes", 0) / (1024 * 1024), 2),
            "current_file": task.get("current_file", ""),
        }
        if task.get("total_files", 0) > 0:
            response["progress"]["percent"] = round(
                (task.get("downloaded_files", 0) / task.get("total_files", 1)) * 100, 1
            )
        else:
            response["progress"]["percent"] = 0

    return JSONResponse(response)


@router.post("/model/delete")
async def delete_model(req: DeleteModelRequest):
    """
    Delete a locally downloaded model.
    """
    model_name = req.model_name

    # Check if model exists in local storage
    model = localModelMgr.find_model(model_name)
    if model is None:
        model_cache = get_model_cache_path(model_name)
        if model_cache is None:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        shutil.rmtree(model_cache, ignore_errors=True)

        return JSONResponse(
            {"status": "OK", "message": f"Model {model_name} deleted from disk"}
        )

    # Unload model if loaded
    if model.is_loaded:
        await model.unload_model()

    # Remove from local model manager
    if model_name in localModelMgr._models_in_memory:
        del localModelMgr._models_in_memory[model_name]

    # Remove from config storage
    if model_name in localModelMgr._model_configs:
        del localModelMgr._model_configs[model_name]

    model_cache = get_model_cache_path(model_name)
    if model_cache is not None:
        shutil.rmtree(model_cache, ignore_errors=True)

    localModelMgr.reload_local_models()

    return JSONResponse(
        {"status": "OK", "message": f"Model {model_name} deleted successfully"}
    )


@router.get("/model/list")
async def list_local_models():
    """
    List all local models (both in-memory and on-disk).
    """
    models_info = localModelMgr.list_model_info()

    cache_dir = Path(get_hf_cache_dir())
    disk_models = []

    if cache_dir.exists():
        for folder in cache_dir.iterdir():
            if folder.is_dir() and folder.name.startswith("models--"):
                model_id = folder.name.replace("models--", "").replace("--", "/")
                if model_id not in localModelMgr._models_in_memory:
                    disk_models.append(
                        {
                            "model_name": model_id,
                            "model_loader": "mlx",
                            "step_processor": "base",
                            "is_loaded": False,
                            "config": {},
                        }
                    )

    return JSONResponse(
        {
            "data": models_info + disk_models,
            "total": len(models_info) + len(disk_models),
        }
    )
