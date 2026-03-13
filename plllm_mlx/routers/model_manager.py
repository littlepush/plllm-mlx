"""
Model manager API router for plllm-mlx.

This module provides endpoints for downloading and managing models from HuggingFace.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from plllm_mlx.logging_config import get_logger
from plllm_mlx.models.local_models import get_local_model_manager, PlLocalModelInfo

logger = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["Model Manager"])

# Get the singleton model manager
localModelMgr = get_local_model_manager()

# Get HuggingFace cache path
HUGGINGFACE_PATH = os.environ.get(
    "HUGGING_FACE_PATH", f"{Path.home()}/.cache/huggingface/hub"
)

# Track download tasks
_download_tasks = {}


class SearchModelResponse(BaseModel):
    id: str
    model_id: str
    downloads: int
    likes: int
    tags: list[str]


class DownloadRequest(BaseModel):
    model_id: str
    model_loader: str = "mlx"


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
        from huggingface_hub import hf_hub_download

        # Try to download a config file to verify the model exists
        # Check for MLX-specific files
        model_id_clean = model_id.replace("/", "--")
        cache_model_dir = Path(HUGGINGFACE_PATH) / f"models--{model_id_clean}"
        if cache_model_dir.exists():
            # Check if snapshots contain mlx files
            snapshots_dir = cache_model_dir / "snapshots"
            if snapshots_dir.exists():
                for snap in snapshots_dir.iterdir():
                    if snap.is_dir():
                        # Check for mlx files or configs
                        mlx_files = list(snap.glob("*.mlx")) + list(
                            snap.glob("mlx/**/*")
                        )
                        if mlx_files:
                            return True
        # Fallback: return True to allow user to try any model
        # The actual download will fail if not compatible
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


async def _download_model_async(task_id: str, model_id: str, model_loader: str):
    """
    Background task to download a model from HuggingFace.
    Uses huggingface_hub to download files without git-lfs dependency.
    """
    try:
        # Update task status
        _download_tasks[task_id]["status"] = "downloading"
        _download_tasks[task_id]["message"] = f"Starting download for {model_id}..."

        # Prepare cache directory
        cache_dir = Path(HUGGINGFACE_PATH)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Build the model path
        model_id_clean = model_id.replace("/", "--")
        target_dir = cache_dir / f"models--{model_id_clean}"

        if target_dir.exists():
            _download_tasks[task_id]["status"] = "completed"
            _download_tasks[task_id]["message"] = "Model already downloaded"
            _download_tasks[task_id]["model_name"] = model_id
            return

        # Create target directory
        target_dir.mkdir(parents=True, exist_ok=True)

        # Use huggingface_hub to download without git-lfs
        _download_tasks[task_id]["message"] = f"Downloading {model_id}..."

        from huggingface_hub import snapshot_download

        # Download the model files in executor to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: snapshot_download(
                repo_id=model_id,
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
            ),
        )

        # Register the model in local model manager
        _download_tasks[task_id]["model_name"] = model_id
        _download_tasks[task_id]["status"] = "completed"
        _download_tasks[task_id]["message"] = (
            f"Model {model_id} downloaded successfully"
        )

        # Reload local models to pick up the new model
        localModelMgr.reload_local_models()

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

    # Validate model_id
    if not model_id or "/" not in model_id:
        # Try to infer the full model id
        model_id = f"mlx-community/{model_id}"

    # Generate task_id
    task_id = str(uuid.uuid4())

    # Create task entry
    _download_tasks[task_id] = {
        "task_id": task_id,
        "model_id": model_id,
        "model_loader": model_loader,
        "status": "pending",
        "message": "Task created",
        "model_name": None,
    }

    # Start background task
    asyncio.create_task(_download_model_async(task_id, model_id, model_loader))

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
    return JSONResponse(
        {
            "task_id": task["task_id"],
            "model_id": task["model_id"],
            "status": task["status"],
            "message": task["message"],
            "model_name": task.get("model_name"),
        }
    )


@router.post("/model/delete")
async def delete_model(req: DeleteModelRequest):
    """
    Delete a locally downloaded model.
    """
    model_name = req.model_name

    # Check if model exists in local storage
    model = localModelMgr.find_model(model_name)
    if model is None:
        # Try to find in on-disk models
        model_id_clean = model_name.replace("/", "--")
        target_dir = Path(HUGGINGFACE_PATH) / f"models--{model_id_clean}"

        if not target_dir.exists():
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        # Delete from disk
        shutil.rmtree(target_dir, ignore_errors=True)

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

    # Delete from disk
    model_id_clean = model_name.replace("/", "--")
    target_dir = Path(HUGGINGFACE_PATH) / f"models--{model_id_clean}"
    if target_dir.exists():
        shutil.rmtree(target_dir, ignore_errors=True)

    # Reload local models
    localModelMgr.reload_local_models()

    return JSONResponse(
        {"status": "OK", "message": f"Model {model_name} deleted successfully"}
    )


@router.get("/model/list")
async def list_local_models():
    """
    List all local models (both in-memory and on-disk).
    """
    # Get models from manager
    models_info = localModelMgr.list_model_info()

    # Also get models that are on disk but not loaded
    cache_dir = Path(HUGGINGFACE_PATH)
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
