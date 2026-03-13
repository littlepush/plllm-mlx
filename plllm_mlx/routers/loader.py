"""
Model loader API router for plllm-mlx.

This module provides endpoints for loading and unloading models.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from plllm_mlx.models.local_models import get_local_model_manager
from plllm_mlx.models.model_loader import PlModelLoader

router = APIRouter(prefix="/v1", tags=["Loader"])

# Get the singleton model manager
localModelMgr = get_local_model_manager()


class LoadModelRequest(BaseModel):
    model_name: str


@router.get("/loader/list")
async def list_loaders():
    loaders = PlModelLoader.listModelLoaders()
    return JSONResponse({"data": loaders})


@router.post("/loader/load")
async def load_model(req: LoadModelRequest):
    model_name = req.model_name

    model = localModelMgr.find_model(model_name)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

    await model.load_model()
    return JSONResponse(
        {"status": "OK", "model_name": model_name, "is_loaded": model.is_loaded}
    )


@router.post("/loader/unload")
async def unload_model(req: LoadModelRequest):
    model_name = req.model_name

    model = localModelMgr.find_model(model_name)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

    await model.unload_model()
    return JSONResponse(
        {"status": "OK", "model_name": model_name, "is_loaded": model.is_loaded}
    )
