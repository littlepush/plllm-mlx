"""
Model management API router for plllm-mlx.

This module provides endpoints for managing local models.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from plllm_mlx.models.local_models import get_local_model_manager

router = APIRouter(prefix="/v1", tags=["Models"])

# Get the singleton model manager
localModelMgr = get_local_model_manager()


@router.get("/model/list")
async def list_models():
    models = localModelMgr.list_model_info()
    return JSONResponse({"data": models})


@router.post("/model/reload")
async def reload_models():
    localModelMgr.reload_local_models()
    return JSONResponse({"status": "OK"})


@router.post("/model/update/stepprocessor")
async def update_step_processor(req: Request):
    body = await req.json()
    model_name = body.get("model_name", "")
    step_processor = body.get("step_processor", None)
    if model_name == "" or not step_processor:
        raise HTTPException(
            status_code=400, detail="model_name and step_processor fields are required"
        )
    result = await localModelMgr.update_step_processor(model_name, step_processor)
    return JSONResponse({"status": "OK" if result else "Failed"})


@router.post("/model/update/modelloader")
async def update_model_loader(req: Request):
    body = await req.json()
    model_name = body.get("model_name", "")
    loader = body.get("model_loader", None)
    if model_name == "" or not loader:
        raise HTTPException(
            status_code=400, detail="model_name and loader fields are required"
        )
    result = await localModelMgr.update_model_loader(model_name, loader)
    return JSONResponse({"status": "OK" if result else "Failed"})


@router.post("/model/update/config")
async def update_model_config(req: Request):
    body = await req.json()
    model_name = body.get("model_name", "")
    config_key = body.get("key", None)
    config_value = body.get("value", None)

    if model_name == "" or config_key is None or config_value is None:
        raise HTTPException(
            status_code=400, detail="model_name and key and value fields are required"
        )
    result = await localModelMgr.update_model_config(
        model_name, config_key, config_value
    )
    return JSONResponse({"status": "OK" if result else "Failed"})


@router.post("/model/load")
async def load_model(req: Request):
    body = await req.json()
    model_name = body.get("model_name", "")
    if model_name == "":
        raise HTTPException(status_code=400, detail="model_name is required")
    model = localModelMgr.find_model(model_name)
    if model is None:
        raise HTTPException(status_code=400, detail=f"model {model_name} not found")
    await model.load_model()
    return JSONResponse({"status": "OK"})


@router.post("/model/unload")
async def unload_model(req: Request):
    body = await req.json()
    model_name = body.get("model_name", "")
    if model_name == "":
        raise HTTPException(status_code=400, detail="model_name is required")
    model = localModelMgr.find_model(model_name)
    if model is None:
        raise HTTPException(status_code=400, detail=f"model {model_name} not found")
    await model.unload_model()
    return JSONResponse({"status": "OK"})
