"""
Model loader API router for plllm-mlx.

This module provides endpoints for loading and unloading models.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from plllm_mlx.logging_config import get_logger
from plllm_mlx.models.local_models import get_local_model_manager
from plllm_mlx.models.model_detector import PlModelDetector
from plllm_mlx.models.model_loader import PlModelLoader

logger = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["Loader"])

localModelMgr = get_local_model_manager()


class LoadModelRequest(BaseModel):
    model_name: str
    loader: Optional[str] = None
    step_processor: Optional[str] = None


@router.get("/loader/list")
async def list_loaders():
    loaders = PlModelLoader.listModelLoaders()
    return JSONResponse({"data": loaders})


@router.post("/loader/load")
async def load_model(req: LoadModelRequest):
    model_name = req.model_name
    loader = req.loader
    step_processor = req.step_processor

    model = localModelMgr.find_model(model_name)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

    model_info = localModelMgr._model_configs.get(model_name)
    current_loader = model_info.model_loader if model_info else "mlx"
    current_stpp = model_info.step_processor if model_info else "base"

    if loader is None and step_processor is None:
        need_detect = current_loader == "mlx" and current_stpp == "base"
        if need_detect:
            logger.info(f"Auto-detecting loader and step_processor for {model_name}")
            try:
                detected = PlModelDetector.detect(model_name)
                if detected.get("loader") and detected["loader"] != current_loader:
                    loader = detected["loader"]
                    logger.info(f"Detected loader: {loader}")
                if (
                    detected.get("step_processor")
                    and detected["step_processor"] != current_stpp
                ):
                    step_processor = detected["step_processor"]
                    logger.info(f"Detected step_processor: {step_processor}")
            except Exception as e:
                logger.warning(f"Auto-detection failed for {model_name}: {e}")

    if loader is not None:
        await localModelMgr.update_model_loader(model_name, loader)
    if step_processor is not None:
        await localModelMgr.update_step_processor(model_name, step_processor)

    if loader is not None or step_processor is not None:
        model = localModelMgr.find_model(model_name)
        if model is None:
            raise HTTPException(
                status_code=500, detail=f"Failed to update model {model_name}"
            )

    await model.load_model()

    model = localModelMgr.find_model(model_name)
    if model is None:
        raise HTTPException(
            status_code=500, detail=f"Failed to get model info for {model_name}"
        )

    return JSONResponse(
        {
            "status": "OK",
            "model_name": model_name,
            "is_loaded": model.is_loaded,
            "loader": type(model).model_loader_name(),
            "step_processor": model.step_processor_clz.step_clz_name(),
        }
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
