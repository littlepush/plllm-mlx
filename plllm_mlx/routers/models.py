"""
Model management API router for plllm-mlx.

This module provides endpoints for managing local models.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from plllm_mlx.logging_config import get_logger
from plllm_mlx.models.local_models import get_local_model_manager
from plllm_mlx.models.model_detector import PlModelDetector
from plllm_mlx.models.model_loader import PlModelLoader

logger = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["Models"])

localModelMgr = get_local_model_manager()

VLM_INSTALL_HINT = (
    "VLM (Vision Language Model) support requires additional dependencies. "
    "Please install with: pip install 'plllm-mlx[vlm]' "
    "or: uv pip install 'plllm-mlx[vlm]'"
)


async def ensure_model_loaded(
    model_name: str, loader: Optional[str] = None, step_processor: Optional[str] = None
) -> dict:
    """
    Ensure model is loaded with proper loader and step_processor.
    Auto-detects if not specified and model uses default values.

    Returns dict with loader and step_processor info.
    """
    model = localModelMgr.find_model(model_name)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

    if model.is_loaded and loader is None and step_processor is None:
        return {
            "model_name": model_name,
            "is_loaded": True,
            "loader": type(model).model_loader_name(),
            "step_processor": model.step_processor_clz.step_clz_name(),
        }

    model_info = localModelMgr._model_configs.get(model_name)
    current_loader = model_info.model_loader if model_info else "mlx"
    current_stpp = model_info.step_processor if model_info else "base"

    if loader is None and step_processor is None:
        need_detect = current_loader == "mlx" and current_stpp == "base"
        if need_detect:
            logger.info(f"Auto-detecting loader and step_processor for {model_name}")
            try:
                detected = PlModelDetector.detect_from_local(model_name)
                if detected.get("loader") and detected["loader"] != current_loader:
                    detected_loader = detected["loader"]
                    available_loaders = PlModelLoader.listModelLoaders()
                    if detected_loader not in available_loaders:
                        if detected_loader == "mlxvlm":
                            raise HTTPException(
                                status_code=400,
                                detail=f"VLM model detected, but 'mlxvlm' loader is not available. {VLM_INSTALL_HINT}",
                            )
                        else:
                            raise HTTPException(
                                status_code=400,
                                detail=f"Detected loader '{detected_loader}' is not available. Available loaders: {available_loaders}",
                            )
                    loader = detected_loader
                    logger.info(f"Detected loader: {loader}")
                if (
                    detected.get("step_processor")
                    and detected["step_processor"] != current_stpp
                ):
                    step_processor = detected["step_processor"]
                    logger.info(f"Detected step_processor: {step_processor}")
            except HTTPException:
                raise
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

    return {
        "model_name": model_name,
        "is_loaded": model.is_loaded,
        "loader": type(model).model_loader_name(),
        "step_processor": model.step_processor_clz.step_clz_name(),
    }


@router.get("/model/list")
async def list_models():
    models = localModelMgr.list_model_info()
    return JSONResponse({"data": models})


@router.get("/models")
async def list_models_openai():
    """OpenAI compatible endpoint - GET /v1/models (only loaded models)"""
    models = localModelMgr.list_model_info()
    openai_models = []
    for m in models:
        if m.get("is_loaded"):
            openai_models.append(
                {
                    "id": m["model_name"],
                    "object": "model",
                    "created": 0,
                    "owned_by": "plllm-mlx",
                }
            )
    return JSONResponse(
        {
            "object": "list",
            "data": openai_models,
        }
    )


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


class LoadModelRequest(BaseModel):
    model_name: str
    loader: Optional[str] = None
    step_processor: Optional[str] = None


@router.post("/model/load")
async def load_model(req: LoadModelRequest):
    result = await ensure_model_loaded(req.model_name, req.loader, req.step_processor)
    result["status"] = "OK"
    return JSONResponse(result)


@router.post("/model/unload")
async def unload_model(req: LoadModelRequest):
    model_name = req.model_name

    model = localModelMgr.find_model(model_name)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

    await model.unload_model()
    return JSONResponse(
        {"status": "OK", "model_name": model_name, "is_loaded": model.is_loaded}
    )
