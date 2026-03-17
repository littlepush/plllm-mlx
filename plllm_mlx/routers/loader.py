"""
Model loader API router for plllm-mlx.

This module provides endpoints for loader information.
Model load/unload operations are handled by /v1/model/load and /v1/model/unload.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from plllm_mlx.subprocess.python.loader import PlModelLoader

router = APIRouter(prefix="/v1", tags=["Loader"])


@router.get("/loader/list")
async def list_loaders():
    """List available model loaders."""
    loaders = PlModelLoader.listModelLoaders()
    return JSONResponse({"data": loaders})
