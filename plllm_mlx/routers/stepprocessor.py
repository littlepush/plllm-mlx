"""
Step processor API router for plllm-mlx.

This module provides endpoints for listing step processors.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from plllm_mlx.subprocess.python.step_processor import PlStepProcessor

router = APIRouter(prefix="/v1", tags=["Step Processor"])


@router.get("/stepprocessor/list")
async def list_processors():
    processors = PlStepProcessor.listStepProcessors()
    return JSONResponse({"data": processors})
