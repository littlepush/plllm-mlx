"""
Package initialization for plllm-mlx routers submodule.

This module provides FastAPI routers for the plllm-mlx service,
including OpenAI-compatible API endpoints.
"""

from __future__ import annotations

from .chat import router as chat_router
from .loader import router as loader_router
from .model_manager import router as model_manager_router
from .models import router as models_router
from .stepprocessor import router as stepprocessor_router

__all__: list[str] = [
    "chat_router",
    "loader_router",
    "model_manager_router",
    "models_router",
    "stepprocessor_router",
]
