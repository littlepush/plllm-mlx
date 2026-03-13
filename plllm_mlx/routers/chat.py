"""
Chat completions API router for plllm-mlx.

This module provides OpenAI-compatible chat completion endpoints.
"""

from __future__ import annotations

import asyncio
import json
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from plllm_mlx.logging_config import get_logger
from plllm_mlx.models.local_models import get_local_model_manager

logger = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["Chat"])

# Get the singleton model manager
localModelMgr = get_local_model_manager()


# CORS preflight handler
@router.options("/{path:path}")
async def preflight_handler(path: str):
    return JSONResponse(content=None, status_code=200)


# Semaphore for chat concurrency control
# Using a large initial value to allow queuing (actual limit handled by wait timeout)
_chat_semaphore: Optional[asyncio.Semaphore] = None


def init_chat_semaphore(max_concurrent: int | None = None):
    """Initialize chat semaphore (max_concurrent is unused now, kept for API compatibility)"""
    global _chat_semaphore
    # Use a large semaphore value to allow unlimited queuing
    # The actual limit is the wait timeout (300 seconds)
    _chat_semaphore = asyncio.Semaphore(1000)
    logger.info("Chat semaphore initialized (unlimited queue, max wait: 300s)")


async def get_chat_semaphore() -> asyncio.Semaphore:
    global _chat_semaphore
    if _chat_semaphore is None:
        _chat_semaphore = asyncio.Semaphore(1000)
    return _chat_semaphore


@router.post("/chat/completions")
async def chat_completions(req: Request):
    body = await req.json()
    # logger.debug(f"chat headers: {json.dumps(dict(req.headers.items()), ensure_ascii=False)}")
    # logger.debug(f"chat: {json.dumps(body, ensure_ascii=False)}")

    is_stream = body.get("stream", False)
    model = body.get("model", "")
    stream_options = body.get("stream_options", None)
    include_usage = False
    if stream_options is not None and stream_options.get("include_usage", False):
        include_usage = True
    if model == "":
        raise HTTPException(status_code=400, detail="model field is required")

    # Directly find model by name (no category lookup)
    local_model = localModelMgr.find_model(model)
    if local_model is None:
        logger.error(f"model {model} not found")
        raise HTTPException(status_code=400, detail=f"model {model} not found")

    # Acquire semaphore for concurrent chat control (with timeout)
    semaphore = await get_chat_semaphore()
    try:
        # Wait up to 300 seconds for a slot
        await asyncio.wait_for(semaphore.acquire(), timeout=300)
        logger.debug("chat acquired semaphore")
    except asyncio.TimeoutError:
        logger.error("Chat request timed out waiting for available slot")
        raise HTTPException(
            status_code=503, detail="Server is busy, please try again later"
        )

    if not is_stream:
        try:
            result = await local_model.chat_completions_restful(body, alias_name=model)
            logger.info(f"[RESTFUL_RESULT]: {json.dumps(result, ensure_ascii=False)}")
            return JSONResponse(result)
        finally:
            semaphore.release()
            logger.debug("chat released semaphore")

    async def _stream_wrapper():
        # Create cancellation event for client disconnect handling
        cancel_event = asyncio.Event()
        try:
            idx = 0
            try:
                # Pass cancel_event so subprocess can be notified when client disconnects
                async for chunk in local_model.chat_completions_stream(
                    body,
                    alias_name=model,
                    include_usage=include_usage,
                    cancel_event=cancel_event,
                ):
                    idx += 1
                    if await req.is_disconnected():
                        logger.info("client disconnected, stop streaming.")
                        cancel_event.set()  # Signal cancellation to subprocess
                        break
                    yield chunk
            except (BrokenPipeError, ConnectionResetError) as e:
                logger.warning(f"client connection broken: {e}")
                cancel_event.set()  # Signal cancellation on connection error
            except Exception as e:
                import traceback

                tb_str = "\n".join(
                    traceback.format_exception(type(e), e, e.__traceback__)
                )
                logger.error(f"unexpected stream error: {e}\n{tb_str}")
                cancel_event.set()  # Signal cancellation on error
        finally:
            semaphore.release()
            logger.debug("chat stream released semaphore")

    return StreamingResponse(
        _stream_wrapper(),
        media_type="text/event-stream; charset=utf-8",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.post("/completions")
async def completions(req: Request):
    body = await req.json()
    logger.debug(
        f"completions headers: {json.dumps(dict(req.headers.items()), ensure_ascii=False)}"
    )
    logger.debug(f"completions: {json.dumps(body, ensure_ascii=False)}")

    is_stream = body.get("stream", False)
    model = body.get("model", "")
    if model == "":
        raise HTTPException(status_code=400, detail="model field is required")

    # Directly find model by name (no category lookup)
    local_model = localModelMgr.find_model(model)
    if local_model is None:
        raise HTTPException(status_code=400, detail=f"model {model} not found")

    # Acquire semaphore for concurrent chat control
    semaphore = await get_chat_semaphore()
    await semaphore.acquire()
    logger.debug("completions acquired semaphore")

    if not is_stream:
        try:
            return JSONResponse(
                await local_model.completions_restful(body, alias_name=model)
            )
        finally:
            semaphore.release()
            logger.debug("completions released semaphore")

    async def _stream_wrapper():
        try:
            try:
                async for chunk in local_model.completions_stream(
                    body, alias_name=model
                ):
                    if await req.is_disconnected():
                        logger.info("client disconnected, stop streaming.")
                        break
                    yield chunk
            except (BrokenPipeError, ConnectionResetError) as e:
                logger.warning(f"client connection broken: {e}")
            except Exception as e:
                logger.error(f"unexpected stream error: {e}")
        finally:
            semaphore.release()
            logger.debug("completions stream released semaphore")

    return StreamingResponse(
        _stream_wrapper(),
        media_type="text/event-stream; charset=utf-8",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
