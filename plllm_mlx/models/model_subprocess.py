"""
Model subprocess entry point.

This module provides the main entry point for model subprocesses that handle
model inference without blocking the main event loop.
"""

from __future__ import annotations

import asyncio
from multiprocessing import Queue

from plllm_mlx.logging_config import get_logger

logger = get_logger(__name__)


def run_subprocess(
    model_id: str,
    loader_name: str,
    step_processor_name: str,
    model_config: dict,
    request_queue: Queue,
    response_queue: Queue,
) -> None:
    """
    Main entry point for model subprocess.

    Runs an event loop to process inference requests.

    Args:
        model_id: The model identifier.
        loader_name: The name of the model loader.
        step_processor_name: The name of the step processor.
        model_config: Model configuration dictionary.
        request_queue: Queue for receiving requests.
        response_queue: Queue for sending responses.
    """
    logger.info(
        f"Starting model subprocess for {model_id} with loader {loader_name}, "
        f"step_processor: {step_processor_name}"
    )

    # Create new event loop for this process
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Load model
    loader = None
    try:
        from .base_step_processor import PlStepProcessor
        from .model_loader import PlModelLoader

        # Use the correct step processor from the main process
        step_p_clz = PlStepProcessor.findStepProcessor(step_processor_name)
        if step_p_clz is None:
            raise RuntimeError(f"Failed to find step processor: {step_processor_name}")

        l_clz = PlModelLoader.__LOADER_MAP__.get(loader_name)  # type: ignore
        if l_clz is None:
            raise RuntimeError(f"Failed to find loader class: {loader_name}")

        loader = l_clz(model_id, step_p_clz)
        logger.info(f"Created loader for {model_id}")

        # Apply model config from main process
        if model_config:
            loader.set_config(model_config)
            logger.info(f"Applied model config for {model_id}")

        # Load model synchronously in the subprocess
        loop.run_until_complete(loader.ensure_model_loaded())
        logger.info(f"Model {model_id} loaded in subprocess")

    except Exception as e:
        logger.error(f"Failed to load model in subprocess: {e}")
        response_queue.put({"type": "error", "message": str(e)})
        loop.close()
        return

    # Process requests in the event loop
    try:
        while True:
            try:
                request = request_queue.get(timeout=60)
            except Exception:
                # Timeout, continue waiting
                continue

            if request is None:
                # Stop signal
                logger.info(f"Received stop signal for {model_id}")
                break

            request_id = request.get("request_id")
            body = request.get("body")

            try:
                logger.info(f"Processing request {request_id}")

                # Prepare prompt
                session_object = loader.prepare_prompt(body)

                # Run inference and stream results
                async def generate():
                    try:
                        async for chunk in loader.stream_generate(session_object):
                            response_queue.put(
                                {
                                    "request_id": request_id,
                                    "type": "chunk",
                                    "data": chunk,
                                }
                            )
                        response_queue.put({"request_id": request_id, "type": "done"})
                    except Exception as e:
                        logger.error(f"Inference error: {e}")
                        response_queue.put(
                            {
                                "request_id": request_id,
                                "type": "error",
                                "message": str(e),
                            }
                        )

                loop.run_until_complete(generate())

            except Exception as e:
                logger.error(f"Request {request_id} error: {e}")
                response_queue.put(
                    {"request_id": request_id, "type": "error", "message": str(e)}
                )

    except KeyboardInterrupt:
        logger.info(f"Subprocess {model_id} interrupted")
    except Exception as e:
        logger.error(f"Subprocess {model_id} error: {e}")
    finally:
        # Cleanup
        if loader is not None:
            try:
                loop.run_until_complete(loader.ensure_model_unloaded())
            except Exception as e:
                logger.warning(f"Failed to unload model: {e}")

        loop.close()
        logger.info(f"Subprocess {model_id} exited")
