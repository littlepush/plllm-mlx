"""
Model proxy for the main process.

This module provides a proxy object that represents a model in the main process,
delegating operations to the subprocess via PlSubprocessHandle.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Optional

from plllm_mlx.logging_config import get_logger

if TYPE_CHECKING:
    from .manager import PlSubprocessManager

logger = get_logger(__name__)


class PlModelProxy:
    """
    Model proxy - represents a model in the main process.

    This class provides a local interface for model operations,
    delegating the actual work to a subprocess via PlSubprocessHandle.
    """

    def __init__(
        self,
        model_name: str,
        loader: str = "mlx",
        step_processor: str = "base",
        manager: Optional["PlSubprocessManager"] = None,
    ):
        """
        Initialize the model proxy.

        Args:
            model_name: Model name or path
            loader: Loader type (mlx, mlxvlm)
            step_processor: Step processor type
            manager: Subprocess manager instance
        """
        self._model_name = model_name
        self._loader = loader
        self._step_processor = step_processor
        self._manager = manager
        self._config: Dict[str, Any] = {}
        self._is_loaded = False

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def loader(self) -> str:
        return self._loader

    @property
    def step_processor(self) -> str:
        return self._step_processor

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def step_processor_clz(self):
        """Get the step processor class (for compatibility)."""
        from plllm_mlx.subprocess.python.step_processor import PlStepProcessor

        return PlStepProcessor.findStepProcessor(self._step_processor)

    def get_config(self) -> Dict[str, Any]:
        """Get the model configuration."""
        return self._config.copy()

    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Set model configuration.

        Args:
            config: Configuration dictionary
        """
        self._config.update(config)

        # If already loaded, update the subprocess
        if self._is_loaded and self._manager:
            handle = self._manager._subprocesses.get(self._model_name)
            if handle:
                asyncio.create_task(handle.update_config(config))

    async def load_model(self) -> bool:
        """
        Load the model.

        Returns:
            True if loaded successfully
        """
        if self._is_loaded:
            return True

        if self._manager is None:
            from .manager import get_subprocess_manager

            self._manager = get_subprocess_manager()

        try:
            handle = await self._manager.get_or_create(
                self._model_name,
                self._loader,
                self._step_processor,
                self._config,
            )
            self._is_loaded = True
            logger.info(f"Model proxy loaded: {self._model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model {self._model_name}: {e}")
            return False

    async def unload_model(self) -> bool:
        """
        Unload the model.

        Returns:
            True if unloaded successfully
        """
        if not self._is_loaded:
            return True

        if self._manager is None:
            return False

        try:
            handle = self._manager._subprocesses.get(self._model_name)
            if handle:
                await handle.unload_model()
            self._is_loaded = False
            logger.info(f"Model proxy unloaded: {self._model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to unload model {self._model_name}: {e}")
            return False

    async def chat_completions_stream(
        self,
        body: Dict[str, Any],
        alias_name: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Stream chat completions.

        Args:
            body: Request body
            alias_name: Optional alias for the model name
            **kwargs: Additional parameters

        Yields:
            SSE data lines
        """
        if not self._is_loaded:
            await self.load_model()

        if self._manager is None:
            raise RuntimeError("Model not loaded")

        handle = self._manager._subprocesses.get(self._model_name)
        if handle is None:
            raise RuntimeError("Subprocess not found")

        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens", 1024)
        temperature = body.get("temperature")
        top_p = body.get("top_p")
        top_k = body.get("top_k")

        async for line in handle.infer(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            **kwargs,
        ):
            yield line

    async def chat_completions_restful(
        self,
        body: Dict[str, Any],
        alias_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Non-streaming chat completions.

        Args:
            body: Request body
            alias_name: Optional alias for the model name

        Returns:
            Complete chat completion response
        """
        import json
        import time
        import uuid

        create_time = int(time.time())
        content = ""
        thinking = ""
        toolcalls = []
        chat_id = f"chatcmpl-{str(uuid.uuid4()).replace('-', '')}"
        usage = None
        finish_reason = None

        async for line in self.chat_completions_stream(body, alias_name=alias_name):
            if line.startswith("data: "):
                line = line[6:]
            if line == "[DONE]":
                break
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue

            usage = chunk.get("usage", usage)
            if "choices" in chunk and len(chunk["choices"]) > 0:
                finish_reason = chunk["choices"][0].get("finish_reason", None)
                delta = chunk["choices"][0].get("delta", {})
                if "reasoning" in delta:
                    thinking += delta["reasoning"]
                elif "content" in delta:
                    content += delta["content"]
                if "tool_calls" in delta:
                    tclist = delta["tool_calls"]
                    if tclist is not None:
                        if isinstance(tclist, list):
                            toolcalls += tclist
                        else:
                            toolcalls.append(tclist)

        chat_obj = {
            "id": chat_id,
            "created": create_time,
            "object": "chat_completion",
            "model": alias_name if alias_name else self._model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None
                        if len(toolcalls) > 0
                        else (content if len(content) > 0 else None),
                        "toolcalls": toolcalls if len(toolcalls) > 0 else None,
                        "reasoning": thinking if len(thinking) > 0 else None,
                    },
                    "finish_reason": finish_reason
                    if finish_reason
                    else ("tool_calls" if len(toolcalls) > 0 else "stop"),
                }
            ],
            "usage": usage,
        }
        return chat_obj

    async def update_step_processor(self, step_processor_name: str) -> bool:
        """
        Update the step processor.

        Args:
            step_processor_name: New step processor name

        Returns:
            True if successful
        """
        from plllm_mlx.subprocess.python.step_processor import PlStepProcessor

        step_clz = PlStepProcessor.findStepProcessor(step_processor_name)
        if step_clz is None:
            logger.error(f"Step processor not found: {step_processor_name}")
            return False

        self._step_processor = step_processor_name

        # If loaded, need to reload the model with new processor
        if self._is_loaded:
            await self.unload_model()
            await self.load_model()

        return True
