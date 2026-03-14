"""
Model loader base class for MLX-based LLM inference.

This module provides the abstract base class for model loaders that handle
model loading, inference, and chat completion streaming.
"""

from __future__ import annotations

import asyncio
import functools
import json
import os
import time
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

from plllm_mlx.helpers import (
    PlChatCompletionHelper,
    PlChunk,
    PlChunkDataType,
    PlFindSpecifialSubclass,
    PlRootPath,
    PlUnpackPath,
)
from plllm_mlx.logging_config import get_logger

if TYPE_CHECKING:
    from .base_step_processor import PlStepProcessor
    from .process_manager import PlProcessManager

logger = get_logger(__name__)


def async_ticker(name: str):
    """
    Decorator for async functions to add timing logging.

    Args:
        name: Name to use in log messages.
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.debug(f"[{name}] {func.__name__} completed in {elapsed:.3f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(
                    f"[{name}] {func.__name__} failed after {elapsed:.3f}s: {e}"
                )
                raise

        return wrapper

    return decorator


def yield_ticker(name: str):
    """
    Decorator for async generator functions to add timing logging.

    Args:
        name: Name to use in log messages.
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                async for item in func(*args, **kwargs):
                    yield item
                elapsed = time.time() - start_time
                logger.debug(f"[{name}] {func.__name__} completed in {elapsed:.3f}s")
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(
                    f"[{name}] {func.__name__} failed after {elapsed:.3f}s: {e}"
                )
                raise

        return wrapper

    return decorator


class PlModelLoader(ABC):
    """
    Abstract base class for model loaders.

    Model loaders handle model loading, configuration, inference, and
    chat completion streaming for different model types.

    Class Attributes:
        __LOADER_MAP__: Registry of all model loader classes.
        _process_manager: Optional process manager for subprocess isolation.
    """

    __LOADER_MAP__: Dict[str, Type["PlModelLoader"]] = {}
    _process_manager: Optional["PlProcessManager"] = None

    def __init__(
        self, model_name: str, step_processor_clz: Type["PlStepProcessor"]
    ) -> None:
        """
        Initialize the model loader.

        Args:
            model_name: The name or path of the model to load.
            step_processor_clz: The step processor class to use for inference.
        """
        self._model_name = model_name
        self._alias = ""
        self._end_tokens: List[str] = []
        self._verbose = False
        self._is_loaded = False
        self._step_processor_clz = step_processor_clz
        self._loader_name = self.model_loader_name()

    @staticmethod
    @abstractmethod
    def model_loader_name() -> str:
        """
        Return the unique name for this model loader.

        Returns:
            The loader name identifier.
        """
        pass

    @staticmethod
    def registerModelLoader(name: str, loader_clz: Type["PlModelLoader"]) -> None:
        """
        Register a model loader class.

        Args:
            name: The name to register under.
            loader_clz: The model loader class to register.
        """
        PlModelLoader.__LOADER_MAP__[name] = loader_clz

    @staticmethod
    def createModel(
        loader_name: str, model_name: str, step_processor_name: str
    ) -> Optional["PlModelLoader"]:
        """
        Create a model loader instance.

        Args:
            loader_name: The name of the loader class to use.
            model_name: The name or path of the model.
            step_processor_name: The name of the step processor to use.

        Returns:
            A model loader instance, or None if creation failed.
        """
        from .base_step_processor import PlStepProcessor

        step_p_clz = PlStepProcessor.findStepProcessor(step_processor_name)
        if step_p_clz is None:
            logger.error(f"Failed to find step processor: {step_processor_name}")
            return None
        l_clz = PlModelLoader.__LOADER_MAP__.get(loader_name, None)
        if l_clz is None:
            logger.error(
                f"Failed to create model: {model_name} with loader: {loader_name}"
            )
            return None
        return l_clz(model_name, step_p_clz)

    @staticmethod
    def listModelLoaders() -> List[str]:
        """
        List all registered model loader names.

        Returns:
            List of registered loader names.
        """
        return list(PlModelLoader.__LOADER_MAP__.keys())

    # Process manager methods
    @classmethod
    def set_process_manager(cls, pm: "PlProcessManager") -> None:
        """Set the process manager for subprocess isolation."""
        cls._process_manager = pm

    @classmethod
    def get_process_manager(cls) -> Optional["PlProcessManager"]:
        """Get the current process manager."""
        return cls._process_manager

    @classmethod
    def is_process_isolation_enabled(cls) -> bool:
        """Check if process isolation is enabled."""
        return cls._process_manager is not None and cls._process_manager.is_enabled()

    # Properties
    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_alias(self) -> str:
        return self._alias

    @model_alias.setter
    def model_alias(self, alias: str) -> None:
        self._alias = alias

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def verbose(self) -> bool:
        return self._verbose

    @verbose.setter
    def verbose(self, v: bool) -> None:
        self._verbose = v

    @property
    def step_processor_clz(self) -> Type["PlStepProcessor"]:
        return self._step_processor_clz

    def update_step_processor(self, step_processor_name: str) -> None:
        """
        Update the step processor for this model.

        Args:
            step_processor_name: The name of the new step processor.
        """
        from .base_step_processor import PlStepProcessor

        step_p_clz = PlStepProcessor.findStepProcessor(step_processor_name)
        if step_p_clz is None:
            logger.error(f"Failed to update step processor: {step_processor_name}")
            return
        self._step_processor_clz = step_p_clz

    # Abstract methods
    @abstractmethod
    async def ensure_model_loaded(self) -> None:
        """Load the model if not already loaded."""
        pass

    async def load_model(self) -> None:
        """Load the model."""
        if self._is_loaded:
            return
        await self.ensure_model_loaded()
        self._is_loaded = True

    @abstractmethod
    async def ensure_model_unloaded(self) -> None:
        """Unload the model."""
        pass

    async def unload_model(self) -> None:
        """Unload the model if loaded."""
        if not self._is_loaded:
            return
        self._is_loaded = False
        await self.ensure_model_unloaded()

    @abstractmethod
    def set_config(self, model_config: dict) -> None:
        """Set model configuration."""
        pass

    @abstractmethod
    def get_config(self) -> dict:
        """Get model configuration."""
        pass

    @abstractmethod
    def prepare_prompt(self, body: dict) -> Any:
        """Prepare prompt from request body."""
        pass

    @abstractmethod
    async def stream_generate(self, session_object: Any):
        """Stream generate tokens from session object."""
        pass

    @abstractmethod
    async def completion_stream_generate(self, session_object: Any):
        """Stream generate for completion (non-chat) mode."""
        pass

    # Process isolation support
    async def stream_generate_via_process(
        self, body: dict, cancel_event: asyncio.Event | None = None
    ):
        """
        Run stream_generate via subprocess when process isolation is enabled.

        Args:
            body: The request body.
            cancel_event: Optional event for cancellation.
        """
        if not self.is_process_isolation_enabled():
            logger.error("Process isolation is not enabled")
            return
            yield  # type: ignore

        if self._process_manager is None:
            logger.error("Process manager is not set")
            return
            yield  # type: ignore

        step_processor_name = self._step_processor_clz.step_clz_name()
        model_config = self.get_config()

        chunk_queue = await self._process_manager.submit_request(
            self._model_name,
            self._loader_name,
            step_processor_name,
            model_config,
            body,
            cancel_event,
        )

        while True:
            chunk = await chunk_queue.get()
            if isinstance(chunk, dict):
                if chunk.get("type") == "done":
                    break
                if chunk.get("type") == "error":
                    raise Exception(chunk.get("message", "Unknown error"))
                continue
            yield chunk

    @yield_ticker("PlModelLoader")
    async def chat_completions_stream_with_isolation(
        self,
        body: dict,
        alias_name: Optional[str] = None,
        return_json: bool = False,
        return_done: bool = True,
        include_usage: bool = False,
        cancel_event: asyncio.Event | None = None,
    ):
        """
        Chat completions stream using process isolation.

        Args:
            body: The request body.
            alias_name: Optional alias for the model.
            return_json: Whether to return JSON format.
            return_done: Whether to send [DONE] at the end.
            include_usage: Whether to include usage statistics.
            cancel_event: Event for cancellation.
        """
        helper = PlChatCompletionHelper(
            alias_name if alias_name else self.model_name, include_usage=include_usage
        )
        helper.prompt_processed()

        cached_finish_reason = None
        total_tokens = 0

        async for chunk in self.stream_generate_via_process(body, cancel_event):
            if self._verbose:
                print(chunk.data, end="")  # type: ignore
            if chunk.data_type == PlChunkDataType.REASONING:  # type: ignore
                helper.update_reason_step(chunk.data, chunk.step)  # type: ignore
                total_tokens += 1
                build_result = helper.build_yield_chunk(return_json)
                yield build_result
                continue

            if chunk.data_type == PlChunkDataType.CONTENT:  # type: ignore
                total_tokens += 1
                helper.update_content_step(chunk.data, chunk.step)  # type: ignore
                yield helper.build_yield_chunk(return_json)
                continue

            if chunk.data_type == PlChunkDataType.TOOLCALL:  # type: ignore
                helper.update_tool_step(chunk.data, chunk.step)  # type: ignore
                helper.finish_step("tool_calls")
                logger.info(
                    f"Get a tool call: {json.dumps(helper.build_chunk(), ensure_ascii=False)}"
                )
                build_result = helper.build_yield_chunk(return_json)
                yield build_result
                if return_done:
                    yield "data: [DONE]\n\n"
                return

            if chunk.finish_reason is not None:  # type: ignore
                cached_finish_reason = chunk.finish_reason  # type: ignore

            if chunk.finish_reason is None and chunk.data_type == PlChunkDataType.NONE:  # type: ignore
                logger.error("Get step data type none when has no finish reason!")
                cached_finish_reason = "stop"
                break

            if cached_finish_reason is not None:
                break

        if cached_finish_reason is None:
            cached_finish_reason = "stop"
        helper.finish_step(cached_finish_reason)
        yield helper.build_yield_chunk(return_json)

        if return_done:
            yield "data: [DONE]\n\n"

    @yield_ticker("PlModelLoader")
    async def chat_completions_stream(
        self,
        body: dict,
        alias_name: Optional[str] = None,
        return_json: bool = False,
        return_done: bool = True,
        include_usage: bool = False,
        cancel_event: asyncio.Event | None = None,
    ):
        """
        Chat completions stream.

        Args:
            body: The request body.
            alias_name: Optional alias for the model.
            return_json: Whether to return JSON format.
            return_done: Whether to send [DONE] at the end.
            include_usage: Whether to include usage statistics.
            cancel_event: Event for cancellation.
        """
        # Check if process isolation is enabled
        if self.is_process_isolation_enabled():
            async for chunk in self.chat_completions_stream_with_isolation(
                body, alias_name, return_json, return_done, include_usage, cancel_event
            ):
                yield chunk
            return

        # Original direct mode
        helper = PlChatCompletionHelper(
            alias_name if alias_name else self.model_name, include_usage=include_usage
        )
        session_object = self.prepare_prompt(body)
        helper.prompt_processed()

        cached_finish_reason = None
        total_tokens = 0

        async for chunk in self.stream_generate(session_object):
            if self._verbose:
                print(chunk.data, end="")  # type: ignore
            if chunk.data_type == PlChunkDataType.REASONING:  # type: ignore
                helper.update_reason_step(chunk.data, chunk.step)  # type: ignore
                total_tokens += 1
                build_result = helper.build_yield_chunk(return_json)
                yield build_result
                continue

            if chunk.data_type == PlChunkDataType.CONTENT:  # type: ignore
                total_tokens += 1
                helper.update_content_step(chunk.data, chunk.step)  # type: ignore
                yield helper.build_yield_chunk(return_json)
                continue

            if chunk.data_type == PlChunkDataType.TOOLCALL:  # type: ignore
                helper.update_tool_step(chunk.data, chunk.step)  # type: ignore
                helper.finish_step("tool_calls")
                logger.info(
                    f"Get a tool call: {json.dumps(helper.build_chunk(), ensure_ascii=False)}"
                )
                build_result = helper.build_yield_chunk(return_json)
                yield build_result
                if return_done:
                    yield "data: [DONE]\n\n"
                return

            if chunk.finish_reason is not None:  # type: ignore
                cached_finish_reason = chunk.finish_reason  # type: ignore

            if chunk.finish_reason is None and chunk.data_type == PlChunkDataType.NONE:  # type: ignore
                logger.error("Get step data type none when has no finish reason!")
                cached_finish_reason = "stop"
                break

            if cached_finish_reason is not None:
                break

        if cached_finish_reason is None:
            cached_finish_reason = "stop"
        helper.finish_step(cached_finish_reason)
        yield helper.build_yield_chunk(return_json)

        if return_done:
            yield "data: [DONE]\n\n"

    @async_ticker("PlModelLoader")
    async def chat_completions_restful(
        self, body: dict, alias_name: Optional[str] = None
    ) -> dict:
        """
        Non-streaming chat completions.

        Args:
            body: The request body.
            alias_name: Optional alias for the model.

        Returns:
            The complete chat completion response.
        """
        create_time = int(time.time())
        content = ""
        thinking = ""
        toolcalls = []
        chat_id = f"chatcmpl-{str(uuid.uuid4()).replace('-', '')}"
        usage = None
        finish_reason = None

        async for chunk in self.chat_completions_stream(
            body, alias_name=alias_name, return_json=True, return_done=False
        ):
            usage = chunk.get("usage", usage)
            finish_reason = chunk["choices"][0].get("finish_reason", None)

            if "choices" not in chunk:
                continue
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
            "model": alias_name if alias_name else self.model_name,
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

    @yield_ticker("PlModelLoader")
    async def completions_stream(
        self,
        body: dict,
        alias_name: Optional[str] = None,
        return_json: bool = False,
        return_done: bool = True,
    ):
        """
        Completions stream (non-chat mode).

        Args:
            body: The request body.
            alias_name: Optional alias for the model.
            return_json: Whether to return JSON format.
            return_done: Whether to send [DONE] at the end.
        """
        helper = PlChatCompletionHelper(alias_name if alias_name else self.model_name)
        session_object = self.prepare_prompt(body)
        helper.prompt_processed()

        cached_finish_reason = None
        total_tokens = 0
        async for chunk in self.completion_stream_generate(session_object):
            if self._verbose:
                print(chunk.data, end="")  # type: ignore
            if chunk.data_type == PlChunkDataType.REASONING:  # type: ignore
                helper.update_reason_step(chunk.data, chunk.step)  # type: ignore
                total_tokens += 1
                build_result = helper.build_yield_text(return_json)
                logger.debug(f"SSE CHUNK REASONING: {build_result}")
                yield build_result
                continue

            if chunk.data_type == PlChunkDataType.CONTENT:  # type: ignore
                total_tokens += 1
                helper.update_content_step(chunk.data, chunk.step)  # type: ignore
                build_result = helper.build_yield_text(return_json)
                logger.debug(f"SSE CHUNK CONTENT: {build_result}")
                yield build_result
                continue

            if chunk.data_type == PlChunkDataType.TOOLCALL:  # type: ignore
                helper.update_tool_step(chunk.data, chunk.step)  # type: ignore
                logger.info(
                    f"Get a tool call: {json.dumps(helper.build_chunk(), ensure_ascii=False)}"
                )
                build_result = helper.build_yield_text(return_json)
                yield build_result
                continue

            if chunk.finish_reason is not None:  # type: ignore
                cached_finish_reason = chunk.finish_reason  # type: ignore

            if chunk.finish_reason is None and chunk.data_type == PlChunkDataType.NONE:  # type: ignore
                logger.error("Get step data type none when has no finish reason!")
                cached_finish_reason = "stop"
                break

            if cached_finish_reason is not None:
                break

        if cached_finish_reason is None:
            cached_finish_reason = "stop"
        helper.finish_step(cached_finish_reason)
        yield helper.build_yield_text(return_json)

        if return_done:
            yield "data: [DONE]\n\n"

    @async_ticker("PlModelLoader")
    async def completions_restful(
        self, body: dict, alias_name: Optional[str] = None
    ) -> dict:
        """
        Non-streaming completions.

        Args:
            body: The request body.
            alias_name: Optional alias for the model.

        Returns:
            The complete completion response.
        """
        chat_id = f"cmpl-{str(uuid.uuid4()).replace('-', '')}"
        create_time = int(time.time())
        content = ""
        usage = None
        finish_reason = None

        async for chunk in self.completions_stream(
            body, alias_name=alias_name, return_json=True, return_done=False
        ):
            if "choices" not in chunk:
                continue
            content += chunk["choices"][0].get("text", "")
            usage = chunk.get("usage", usage)
            finish_reason = chunk["choices"][0].get("finish_reason", None)

        result = {
            "id": chat_id,
            "object": "text_completion",
            "created": create_time,
            "model": alias_name if alias_name else self.model_name,
            "choices": [{"index": 0, "text": content, "finish_reason": finish_reason}],
            "usage": usage,
        }
        return result


def _load_all_model_loaders() -> None:
    """
    Load all model loaders from the models directory.

    This function imports all *_loader.py files from the models directory,
    which will trigger their registration via PlModelLoader.registerModelLoader().
    """
    import importlib

    models_path = os.path.join(PlRootPath(), "models")
    if not os.path.exists(models_path):
        return

    all_python_files = PlUnpackPath(models_path, recursive=False)
    for script in all_python_files:
        filename = os.path.basename(script)
        if filename == "model_loader.py":
            continue
        if script.endswith("_loader.py"):
            module_name = filename[:-3]  # Remove .py extension
            try:
                # Import the module using the package-relative path
                importlib.import_module(f".{module_name}", package="plllm_mlx.models")
                logger.debug(f"Loaded model loader module: {module_name}")
            except ImportError as e:
                # Log the error but don't fail - the loader might have missing dependencies
                logger.debug(f"Could not import model loader {module_name}: {e}")
            except Exception as e:
                logger.error(f"Failed to load model loader from {script}: {e}")


# Auto-load model loaders when module is imported
_load_all_model_loaders()
