"""
Base step processor for handling LLM generation steps.

This module provides the abstract base class for step processors that handle
individual generation steps from LLM inference, including token processing,
tool call detection, and finish reason handling.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

from plllm_mlx.helpers import PlChunk
from plllm_mlx.logging_config import get_logger

logger = get_logger(__name__)


class PlStepProcessor(ABC):
    """
    Abstract base class for step processors.

    Step processors handle individual generation steps from LLM inference,
    processing tokens, detecting tool calls, and managing finish reasons.

    Attributes:
        total_tokens: Total number of tokens processed.
        is_running: Whether the generation is still running.
        unprocessed_text: Text that hasn't been processed yet.

    Class Attributes:
        __PROCESSOR_MAP__: Registry of all step processor classes.
    """

    __PROCESSOR_MAP__: Dict[str, Type["PlStepProcessor"]] = {}

    def __init__(self) -> None:
        """Initialize the step processor."""
        self.total_tokens: int = 0
        self.is_running: bool = True
        self.unprocessed_text: str = ""

    @staticmethod
    @abstractmethod
    def step_clz_name() -> str:
        """
        Return the name of this step processor class.

        Returns:
            The unique name identifier for this processor.
        """
        pass

    @staticmethod
    def registerStepProcessor(
        name: str, processor_clz: Type["PlStepProcessor"]
    ) -> None:
        """
        Register a step processor class.

        Args:
            name: The name to register under.
            processor_clz: The step processor class to register.
        """
        PlStepProcessor.__PROCESSOR_MAP__[name] = processor_clz

    @staticmethod
    def findStepProcessor(name: str) -> Optional[Type["PlStepProcessor"]]:
        """
        Find a step processor class by name.

        Args:
            name: The name of the step processor to find.

        Returns:
            The step processor class, or None if not found.
        """
        return PlStepProcessor.__PROCESSOR_MAP__.get(name)

    @staticmethod
    def listStepProcessors() -> List[str]:
        """
        List all registered step processor names.

        Returns:
            List of registered step processor names.
        """
        return list(PlStepProcessor.__PROCESSOR_MAP__.keys())

    @abstractmethod
    def step(self, generation_result: Any) -> Optional[PlChunk]:
        """
        Process a single generation step.

        Args:
            generation_result: The generation result from the LLM.

        Returns:
            A PlChunk if there's output to yield, None otherwise.
        """
        pass

    @abstractmethod
    def tool_calls(self) -> List[PlChunk]:
        """
        Get any tool calls detected during processing.

        Returns:
            List of PlChunk objects containing tool call data.
        """
        pass

    @abstractmethod
    def finish(self) -> PlChunk:
        """
        Finish processing and return the final chunk.

        Returns:
            The final PlChunk with finish reason.
        """
        pass


def _load_all_step_processors() -> None:
    """
    Load all step processors from the models directory.

    This function imports all *_step_processor.py files from the models directory,
    which will trigger their registration via PlStepProcessor.registerStepProcessor().
    """
    import importlib
    import os

    from plllm_mlx.helpers.clz_helper import PlRootPath

    models_path = os.path.join(PlRootPath(), "models")
    if not os.path.exists(models_path):
        return

    for filename in os.listdir(models_path):
        if filename == "base_step_processor.py":
            continue
        if filename.endswith("_step_processor.py"):
            module_name = filename[:-3]  # Remove .py extension
            try:
                # Import the module using the package-relative path
                importlib.import_module(f".{module_name}", package="plllm_mlx.models")
                logger.debug(f"Loaded step processor module: {module_name}")
            except ImportError as e:
                # Log the error but don't fail - the processor might have missing dependencies
                logger.debug(f"Could not import step processor {module_name}: {e}")
            except Exception as e:
                logger.error(f"Failed to load step processor from {filename}: {e}")


# Define and register a default step processor
class PlDefaultStepProcessor(PlStepProcessor):
    """
    Default step processor for basic LLM generation.

    This processor handles standard text generation without special processing.
    It simply yields content chunks and handles finish reasons.
    """

    def __init__(self) -> None:
        """Initialize the default step processor."""
        super().__init__()
        self._finish_reason: Optional[str] = None

    @staticmethod
    def step_clz_name() -> str:
        """
        Return the name of this step processor class.

        Returns:
            The unique name identifier for this processor.
        """
        return "default"

    def step(self, generation_result: Any) -> Optional[PlChunk]:
        """
        Process a single generation step.

        Args:
            generation_result: The generation result from the LLM.

        Returns:
            A PlChunk with the generated text, or None if no output.
        """
        from plllm_mlx.helpers import PlStepUsage

        gr = generation_result
        self.total_tokens += 1

        try:
            # Check for finish reason
            if gr.finish_reason is not None:
                self._finish_reason = gr.finish_reason
                self.is_running = False
                return None

            # Build usage info
            usage = PlStepUsage()
            usage.prompt_tokens = getattr(gr, "prompt_tokens", 0)
            usage.completion_tokens = getattr(gr, "generation_tokens", 0)
            usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
            usage.prompt_tps = getattr(gr, "prompt_tps", 0.0)
            usage.generation_tps = getattr(gr, "generation_tps", 0.0)

            # Create content chunk
            chunk = PlChunk(
                data=gr.text,
                data_type=PlChunkDataType.CONTENT,
                step=usage,
            )
            return chunk

        except Exception as e:
            logger.error(f"Step processing error: {e}")
            self.is_running = False
            return None

    def tool_calls(self) -> List[PlChunk]:
        """
        Get any tool calls detected during processing.

        Returns:
            Empty list as default processor doesn't handle tool calls.
        """
        return []

    def finish(self) -> PlChunk:
        """
        Finish processing and return the final chunk.

        Returns:
            A PlChunk with the finish reason.
        """
        finish_reason = self._finish_reason or "stop"
        return PlChunk(finish_reason=finish_reason)


# Register default step processor
PlStepProcessor.registerStepProcessor("default", PlDefaultStepProcessor)
PlStepProcessor.registerStepProcessor(
    "base", PlDefaultStepProcessor
)  # Alias for compatibility

# Auto-load step processors when module is imported
_load_all_step_processors()
