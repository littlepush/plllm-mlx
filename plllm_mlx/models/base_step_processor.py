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


# Auto-load step processors when module is imported
_load_all_step_processors()
