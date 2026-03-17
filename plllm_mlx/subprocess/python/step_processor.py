import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List, Optional

from plllm_mlx.helpers import PlChunk, PlRootPath, PlUnpackPath
from plllm_mlx.logging_config import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from .special_tokens import SpecialTokens


class PlStepProcessor(ABC):
    def __init__(self, special_tokens: Optional["SpecialTokens"] = None):
        self._special_tokens = special_tokens
        self._can_step_continue = True
        self._total_tokens = 0
        self._unprocessed_text = ""

    __STEPP_MAP__ = {}

    @staticmethod
    def registerStepProcessor(step_p_name: str, step_p_clz):
        PlStepProcessor.__STEPP_MAP__[step_p_name] = step_p_clz

    @staticmethod
    def findStepProcessor(step_p_name: str):
        return PlStepProcessor.__STEPP_MAP__.get(step_p_name, None)

    @staticmethod
    def listStepProcessors():
        return [s for s in PlStepProcessor.__STEPP_MAP__.keys()]

    @staticmethod
    @abstractmethod
    def step_clz_name() -> str:
        pass

    @property
    def special_tokens(self) -> "SpecialTokens":
        if self._special_tokens is None:
            from .special_tokens import SpecialTokens

            self._special_tokens = SpecialTokens()
        return self._special_tokens

    @property
    def is_running(self):
        return self._can_step_continue

    def stop(self):
        self._can_step_continue = False

    @property
    def total_tokens(self):
        return self._total_tokens

    @total_tokens.setter
    def total_tokens(self, tt):
        self._total_tokens = tt

    @property
    def unprocessed_text(self):
        return self._unprocessed_text

    @unprocessed_text.setter
    def unprocessed_text(self, ut):
        self._unprocessed_text = ut

    @abstractmethod
    def step(self, generate_response: Optional[Any] = None) -> Optional[PlChunk]:
        pass

    @abstractmethod
    def tool_calls(self) -> List[PlChunk]:
        pass

    @abstractmethod
    def finish(self) -> PlChunk:
        pass


# Load all step processors when current script is imported
def _load_all_step_processors():
    import importlib

    stepps_path = os.path.join(PlRootPath(), "subprocess", "python", "stepps")
    if not os.path.exists(stepps_path):
        return

    all_python_codes = PlUnpackPath(stepps_path, recursive=False)
    for script in all_python_codes:
        filename = os.path.basename(script)
        if filename in ("__init__.py",):
            continue
        if filename.endswith("_step_processor.py"):
            module_name = filename[:-3]
            try:
                mod = importlib.import_module(
                    f".{module_name}", package="plllm_mlx.subprocess.python.stepps"
                )
                for attr_name in dir(mod):
                    attr = getattr(mod, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, PlStepProcessor)
                        and attr is not PlStepProcessor
                    ):
                        logger.info(f"add step processor: {attr.step_clz_name()}")
                        PlStepProcessor.registerStepProcessor(
                            attr.step_clz_name(), attr
                        )
            except ImportError as e:
                logger.debug(f"Could not import step processor {module_name}: {e}")


_load_all_step_processors()
