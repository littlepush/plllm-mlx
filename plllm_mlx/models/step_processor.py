from abc import ABC, abstractmethod
from plllm_mlx.helpers import *
from typing import Any, Optional, List
from plllm_mlx.logging_config import get_logger
logger = get_logger(__name__)
import os

class PlStepProcessor(ABC):
  def __init__(self):
    self._can_step_continue = True
    self._total_tokens = 0
    self._unprocessed_text = ""
    
  __STEPP_MAP__ = {}
  
  @staticmethod
  def registerStepProcessor(step_p_name:str, step_p_clz):
    PlStepProcessor.__STEPP_MAP__[step_p_name] = step_p_clz
    
  @staticmethod
  def findStepProcessor(step_p_name:str):
    return PlStepProcessor.__STEPP_MAP__.get(step_p_name, None)
  
  @staticmethod
  def listStepProcessors():
    return [s for s in PlStepProcessor.__STEPP_MAP__.keys()]
    
  @staticmethod
  @abstractmethod
  def step_clz_name() -> str:
    pass
    
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
  def step(self, generate_response:Optional[Any] = None) -> Optional[PlChunk]:
    pass
  
  @abstractmethod
  def tool_calls(self) -> List[PlChunk]:
    pass
  
  @abstractmethod
  def finish(self) -> PlChunk:
    pass


# Load all step processors when current script is imported
_all_python_codes = PlUnpackPath(os.path.join(PlRootPath(), 'models'), recursive=False)
for script in _all_python_codes:
  if script.endswith("_step_processor.py"):
    step_processor_clz = PlFindSpecifialSubclass(script, PlStepProcessor)
    for clz in step_processor_clz:
      pl_log.info(f"add step processor: {clz.step_clz_name()}")
      PlStepProcessor.registerStepProcessor(clz.step_clz_name(), clz)
