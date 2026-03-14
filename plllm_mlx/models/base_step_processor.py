from plllm_mlx.logging_config import get_logger
logger = get_logger(__name__)
from plllm_mlx.helpers import *
from plllm_mlx.models.step_processor import PlStepProcessor
from typing import Any, Optional, List
import time

class PlBaseStepProcessor(PlStepProcessor):
  def __init__(self):
    super().__init__()
    self.toolcall_buffer = []
    self.first_token_time = None
    self.stop_reason = ""
    self.is_stop_by_length = False
    self.full_content = ""
    
  @staticmethod
  def step_clz_name():
    return "base"
  
  def step(self, generate_response:Any) -> Optional[PlChunk]:
    gr = generate_response
    self.total_tokens += 1
    self.full_content += gr.text
    
    if self.first_token_time is None:
      self.first_token_time = time.time()
    
    # Just stop the generation
    if PlMlxGetFinishReason(gr) is not None:
      self.is_stop_by_length = True if PlMlxGetFinishReason(gr) == "length" else False
      self.stop_reason = PlMlxGetFinishReason(gr) if PlMlxGetFinishReason(gr) else "stop"
      self.stop()
      return None
    
    # Begin of tool call
    if gr.text == "<tool_call>":
      self.toolcall_buffer.append(gr.text)
      return None
    
    # If already in tool call mode
    if len(self.toolcall_buffer) > 0:
      self.toolcall_buffer.append(gr.text)
      # End of tool call, all generation should stop
      if gr.text == "</tool_call>":
        self.stop()
      return None
    
    # Prepare response chunk (regular content)
    sr = PlStepUsage()
    sr.prompt_tokens = gr.prompt_tokens
    sr.completion_tokens = gr.generation_tokens
    sr.total_tokens = gr.prompt_tokens + gr.generation_tokens
    sr.prompt_tps = gr.prompt_tps
    sr.generation_tps = gr.generation_tps
    if sr.generation_tps is None:
      if self.total_tokens == 1:
        sr.generation_tps = 1
      else:
        sr.generation_tps = self.total_tokens / (time.time() - self.first_token_time)
    
    chunk = PlChunk(step=sr)
    chunk.data = gr.text
    chunk.data_type = PlChunkDataType.CONTENT
    return chunk
  
  def tool_calls(self) -> List[PlChunk]:
    result = []
    if len(self.toolcall_buffer) > 0:
      # Pass buffer content (excluding <tool_call> and </tool_call>) to parser
      tool_call_chunk = PlCommonToolcallParser(self.toolcall_buffer[1:-1])
      if tool_call_chunk is not None:
        result.append(tool_call_chunk)
        self.stop_reason = "tool_calls"
    return result
  
  def finish(self) -> PlChunk:
    logger.debug(f"full content: {self.full_content}")
    finish_chunk = PlChunk(finish_reason=self.stop_reason)
    return finish_chunk
