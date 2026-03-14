from plllm_mlx.logging_config import get_logger
logger = get_logger(__name__)
from plllm_mlx.helpers import *
from plllm_mlx.models.base_step_processor import PlStepProcessor
from typing import Any, Optional, List
import re
import json

class PlOpenAIStepProcessor(PlStepProcessor):
  def __init__(self):
    super().__init__()
    self.channel_buffer = {}
    self.current_channel_name = None
    self.current_role = "assistant"
    self.current_key_in_channel = None
    self.is_waiting_for_role = False
    self.is_waiting_for_channel_name = False
    self.is_stop_by_length = False
    self.full_content = ""
    self.stop_reason = ""
    
  @staticmethod
  def step_clz_name():
    return "openai"
  
  def step(self, generate_response:Any) -> Optional[PlChunk]:
    gr = generate_response
    self.total_tokens += 1
    try:
      self.full_content += gr.text
      # Just stop the generation
      if gr.finish_reason is not None:
        self.is_stop_by_length = True if gr.finish_reason == "length" else False
        self.stop()
        return None
      
      if gr.text == "<|end|>":
        if self.current_channel_name == "final":
          self.stop()
          return None
        self.current_channel_name = None
        self.current_key_in_channel = None
        return None
      if gr.text == "<|start|>":
        # wait for a role
        self.is_waiting_for_role = True
        return None
      if self.is_waiting_for_role and not gr.text.startswith("<|"):
        self.current_role = gr.text
        self.is_waiting_for_role = False
        return None
      if gr.text == "<|channel|>":
        self.is_waiting_for_channel_name = True
        self.current_channel_name = ""
        return None
      if self.is_waiting_for_channel_name:
        if gr.text.startswith("<|") and gr.text.endswith("|>"):
          self.is_waiting_for_channel_name = False
          # end of channel name, save to buffer cache
          self.current_channel_name = self.current_channel_name.strip()
          self.current_key_in_channel = re.sub(r"<\|([^|]+)\|>", r"\1", gr.text)
          self.channel_buffer[self.current_channel_name] = {
            "roll": self.current_role,
            "channel": self.current_channel_name,
            self.current_key_in_channel: ""
          }
        else:
          self.current_channel_name += gr.text
        return None
      
      strip_token = gr.text.strip()
      if strip_token.startswith("<|"):
        if not strip_token.endswith("|>"):
          logger.debug(f"get specifial flag but is not validate: gr.text: [{gr.text}], strip_token: [{strip_token}]")
      if strip_token.startswith("<|") and strip_token.endswith("|>"):
        # find a new channel
        self.current_key_in_channel = re.sub(r"<\|([^|]+)\|>", r"\1", gr.text)
        self.channel_buffer[self.current_channel_name][self.current_key_in_channel] = ""

      # prepare response chunk
      sr = PlStepUsage()
      sr.prompt_tokens = gr.prompt_tokens
      sr.completion_tokens = gr.generation_tokens
      sr.total_tokens = gr.prompt_tokens + gr.generation_tokens
      sr.prompt_tps = gr.prompt_tps
      sr.generation_tps = gr.generation_tps
      chunk = PlChunk(step=sr)
      
      if self.current_channel_name == "analysis":
        if self.current_key_in_channel == "message":
          chunk.data = gr.text
          chunk.data_type = PlChunkDataType.REASONING
          return chunk
        else:
          logger.error(f"get token in channel analysis, but not in key message: {self.current_channel_name}.{self.current_key_in_channel}: {gr.text}")
        return None
      
      if self.current_channel_name == "final":
        if self.current_key_in_channel == "message":
          chunk.data = gr.text
          chunk.data_type = PlChunkDataType.CONTENT
          return chunk
        else:
          logger.error(f"get token in channel final, but not in key message: {self.current_channel_name}.{self.current_key_in_channel}: {gr.text}")
        return None
      
      if self.current_channel_name is None or self.current_key_in_channel is None:
        return None
      
      self.channel_buffer[self.current_channel_name][self.current_key_in_channel] += gr.text
      return None
    
    except Exception as e:
      logger.debug(f"Step error: {str(e)}")
      logger.debug(f"full content: {self.full_content}")
      logger.debug(f"new token: {gr.text}")
      logger.debug(f"self status: {json.dumps({
        "current_channel_name": self.current_channel_name,
        "current_role": self.current_role,
        "current_key_in_channel": self.current_key_in_channel,
        "is_waiting_for_role": self.is_waiting_for_role,
        "is_waiting_for_channel_name": self.is_waiting_for_channel_name,
        "channel_buffer": self.channel_buffer
      })}")
      self.stop()
      return None
  
  def tool_calls(self) -> List[PlChunk]:
    self.stop_reason = "length" if self.is_stop_by_length else "stop"
    logger.debug(f"channel_buffer: {json.dumps(self.channel_buffer, ensure_ascii=False)}")
    result = []
    for channel, data in self.channel_buffer.items():
      if channel.startswith("commentary to="):
        tool_call = {
          "name": channel[channel.find('=') + 1:],
          "parameters": data.get("message", "")
        }
        chunk = PlChunk(data=tool_call, data_type=PlChunkDataType.TOOLCALL)
        logger.debug(f"get to tool call: {tool_call}")
        result.append(chunk)
        self.stop_reason = "tool_calls"
    return result
  
  def finish(self) -> PlChunk:
    logger.debug(f"full content: {self.full_content}")
    finish_chunk = PlChunk(finish_reason=self.stop_reason)
    return finish_chunk

# Register the step processor
PlStepProcessor.registerStepProcessor("openai", PlOpenAIStepProcessor)
