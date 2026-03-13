"""
Tool call parsing utilities.

This module provides parsers for extracting tool/function calls
from LLM outputs, supporting both JSON format and XML-style format.
"""

import json
from typing import Optional

from .chunk_helper import PlChunk, PlChunkDataType
from plllm_mlx.logging_config import get_logger

logger = get_logger(__name__)


def PlCommonToolcallParser(toolcall_content_buffer: list) -> Optional[PlChunk]:
    """
    Parse tool call content from a buffer of string fragments.

    This function attempts to parse tool calls in two formats:
    1. JSON format: `{"name": "tool_name", "arguments": {...}}`
    2. XML-style format: `<function=tool_name><parameter=name>value</parameter>...`

    Args:
        toolcall_content_buffer: List of string fragments that form the tool call content.

    Returns:
        PlChunk with TOOLCALL data type if parsing succeeds, None otherwise.

    Example:
        >>> buffer = ['{"name": "get_weather", "arguments": "{\\"city\\": \\"Beijing\\"}"}']
        >>> chunk = PlCommonToolcallParser(buffer)
        >>> chunk.data
        {'name': 'get_weather', 'arguments': '{"city": "Beijing"}'}
    """
    string_content = "".join(toolcall_content_buffer).strip()

    if not string_content:
        return None

    # Try JSON format
    if string_content[0] == "{":
        logger.debug(f"Parsing JSON tool call: {string_content}")
        try:
            json_toolcall = json.loads(string_content)
            tool_name = json_toolcall.get("name", None)
            if tool_name is None:
                logger.error(f"Tool call JSON missing 'name' field: {string_content}")
                return None
            tool_parameters = json_toolcall.get("arguments", None)
            if tool_parameters is None:
                tool_parameters = json_toolcall.get("parameters", "")
            if not isinstance(tool_parameters, str):
                tool_parameters = json.dumps(tool_parameters, ensure_ascii=False)
            logger.debug(
                f"Extracted tool call - name: {tool_name}, parameters: {tool_parameters}"
            )
            tool_call = {"name": tool_name, "arguments": tool_parameters}
            chunk = PlChunk(data=tool_call, data_type=PlChunkDataType.TOOLCALL)
            return chunk
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse tool call JSON: {string_content}, error: {str(e)}"
            )
            return None

    # Try XML-style format
    s = string_content
    fs = s.find("<function=")
    if fs == -1:
        return None
    fe = s.find(">", fs)
    if fe == -1:
        return None
    name = s[fs + len("<function=") : fe]
    args = {}
    pos = 0
    while True:
        ps = s.find("<parameter=", pos)
        if ps == -1:
            break
        pe = s.find(">", ps)
        if pe == -1:
            break
        pname = s[ps + len("<parameter=") : pe]
        pval_start = pe + 1
        pve = s.find("<" + "/parameter>", pval_start)
        if pve == -1:
            break
        pval = s[pval_start:pve].strip()
        args[pname] = pval
        pos = pve + len("<" + "/parameter>")
    tool_call = {"name": name, "arguments": json.dumps(args) if args else "{}"}
    logger.debug(
        f"Parsed non-JSON tool call - name: {name}, arguments: {tool_call['arguments']}"
    )
    chunk = PlChunk(data=tool_call, data_type=PlChunkDataType.TOOLCALL)
    return chunk
