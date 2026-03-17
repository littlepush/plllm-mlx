"""Tests for chat helper module."""

import time


from plllm_mlx.helpers import PlChatCompletionHelper
from plllm_mlx.helpers.step_info import PlStepUsage


class TestPlChatCompletionHelper:
    """Tests for PlChatCompletionHelper."""

    def test_init(self):
        """Test initialization."""
        helper = PlChatCompletionHelper("test-model")
        assert helper._model_name == "test-model"
        assert helper._is_finished is False
        assert helper._finish_reason is None
        assert helper._is_first_chunk is True

    def test_init_with_include_usage(self):
        """Test initialization with include_usage flag."""
        helper = PlChatCompletionHelper("test-model", include_usage=True)
        assert helper._include_usage is True

    def test_prompt_processed(self):
        """Test prompt_processed method."""
        helper = PlChatCompletionHelper("test-model")
        helper.prompt_processed()
        assert helper._step_helper._prompt_processed is True

    def test_update_reason_step_first_chunk(self):
        """Test update_reason_step for first chunk."""
        helper = PlChatCompletionHelper("test-model")
        helper.update_reason_step("thinking...")
        assert helper._last_delta == {"reasoning": "thinking...", "role": "assistant"}
        assert helper._is_first_chunk is False

    def test_update_reason_step_subsequent(self):
        """Test update_reason_step for subsequent chunks."""
        helper = PlChatCompletionHelper("test-model")
        helper.update_reason_step("thinking...")
        helper.update_reason_step("more thinking...")
        assert helper._last_delta == {"reasoning": "more thinking..."}
        assert "role" not in helper._last_delta

    def test_update_content_step_first_chunk(self):
        """Test update_content_step for first chunk."""
        helper = PlChatCompletionHelper("test-model")
        helper.update_content_step("Hello")
        assert helper._last_delta == {"content": "Hello", "role": "assistant"}
        assert helper._is_first_chunk is False

    def test_update_content_step_subsequent(self):
        """Test update_content_step for subsequent chunks."""
        helper = PlChatCompletionHelper("test-model")
        helper.update_content_step("Hello")
        helper.update_content_step(" World")
        assert helper._last_delta == {"content": " World"}

    def test_update_tool_step(self):
        """Test update_tool_step."""
        helper = PlChatCompletionHelper("test-model")
        tool_call = {"name": "get_weather", "arguments": '{"city": "Beijing"}'}
        helper.update_tool_step(tool_call)
        assert "tool_calls" in helper._last_delta
        assert helper._last_delta["tool_calls"][0]["function"] == tool_call
        assert helper._last_delta["role"] == "assistant"

    def test_update_step_with_usage(self):
        """Test update methods with step usage."""
        helper = PlChatCompletionHelper("test-model")
        step = PlStepUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        helper.update_content_step("Hello", step=step)
        assert helper._step_helper._last_step_usage.prompt_tokens == 10

    def test_finish_step(self):
        """Test finish_step method."""
        helper = PlChatCompletionHelper("test-model")
        helper.finish_step("stop")
        assert helper._is_finished is True
        assert helper._finish_reason == "stop"

    def test_finish_step_tool_calls_keeps_delta(self):
        """Test finish_step with tool_calls keeps delta."""
        helper = PlChatCompletionHelper("test-model")
        tool_call = {"name": "test", "arguments": "{}"}
        helper.update_tool_step(tool_call)
        helper.finish_step("tool_calls")
        assert helper._last_delta != {}

    def test_finish_step_stop_clears_delta(self):
        """Test finish_step with stop clears delta."""
        helper = PlChatCompletionHelper("test-model")
        helper.update_content_step("Hello")
        helper.finish_step("stop")
        assert helper._last_delta == {}

    def test_no_update_after_finish(self):
        """Test that updates are ignored after finish."""
        helper = PlChatCompletionHelper("test-model")
        helper.finish_step("stop")
        helper.update_content_step("Should be ignored")
        assert helper._last_delta == {}

    def test_build_chunk(self):
        """Test build_chunk method."""
        helper = PlChatCompletionHelper("test-model")
        helper.update_content_step("Hello")
        chunk = helper.build_chunk()
        assert "id" in chunk
        assert chunk["id"].startswith("chatcmpl-")
        assert chunk["object"] == "chat.completion.chunk"
        assert chunk["model"] == "test-model"
        assert "choices" in chunk
        assert len(chunk["choices"]) == 1
        assert chunk["choices"][0]["delta"]["content"] == "Hello"

    def test_build_chunk_with_finish_reason(self):
        """Test build_chunk with finish reason."""
        helper = PlChatCompletionHelper("test-model")
        helper.update_content_step("Hello")
        helper.finish_step("stop")
        chunk = helper.build_chunk()
        assert chunk["choices"][0]["finish_reason"] == "stop"

    def test_build_chunk_with_usage(self):
        """Test build_chunk with usage statistics."""
        helper = PlChatCompletionHelper("test-model", include_usage=True)
        step = PlStepUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        helper.update_content_step("Hello", step=step)
        helper.finish_step("stop")
        chunk = helper.build_chunk()
        assert "usage" in chunk
        assert chunk["usage"]["prompt_tokens"] == 10
        assert chunk["usage"]["completion_tokens"] == 5

    def test_build_chunk_without_usage_when_not_finished(self):
        """Test build_chunk doesn't include usage when not finished."""
        helper = PlChatCompletionHelper("test-model", include_usage=True)
        step = PlStepUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        helper.update_content_step("Hello", step=step)
        chunk = helper.build_chunk()
        assert "usage" not in chunk

    def test_build_text(self):
        """Test build_text method."""
        helper = PlChatCompletionHelper("test-model")
        helper.update_content_step("Hello")
        text_chunk = helper.build_text()
        assert text_chunk["id"].startswith("cmpl-")
        assert text_chunk["object"] == "text_completion.chunk"
        assert text_chunk["choices"][0]["text"] == "Hello"

    def test_build_yield_chunk_string(self):
        """Test build_yield_chunk returns SSE string."""
        helper = PlChatCompletionHelper("test-model")
        helper.update_content_step("Hello")
        result = helper.build_yield_chunk()
        assert isinstance(result, str)
        assert result.startswith("data: ")
        assert "\n\n" in result

    def test_build_yield_chunk_dict(self):
        """Test build_yield_chunk returns dict with direct_json=True."""
        helper = PlChatCompletionHelper("test-model")
        helper.update_content_step("Hello")
        result = helper.build_yield_chunk(direct_json=True)
        assert isinstance(result, dict)
        assert result["model"] == "test-model"

    def test_build_yield_text_string(self):
        """Test build_yield_text returns SSE string."""
        helper = PlChatCompletionHelper("test-model")
        helper.update_content_step("Hello")
        result = helper.build_yield_text()
        assert isinstance(result, str)
        assert result.startswith("data: ")

    def test_build_yield_text_dict(self):
        """Test build_yield_text returns dict with direct_json=True."""
        helper = PlChatCompletionHelper("test-model")
        helper.update_content_step("Hello")
        result = helper.build_yield_text(direct_json=True)
        assert isinstance(result, dict)
        assert result["object"] == "text_completion.chunk"

    def test_full_chat_completion_flow(self):
        """Test a full chat completion flow."""
        helper = PlChatCompletionHelper("test-model", include_usage=True)
        helper.prompt_processed()
        helper.update_content_step("Hello")
        chunk1 = helper.build_yield_chunk(direct_json=True)
        assert chunk1["choices"][0]["delta"]["content"] == "Hello"
        helper.update_content_step(" World")
        chunk2 = helper.build_yield_chunk(direct_json=True)
        assert chunk2["choices"][0]["delta"]["content"] == " World"
        step = PlStepUsage(prompt_tokens=5, completion_tokens=2, total_tokens=7)
        helper.finish_step("stop")
        helper._step_helper.update_step(step)
        final_chunk = helper.build_yield_chunk(direct_json=True)
        assert final_chunk["choices"][0]["finish_reason"] == "stop"
        assert "usage" in final_chunk

    def test_reasoning_then_content_flow(self):
        """Test reasoning followed by content."""
        helper = PlChatCompletionHelper("test-model")
        helper.update_reason_step("Let me think...")
        chunk1 = helper.build_chunk()
        assert chunk1["choices"][0]["delta"]["reasoning"] == "Let me think..."
        helper.update_content_step("The answer is...")
        chunk2 = helper.build_chunk()
        assert chunk2["choices"][0]["delta"]["content"] == "The answer is..."

    def test_chunk_id_consistency(self):
        """Test that chunk IDs are consistent within a completion."""
        helper = PlChatCompletionHelper("test-model")
        helper.update_content_step("Hello")
        chunk1 = helper.build_chunk()
        helper.update_content_step(" World")
        chunk2 = helper.build_chunk()
        assert chunk1["id"] == chunk2["id"]

    def test_chunk_timestamps(self):
        """Test that chunk timestamps are valid."""
        helper = PlChatCompletionHelper("test-model")
        before = int(time.time())
        helper.update_content_step("Hello")
        chunk = helper.build_chunk()
        after = int(time.time())
        assert before <= chunk["created"] <= after
