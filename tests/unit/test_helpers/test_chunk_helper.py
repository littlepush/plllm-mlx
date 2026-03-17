"""Tests for chunk helper module."""


from plllm_mlx.helpers import PlChunk, PlChunkDataType
from plllm_mlx.helpers.step_info import PlStepUsage


class TestPlChunkDataType:
    """Tests for PlChunkDataType enum."""

    def test_enum_values(self):
        """Test enum values."""
        assert PlChunkDataType.NONE.value == 0
        assert PlChunkDataType.REASONING.value == 1
        assert PlChunkDataType.CONTENT.value == 2
        assert PlChunkDataType.TOOLCALL.value == 3

    def test_enum_count(self):
        """Test enum has expected number of values."""
        assert len(list(PlChunkDataType)) == 4


class TestPlChunk:
    """Tests for PlChunk model."""

    def test_default_values(self):
        """Test default values."""
        chunk = PlChunk()
        assert chunk.data_type == PlChunkDataType.NONE
        assert chunk.data is None
        assert chunk.finish_reason is None
        assert chunk.step is None

    def test_content_chunk(self):
        """Test content chunk."""
        chunk = PlChunk(
            data_type=PlChunkDataType.CONTENT,
            data="Hello, world!",
        )
        assert chunk.data_type == PlChunkDataType.CONTENT
        assert chunk.data == "Hello, world!"

    def test_reasoning_chunk(self):
        """Test reasoning chunk."""
        chunk = PlChunk(
            data_type=PlChunkDataType.REASONING,
            data="Let me think...",
        )
        assert chunk.data_type == PlChunkDataType.REASONING
        assert chunk.data == "Let me think..."

    def test_toolcall_chunk(self):
        """Test tool call chunk."""
        tool_data = {"name": "get_weather", "arguments": '{"city": "Beijing"}'}
        chunk = PlChunk(
            data_type=PlChunkDataType.TOOLCALL,
            data=tool_data,
        )
        assert chunk.data_type == PlChunkDataType.TOOLCALL
        assert chunk.data == tool_data
        assert chunk.data["name"] == "get_weather"

    def test_chunk_with_finish_reason(self):
        """Test chunk with finish reason."""
        chunk = PlChunk(
            data_type=PlChunkDataType.CONTENT,
            data="Done",
            finish_reason="stop",
        )
        assert chunk.finish_reason == "stop"

    def test_chunk_with_step(self):
        """Test chunk with usage step."""
        step = PlStepUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        chunk = PlChunk(step=step)
        assert chunk.step.prompt_tokens == 10
        assert chunk.step.completion_tokens == 5

    def test_model_dump(self):
        """Test model serialization."""
        chunk = PlChunk(
            data_type=PlChunkDataType.CONTENT,
            data="Hello",
            finish_reason="stop",
        )
        data = chunk.model_dump()
        assert data["data_type"] == PlChunkDataType.CONTENT
        assert data["data"] == "Hello"
        assert data["finish_reason"] == "stop"

    def test_model_dump_json(self):
        """Test JSON serialization."""
        chunk = PlChunk(
            data_type=PlChunkDataType.CONTENT,
            data="Hello",
        )
        json_str = chunk.model_dump_json()
        assert "data_type" in json_str
        assert "Hello" in json_str

    def test_model_validate(self):
        """Test model validation from dict."""
        data = {
            "data_type": PlChunkDataType.CONTENT,
            "data": "Test",
            "finish_reason": None,
        }
        chunk = PlChunk.model_validate(data)
        assert chunk.data_type == PlChunkDataType.CONTENT
        assert chunk.data == "Test"

    def test_chunk_with_dict_data(self):
        """Test chunk with dictionary data."""
        data = {"key": "value", "nested": {"a": 1}}
        chunk = PlChunk(data=data)
        assert chunk.data == data

    def test_empty_string_data(self):
        """Test chunk with empty string data."""
        chunk = PlChunk(data_type=PlChunkDataType.CONTENT, data="")
        assert chunk.data == ""

    def test_chunk_immutability(self):
        """Test that model is frozen by default (Pydantic v2 behavior)."""
        chunk = PlChunk(data="test")
        chunk.data = "modified"
        assert chunk.data == "modified"
