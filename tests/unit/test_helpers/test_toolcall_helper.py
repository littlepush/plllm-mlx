"""Tests for tool call helper module."""


from plllm_mlx.helpers import PlCommonToolcallParser
from plllm_mlx.helpers.chunk_helper import PlChunkDataType


class TestPlCommonToolcallParser:
    """Tests for PlCommonToolcallParser function."""

    def test_empty_buffer(self):
        """Test parsing empty buffer."""
        result = PlCommonToolcallParser([])
        assert result is None

    def test_whitespace_only(self):
        """Test parsing whitespace only."""
        result = PlCommonToolcallParser(["   ", "  "])
        assert result is None

    def test_json_format_basic(self):
        """Test parsing basic JSON format."""
        buffer = ['{"name": "get_weather", "arguments": "{\\"city\\": \\"Beijing\\"}"}']
        result = PlCommonToolcallParser(buffer)
        assert result is not None
        assert result.data_type == PlChunkDataType.TOOLCALL
        assert result.data["name"] == "get_weather"
        assert result.data["arguments"] == '{"city": "Beijing"}'

    def test_json_format_with_parameters(self):
        """Test parsing JSON format with parameters field."""
        buffer = ['{"name": "test_tool", "parameters": {"key": "value"}}']
        result = PlCommonToolcallParser(buffer)
        assert result is not None
        assert result.data["name"] == "test_tool"
        assert "key" in result.data["arguments"]

    def test_json_format_missing_name(self):
        """Test parsing JSON format without name field."""
        buffer = ['{"arguments": "{}"}']
        result = PlCommonToolcallParser(buffer)
        assert result is None

    def test_json_format_invalid_json(self):
        """Test parsing invalid JSON."""
        buffer = ['{"name": "test", invalid}']
        result = PlCommonToolcallParser(buffer)
        assert result is None

    def test_json_format_split_buffer(self):
        """Test parsing JSON from split buffer."""
        buffer = ['{"name": "', "get_weather", '", "arguments": "{}"}']
        result = PlCommonToolcallParser(buffer)
        assert result is not None
        assert result.data["name"] == "get_weather"

    def test_xml_format_basic(self):
        """Test parsing XML format."""
        buffer = [
            "<function=get_weather><parameter=city>Beijing</parameter></function>"
        ]
        result = PlCommonToolcallParser(buffer)
        assert result is not None
        assert result.data_type == PlChunkDataType.TOOLCALL
        assert result.data["name"] == "get_weather"

    def test_xml_format_multiple_parameters(self):
        """Test parsing XML format with multiple parameters."""
        buffer = [
            "<function=search>",
            "<parameter=query>hello</parameter>",
            "<parameter=limit>10</parameter>",
            "</function>",
        ]
        result = PlCommonToolcallParser(buffer)
        assert result is not None
        assert result.data["name"] == "search"

    def test_xml_format_no_parameters(self):
        """Test parsing XML format without parameters."""
        buffer = ["<function=ping></function>"]
        result = PlCommonToolcallParser(buffer)
        assert result is not None
        assert result.data["name"] == "ping"
        assert result.data["arguments"] == "{}"

    def test_xml_format_incomplete(self):
        """Test parsing incomplete XML format."""
        buffer = ["<function=test"]
        result = PlCommonToolcallParser(buffer)
        assert result is None

    def test_xml_format_no_function_tag(self):
        """Test parsing string without function tag."""
        buffer = ["some random text"]
        result = PlCommonToolcallParser(buffer)
        assert result is None

    def test_unknown_format(self):
        """Test parsing unknown format."""
        buffer = ["some text without any format"]
        result = PlCommonToolcallParser(buffer)
        assert result is None

    def test_json_dict_arguments(self):
        """Test JSON with dict arguments (not string)."""
        buffer = ['{"name": "test", "arguments": {"key": "value"}}']
        result = PlCommonToolcallParser(buffer)
        assert result is not None
        assert result.data["name"] == "test"
        assert isinstance(result.data["arguments"], str)

    def test_chunk_is_plchunk(self):
        """Test that result is a PlChunk instance."""
        buffer = ['{"name": "test", "arguments": "{}"}']
        result = PlCommonToolcallParser(buffer)
        assert result is not None
        from plllm_mlx.helpers import PlChunk

        assert isinstance(result, PlChunk)

    def test_json_with_special_characters(self):
        """Test JSON with special characters in arguments."""
        buffer = ['{"name": "test", "arguments": "{\\"msg\\": \\"Hello, World!\\"}"}']
        result = PlCommonToolcallParser(buffer)
        assert result is not None
        assert result.data["name"] == "test"

    def test_empty_arguments(self):
        """Test JSON with empty arguments."""
        buffer = ['{"name": "test", "arguments": ""}']
        result = PlCommonToolcallParser(buffer)
        assert result is not None
        assert result.data["name"] == "test"
        assert result.data["arguments"] == ""

    def test_json_arguments_as_dict(self):
        """Test JSON where arguments is already a dict."""
        buffer = ['{"name": "test", "arguments": {"nested": {"key": "value"}}}']
        result = PlCommonToolcallParser(buffer)
        assert result is not None
        assert result.data["name"] == "test"
        assert isinstance(result.data["arguments"], str)
