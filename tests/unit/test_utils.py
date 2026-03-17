"""Tests for utility functions."""


from plllm_mlx.utils import (
    format_bytes,
    format_config,
    format_number,
    parse_value,
    print_table,
)


class TestPrintTable:
    """Tests for print_table function."""

    def test_empty_rows(self, capsys):
        """Test printing empty table."""
        print_table([])
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_simple_rows(self, capsys):
        """Test printing simple rows."""
        print_table([["a", "b"], ["c", "d"]])
        captured = capsys.readouterr()
        assert "a" in captured.out
        assert "b" in captured.out

    def test_with_headers(self, capsys):
        """Test printing table with headers."""
        print_table([["1", "2"]], headers=["Col1", "Col2"])
        captured = capsys.readouterr()
        assert "Col1" in captured.out
        assert "Col2" in captured.out
        assert "-" in captured.out

    def test_uneven_columns(self, capsys):
        """Test printing table with uneven columns."""
        print_table([["a", "b", "c"]])
        captured = capsys.readouterr()
        assert "a" in captured.out


class TestFormatConfig:
    """Tests for format_config function."""

    def test_empty_config(self):
        """Test formatting empty config."""
        result = format_config({})
        assert result == "(default)"

    def test_none_config(self):
        """Test formatting None config."""
        result = format_config(None)
        assert result == "(default)"

    def test_temperature_only(self):
        """Test formatting config with temperature only."""
        result = format_config({"temperature": 0.7})
        assert "temperature=0.7" in result

    def test_max_tokens_only(self):
        """Test formatting config with max_tokens only."""
        result = format_config({"max_tokens": 2048})
        assert "max_tokens=2048" in result

    def test_top_p_only(self):
        """Test formatting config with top_p only."""
        result = format_config({"top_p": 0.9})
        assert "top_p=0.9" in result

    def test_multiple_values(self):
        """Test formatting config with multiple values."""
        result = format_config({"temperature": 0.7, "max_tokens": 2048, "top_p": 0.9})
        assert "temperature=0.7" in result
        assert "max_tokens=2048" in result
        assert "top_p=0.9" in result

    def test_ignored_keys(self):
        """Test that some keys are ignored."""
        result = format_config({"other_key": "value"})
        assert result == "(default)"


class TestParseValue:
    """Tests for parse_value function."""

    def test_parse_bool_true(self):
        """Test parsing boolean true values."""
        assert parse_value("true") is True
        assert parse_value("True") is True
        assert parse_value("TRUE") is True
        assert parse_value("yes") is True
        assert parse_value("Yes") is True
        assert parse_value("YES") is True

    def test_parse_bool_false(self):
        """Test parsing boolean false values."""
        assert parse_value("false") is False
        assert parse_value("False") is False
        assert parse_value("FALSE") is False
        assert parse_value("no") is False
        assert parse_value("No") is False
        assert parse_value("NO") is False

    def test_parse_int(self):
        """Test parsing integer values."""
        assert parse_value("123") == 123
        assert parse_value("-456") == -456
        assert parse_value("0") == 0

    def test_parse_float(self):
        """Test parsing float values."""
        assert parse_value("3.14") == 3.14
        assert parse_value("-2.5") == -2.5
        assert parse_value("0.0") == 0.0

    def test_parse_string(self):
        """Test parsing string values."""
        assert parse_value("hello") == "hello"
        assert parse_value("world123") == "world123"

    def test_int_takes_precedence_over_float(self):
        """Test that int is tried before float."""
        result = parse_value("42")
        assert result == 42
        assert isinstance(result, int)


class TestFormatBytes:
    """Tests for format_bytes function."""

    def test_bytes(self):
        """Test formatting bytes."""
        assert format_bytes(500) == "500.0 B"

    def test_kilobytes(self):
        """Test formatting kilobytes."""
        assert format_bytes(1024) == "1.0 KB"
        assert format_bytes(1536) == "1.5 KB"

    def test_megabytes(self):
        """Test formatting megabytes."""
        assert format_bytes(1048576) == "1.0 MB"
        assert format_bytes(1572864) == "1.5 MB"

    def test_gigabytes(self):
        """Test formatting gigabytes."""
        assert format_bytes(1073741824) == "1.0 GB"

    def test_terabytes(self):
        """Test formatting terabytes."""
        assert format_bytes(1099511627776) == "1.0 TB"

    def test_petabytes(self):
        """Test formatting petabytes."""
        assert format_bytes(1125899906842624) == "1.0 PB"

    def test_zero(self):
        """Test formatting zero bytes."""
        assert format_bytes(0) == "0.0 B"


class TestFormatNumber:
    """Tests for format_number function."""

    def test_small_number(self):
        """Test formatting small numbers."""
        assert format_number(0) == "0"
        assert format_number(100) == "100"
        assert format_number(999) == "999"

    def test_thousands(self):
        """Test formatting thousands."""
        assert format_number(1000) == "1,000"
        assert format_number(1234) == "1,234"

    def test_millions(self):
        """Test formatting millions."""
        assert format_number(1000000) == "1,000,000"
        assert format_number(1234567) == "1,234,567"

    def test_negative_number(self):
        """Test formatting negative numbers."""
        assert format_number(-1234) == "-1,234"
