"""
Utility functions for plllm-mlx CLI.
"""

from __future__ import annotations

from typing import Any, List


def print_table(rows: List[List[str]], headers: List[str] = None):
    """
    Print a formatted table.

    Args:
        rows: Table rows (each row is a list of strings).
        headers: Optional headers.
    """
    if not rows:
        return

    # Add headers if provided
    if headers:
        rows = [headers] + rows

    # Calculate column widths
    if not rows:
        return

    num_cols = len(rows[0])
    col_widths = [0] * num_cols

    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    # Print table
    for i, row in enumerate(rows):
        # Format cells with padding
        cells = []
        for j, cell in enumerate(row):
            cells.append(str(cell).ljust(col_widths[j]))

        line = "  ".join(cells)
        print(line)

        # Print separator after headers
        if headers and i == 0:
            separator = "  ".join("-" * w for w in col_widths)
            print(separator)


def format_config(config: Dict[str, Any]) -> str:
    """
    Format config dict for display.

    Args:
        config: Config dictionary.

    Returns:
        Formatted string.
    """
    if not config:
        return "(default)"

    parts = []
    for key, value in config.items():
        if key in ["temperature", "max_tokens", "top_p"]:
            parts.append(f"{key}={value}")

    return ", ".join(parts) if parts else "(default)"


def parse_value(value_str: str) -> Any:
    """
    Parse string value to appropriate type.

    Args:
        value_str: String value.

    Returns:
        Parsed value (int, float, bool, or str).
    """
    # Try bool
    if value_str.lower() in ["true", "yes"]:
        return True
    if value_str.lower() in ["false", "no"]:
        return False

    # Try int
    try:
        return int(value_str)
    except ValueError:
        pass

    # Try float
    try:
        return float(value_str)
    except ValueError:
        pass

    # Return as string
    return value_str


def format_bytes(size: int) -> str:
    """
    Format bytes to human readable string.

    Args:
        size: Size in bytes.

    Returns:
        Formatted string (e.g., "1.5 GB").
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def format_number(num: int) -> str:
    """
    Format number with commas.

    Args:
        num: Number.

    Returns:
        Formatted string (e.g., "1,234,567").
    """
    return f"{num:,}"
