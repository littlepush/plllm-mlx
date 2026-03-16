"""
Response helper utilities for MLX model inference.

This module provides utilities for extracting finish reasons and
other response data from MLX generation results.
"""

from __future__ import annotations

from typing import Any, Optional


def get_finish_reason(gr: Any) -> Optional[str]:
    """
    Get the finish reason from a generation result.

    Args:
        gr: The generation result object from MLX.

    Returns:
        The finish reason string if available, None otherwise.
    """
    if getattr(gr, "finish_reason", None) is not None:
        return gr.finish_reason
    return None


# Alias for backward compatibility
PlMlxGetFinishReason = get_finish_reason
