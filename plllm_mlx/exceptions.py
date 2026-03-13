"""
Custom exceptions for plllm-mlx service.

This module defines all custom exceptions used throughout the plllm-mlx service.
All exceptions inherit from PlMlxException for unified error handling.
"""


class PlMlxException(Exception):
    """
    Base exception for all plllm-mlx exceptions.

    All custom exceptions in plllm-mlx should inherit from this class
    for unified error handling and identification.

    Attributes:
        message: Human-readable error message.
        error_code: Optional error code for programmatic handling.
    """

    def __init__(self, message: str, error_code: str | None = None) -> None:
        """
        Initialize the exception.

        Args:
            message: Human-readable error message.
            error_code: Optional error code for programmatic handling.
        """
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return string representation of the exception."""
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ModelNotFoundError(PlMlxException):
    """
    Exception raised when a model is not found.

    This exception is raised when attempting to load a model that does not exist
    or cannot be found at the specified path.
    """

    def __init__(self, model_name: str) -> None:
        """
        Initialize ModelNotFoundError.

        Args:
            model_name: Name or path of the model that was not found.
        """
        super().__init__(
            message=f"Model not found: {model_name}",
            error_code="MODEL_NOT_FOUND",
        )


class ModelLoadError(PlMlxException):
    """
    Exception raised when model loading fails.

    This exception is raised when there is an error during model loading,
    such as insufficient memory or corrupted model files.
    """

    def __init__(self, model_name: str, reason: str | None = None) -> None:
        """
        Initialize ModelLoadError.

        Args:
            model_name: Name or path of the model that failed to load.
            reason: Optional reason for the failure.
        """
        message = f"Failed to load model: {model_name}"
        if reason:
            message = f"{message}. Reason: {reason}"
        super().__init__(message=message, error_code="MODEL_LOAD_ERROR")


class ConfigurationError(PlMlxException):
    """
    Exception raised for configuration errors.

    This exception is raised when there is an error in the configuration,
    such as missing required fields or invalid values.
    """

    def __init__(self, message: str) -> None:
        """
        Initialize ConfigurationError.

        Args:
            message: Description of the configuration error.
        """
        super().__init__(message=message, error_code="CONFIG_ERROR")


class CacheError(PlMlxException):
    """
    Exception raised for cache-related errors.

    This exception is raised when there is an error with the KV cache,
    such as memory allocation failures or cache corruption.
    """

    def __init__(self, message: str) -> None:
        """
        Initialize CacheError.

        Args:
            message: Description of the cache error.
        """
        super().__init__(message=message, error_code="CACHE_ERROR")


class GenerationError(PlMlxException):
    """
    Exception raised during text generation.

    This exception is raised when there is an error during text generation,
    such as invalid parameters or generation failures.
    """

    def __init__(self, message: str) -> None:
        """
        Initialize GenerationError.

        Args:
            message: Description of the generation error.
        """
        super().__init__(message=message, error_code="GENERATION_ERROR")


class ValidationError(PlMlxException):
    """
    Exception raised for input validation errors.

    This exception is raised when input validation fails,
    such as invalid request parameters or malformed data.
    """

    def __init__(self, message: str, field: str | None = None) -> None:
        """
        Initialize ValidationError.

        Args:
            message: Description of the validation error.
            field: Optional field name that failed validation.
        """
        if field:
            message = f"Validation error in '{field}': {message}"
        super().__init__(message=message, error_code="VALIDATION_ERROR")
