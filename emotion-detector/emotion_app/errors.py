"""Custom error types for clear error handling."""

class InvalidTextError(ValueError):
    """Raised when input text is missing or invalid."""


class ServiceUnavailableError(RuntimeError):
    """Raised when the external service cannot be reached."""
