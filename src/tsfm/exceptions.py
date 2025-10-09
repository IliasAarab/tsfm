class TSFMError(Exception):
    """Base class for exceptions in TSFM models."""


class InvalidInputError(TSFMError):
    """Raised when input data is invalid for a specific model."""

    def __init__(self, message: str):
        super().__init__(message)
