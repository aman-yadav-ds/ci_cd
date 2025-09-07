class CustomException(Exception):
    """Custom exception for application-specific errors."""

    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = errors

    def __str__(self):
        if self.errors:
            return f"{super().__str__()} | Details: {self.errors}"
        return super().__str__()