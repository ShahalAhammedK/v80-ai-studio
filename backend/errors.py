class UserFacingError(Exception):
    """An error whose message is safe to show directly to the browser."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
