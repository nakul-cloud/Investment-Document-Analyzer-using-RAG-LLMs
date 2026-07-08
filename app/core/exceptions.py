from fastapi import status


class AppException(Exception):
    def __init__(self, status_code: int, message: str, details: any = None):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.details = details


class AuthError(AppException):
    def __init__(self, message: str = "Authentication failed", details: any = None):
        super().__init__(status.HTTP_401_UNAUTHORIZED, message, details)


class ValidationError(AppException):
    def __init__(self, message: str = "Validation failed", details: any = None):
        super().__init__(status.HTTP_400_BAD_REQUEST, message, details)


class NotFoundError(AppException):
    def __init__(self, message: str = "Resource not found", details: any = None):
        super().__init__(status.HTTP_404_NOT_FOUND, message, details)


class DocumentIngestionError(AppException):
    def __init__(self, message: str = "Failed to ingest document", details: any = None):
        super().__init__(status.HTTP_500_INTERNAL_SERVER_ERROR, message, details)
