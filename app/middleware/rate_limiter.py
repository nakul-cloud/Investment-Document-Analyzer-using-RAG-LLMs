import time
from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from app.core.config import settings
from app.core.logging import logger


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.calls_per_minute = settings.RATE_LIMIT_CALLS
        self.ip_records = {}  # Map of client_ip -> list of epoch timestamps

    async def dispatch(self, request: Request, call_next):
        # Exclude API documentation and static assets from rate limits
        path = request.url.path
        if path.startswith(("/docs", "/redoc", "/openapi.json", "/static")):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        # Initialize list for IP if not present
        if client_ip not in self.ip_records:
            self.ip_records[client_ip] = []

        # Filter out requests that are older than 60 seconds (1 minute window)
        self.ip_records[client_ip] = [
            timestamp for timestamp in self.ip_records[client_ip] if now - timestamp < 60
        ]

        # Check limit threshold
        if len(self.ip_records[client_ip]) >= self.calls_per_minute:
            logger.warning(f"Rate limit triggered for IP {client_ip} on path {path}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "status": "error",
                    "msg": "Rate limit exceeded. Please wait a moment before sending more requests."
                }
            )

        # Record this hit
        self.ip_records[client_ip].append(now)
        return await call_next(request)
