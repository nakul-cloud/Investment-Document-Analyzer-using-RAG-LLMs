import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from app.core.logging import logger


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()
        
        # Process request
        response = await call_next(request)
        
        process_time = (time.perf_counter() - start_time) * 1000
        client_ip = request.client.host if request.client else "unknown"
        
        logger.info(
            f"Client: {client_ip} | {request.method} {request.url.path} "
            f"| Status: {response.status_code} | Duration: {process_time:.2f}ms"
        )
        
        return response
