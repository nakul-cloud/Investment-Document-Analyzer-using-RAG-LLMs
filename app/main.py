import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.exceptions import AppException
from app.core.logging import logger
from app.db.session import engine, Base
from app.api.v1.router import api_router
from app.middleware.rate_limiter import RateLimitMiddleware
from app.middleware.request_logging import RequestLoggingMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup Actions
    logger.info("Initializing database tables...")
    async with engine.begin() as conn:
        # Create all tables asynchronously
        await conn.run_sync(Base.metadata.create_all)
        
    logger.info("Pre-loading/warming sentence embedding model...")
    from app.rag.embeddings import get_embedding_model
    get_embedding_model()
    
    logger.info("Application startup completed successfully.")
    yield
    # Shutdown Actions
    logger.info("Application shutting down.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom Middlewares
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RateLimitMiddleware)

# Mount unified API routes
app.mount(settings.API_V1_STR, api_router)


# Global Exception Handlers
@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    """Custom application exception mapper."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "msg": exc.message}
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Custom mapper for FastAPI Pydantic schema validation failures."""
    errors = exc.errors()
    msg = "Validation failed"
    if errors:
        error_loc = " -> ".join(str(x) for x in errors[0]["loc"])
        msg = f"Validation failed: {errors[0]['msg']} ({error_loc})"
    return JSONResponse(
        status_code=400,
        content={"status": "error", "msg": msg}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Fallback handler for unhandled generic system errors."""
    logger.exception("An unhandled system exception occurred:")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "msg": "An unexpected internal server error occurred."
        }
    )


# Root Frontend Route
@app.get("/", response_class=HTMLResponse)
async def serve_home():
    """Serves the main single page dashboard interface."""
    template_path = os.path.join("templates", "index.html")
    if not os.path.exists(template_path):
        return HTMLResponse("index.html template not found in templates directory", status_code=404)
        
    with open(template_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)
