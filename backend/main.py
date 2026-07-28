import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.v1.auth import router as auth_router
from backend.api.v1.conversations import router as conversations_router
from backend.api.v1.documents import router as documents_router
from backend.api.v1.graph import router as graph_router
from backend.api.v1.members import router as members_router
from backend.api.v1.reports import router as reports_router
from backend.api.v1.search import router as search_router
from backend.api.v1.versions import router as versions_router
from backend.api.v1.workspaces import router as workspaces_router
from backend.core.health import router as health_router
from backend.database.connection import init_db
from config.settings import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ekn.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Handle application startup and shutdown lifecycle."""
    settings = get_settings()
    logger.info("Starting %s in %s environment", settings.app_name, settings.environment)

    # Security check: Refuse to start in production with unsafe default signing secret
    if settings.environment.lower() == "production":
        default_secret = "enterprise-knowledge-navigator-jwt-secret-key-32bytes-min"
        if not settings.jwt_secret_key or settings.jwt_secret_key == default_secret or len(settings.jwt_secret_key) < 32:
            raise RuntimeError(
                "CRITICAL SECURITY CONFIGURATION: Production mode requires a secure, non-default "
                "JWT_SECRET_KEY of at least 32 characters. Run 'python -m backend.cli generate-secret' "
                "to generate a strong key and set it in your environment."
            )

    # Initialize database tables if database engine is configured
    try:
        init_db()
    except Exception as exc:
        logger.warning("Could not auto-initialize database on startup: %s", exc)
    yield
    logger.info("Shutting down %s", settings.app_name)


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="Enterprise Document Intelligence Platform backend API",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS configuration for local development and authorized frontend origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.effective_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Security Headers Middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Enforce standard enterprise security headers on all HTTP responses."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


# Exception handler for unhandled server exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception processing %s %s: %s", request.method, request.url.path, exc, exc_info=True)
    # Mask raw exception message in production/non-debug environments to prevent information disclosure
    safe_message = str(exc) if settings.debug and settings.environment.lower() != "production" else "An unexpected error occurred. Please contact system support."
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An internal server error occurred.",
            "error_type": exc.__class__.__name__ if settings.debug else "InternalServerError",
            "message": safe_message,
        },
    )


# Include health and readiness routes at root level
app.include_router(health_router)

# Include v1 Authentication and Member routers
app.include_router(auth_router)
app.include_router(members_router)

# Include v1 Workspace, Document, Search, and Conversation routers
app.include_router(workspaces_router)
app.include_router(documents_router)
app.include_router(search_router)
app.include_router(conversations_router)
app.include_router(graph_router)
app.include_router(reports_router)
app.include_router(versions_router)


@app.get("/", summary="Root Index", tags=["Health & Readiness"])
def root_index() -> dict[str, str]:
    """Root endpoint providing service information and core navigation links."""
    return {
        "service": settings.app_name,
        "environment": settings.environment,
        "health_check": "/health",
        "readiness_check": "/ready",
        "workspaces": "/api/v1/workspaces",
        "documentation": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )
