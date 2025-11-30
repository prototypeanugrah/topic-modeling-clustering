"""FastAPI application factory."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from backend.api.routes import api_router
from backend.cache.manager import is_cache_complete, get_cache_status
from backend.models.responses import HealthResponse, StatusResponse


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Topic Modeling & Clustering Dashboard API",
        description="API for interactive topic modeling (LDA) and K-means clustering on 20 Newsgroups dataset",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # GZip compression for responses > 1KB (reduces payload size significantly)
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # CORS middleware for frontend development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",  # Vite dev server
            "http://localhost:3000",  # Alternative port
            "http://127.0.0.1:5173",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health check endpoint
    @app.get("/api/health", response_model=HealthResponse, tags=["health"])
    async def health_check():
        """Check API health and cache status."""
        return HealthResponse(
            status="healthy",
            cache_complete=is_cache_complete(),
        )

    # Status endpoint
    @app.get("/api/status", response_model=StatusResponse, tags=["health"])
    async def get_status():
        """Get detailed cache status."""
        status = get_cache_status()
        return StatusResponse(**status)

    # Include API routes
    app.include_router(api_router)

    return app


# Create app instance for uvicorn
app = create_app()
