"""API route aggregation."""

from fastapi import APIRouter

from backend.api.topics import router as topics_router
from backend.api.clustering import router as clustering_router
from backend.api.gmm import router as gmm_router
from backend.api.visualization import router as visualization_router
from backend.api.precompute import router as precompute_router
from backend.api.eda import router as eda_router

# Main API router
api_router = APIRouter(prefix="/api")

# Include all routers
api_router.include_router(topics_router)
api_router.include_router(clustering_router)
api_router.include_router(gmm_router)
api_router.include_router(visualization_router)
api_router.include_router(precompute_router)
api_router.include_router(eda_router)
