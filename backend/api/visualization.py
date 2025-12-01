"""Visualization API endpoints."""

from fastapi import APIRouter, HTTPException

from backend.cache.manager import load_umap_projection, load_doc_topic_distribution
from backend.core.clustering import perform_kmeans
from backend.models.requests import VisualizationRequest
from backend.models.responses import VisualizationResponse, ClusteredVisualizationResponse
from backend.config import MIN_TOPICS, MAX_TOPICS

router = APIRouter(prefix="/visualization", tags=["visualization"])


@router.get("/{n_topics}", response_model=VisualizationResponse)
async def get_visualization(n_topics: int, dataset: str = "train"):
    """
    Get UMAP 2D projections for visualization.

    Args:
        n_topics: Number of topics
        dataset: Which dataset to visualize (train or test)

    Returns pre-computed UMAP coordinates for all documents.
    """
    if n_topics < MIN_TOPICS or n_topics > MAX_TOPICS:
        raise HTTPException(
            status_code=400,
            detail=f"n_topics must be between {MIN_TOPICS} and {MAX_TOPICS}"
        )

    if dataset not in ["train", "test"]:
        raise HTTPException(
            status_code=400,
            detail="dataset must be 'train' or 'test'"
        )

    projection = load_umap_projection(n_topics, dataset)

    if projection is None:
        raise HTTPException(
            status_code=503,
            detail=f"UMAP projection for {n_topics} topics ({dataset}) not available. Run precomputation first."
        )

    # Round to 4 decimal places to reduce payload size (~30% smaller)
    rounded_projections = [[round(x, 4), round(y, 4)] for x, y in projection.tolist()]

    return VisualizationResponse(
        n_topics=n_topics,
        projections=rounded_projections,
        document_ids=list(range(len(projection))),
        dataset=dataset,
    )


@router.post("/clustered", response_model=ClusteredVisualizationResponse)
async def get_clustered_visualization(request: VisualizationRequest):
    """
    Get UMAP projections with cluster labels.

    Combines pre-computed UMAP projections with real-time K-Means clustering.
    Supports both train and test datasets.
    """
    dataset = request.dataset

    projection = load_umap_projection(request.n_topics, dataset)
    distribution = load_doc_topic_distribution(request.n_topics, dataset)

    if projection is None:
        raise HTTPException(
            status_code=503,
            detail=f"UMAP projection for {request.n_topics} topics ({dataset}) not available. Run precomputation first."
        )

    if distribution is None:
        raise HTTPException(
            status_code=503,
            detail=f"Distribution for {request.n_topics} topics ({dataset}) not available. Run precomputation first."
        )

    # Perform clustering
    result = perform_kmeans(distribution, request.n_clusters)

    # Round to 4 decimal places to reduce payload size (~30% smaller)
    rounded_projections = [[round(x, 4), round(y, 4)] for x, y in projection.tolist()]

    return ClusteredVisualizationResponse(
        n_topics=request.n_topics,
        n_clusters=request.n_clusters,
        projections=rounded_projections,
        cluster_labels=result.labels.tolist(),
        document_ids=list(range(len(projection))),
        dataset=dataset,
    )
