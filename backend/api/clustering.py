"""Clustering API endpoints."""

from fastapi import APIRouter, HTTPException

from backend.cache.manager import load_doc_topic_distribution
from backend.core.clustering import perform_kmeans, get_cluster_sizes
from backend.core.metrics import calculate_silhouette, compute_metrics_for_all_clusters
from backend.models.requests import ClusteringRequest
from backend.models.responses import ClusteringResponse, ClusterMetricsResponse
from backend.config import MIN_TOPICS, MAX_TOPICS, MIN_CLUSTERS, MAX_CLUSTERS

router = APIRouter(prefix="/clustering", tags=["clustering"])


@router.post("", response_model=ClusteringResponse)
async def cluster_documents(request: ClusteringRequest):
    """
    Perform K-Means clustering on document-topic distributions.

    This is computed in real-time (not cached) since K-Means is fast.
    """
    distribution = load_doc_topic_distribution(request.n_topics)

    if distribution is None:
        raise HTTPException(
            status_code=503,
            detail=f"Distribution for {request.n_topics} topics not available. Run precomputation first."
        )

    # Perform clustering
    result = perform_kmeans(distribution, request.n_clusters)

    # Calculate silhouette score
    silhouette = calculate_silhouette(distribution, result.labels)

    # Get cluster sizes
    sizes = get_cluster_sizes(result.labels)

    return ClusteringResponse(
        n_topics=request.n_topics,
        n_clusters=request.n_clusters,
        labels=result.labels.tolist(),
        silhouette=silhouette,
        inertia=result.inertia,
        cluster_sizes=sizes,
    )


@router.get("/metrics/{n_topics}", response_model=ClusterMetricsResponse)
async def get_cluster_metrics(
    n_topics: int,
    min_clusters: int = MIN_CLUSTERS,
    max_clusters: int = MAX_CLUSTERS,
):
    """
    Get clustering metrics (silhouette, inertia) for a range of cluster counts.

    Used for the "optimal number of clusters" chart.
    """
    if n_topics < MIN_TOPICS or n_topics > MAX_TOPICS:
        raise HTTPException(
            status_code=400,
            detail=f"n_topics must be between {MIN_TOPICS} and {MAX_TOPICS}"
        )

    if min_clusters < MIN_CLUSTERS:
        min_clusters = MIN_CLUSTERS
    if max_clusters > MAX_CLUSTERS:
        max_clusters = MAX_CLUSTERS

    distribution = load_doc_topic_distribution(n_topics)

    if distribution is None:
        raise HTTPException(
            status_code=503,
            detail=f"Distribution for {n_topics} topics not available. Run precomputation first."
        )

    # Compute metrics for all cluster counts
    metrics = compute_metrics_for_all_clusters(
        distribution,
        min_clusters=min_clusters,
        max_clusters=max_clusters,
    )

    return ClusterMetricsResponse(
        n_topics=n_topics,
        cluster_counts=metrics["cluster_counts"],
        silhouette_scores=metrics["silhouette_scores"],
        inertia_scores=metrics["inertia_scores"],
        elbow_point=metrics["elbow_point"],
    )
