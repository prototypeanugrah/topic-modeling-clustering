"""Visualization API endpoints."""

import numpy as np
from fastapi import APIRouter, HTTPException

from backend.cache.manager import (
    load_umap_projection,
    load_doc_topic_distribution,
    load_document_labels,
    load_lda_model,
    load_cluster_labels,
    load_document_enrichment,
    load_gmm_labels,
    load_gmm_probabilities,
)
from backend.core.clustering import perform_kmeans
from backend.core.gmm import perform_gmm
from backend.models.requests import VisualizationRequest, GMMVisualizationRequest
from backend.models.responses import (
    VisualizationResponse,
    ClusteredVisualizationResponse,
    DocumentTopicInfo,
    GMMClusteredVisualizationResponse,
    ClusterProbability,
)
from backend.config import MIN_TOPICS, MAX_TOPICS

router = APIRouter(prefix="/visualization", tags=["visualization"])


def _compute_cluster_centers_2d(
    projections: np.ndarray, labels: np.ndarray, n_clusters: int
) -> list[list[float]]:
    """
    Compute cluster centers in 2D UMAP space.

    Args:
        projections: UMAP projections of shape (n_samples, 2)
        labels: Cluster labels for each sample
        n_clusters: Number of clusters

    Returns:
        List of [x, y] coordinates for each cluster center
    """
    centers = []
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        if np.any(mask):
            center = projections[mask].mean(axis=0)
            centers.append([round(float(center[0]), 4), round(float(center[1]), 4)])
        else:
            # Empty cluster - use origin as fallback
            centers.append([0.0, 0.0])
    return centers


def _compute_cluster_covariances_2d(
    projections: np.ndarray, labels: np.ndarray, n_clusters: int
) -> list[list[list[float]]]:
    """
    Compute cluster covariance matrices in 2D UMAP space.

    Args:
        projections: UMAP projections of shape (n_samples, 2)
        labels: Cluster labels for each sample
        n_clusters: Number of clusters

    Returns:
        List of 2x2 covariance matrices for each cluster
    """
    covariances = []
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        if np.sum(mask) > 2:  # Need at least 3 points for meaningful covariance
            cluster_points = projections[mask]
            cov = np.cov(cluster_points.T)
            # Ensure it's 2x2 (np.cov returns scalar for 1D)
            if cov.ndim == 0:
                cov = np.array([[float(cov), 0], [0, float(cov)]])
            covariances.append([
                [round(float(cov[0, 0]), 6), round(float(cov[0, 1]), 6)],
                [round(float(cov[1, 0]), 6), round(float(cov[1, 1]), 6)],
            ])
        else:
            # Not enough points - use identity matrix as fallback
            covariances.append([[1.0, 0.0], [0.0, 1.0]])
    return covariances


@router.get("/{n_topics}", response_model=VisualizationResponse)
async def get_visualization(n_topics: int):
    """
    Get UMAP 2D projections for visualization.

    Args:
        n_topics: Number of topics

    Returns pre-computed UMAP coordinates for all documents.
    """
    if n_topics < MIN_TOPICS or n_topics > MAX_TOPICS:
        raise HTTPException(
            status_code=400,
            detail=f"n_topics must be between {MIN_TOPICS} and {MAX_TOPICS}"
        )

    projection = load_umap_projection(n_topics)

    if projection is None:
        raise HTTPException(
            status_code=503,
            detail=f"UMAP projection for {n_topics} topics not available. Run precomputation first."
        )

    # Round to 4 decimal places to reduce payload size (~30% smaller)
    rounded_projections = [[round(x, 4), round(y, 4)] for x, y in projection.tolist()]

    return VisualizationResponse(
        n_topics=n_topics,
        projections=rounded_projections,
        document_ids=list(range(len(projection))),
    )


@router.post("/clustered", response_model=ClusteredVisualizationResponse)
async def get_clustered_visualization(request: VisualizationRequest):
    """
    Get UMAP projections with cluster labels, newsgroup labels, and topic info.

    Uses precomputed data when available for fast response times.
    """
    projection = load_umap_projection(request.n_topics)

    if projection is None:
        raise HTTPException(
            status_code=503,
            detail=f"UMAP projection for {request.n_topics} topics not available. Run precomputation first."
        )

    # Try precomputed cluster labels first
    cluster_labels = load_cluster_labels(request.n_topics, request.n_clusters)
    if cluster_labels is None:
        # Fallback to runtime computation
        distribution = load_doc_topic_distribution(request.n_topics)
        if distribution is None:
            raise HTTPException(
                status_code=503,
                detail=f"Distribution for {request.n_topics} topics not available. Run precomputation first."
            )
        result = perform_kmeans(distribution, request.n_clusters)
        cluster_labels = result.labels

    # Round to 4 decimal places to reduce payload size (~30% smaller)
    rounded_projections = [[round(x, 4), round(y, 4)] for x, y in projection.tolist()]

    # Load newsgroup labels (optional - may not be cached)
    newsgroup_labels = load_document_labels()

    # Try precomputed document enrichment first
    enrichment = load_document_enrichment(request.n_topics)
    if enrichment is not None:
        top_topics = [
            [DocumentTopicInfo(topic_id=t["topic_id"], probability=t["probability"])
             for t in doc_topics]
            for doc_topics in enrichment["top_topics"]
        ]
        dominant_topic_words = enrichment["dominant_topic_words"]
    else:
        # Fallback to runtime computation
        top_topics = None
        dominant_topic_words = None

        distribution = load_doc_topic_distribution(request.n_topics)
        model = load_lda_model(request.n_topics)

        if model is not None and distribution is not None:
            # Get top 3 topics for each document (sorted by probability descending)
            top_3_indices = np.argsort(distribution, axis=1)[:, -3:][:, ::-1]
            top_topics = []
            for i, doc_dist in enumerate(distribution):
                doc_top_topics = [
                    DocumentTopicInfo(
                        topic_id=int(idx),
                        probability=round(float(doc_dist[idx]), 4)
                    )
                    for idx in top_3_indices[i]
                ]
                top_topics.append(doc_top_topics)

            # Get top 5 words for each topic (cache to avoid repeated calls)
            topic_word_cache: dict[int, list[str]] = {}
            for topic_id in range(request.n_topics):
                words = model.show_topic(topic_id, topn=5)
                topic_word_cache[topic_id] = [word for word, _ in words]

            # Map each document to its dominant topic's words
            dominant_topics = np.argmax(distribution, axis=1)
            dominant_topic_words = [
                topic_word_cache[int(topic_id)] for topic_id in dominant_topics
            ]

    # Compute cluster geometry in UMAP 2D space
    labels_array = cluster_labels if isinstance(cluster_labels, np.ndarray) else np.array(cluster_labels)
    cluster_centers = _compute_cluster_centers_2d(projection, labels_array, request.n_clusters)
    cluster_covariances = _compute_cluster_covariances_2d(projection, labels_array, request.n_clusters)

    return ClusteredVisualizationResponse(
        n_topics=request.n_topics,
        n_clusters=request.n_clusters,
        projections=rounded_projections,
        cluster_labels=cluster_labels.tolist() if hasattr(cluster_labels, 'tolist') else list(cluster_labels),
        document_ids=list(range(len(projection))),
        cluster_centers=cluster_centers,
        cluster_covariances=cluster_covariances,
        newsgroup_labels=newsgroup_labels,
        top_topics=top_topics,
        dominant_topic_words=dominant_topic_words,
    )


def _get_top_probabilities(
    probs: np.ndarray, top_n: int = 3
) -> list[list[ClusterProbability]]:
    """
    Extract top N cluster probabilities for each document.

    Args:
        probs: Probability matrix of shape (n_samples, n_clusters)
        top_n: Number of top probabilities to return per document

    Returns:
        List of lists of ClusterProbability for each document
    """
    result = []
    for doc_probs in probs:
        # Get indices of top N probabilities
        top_indices = np.argsort(doc_probs)[::-1][:top_n]
        doc_top_probs = [
            ClusterProbability(
                cluster_id=int(idx),
                probability=round(float(doc_probs[idx]), 4),
            )
            for idx in top_indices
            if doc_probs[idx] > 0.01  # Only include if probability > 1%
        ]
        result.append(doc_top_probs)
    return result


@router.post("/gmm-clustered", response_model=GMMClusteredVisualizationResponse)
async def get_gmm_clustered_visualization(request: GMMVisualizationRequest):
    """
    Get UMAP projections with GMM cluster labels, probabilities, and document info.

    Uses precomputed data when available for fast response times.
    """
    projection = load_umap_projection(request.n_topics)

    if projection is None:
        raise HTTPException(
            status_code=503,
            detail=f"UMAP projection for {request.n_topics} topics not available. Run precomputation first.",
        )

    # Try precomputed GMM labels and probabilities first
    cluster_labels = load_gmm_labels(
        request.n_topics, request.n_clusters, request.covariance_type
    )
    probs = load_gmm_probabilities(
        request.n_topics, request.n_clusters, request.covariance_type
    )

    if cluster_labels is None or probs is None:
        # Fallback to runtime computation
        distribution = load_doc_topic_distribution(request.n_topics)
        if distribution is None:
            raise HTTPException(
                status_code=503,
                detail=f"Distribution for {request.n_topics} topics not available. Run precomputation first.",
            )
        result = perform_gmm(
            distribution, request.n_clusters, request.covariance_type
        )
        cluster_labels = result.labels
        probs = result.probabilities

    # Round projections to 4 decimal places to reduce payload size
    rounded_projections = [
        [round(x, 4), round(y, 4)] for x, y in projection.tolist()
    ]

    # Get top 3 probabilities per document
    cluster_probabilities = _get_top_probabilities(probs, top_n=3)

    # Load newsgroup labels (optional)
    newsgroup_labels = load_document_labels()

    # Try precomputed document enrichment
    enrichment = load_document_enrichment(request.n_topics)
    if enrichment is not None:
        top_topics = [
            [
                DocumentTopicInfo(topic_id=t["topic_id"], probability=t["probability"])
                for t in doc_topics
            ]
            for doc_topics in enrichment["top_topics"]
        ]
        dominant_topic_words = enrichment["dominant_topic_words"]
    else:
        # Fallback to runtime computation
        top_topics = None
        dominant_topic_words = None

        distribution = load_doc_topic_distribution(request.n_topics)
        model = load_lda_model(request.n_topics)

        if model is not None and distribution is not None:
            # Get top 3 topics for each document
            top_3_indices = np.argsort(distribution, axis=1)[:, -3:][:, ::-1]
            top_topics = []
            for i, doc_dist in enumerate(distribution):
                doc_top_topics = [
                    DocumentTopicInfo(
                        topic_id=int(idx),
                        probability=round(float(doc_dist[idx]), 4),
                    )
                    for idx in top_3_indices[i]
                ]
                top_topics.append(doc_top_topics)

            # Get top 5 words for each topic
            topic_word_cache: dict[int, list[str]] = {}
            for topic_id in range(request.n_topics):
                words = model.show_topic(topic_id, topn=5)
                topic_word_cache[topic_id] = [word for word, _ in words]

            # Map each document to its dominant topic's words
            dominant_topics = np.argmax(distribution, axis=1)
            dominant_topic_words = [
                topic_word_cache[int(topic_id)] for topic_id in dominant_topics
            ]

    # Compute cluster geometry in UMAP 2D space
    labels_array = cluster_labels if isinstance(cluster_labels, np.ndarray) else np.array(cluster_labels)
    cluster_means = _compute_cluster_centers_2d(projection, labels_array, request.n_clusters)
    cluster_covariances = _compute_cluster_covariances_2d(projection, labels_array, request.n_clusters)

    return GMMClusteredVisualizationResponse(
        n_topics=request.n_topics,
        n_clusters=request.n_clusters,
        covariance_type=request.covariance_type,
        projections=rounded_projections,
        cluster_labels=(
            cluster_labels.tolist()
            if hasattr(cluster_labels, "tolist")
            else list(cluster_labels)
        ),
        cluster_probabilities=cluster_probabilities,
        document_ids=list(range(len(projection))),
        cluster_means=cluster_means,
        cluster_covariances=cluster_covariances,
        newsgroup_labels=newsgroup_labels,
        top_topics=top_topics,
        dominant_topic_words=dominant_topic_words,
    )
