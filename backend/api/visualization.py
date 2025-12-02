"""Visualization API endpoints."""

import numpy as np
from fastapi import APIRouter, HTTPException

from backend.cache.manager import (
    load_umap_projection,
    load_doc_topic_distribution,
    load_document_labels,
    load_lda_model,
)
from backend.core.clustering import perform_kmeans
from backend.models.requests import VisualizationRequest
from backend.models.responses import (
    VisualizationResponse,
    ClusteredVisualizationResponse,
    DocumentTopicInfo,
)
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
    Get UMAP projections with cluster labels, newsgroup labels, and topic info.

    Combines pre-computed UMAP projections with real-time K-Means clustering.
    Includes original newsgroup labels and top topics for tooltip enrichment.
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

    # Load newsgroup labels (optional - may not be cached)
    newsgroup_labels = load_document_labels(dataset)

    # Compute top 3 topics for each document
    top_topics = None
    dominant_topic_words = None

    model = load_lda_model(request.n_topics)
    if model is not None:
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

    return ClusteredVisualizationResponse(
        n_topics=request.n_topics,
        n_clusters=request.n_clusters,
        projections=rounded_projections,
        cluster_labels=result.labels.tolist(),
        document_ids=list(range(len(projection))),
        dataset=dataset,
        newsgroup_labels=newsgroup_labels,
        top_topics=top_topics,
        dominant_topic_words=dominant_topic_words,
    )
