"""Topic-related API endpoints."""

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import HTMLResponse

from backend.cache.manager import (
    load_coherence_scores,
    load_lda_model,
    load_doc_topic_distribution,
    load_umap_projection,
    load_pyldavis_html,
)
from backend.core.clustering import perform_kmeans
from backend.core.metrics import compute_metrics_for_all_clusters
from backend.models.responses import (
    CoherenceResponse,
    TopicWord,
    TopicWordsResponse,
    ClusterMetricsResponse,
    ClusteredVisualizationResponse,
    TopicBundleResponse,
)
from backend.config import MIN_TOPICS, MAX_TOPICS, MIN_CLUSTERS, MAX_CLUSTERS

router = APIRouter(prefix="/topics", tags=["topics"])

# Cache headers for precomputed data (1 hour)
CACHE_HEADERS = {"Cache-Control": "public, max-age=3600"}


@router.get("/coherence", response_model=CoherenceResponse)
async def get_coherence_scores(response: Response):
    """
    Get coherence scores for all topic counts.

    Returns both validation (5-fold CV averaged) and test (held-out) scores.
    Used for the "optimal number of topics" chart.
    """
    # Load validation scores (from CV)
    coherence_val = load_coherence_scores("val")

    # Load test scores (final evaluation)
    coherence_test = load_coherence_scores("test")

    if coherence_test is None:
        raise HTTPException(
            status_code=503,
            detail="Coherence scores not available. Run precomputation first."
        )

    # Add cache headers - this data doesn't change after precomputation
    response.headers.update(CACHE_HEADERS)

    topic_counts = sorted(coherence_test.keys())

    # Build response arrays
    coherence_val_values = [coherence_val.get(k, 0) for k in topic_counts] if coherence_val else []
    coherence_test_values = [coherence_test[k] for k in topic_counts]

    # Find optimal based on test coherence
    optimal_topics = max(coherence_test, key=coherence_test.get)

    return CoherenceResponse(
        topic_counts=topic_counts,
        coherence_val=coherence_val_values,
        coherence_test=coherence_test_values,
        optimal_topics=optimal_topics,
    )


@router.get("/{n_topics}/words", response_model=TopicWordsResponse)
async def get_topic_words(n_topics: int, response: Response, num_words: int = 10):
    """
    Get top words for each topic.

    Args:
        n_topics: Number of topics in the model
        num_words: Number of top words per topic (default 10)
    """
    if n_topics < MIN_TOPICS or n_topics > MAX_TOPICS:
        raise HTTPException(
            status_code=400,
            detail=f"n_topics must be between {MIN_TOPICS} and {MAX_TOPICS}"
        )

    model = load_lda_model(n_topics)

    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"LDA model for {n_topics} topics not available. Run precomputation first."
        )

    # Add cache headers
    response.headers.update(CACHE_HEADERS)

    topics = []
    for topic_id in range(n_topics):
        topic_words = model.show_topic(topic_id, topn=num_words)
        topics.append([
            TopicWord(word=word, probability=prob)
            for word, prob in topic_words
        ])

    return TopicWordsResponse(
        n_topics=n_topics,
        topics=topics,
    )


@router.get("/{n_topics}/distribution")
async def get_topic_distribution(n_topics: int, dataset: str = "train"):
    """
    Get document-topic distribution matrix.

    Args:
        n_topics: Number of topics
        dataset: Which dataset to get distribution for (train or test)

    Returns the topic distribution for all documents.
    Note: This can be a large response (~11K train or ~7.5K test documents x n_topics floats).
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

    distribution = load_doc_topic_distribution(n_topics, dataset)

    if distribution is None:
        raise HTTPException(
            status_code=503,
            detail=f"Distribution for {n_topics} topics ({dataset}) not available. Run precomputation first."
        )

    return {
        "n_topics": n_topics,
        "dataset": dataset,
        "n_documents": distribution.shape[0],
        "distribution": distribution.tolist(),
    }


@router.get("/{n_topics}/pyldavis", response_class=HTMLResponse)
async def get_pyldavis(n_topics: int):
    """
    Get pyLDAvis HTML visualization for a topic model.

    Returns an interactive HTML visualization of topic-word distributions.
    """
    if n_topics < MIN_TOPICS or n_topics > MAX_TOPICS:
        raise HTTPException(
            status_code=400,
            detail=f"n_topics must be between {MIN_TOPICS} and {MAX_TOPICS}"
        )

    html = load_pyldavis_html(n_topics)

    if html is None:
        raise HTTPException(
            status_code=503,
            detail=f"pyLDAvis for {n_topics} topics not available. Run precomputation first."
        )

    return HTMLResponse(content=html)


@router.get("/{n_topics}/bundle", response_model=TopicBundleResponse)
async def get_topic_bundle(n_topics: int, n_clusters: int = 5, num_words: int = 10, dataset: str = "train"):
    """
    Get bundled topic data in a single request (reduces round trips).

    Args:
        n_topics: Number of topics
        n_clusters: Number of clusters
        num_words: Number of top words per topic
        dataset: Which dataset to use for visualization (train or test)

    Returns topic words, cluster metrics, and visualization data together.
    This is optimized for the dashboard to minimize latency.
    """
    if n_topics < MIN_TOPICS or n_topics > MAX_TOPICS:
        raise HTTPException(
            status_code=400,
            detail=f"n_topics must be between {MIN_TOPICS} and {MAX_TOPICS}"
        )

    if n_clusters < MIN_CLUSTERS or n_clusters > MAX_CLUSTERS:
        raise HTTPException(
            status_code=400,
            detail=f"n_clusters must be between {MIN_CLUSTERS} and {MAX_CLUSTERS}"
        )

    if dataset not in ["train", "test"]:
        raise HTTPException(
            status_code=400,
            detail="dataset must be 'train' or 'test'"
        )

    # Load all required data
    model = load_lda_model(n_topics)
    distribution = load_doc_topic_distribution(n_topics, dataset)
    projection = load_umap_projection(n_topics, dataset)

    if model is None or distribution is None or projection is None:
        raise HTTPException(
            status_code=503,
            detail=f"Data for {n_topics} topics ({dataset}) not available. Run precomputation first."
        )

    # 1. Topic words
    topics = []
    for topic_id in range(n_topics):
        topic_words = model.show_topic(topic_id, topn=num_words)
        topics.append([
            TopicWord(word=word, probability=prob)
            for word, prob in topic_words
        ])
    words_response = TopicWordsResponse(n_topics=n_topics, topics=topics)

    # 2. Cluster metrics (computed on the selected dataset)
    metrics = compute_metrics_for_all_clusters(
        distribution,
        min_clusters=MIN_CLUSTERS,
        max_clusters=MAX_CLUSTERS,
    )
    cluster_metrics_response = ClusterMetricsResponse(
        n_topics=n_topics,
        cluster_counts=metrics["cluster_counts"],
        silhouette_scores=metrics["silhouette_scores"],
        inertia_scores=metrics["inertia_scores"],
        elbow_point=metrics["elbow_point"],
    )

    # 3. Clustered visualization (with reduced precision for smaller payload)
    result = perform_kmeans(distribution, n_clusters)
    # Round projections to 4 decimal places to reduce payload size
    rounded_projections = [[round(x, 4), round(y, 4)] for x, y in projection.tolist()]
    visualization_response = ClusteredVisualizationResponse(
        n_topics=n_topics,
        n_clusters=n_clusters,
        projections=rounded_projections,
        cluster_labels=result.labels.tolist(),
        document_ids=list(range(len(projection))),
        dataset=dataset,
    )

    return TopicBundleResponse(
        words=words_response,
        cluster_metrics=cluster_metrics_response,
        visualization=visualization_response,
    )
