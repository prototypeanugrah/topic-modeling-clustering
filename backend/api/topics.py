"""Topic-related API endpoints."""

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import HTMLResponse

from backend.cache.manager import (
    load_coherence_scores,
    load_lda_model,
    load_doc_topic_distribution,
    load_umap_projection,
    load_pyldavis_html,
    load_topic_words,
    load_cluster_metrics,
    load_cluster_labels,
    load_document_enrichment,
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

    Used for the "optimal number of topics" chart.
    """
    coherence = load_coherence_scores()

    if coherence is None:
        raise HTTPException(
            status_code=503,
            detail="Coherence scores not available. Run precomputation first."
        )

    # Add cache headers - this data doesn't change after precomputation
    response.headers.update(CACHE_HEADERS)

    topic_counts = sorted(coherence.keys())
    coherence_values = [coherence[k] for k in topic_counts]

    # Find optimal based on coherence
    optimal_topics = max(coherence, key=coherence.get)

    return CoherenceResponse(
        topic_counts=topic_counts,
        coherence=coherence_values,
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

    # Try precomputed cache first (for num_words <= 10)
    cached = load_topic_words(n_topics)
    if cached is not None and num_words <= 10:
        response.headers.update(CACHE_HEADERS)
        topics = [
            [TopicWord(word=w["word"], probability=w["probability"])
             for w in topic[:num_words]]
            for topic in cached["topics"]
        ]
        return TopicWordsResponse(n_topics=n_topics, topics=topics)

    # Fallback to model loading (for num_words > 10 or missing cache)
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
async def get_topic_distribution(n_topics: int):
    """
    Get document-topic distribution matrix.

    Args:
        n_topics: Number of topics

    Returns the topic distribution for all documents.
    Note: This can be a large response (~18.8K documents x n_topics floats).
    """
    if n_topics < MIN_TOPICS or n_topics > MAX_TOPICS:
        raise HTTPException(
            status_code=400,
            detail=f"n_topics must be between {MIN_TOPICS} and {MAX_TOPICS}"
        )

    distribution = load_doc_topic_distribution(n_topics)

    if distribution is None:
        raise HTTPException(
            status_code=503,
            detail=f"Distribution for {n_topics} topics not available. Run precomputation first."
        )

    return {
        "n_topics": n_topics,
        "n_documents": distribution.shape[0],
        "distribution": distribution.tolist(),
    }


@router.get("/{n_topics}/pyldavis", response_class=HTMLResponse)
async def get_pyldavis(n_topics: int, theme: str = "light"):
    """
    Get pyLDAvis HTML visualization for a topic model.

    Returns an interactive HTML visualization of topic-word distributions.

    Args:
        n_topics: Number of topics in the model
        theme: Color theme ("light" or "dark")
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

    # Inject dark mode CSS if theme is dark
    if theme == "dark":
        dark_mode_css = """
<style>
    /* Dark mode - force all backgrounds dark */
    body, html {
        background-color: #21262d !important;
        color: #c9d1d9 !important;
    }

    /* Target all divs and common containers */
    div {
        background-color: #21262d !important;
        color: #c9d1d9 !important;
    }

    /* SVG text elements */
    svg text {
        fill: #c9d1d9 !important;
    }

    /* Axis styling */
    .xaxis text, .yaxis text, .axis text, .slideraxis text {
        fill: #8b949e !important;
    }
    .xaxis line, .yaxis line, .axis line, .slideraxis line,
    .xaxis path, .yaxis path, .axis path, .slideraxis path {
        stroke: #484f58 !important;
    }

    /* Inputs and buttons */
    input, button, select {
        background-color: #30363d !important;
        color: #c9d1d9 !important;
        border: 1px solid #484f58 !important;
    }

    /* Labels and text */
    label, span, p {
        color: #c9d1d9 !important;
    }
</style>
"""
        # Insert dark mode CSS at the beginning
        html = dark_mode_css + html

        # Replace the hardcoded white background on the main div
        html = html.replace('style="background-color:white;"', 'style="background-color:#21262d;"')

    return HTMLResponse(content=html)


@router.get("/{n_topics}/bundle", response_model=TopicBundleResponse)
async def get_topic_bundle(n_topics: int, n_clusters: int = 5, num_words: int = 10):
    """
    Get bundled topic data in a single request (reduces round trips).

    Args:
        n_topics: Number of topics
        n_clusters: Number of clusters
        num_words: Number of top words per topic

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

    # 1. Topic words (try precomputed cache first)
    cached_words = load_topic_words(n_topics)
    if cached_words is not None and num_words <= 10:
        topics = [
            [TopicWord(word=w["word"], probability=w["probability"])
             for w in topic[:num_words]]
            for topic in cached_words["topics"]
        ]
    else:
        model = load_lda_model(n_topics)
        if model is None:
            raise HTTPException(
                status_code=503,
                detail=f"Data for {n_topics} topics not available. Run precomputation first."
            )
        topics = []
        for topic_id in range(n_topics):
            topic_words = model.show_topic(topic_id, topn=num_words)
            topics.append([
                TopicWord(word=word, probability=prob)
                for word, prob in topic_words
            ])
    words_response = TopicWordsResponse(n_topics=n_topics, topics=topics)

    # 2. Cluster metrics (try precomputed cache first)
    cached_metrics = load_cluster_metrics(n_topics)
    if cached_metrics is not None:
        cluster_metrics_response = ClusterMetricsResponse(
            n_topics=n_topics,
            cluster_counts=cached_metrics["cluster_counts"],
            silhouette_scores=cached_metrics["silhouette_scores"],
            inertia_scores=cached_metrics["inertia_scores"],
            elbow_point=cached_metrics["elbow_point"],
        )
    else:
        distribution = load_doc_topic_distribution(n_topics)
        if distribution is None:
            raise HTTPException(
                status_code=503,
                detail=f"Distribution for {n_topics} topics not available. Run precomputation first."
            )
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

    # 3. Clustered visualization
    projection = load_umap_projection(n_topics)
    if projection is None:
        raise HTTPException(
            status_code=503,
            detail=f"UMAP projection for {n_topics} topics not available. Run precomputation first."
        )

    # Try precomputed cluster labels
    cluster_labels = load_cluster_labels(n_topics, n_clusters)
    if cluster_labels is None:
        distribution = load_doc_topic_distribution(n_topics)
        if distribution is None:
            raise HTTPException(
                status_code=503,
                detail=f"Distribution for {n_topics} topics not available. Run precomputation first."
            )
        result = perform_kmeans(distribution, n_clusters)
        cluster_labels = result.labels

    rounded_projections = [[round(x, 4), round(y, 4)] for x, y in projection.tolist()]
    visualization_response = ClusteredVisualizationResponse(
        n_topics=n_topics,
        n_clusters=n_clusters,
        projections=rounded_projections,
        cluster_labels=cluster_labels.tolist() if hasattr(cluster_labels, 'tolist') else list(cluster_labels),
        document_ids=list(range(len(projection))),
    )

    return TopicBundleResponse(
        words=words_response,
        cluster_metrics=cluster_metrics_response,
        visualization=visualization_response,
    )
