"""Topic-related API endpoints."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from backend.cache.manager import (
    load_coherence_scores,
    load_perplexity_scores,
    load_lda_model,
    load_doc_topic_distribution,
    load_pyldavis_html,
)
from backend.models.responses import (
    CoherenceResponse,
    TopicWord,
    TopicWordsResponse,
)
from backend.config import MIN_TOPICS, MAX_TOPICS

router = APIRouter(prefix="/topics", tags=["topics"])


@router.get("/coherence", response_model=CoherenceResponse)
async def get_coherence_scores():
    """
    Get coherence and perplexity scores for all topic counts.

    Used for the "optimal number of topics" chart.
    """
    scores = load_coherence_scores()
    perplexity = load_perplexity_scores()

    if scores is None:
        raise HTTPException(
            status_code=503,
            detail="Coherence scores not available. Run precomputation first."
        )

    topic_counts = sorted(scores.keys())
    coherence_values = [scores[k] for k in topic_counts]
    perplexity_values = [perplexity[k] for k in topic_counts] if perplexity else []

    # Find optimal (highest coherence)
    optimal_topics = max(scores, key=scores.get)

    return CoherenceResponse(
        topic_counts=topic_counts,
        coherence_scores=coherence_values,
        perplexity_scores=perplexity_values,
        optimal_topics=optimal_topics,
    )


@router.get("/{n_topics}/words", response_model=TopicWordsResponse)
async def get_topic_words(n_topics: int, num_words: int = 10):
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

    Returns the topic distribution for all documents.
    Note: This can be a large response (~18K documents x n_topics floats).
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
