"""Pydantic response models."""

from pydantic import BaseModel
from typing import Optional


class HealthResponse(BaseModel):
    """Response for health check endpoint."""

    status: str
    cache_complete: bool


class StatusResponse(BaseModel):
    """Response for status endpoint."""

    complete: bool
    dictionary: bool
    corpus_train: bool
    corpus_test: bool
    tokenized_train: bool
    tokenized_test: bool
    coherence_val: bool
    coherence_test: bool
    perplexity_val: bool
    perplexity_test: bool
    models: dict[int, bool]
    distributions_train: dict[int, bool]
    distributions_test: dict[int, bool]
    projections_train: dict[int, bool]
    projections_test: dict[int, bool]


class CoherenceResponse(BaseModel):
    """Response for coherence scores endpoint with val and test scores."""

    topic_counts: list[int]
    # Validation scores (averaged from 5-fold CV)
    coherence_val: list[float]
    perplexity_val: list[float]
    # Test scores (final evaluation on held-out set)
    coherence_test: list[float]
    perplexity_test: list[float]
    optimal_topics: int  # Based on test coherence


class TopicWord(BaseModel):
    """A word in a topic with its probability."""

    word: str
    probability: float


class TopicWordsResponse(BaseModel):
    """Response for topic words endpoint."""

    n_topics: int
    topics: list[list[TopicWord]]


class ClusteringResponse(BaseModel):
    """Response for clustering endpoint."""

    n_topics: int
    n_clusters: int
    labels: list[int]
    silhouette: float
    inertia: float
    cluster_sizes: list[int]


class ClusterMetricsResponse(BaseModel):
    """Response for cluster metrics endpoint."""

    n_topics: int
    cluster_counts: list[int]
    silhouette_scores: list[float]
    inertia_scores: list[float]
    elbow_point: Optional[int]


class VisualizationResponse(BaseModel):
    """Response for visualization endpoint."""

    n_topics: int
    projections: list[list[float]]  # [[x, y], ...]
    document_ids: list[int]
    dataset: str = "train"  # "train" or "test"


class ClusteredVisualizationResponse(BaseModel):
    """Response for clustered visualization endpoint."""

    n_topics: int
    n_clusters: int
    projections: list[list[float]]  # [[x, y], ...]
    cluster_labels: list[int]
    document_ids: list[int]
    dataset: str = "train"  # "train" or "test"


class PrecomputeProgressResponse(BaseModel):
    """Response for precomputation progress."""

    in_progress: bool
    current_step: Optional[str]
    progress_percent: int
    completed_topics: list[int]
    error: Optional[str]


class TopicBundleResponse(BaseModel):
    """Batch response for topic-related data (reduces API round trips)."""

    words: TopicWordsResponse
    cluster_metrics: ClusterMetricsResponse
    visualization: ClusteredVisualizationResponse


class StageStats(BaseModel):
    """Statistics for one preprocessing stage."""

    n_documents: int
    avg_length: float
    median_length: float
    min_length: int
    max_length: int
    std_length: float
    empty_count: int
    empty_pct: float
    percentiles: dict[int, float]  # {10, 25, 50, 75, 90, 95, 99}
    # For histogram (binned data to reduce payload)
    histogram_bins: list[float]  # bin edges
    histogram_counts: list[int]  # counts per bin


class EDAResponse(BaseModel):
    """Full EDA response with 3 preprocessing stages."""

    # Stage 1: Raw documents (character lengths)
    raw_train: StageStats
    raw_test: StageStats

    # Stage 2: Tokenized (before filter_extremes)
    vocab_before_filter: int
    tokenized_train: StageStats
    tokenized_test: StageStats

    # Stage 3: Filtered (corpus for LDA)
    vocab_after_filter: int
    filtered_train: StageStats
    filtered_test: StageStats

    # Summary metrics
    vocab_reduction_pct: float
    token_reduction_pct: float
