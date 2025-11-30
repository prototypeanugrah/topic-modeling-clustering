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
    corpus: bool
    tokenized_docs: bool
    coherence_scores: bool
    models: dict[int, bool]
    distributions: dict[int, bool]
    projections: dict[int, bool]


class CoherenceResponse(BaseModel):
    """Response for coherence scores endpoint."""

    topic_counts: list[int]
    coherence_scores: list[float]
    perplexity_scores: list[float]
    optimal_topics: int


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


class ClusteredVisualizationResponse(BaseModel):
    """Response for clustered visualization endpoint."""

    n_topics: int
    n_clusters: int
    projections: list[list[float]]  # [[x, y], ...]
    cluster_labels: list[int]
    document_ids: list[int]


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
