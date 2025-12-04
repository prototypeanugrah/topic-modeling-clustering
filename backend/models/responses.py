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
    tokenized: bool
    coherence: bool
    models: dict[int, bool]
    distributions: dict[int, bool]
    projections: dict[int, bool]


class CoherenceResponse(BaseModel):
    """Response for coherence scores endpoint."""

    topic_counts: list[int]
    coherence: list[float]
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


class DocumentTopicInfo(BaseModel):
    """Top topic information for a single document."""

    topic_id: int
    probability: float


class ClusteredVisualizationResponse(BaseModel):
    """Response for clustered visualization endpoint."""

    n_topics: int
    n_clusters: int
    projections: list[list[float]]  # [[x, y], ...]
    cluster_labels: list[int]
    document_ids: list[int]

    # Optional enrichment fields for tooltip
    newsgroup_labels: Optional[list[str]] = None  # Original 20 newsgroups labels
    top_topics: Optional[list[list[DocumentTopicInfo]]] = None  # Top 3 topics per doc
    dominant_topic_words: Optional[list[list[str]]] = None  # Top 5 words from dominant topic


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
    mean: float
    median: float
    min: int
    max: int
    std: float
    empty_count: int
    empty_pct: float
    percentiles: dict[str, float]  # {"10", "25", "50", "75", "90", "95", "99"}
    # For histogram (binned data to reduce payload)
    histogram_bins: list[float]  # bin edges
    histogram_counts: list[int]  # counts per bin


class EDAResponse(BaseModel):
    """Full EDA response with 4 preprocessing stages."""

    # Stage 1: Raw documents (token counts via whitespace split)
    raw: StageStats

    # Stage 2: Tokenized (after preprocessing, before filter_extremes)
    vocab_before_filter: int
    tokenized: StageStats

    # Stage 3: After filter_extremes
    vocab_after_filter: int
    filtered: StageStats
    vocab_reduction_pct: float

    # Stage 4: After document filtering (final corpus)
    final: StageStats
    min_tokens_threshold: int
    docs_removed: int

    # Filter settings used
    filter_no_below: int
    filter_no_above: float


class BoxPlotData(BaseModel):
    """Box plot data for EDA visualizations."""

    # Token counts by preprocessing stage (raw arrays for Plotly box plots)
    stage_token_counts: dict[str, list[int]]  # {"raw": [...], "tokenized": [...], "filtered": [...], "final": [...]}

    # Token counts by newsgroup category
    category_token_counts: dict[str, list[int]]  # {"alt.atheism": [...], "comp.graphics": [...], ...}
