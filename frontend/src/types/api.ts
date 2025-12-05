// API Response Types

export interface HealthResponse {
  status: string;
  cache_complete: boolean;
}

export interface StatusResponse {
  complete: boolean;
  dictionary: boolean;
  corpus: boolean;
  tokenized: boolean;
  coherence: boolean;
  models: Record<number, boolean>;
  distributions: Record<number, boolean>;
  projections: Record<number, boolean>;
}

export interface CoherenceResponse {
  topic_counts: number[];
  coherence: number[];
  optimal_topics: number;
}

export interface TopicWord {
  word: string;
  probability: number;
}

export interface TopicWordsResponse {
  n_topics: number;
  topics: TopicWord[][];
}

export interface ClusteringRequest {
  n_topics: number;
  n_clusters: number;
}

export interface ClusteringResponse {
  n_topics: number;
  n_clusters: number;
  labels: number[];
  silhouette: number;
  inertia: number;
  cluster_sizes: number[];
}

export interface ClusterMetricsResponse {
  n_topics: number;
  cluster_counts: number[];
  silhouette_scores: number[];
  inertia_scores: number[];
  elbow_point: number | null;
}

export interface VisualizationResponse {
  n_topics: number;
  projections: number[][];
  document_ids: number[];
}

export interface DocumentTopicInfo {
  topic_id: number;
  probability: number;
}

export interface ClusteredVisualizationResponse {
  n_topics: number;
  n_clusters: number;
  projections: number[][];
  cluster_labels: number[];
  document_ids: number[];

  // Cluster geometry for boundary visualization
  cluster_centers?: number[][]; // [[x, y], ...] per cluster
  cluster_covariances?: number[][][]; // [[[a,b],[c,d]], ...] 2x2 per cluster

  // Optional tooltip enrichment fields
  newsgroup_labels?: string[]; // Original 20 newsgroups labels
  top_topics?: DocumentTopicInfo[][]; // Top 3 topics per document
  dominant_topic_words?: string[][]; // Top 5 words from dominant topic
}

export interface PrecomputeProgressResponse {
  in_progress: boolean;
  current_step: string | null;
  progress_percent: number;
  completed_topics: number[];
  error: string | null;
}

// Batch response combining topic words, cluster metrics, and visualization
export interface TopicBundleResponse {
  words: TopicWordsResponse;
  cluster_metrics: ClusterMetricsResponse;
  visualization: ClusteredVisualizationResponse;
}

// EDA (Exploratory Data Analysis) types
export interface StageStats {
  n_documents: number;
  mean: number;
  median: number;
  min: number;
  max: number;
  std: number;
  empty_count: number;
  empty_pct: number;
  percentiles: Record<string, number>;
  histogram_bins: number[];
  histogram_counts: number[];
}

export interface EDAResponse {
  // Stage 1: Raw documents (token counts via whitespace split)
  raw: StageStats;

  // Stage 2: Tokenized (after preprocessing, before filter_extremes)
  vocab_before_filter: number;
  tokenized: StageStats;

  // Stage 3: After filter_extremes
  vocab_after_filter: number;
  filtered: StageStats;
  vocab_reduction_pct: number;

  // Stage 4: After document filtering (final corpus)
  final: StageStats;
  min_tokens_threshold: number;
  docs_removed: number;

  // Filter settings used
  filter_no_below: number;
  filter_no_above: number;
}

export interface BoxPlotData {
  // Token counts by preprocessing stage (raw arrays for Plotly box plots)
  stage_token_counts: Record<string, number[]>;

  // Token counts by newsgroup category
  category_token_counts: Record<string, number[]>;
}

// === GMM Types ===

export type CovarianceType = "full" | "diag" | "spherical";

export interface GMMRequest {
  n_topics: number;
  n_clusters: number;
  covariance_type: CovarianceType;
}

export interface ClusterProbability {
  cluster_id: number;
  probability: number;
}

export interface GMMResponse {
  n_topics: number;
  n_clusters: number;
  covariance_type: string;
  labels: number[];
  probabilities: ClusterProbability[][];
  bic: number;
  aic: number;
  cluster_sizes: number[];
}

export interface GMMMetricsResponse {
  n_topics: number;
  covariance_type: string;
  cluster_counts: number[];
  bic_scores: number[];
  aic_scores: number[];
  optimal_bic: number;
  optimal_aic: number;
}

export interface GMMAllCovarianceMetricsResponse {
  n_topics: number;
  full: GMMMetricsResponse;
  diag: GMMMetricsResponse;
  spherical: GMMMetricsResponse;
}

export interface GMMClusteredVisualizationResponse {
  n_topics: number;
  n_clusters: number;
  covariance_type: string;
  projections: number[][];
  cluster_labels: number[];
  cluster_probabilities: ClusterProbability[][];
  document_ids: number[];

  // Cluster geometry for ellipse visualization (in UMAP 2D space)
  cluster_means?: number[][]; // [[x, y], ...] per cluster
  cluster_covariances?: number[][][]; // [[[a,b],[c,d]], ...] 2x2 per cluster

  // Optional tooltip enrichment fields
  newsgroup_labels?: string[];
  top_topics?: DocumentTopicInfo[][];
  dominant_topic_words?: string[][];
}
