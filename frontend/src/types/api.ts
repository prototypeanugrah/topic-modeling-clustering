// API Response Types

export interface HealthResponse {
  status: string;
  cache_complete: boolean;
}

export interface StatusResponse {
  complete: boolean;
  dictionary: boolean;
  corpus_train: boolean;
  corpus_test: boolean;
  tokenized_train: boolean;
  tokenized_test: boolean;
  coherence_val: boolean;
  coherence_test: boolean;
  perplexity_val: boolean;
  perplexity_test: boolean;
  models: Record<number, boolean>;
  distributions_train: Record<number, boolean>;
  distributions_test: Record<number, boolean>;
  projections_train: Record<number, boolean>;
  projections_test: Record<number, boolean>;
}

export interface CoherenceResponse {
  topic_counts: number[];
  // Validation scores (averaged from 5-fold CV)
  coherence_val: number[];
  perplexity_val: number[];
  // Test scores (final evaluation on held-out set)
  coherence_test: number[];
  perplexity_test: number[];
  optimal_topics: number; // Based on test coherence
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
  dataset?: "train" | "test";
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
  dataset: "train" | "test";
}

export interface ClusteredVisualizationResponse {
  n_topics: number;
  n_clusters: number;
  projections: number[][];
  cluster_labels: number[];
  document_ids: number[];
  dataset: "train" | "test";
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
  avg_length: number;
  median_length: number;
  min_length: number;
  max_length: number;
  std_length: number;
  empty_count: number;
  empty_pct: number;
  percentiles: Record<number, number>;
  histogram_bins: number[];
  histogram_counts: number[];
}

export interface EDAResponse {
  // Stage 1: Raw documents (character lengths)
  raw_train: StageStats;
  raw_test: StageStats;

  // Stage 2: Tokenized (before filter_extremes)
  vocab_before_filter: number;
  tokenized_train: StageStats;
  tokenized_test: StageStats;

  // Stage 3: Filtered (corpus for LDA)
  vocab_after_filter: number;
  filtered_train: StageStats;
  filtered_test: StageStats;

  // Summary metrics
  vocab_reduction_pct: number;
  token_reduction_pct: number;
}
