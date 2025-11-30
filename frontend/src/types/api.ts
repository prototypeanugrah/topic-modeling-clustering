// API Response Types

export interface HealthResponse {
  status: string;
  cache_complete: boolean;
}

export interface StatusResponse {
  complete: boolean;
  dictionary: boolean;
  corpus: boolean;
  tokenized_docs: boolean;
  coherence_scores: boolean;
  models: Record<number, boolean>;
  distributions: Record<number, boolean>;
  projections: Record<number, boolean>;
}

export interface CoherenceResponse {
  topic_counts: number[];
  coherence_scores: number[];
  perplexity_scores: number[];
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

export interface ClusteredVisualizationResponse {
  n_topics: number;
  n_clusters: number;
  projections: number[][];
  cluster_labels: number[];
  document_ids: number[];
}

export interface PrecomputeProgressResponse {
  in_progress: boolean;
  current_step: string | null;
  progress_percent: number;
  completed_topics: number[];
  error: string | null;
}
