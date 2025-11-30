// API Client

const API_BASE_URL = "http://localhost:8000/api";

class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function fetchApi<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;

  const response = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new ApiError(
      response.status,
      errorData.detail || `HTTP error ${response.status}`
    );
  }

  return response.json();
}

export const api = {
  // Health & Status
  getHealth: () => fetchApi<import("../types/api").HealthResponse>("/health"),

  getStatus: () => fetchApi<import("../types/api").StatusResponse>("/status"),

  // Topics
  getCoherence: () =>
    fetchApi<import("../types/api").CoherenceResponse>("/topics/coherence"),

  getTopicWords: (nTopics: number, numWords: number = 10) =>
    fetchApi<import("../types/api").TopicWordsResponse>(
      `/topics/${nTopics}/words?num_words=${numWords}`
    ),

  // Clustering
  cluster: (request: import("../types/api").ClusteringRequest) =>
    fetchApi<import("../types/api").ClusteringResponse>("/clustering", {
      method: "POST",
      body: JSON.stringify(request),
    }),

  getClusterMetrics: (nTopics: number) =>
    fetchApi<import("../types/api").ClusterMetricsResponse>(
      `/clustering/metrics/${nTopics}`
    ),

  // Visualization
  getVisualization: (nTopics: number) =>
    fetchApi<import("../types/api").VisualizationResponse>(
      `/visualization/${nTopics}`
    ),

  getClusteredVisualization: (
    request: import("../types/api").ClusteringRequest
  ) =>
    fetchApi<import("../types/api").ClusteredVisualizationResponse>(
      "/visualization/clustered",
      {
        method: "POST",
        body: JSON.stringify(request),
      }
    ),

  // Precomputation
  startPrecompute: () =>
    fetchApi<{ status: string; message: string }>("/precompute/start", {
      method: "POST",
    }),

  getPrecomputeProgress: () =>
    fetchApi<import("../types/api").PrecomputeProgressResponse>(
      "/precompute/progress"
    ),
};

export { ApiError };
