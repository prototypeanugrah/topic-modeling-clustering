// API Client with SWR support

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

// SWR fetcher function
export const swrFetcher = async (url: string) => {
  const response = await fetch(`${API_BASE_URL}${url}`, {
    headers: { "Content-Type": "application/json" },
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new ApiError(
      response.status,
      errorData.detail || `HTTP error ${response.status}`
    );
  }

  return response.json();
};

// POST fetcher for SWR
export const swrPostFetcher = async ([url, body]: [string, object]) => {
  const response = await fetch(`${API_BASE_URL}${url}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new ApiError(
      response.status,
      errorData.detail || `HTTP error ${response.status}`
    );
  }

  return response.json();
};

// Prefetch helper - fetches and warms the browser cache
export const prefetch = (endpoint: string) => {
  const url = `${API_BASE_URL}${endpoint}`;
  fetch(url, {
    headers: { "Content-Type": "application/json" },
  }).catch(() => {
    // Silently ignore prefetch errors
  });
};

// Prefetch POST endpoint
export const prefetchPost = (endpoint: string, body: object) => {
  fetch(`${API_BASE_URL}${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).catch(() => {
    // Silently ignore prefetch errors
  });
};

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

  // Topic Bundle - batch endpoint for all topic-related data
  getTopicBundle: (nTopics: number, nClusters: number) =>
    fetchApi<import("../types/api").TopicBundleResponse>(
      `/topics/${nTopics}/bundle?n_clusters=${nClusters}`
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

  // EDA
  getEDA: () => fetchApi<import("../types/api").EDAResponse>("/eda"),

  // Prefetch helpers
  prefetch,
  prefetchPost,
};

export { ApiError, API_BASE_URL };
