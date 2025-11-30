import useSWR from "swr";
import { useEffect } from "react";
import { swrPostFetcher, prefetchPost } from "../api/client";
import type { ClusteredVisualizationResponse } from "../types/api";

// SWR configuration for optimal caching
const swrConfig = {
  revalidateOnFocus: false,
  revalidateOnReconnect: false,
  dedupingInterval: 60000,
  keepPreviousData: true,  // Show previous data while loading new - enables optimistic UI
};

export function useClusteredVisualization(nTopics: number, nClusters: number) {
  // Use array key for POST requests with SWR
  const { data, error, isLoading, isValidating } = useSWR<ClusteredVisualizationResponse>(
    ["/visualization/clustered", { n_topics: nTopics, n_clusters: nClusters }],
    swrPostFetcher,
    swrConfig
  );

  // Prefetch adjacent topic and cluster combinations
  useEffect(() => {
    const timer = setTimeout(() => {
      // Prefetch adjacent topic counts with same cluster count
      if (nTopics > 2) {
        prefetchPost("/visualization/clustered", { n_topics: nTopics - 1, n_clusters: nClusters });
      }
      if (nTopics < 20) {
        prefetchPost("/visualization/clustered", { n_topics: nTopics + 1, n_clusters: nClusters });
      }
      // Prefetch adjacent cluster counts with same topic count
      if (nClusters > 2) {
        prefetchPost("/visualization/clustered", { n_topics: nTopics, n_clusters: nClusters - 1 });
      }
      if (nClusters < 15) {
        prefetchPost("/visualization/clustered", { n_topics: nTopics, n_clusters: nClusters + 1 });
      }
    }, 200);  // Slightly longer delay for visualization (larger payload)

    return () => clearTimeout(timer);
  }, [nTopics, nClusters]);

  return {
    data: data ?? null,
    loading: isLoading,
    isValidating,
    error: error?.message ?? null,
  };
}
