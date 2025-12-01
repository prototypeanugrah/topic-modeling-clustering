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

export function useClusteredVisualization(
  nTopics: number,
  nClusters: number,
  dataset: "train" | "test" = "train"
) {
  // Use array key for POST requests with SWR
  const { data, error, isLoading, isValidating } = useSWR<ClusteredVisualizationResponse>(
    ["/visualization/clustered", { n_topics: nTopics, n_clusters: nClusters, dataset }],
    swrPostFetcher,
    swrConfig
  );

  // Prefetch adjacent topic and cluster combinations
  useEffect(() => {
    const timer = setTimeout(() => {
      // Prefetch adjacent topic counts with same cluster count
      if (nTopics > 2) {
        prefetchPost("/visualization/clustered", { n_topics: nTopics - 1, n_clusters: nClusters, dataset });
      }
      if (nTopics < 20) {
        prefetchPost("/visualization/clustered", { n_topics: nTopics + 1, n_clusters: nClusters, dataset });
      }
      // Prefetch adjacent cluster counts with same topic count
      if (nClusters > 2) {
        prefetchPost("/visualization/clustered", { n_topics: nTopics, n_clusters: nClusters - 1, dataset });
      }
      if (nClusters < 15) {
        prefetchPost("/visualization/clustered", { n_topics: nTopics, n_clusters: nClusters + 1, dataset });
      }
      // Prefetch the other dataset (train/test toggle)
      const otherDataset = dataset === "train" ? "test" : "train";
      prefetchPost("/visualization/clustered", { n_topics: nTopics, n_clusters: nClusters, dataset: otherDataset });
    }, 200);  // Slightly longer delay for visualization (larger payload)

    return () => clearTimeout(timer);
  }, [nTopics, nClusters, dataset]);

  return {
    data: data ?? null,
    loading: isLoading,
    isValidating,
    error: error?.message ?? null,
  };
}
