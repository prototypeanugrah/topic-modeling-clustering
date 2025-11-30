import useSWR, { preload } from "swr";
import { useEffect } from "react";
import { swrFetcher, prefetch } from "../api/client";
import type {
  CoherenceResponse,
  TopicWordsResponse,
  ClusterMetricsResponse,
} from "../types/api";

// SWR configuration for optimal caching
const swrConfig = {
  revalidateOnFocus: false,      // Don't refetch when window regains focus
  revalidateOnReconnect: false,  // Don't refetch on network reconnect
  dedupingInterval: 60000,       // Dedupe requests within 60 seconds
  keepPreviousData: true,        // Show stale data while fetching new data
};

export function useCoherence() {
  const { data, error, isLoading, isValidating } = useSWR<CoherenceResponse>(
    "/topics/coherence",
    swrFetcher,
    {
      ...swrConfig,
      revalidateIfStale: false,  // Coherence data doesn't change
    }
  );

  return {
    data: data ?? null,
    loading: isLoading,
    isValidating,
    error: error?.message ?? null,
  };
}

export function useTopicWords(nTopics: number, numWords: number = 10) {
  const { data, error, isLoading, isValidating } = useSWR<TopicWordsResponse>(
    `/topics/${nTopics}/words?num_words=${numWords}`,
    swrFetcher,
    {
      ...swrConfig,
      revalidateIfStale: false,  // Topic words don't change for same k
    }
  );

  // Prefetch adjacent topic counts in the background
  useEffect(() => {
    // Small delay to let current request complete first
    const timer = setTimeout(() => {
      // Prefetch k-1 and k+1
      if (nTopics > 2) {
        prefetch(`/topics/${nTopics - 1}/words?num_words=${numWords}`);
      }
      if (nTopics < 20) {
        prefetch(`/topics/${nTopics + 1}/words?num_words=${numWords}`);
      }
    }, 100);

    return () => clearTimeout(timer);
  }, [nTopics, numWords]);

  return {
    data: data ?? null,
    loading: isLoading,
    isValidating,
    error: error?.message ?? null,
  };
}

export function useClusterMetrics(nTopics: number) {
  const { data, error, isLoading, isValidating } = useSWR<ClusterMetricsResponse>(
    `/clustering/metrics/${nTopics}`,
    swrFetcher,
    swrConfig
  );

  // Prefetch adjacent topic counts
  useEffect(() => {
    const timer = setTimeout(() => {
      if (nTopics > 2) {
        prefetch(`/clustering/metrics/${nTopics - 1}`);
      }
      if (nTopics < 20) {
        prefetch(`/clustering/metrics/${nTopics + 1}`);
      }
    }, 100);

    return () => clearTimeout(timer);
  }, [nTopics]);

  return {
    data: data ?? null,
    loading: isLoading,
    isValidating,
    error: error?.message ?? null,
  };
}

// Preload function for warming the cache
export function preloadTopicData(nTopics: number, numWords: number = 10) {
  preload(`/topics/${nTopics}/words?num_words=${numWords}`, swrFetcher);
  preload(`/clustering/metrics/${nTopics}`, swrFetcher);
}
