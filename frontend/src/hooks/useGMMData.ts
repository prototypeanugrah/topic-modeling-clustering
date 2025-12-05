import useSWR from "swr";
import { useEffect } from "react";
import { swrFetcher, swrPostFetcher, prefetch, prefetchPost } from "../api/client";
import type {
  GMMMetricsResponse,
  GMMAllCovarianceMetricsResponse,
  GMMClusteredVisualizationResponse,
  CovarianceType,
} from "../types/api";

// SWR configuration for optimal caching
const swrConfig = {
  revalidateOnFocus: false,
  revalidateOnReconnect: false,
  dedupingInterval: 60000,
  keepPreviousData: true,
};

/**
 * Hook for fetching GMM metrics for a specific covariance type.
 */
export function useGMMMetrics(nTopics: number, covarianceType: CovarianceType) {
  const { data, error, isLoading, isValidating } = useSWR<GMMMetricsResponse>(
    `/gmm/metrics/${nTopics}?covariance_type=${covarianceType}`,
    swrFetcher,
    swrConfig
  );

  // Prefetch other covariance types
  useEffect(() => {
    const timer = setTimeout(() => {
      const covTypes: CovarianceType[] = ["full", "diag", "spherical"];
      covTypes
        .filter((ct) => ct !== covarianceType)
        .forEach((ct) => {
          prefetch(`/gmm/metrics/${nTopics}?covariance_type=${ct}`);
        });
    }, 200);

    return () => clearTimeout(timer);
  }, [nTopics, covarianceType]);

  return {
    data: data ?? null,
    loading: isLoading,
    isValidating,
    error: error?.message ?? null,
  };
}

/**
 * Hook for fetching GMM metrics for all covariance types at once.
 */
export function useGMMAllMetrics(nTopics: number) {
  const { data, error, isLoading, isValidating } =
    useSWR<GMMAllCovarianceMetricsResponse>(
      `/gmm/metrics/all/${nTopics}`,
      swrFetcher,
      swrConfig
    );

  // Prefetch adjacent topic counts
  useEffect(() => {
    const timer = setTimeout(() => {
      if (nTopics > 2) {
        prefetch(`/gmm/metrics/all/${nTopics - 1}`);
      }
      if (nTopics < 20) {
        prefetch(`/gmm/metrics/all/${nTopics + 1}`);
      }
    }, 200);

    return () => clearTimeout(timer);
  }, [nTopics]);

  return {
    data: data ?? null,
    loading: isLoading,
    isValidating,
    error: error?.message ?? null,
  };
}

/**
 * Hook for fetching GMM clustered visualization data.
 */
export function useGMMClusteredVisualization(
  nTopics: number,
  nClusters: number,
  covarianceType: CovarianceType
) {
  const { data, error, isLoading, isValidating } =
    useSWR<GMMClusteredVisualizationResponse>(
      [
        "/visualization/gmm-clustered",
        { n_topics: nTopics, n_clusters: nClusters, covariance_type: covarianceType },
      ],
      swrPostFetcher,
      swrConfig
    );

  // Prefetch adjacent cluster counts
  useEffect(() => {
    const timer = setTimeout(() => {
      // Prefetch adjacent cluster counts
      if (nClusters > 2) {
        prefetchPost("/visualization/gmm-clustered", {
          n_topics: nTopics,
          n_clusters: nClusters - 1,
          covariance_type: covarianceType,
        });
      }
      if (nClusters < 15) {
        prefetchPost("/visualization/gmm-clustered", {
          n_topics: nTopics,
          n_clusters: nClusters + 1,
          covariance_type: covarianceType,
        });
      }
    }, 200);

    return () => clearTimeout(timer);
  }, [nTopics, nClusters, covarianceType]);

  return {
    data: data ?? null,
    loading: isLoading,
    isValidating,
    error: error?.message ?? null,
  };
}
