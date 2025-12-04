import useSWR from "swr";
import { swrFetcher } from "../api/client";
import type { BoxPlotData, EDAResponse } from "../types/api";

// SWR configuration - EDA data is static
const swrConfig = {
  revalidateOnFocus: false,
  revalidateOnReconnect: false,
  dedupingInterval: 60000,
  keepPreviousData: true,
  revalidateIfStale: false, // EDA data doesn't change
};

export function useEDA() {
  const { data, error, isLoading, isValidating } = useSWR<EDAResponse>(
    "/eda",
    swrFetcher,
    swrConfig
  );

  return {
    data: data ?? null,
    loading: isLoading,
    isValidating,
    error: error?.message ?? null,
  };
}

export function useBoxPlotData() {
  const { data, error, isLoading, isValidating } = useSWR<BoxPlotData>(
    "/eda/boxplot",
    swrFetcher,
    swrConfig
  );

  return {
    data: data ?? null,
    loading: isLoading,
    isValidating,
    error: error?.message ?? null,
  };
}
