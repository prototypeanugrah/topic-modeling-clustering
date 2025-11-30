import { useState, useEffect } from "react";
import { api } from "../api/client";
import type { ClusteredVisualizationResponse } from "../types/api";

export function useClusteredVisualization(nTopics: number, nClusters: number) {
  const [data, setData] = useState<ClusteredVisualizationResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    api
      .getClusteredVisualization({ n_topics: nTopics, n_clusters: nClusters })
      .then(setData)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [nTopics, nClusters]);

  return { data, loading, error };
}
