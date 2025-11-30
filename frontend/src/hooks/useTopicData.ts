import { useState, useEffect } from "react";
import { api } from "../api/client";
import type {
  CoherenceResponse,
  TopicWordsResponse,
  ClusterMetricsResponse,
} from "../types/api";

export function useCoherence() {
  const [data, setData] = useState<CoherenceResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    console.log("Fetching coherence...");
    api
      .getCoherence()
      .then((result) => {
        console.log("Coherence result:", result);
        setData(result);
      })
      .catch((err) => {
        console.error("Coherence error:", err);
        setError(err.message);
      })
      .finally(() => {
        console.log("Coherence loading complete");
        setLoading(false);
      });
  }, []);

  return { data, loading, error };
}

export function useTopicWords(nTopics: number, numWords: number = 10) {
  const [data, setData] = useState<TopicWordsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    api
      .getTopicWords(nTopics, numWords)
      .then(setData)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [nTopics, numWords]);

  return { data, loading, error };
}

export function useClusterMetrics(nTopics: number) {
  const [data, setData] = useState<ClusterMetricsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    api
      .getClusterMetrics(nTopics)
      .then(setData)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [nTopics]);

  return { data, loading, error };
}
