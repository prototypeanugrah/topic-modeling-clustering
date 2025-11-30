import { useState, useEffect } from "react";
import { ControlPanel } from "./ControlPanel";
import { CoherenceChart } from "./Charts/CoherenceChart";
import { ClusterMetricsChart } from "./Charts/ClusterMetricsChart";
import { ScatterPlot } from "./Visualization/ScatterPlot";
import { TopicWordList } from "./Topics/TopicWordList";
import { PyLDAvisViewer } from "./Topics/PyLDAvisViewer";
import { LoadingSkeleton } from "./ui/LoadingSkeleton";
import { useDebounce } from "../hooks/useDebounce";
import { useCoherence, useTopicWords, useClusterMetrics } from "../hooks/useTopicData";
import { useClusteredVisualization } from "../hooks/useClusterData";
import { api } from "../api/client";
import type { HealthResponse } from "../types/api";

export function Dashboard() {
  // State - start with minimum values
  const [nTopics, setNTopics] = useState(2);
  const [nClusters, setNClusters] = useState(2);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthLoading, setHealthLoading] = useState(true);
  const [healthError, setHealthError] = useState<string | null>(null);

  // Debounce slider values - reduced delay for snappier feel
  const debouncedTopics = useDebounce(nTopics, 150);
  const debouncedClusters = useDebounce(nClusters, 150);

  // Fetch data - all independent, no blocking
  const { data: coherenceData, loading: coherenceLoading, error: coherenceError } = useCoherence();
  const { data: topicWordsData, loading: topicWordsLoading } = useTopicWords(debouncedTopics);
  const { data: clusterMetricsData, loading: clusterMetricsLoading } = useClusterMetrics(debouncedTopics);
  const { data: visualizationData, loading: visualizationLoading } = useClusteredVisualization(
    debouncedTopics,
    debouncedClusters
  );

  // Check health on mount
  useEffect(() => {
    api
      .getHealth()
      .then(setHealth)
      .catch((err) => setHealthError(err.message))
      .finally(() => setHealthLoading(false));
  }, []);

  // Show error if backend is down
  if (healthError) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 flex items-center justify-center p-8">
        <div className="bg-white rounded-2xl shadow-xl p-8 max-w-md text-center border border-red-100">
          <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg className="w-8 h-8 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          </div>
          <h1 className="text-xl font-bold text-gray-800 mb-2">
            Backend Unavailable
          </h1>
          <p className="text-gray-600 mb-4">
            Could not connect to the API server.
          </p>
          <code className="block bg-gray-100 rounded-lg p-3 text-sm font-mono text-gray-700">
            uv run uvicorn backend.app:app --reload
          </code>
        </div>
      </div>
    );
  }

  // Show precomputation required only if coherence explicitly failed (not just loading)
  if (!coherenceLoading && coherenceError) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 flex items-center justify-center p-8">
        <div className="bg-white rounded-2xl shadow-xl p-8 max-w-md text-center border border-amber-100">
          <div className="w-16 h-16 bg-amber-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg className="w-8 h-8 text-amber-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          </div>
          <h1 className="text-xl font-bold text-gray-800 mb-2">
            Precomputation Required
          </h1>
          <p className="text-gray-600 mb-6">
            The topic models need to be precomputed before using the dashboard.
          </p>
          <div className="space-y-4">
            <code className="block bg-gray-100 rounded-lg p-3 text-sm font-mono text-gray-700">
              uv run python scripts/precompute.py
            </code>
            <p className="text-xs text-gray-400">
              For testing: --min-topics 2 --max-topics 4
            </p>
          </div>
        </div>
      </div>
    );
  }

  // Derive min/max from coherence data or use defaults
  const minTopics = coherenceData ? Math.min(...coherenceData.topic_counts) : 2;
  const maxTopics = coherenceData ? Math.max(...coherenceData.topic_counts) : 20;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      <div className="max-w-[1600px] mx-auto px-4 py-6 md:px-6 md:py-8">
        {/* Header */}
        <header className="bg-gradient-to-r from-indigo-600 to-purple-600 rounded-2xl shadow-lg p-6 mb-6 text-white">
          <h1 className="text-2xl md:text-3xl font-bold">
            Topic Modeling & Clustering Dashboard
          </h1>
          <p className="text-indigo-100 text-sm mt-2">
            Explore topic distributions and document clusters in the 20 Newsgroups dataset
          </p>
        </header>

        {/* Row 1: Controls + Top Words per Topic */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
          {/* Control Panel */}
          <div className="lg:col-span-1">
            {coherenceLoading ? (
              <LoadingSkeleton height="h-full" title="Controls" message="Loading configuration..." />
            ) : (
              <ControlPanel
                nTopics={nTopics}
                nClusters={nClusters}
                onTopicsChange={setNTopics}
                onClustersChange={setNClusters}
                optimalTopics={coherenceData?.optimal_topics}
                optimalClusters={clusterMetricsData?.elbow_point ?? undefined}
                minTopics={minTopics}
                maxTopics={maxTopics}
              />
            )}
          </div>

          {/* Topic Words - spans 2 columns */}
          <div className="lg:col-span-2">
            <TopicWordList
              data={topicWordsData}
              loading={topicWordsLoading}
            />
          </div>
        </div>

        {/* Row 2: Charts */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <CoherenceChart
            data={coherenceData}
            loading={coherenceLoading}
            selectedTopics={nTopics}
            onSelectTopics={setNTopics}
          />
          <ClusterMetricsChart
            data={clusterMetricsData}
            loading={clusterMetricsLoading}
            selectedClusters={nClusters}
            onSelectClusters={setNClusters}
          />
        </div>

        {/* Row 3: Scatter plot */}
        <div className="mb-6">
          <ScatterPlot
            data={visualizationData}
            loading={visualizationLoading}
          />
        </div>

        {/* Row 4: pyLDAvis - full width */}
        <div className="mb-6">
          <PyLDAvisViewer
            nTopics={debouncedTopics}
          />
        </div>

        {/* Footer */}
        <footer className="text-center text-sm text-gray-500 py-6">
          Topic Modeling with LDA | K-Means Clustering | UMAP Visualization
        </footer>
      </div>
    </div>
  );
}
