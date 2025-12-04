import { useState, useEffect } from "react";
import { ControlPanel } from "./ControlPanel";
import { CoherenceChart } from "./Charts/CoherenceChart";
import { ClusterMetricsChart } from "./Charts/ClusterMetricsChart";
import { ScatterPlot } from "./Visualization/ScatterPlot";
import { TopicWordList } from "./Topics/TopicWordList";
import { PyLDAvisViewer } from "./Topics/PyLDAvisViewer";
import { DatasetOverview } from "./EDA/DatasetOverview";
import { NewsgroupCategoriesBoxPlot } from "./EDA/NewsgroupCategoriesBoxPlot";
import { LoadingSkeleton } from "./ui/LoadingSkeleton";
import { ThemeToggle } from "./ui/ThemeToggle";
import { useDebounce } from "../hooks/useDebounce";
import { useCoherence, useTopicWords, useClusterMetrics } from "../hooks/useTopicData";
import { useClusteredVisualization } from "../hooks/useClusterData";
import { useEDA, useBoxPlotData } from "../hooks/useEDA";
import { api } from "../api/client";
import type { HealthResponse } from "../types/api";

export function Dashboard() {
  // State - start with minimum values
  const [nTopics, setNTopics] = useState(2);
  const [nClusters, setNClusters] = useState(2);
  const [_health, setHealth] = useState<HealthResponse | null>(null);
  const [_healthLoading, setHealthLoading] = useState(true);
  const [healthError, setHealthError] = useState<string | null>(null);

  // Debounce slider values - reduced delay for snappier feel
  const debouncedTopics = useDebounce(nTopics, 150);
  const debouncedClusters = useDebounce(nClusters, 150);

  // Fetch data - all independent, no blocking
  // isValidating = background revalidation (show stale data with subtle indicator)
  const { data: coherenceData, loading: coherenceLoading, error: coherenceError, isValidating: coherenceValidating } = useCoherence();
  const { data: topicWordsData, loading: topicWordsLoading, isValidating: topicWordsValidating } = useTopicWords(debouncedTopics);
  const { data: clusterMetricsData, loading: clusterMetricsLoading, isValidating: clusterMetricsValidating } = useClusterMetrics(debouncedTopics);
  const { data: visualizationData, loading: visualizationLoading, isValidating: visualizationValidating } = useClusteredVisualization(
    debouncedTopics,
    debouncedClusters
  );
  const { data: edaData, loading: edaLoading, isValidating: edaValidating } = useEDA();
  const { data: boxPlotData, loading: boxPlotLoading, isValidating: boxPlotValidating } = useBoxPlotData();

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
      <div className="min-h-screen flex items-center justify-center p-8" style={{ background: 'var(--bg-primary)' }}>
        <div className="terminal-panel max-w-md text-center">
          <div className="terminal-panel-header">
            <span className="status-dot status-dot--error"></span>
            System Error
          </div>
          <div className="terminal-panel-content">
            <div
              className="w-16 h-16 rounded flex items-center justify-center mx-auto mb-4"
              style={{ background: 'var(--bg-secondary)' }}
            >
              <svg className="w-8 h-8" style={{ color: 'var(--status-error)' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
            </div>
            <h1 className="text-xl font-semibold mb-2 font-mono" style={{ color: 'var(--text-primary)' }}>
              Backend Unavailable
            </h1>
            <p className="mb-4 text-sm" style={{ color: 'var(--text-secondary)' }}>
              Could not connect to the API server.
            </p>
            <code
              className="block rounded p-3 text-sm font-mono"
              style={{ background: 'var(--bg-secondary)', color: 'var(--accent-primary)' }}
            >
              uv run uvicorn backend.app:app --reload
            </code>
          </div>
        </div>
      </div>
    );
  }

  // Show precomputation required only if coherence explicitly failed (not just loading)
  if (!coherenceLoading && coherenceError) {
    return (
      <div className="min-h-screen flex items-center justify-center p-8" style={{ background: 'var(--bg-primary)' }}>
        <div className="terminal-panel max-w-md text-center">
          <div className="terminal-panel-header">
            <span className="status-dot status-dot--warning"></span>
            Initialization Required
          </div>
          <div className="terminal-panel-content">
            <div
              className="w-16 h-16 rounded flex items-center justify-center mx-auto mb-4"
              style={{ background: 'var(--bg-secondary)' }}
            >
              <svg className="w-8 h-8" style={{ color: 'var(--status-warning)' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <h1 className="text-xl font-semibold mb-2 font-mono" style={{ color: 'var(--text-primary)' }}>
              Precomputation Required
            </h1>
            <p className="mb-6 text-sm" style={{ color: 'var(--text-secondary)' }}>
              The topic models need to be precomputed before using the dashboard.
            </p>
            <div className="space-y-4">
              <code
                className="block rounded p-3 text-sm font-mono"
                style={{ background: 'var(--bg-secondary)', color: 'var(--accent-primary)' }}
              >
                uv run python scripts/precompute.py
              </code>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                For testing: --min-topics 2 --max-topics 4
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Derive min/max from backend data or use defaults while loading
  const minTopics = coherenceData ? Math.min(...coherenceData.topic_counts) : 2;
  const maxTopics = coherenceData ? Math.max(...coherenceData.topic_counts) : 20;
  const minClusters = clusterMetricsData ? Math.min(...clusterMetricsData.cluster_counts) : 2;
  const maxClusters = clusterMetricsData ? Math.max(...clusterMetricsData.cluster_counts) : 15;

  return (
    <div className="min-h-screen" style={{ background: 'var(--bg-primary)' }}>
      {/* Fixed Sidebar - hidden on mobile, visible on lg+ */}
      <aside
        className="hidden lg:flex lg:flex-col lg:fixed lg:left-0 lg:top-0 lg:h-screen lg:w-72 lg:z-40"
        style={{
          background: 'var(--bg-card)',
          borderRight: '1px solid var(--border-color)',
          boxShadow: 'var(--shadow-lg)'
        }}
      >
        {/* Sidebar Header */}
        <div
          className="p-4"
          style={{
            background: 'var(--bg-secondary)',
            borderBottom: '1px solid var(--border-color)'
          }}
        >
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <div className="status-dot status-dot--active"></div>
              <span className="font-mono text-xs uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>
                System Active
              </span>
            </div>
            <ThemeToggle />
          </div>
          <h1 className="text-lg font-semibold font-mono" style={{ color: 'var(--text-primary)' }}>
            Topic Modeling
          </h1>
          <p className="text-xs mt-1 font-mono" style={{ color: 'var(--text-muted)' }}>
            20 Newsgroups Dataset
          </p>
        </div>

        {/* Controls */}
        <div className="flex-1 p-4 overflow-y-auto custom-scrollbar">
          {coherenceLoading ? (
            <LoadingSkeleton height="h-40" title="Controls" message="Loading..." />
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
              minClusters={minClusters}
              maxClusters={maxClusters}
            />
          )}
        </div>
      </aside>

      {/* Main Content - offset by sidebar width on lg+ */}
      <main className="lg:ml-72">
        <div className="max-w-[1400px] mx-auto px-4 py-6 md:px-6 md:py-8">
          {/* Mobile Header - visible only on mobile */}
          <header
            className="lg:hidden terminal-panel mb-6"
          >
            <div className="terminal-panel-header justify-between">
              <div className="flex items-center gap-2">
                <span>Topic Modeling Dashboard</span>
              </div>
              <ThemeToggle />
            </div>
            <div className="terminal-panel-content">
              <h1 className="text-xl font-semibold font-mono" style={{ color: 'var(--text-primary)' }}>
                Topic Modeling & Clustering
              </h1>
              <p className="text-sm mt-1 font-mono" style={{ color: 'var(--text-muted)' }}>
                20 Newsgroups dataset
              </p>
            </div>
          </header>

          {/* Mobile Controls - visible only on mobile */}
          <div className="lg:hidden mb-6">
            {coherenceLoading ? (
              <LoadingSkeleton height="h-40" title="Controls" message="Loading..." />
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
                minClusters={minClusters}
                maxClusters={maxClusters}
              />
            )}
          </div>

          {/* Dataset Overview (EDA) */}
          <div className="mb-6 animate-slide-up delay-1">
            <DatasetOverview
              data={edaData}
              loading={edaLoading}
              isValidating={edaValidating}
            />
          </div>

          {/* Category Box Plot */}
          <div className="mb-6 animate-slide-up delay-2">
            <NewsgroupCategoriesBoxPlot
              data={boxPlotData}
              loading={boxPlotLoading}
              isValidating={boxPlotValidating}
            />
          </div>

          {/* Top Words per Topic - Full Width */}
          <div className="mb-6 animate-slide-up delay-3">
            <TopicWordList
              data={topicWordsData}
              loading={topicWordsLoading}
              isValidating={topicWordsValidating}
            />
          </div>

          {/* Charts Row */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            <div className="animate-slide-up delay-4">
              <CoherenceChart
                data={coherenceData}
                loading={coherenceLoading}
                isValidating={coherenceValidating}
                selectedTopics={nTopics}
                onSelectTopics={setNTopics}
              />
            </div>
            <div className="animate-slide-up delay-5">
              <ClusterMetricsChart
                data={clusterMetricsData}
                loading={clusterMetricsLoading}
                isValidating={clusterMetricsValidating}
                selectedClusters={nClusters}
                onSelectClusters={setNClusters}
              />
            </div>
          </div>

          {/* Scatter plot */}
          <div className="mb-6 animate-fade-in">
            <ScatterPlot
              data={visualizationData}
              loading={visualizationLoading}
              isValidating={visualizationValidating}
            />
          </div>

          {/* pyLDAvis */}
          <div className="mb-6 animate-fade-in">
            <PyLDAvisViewer
              nTopics={debouncedTopics}
            />
          </div>

        </div>
      </main>
    </div>
  );
}
