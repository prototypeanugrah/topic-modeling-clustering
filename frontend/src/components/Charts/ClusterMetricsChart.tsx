import Plot from "react-plotly.js";
import type { ClusterMetricsResponse } from "../../types/api";
import { useTheme } from "../../hooks/useTheme";

interface ClusterMetricsChartProps {
  data: ClusterMetricsResponse | null;
  loading: boolean;
  isValidating?: boolean;
  selectedClusters: number;
  onSelectClusters?: (n: number) => void;
}

const CHART_HEIGHT = 350;

export function ClusterMetricsChart({
  data,
  loading,
  isValidating,
  selectedClusters,
  onSelectClusters,
}: ClusterMetricsChartProps) {
  const { isDark } = useTheme();

  // Theme-aware colors
  const colors = {
    bg: isDark ? '#21262d' : '#ffffff',
    text: isDark ? '#e6edf3' : '#1a1a1a',
    textMuted: isDark ? '#8b949e' : '#6b6b6b',
    grid: isDark ? '#30363d' : '#e5e5e0',
    silhouette: '#a855f7',
    inertia: isDark ? '#6e7681' : '#6b6b6b',
    success: '#00d4aa',
    selected: '#ff6b35',
  };

  // Show full skeleton only on initial load (no cached data)
  if (loading && !data) {
    return (
      <div className="terminal-panel h-[480px] flex flex-col">
        <div className="terminal-panel-header">
          Cluster Metrics
        </div>
        <div className="terminal-panel-content flex-1 flex flex-col items-center justify-center">
          <div className="terminal-loading">Computing metrics</div>
          <div className="loading-bar w-48 mt-4">
            <div className="loading-bar-progress"></div>
          </div>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="terminal-panel h-[480px] flex flex-col">
        <div className="terminal-panel-header">
          <span className="status-dot status-dot--error"></span>
          Cluster Metrics
        </div>
        <div className="terminal-panel-content flex-1 flex flex-col items-center justify-center">
          <svg className="w-12 h-12 mb-3" style={{ color: 'var(--status-error)' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
          <span className="font-mono text-sm" style={{ color: 'var(--status-error)' }}>Failed to load cluster metrics</span>
        </div>
      </div>
    );
  }

  const selectedIndex = data.cluster_counts.indexOf(selectedClusters);
  const showValidatingIndicator = isValidating && data;

  return (
    <div className={`terminal-panel h-[480px] flex flex-col relative transition-opacity ${showValidatingIndicator ? 'opacity-80' : ''}`}>
      {showValidatingIndicator && (
        <div className="absolute top-3 right-3 z-10 flex items-center gap-2 font-mono text-xs" style={{ color: 'var(--text-muted)' }}>
          <div className="status-dot status-dot--active" style={{ width: 6, height: 6 }} />
          <span>Updating...</span>
        </div>
      )}
      <div className="terminal-panel-header">
        Optimal Clusters
        <span className="ml-2 font-mono text-xs" style={{ color: 'var(--text-muted)' }}>
          {selectedClusters} clusters
        </span>
        {data.elbow_point && (
          <span className="ml-2 px-2 py-0.5 rounded text-xs" style={{ background: 'var(--accent-secondary)', color: 'white' }}>
            elbow={data.elbow_point}
          </span>
        )}
      </div>
      <div className="terminal-panel-content flex-1 min-h-0">
        <Plot
          key={isDark ? 'dark' : 'light'}
          data={[
            // Silhouette score (primary y-axis)
            {
              x: data.cluster_counts,
              y: data.silhouette_scores,
              type: "scatter",
              mode: "lines+markers",
              marker: { color: colors.silhouette, size: 8 },
              line: { color: colors.silhouette, width: 2 },
              name: "Silhouette",
              yaxis: "y",
              hovertemplate: "<b>Metric:</b> Silhouette<br><b>Clusters:</b> %{x}<br><b>Value:</b> %{y:.6f}<extra></extra>",
            },
            // Inertia (secondary y-axis)
            {
              x: data.cluster_counts,
              y: data.inertia_scores,
              type: "scatter",
              mode: "lines+markers",
              marker: { color: colors.inertia, size: 6 },
              line: { color: colors.inertia, width: 1, dash: "dash" },
              name: "Inertia",
              yaxis: "y2",
              hovertemplate: "<b>Metric:</b> Inertia<br><b>Clusters:</b> %{x}<br><b>Value:</b> %{y:.2f}<extra></extra>",
            },
            // Highlight elbow point on the Inertia line
            data.elbow_point
              ? {
                  x: [data.elbow_point],
                  y: [data.inertia_scores[data.cluster_counts.indexOf(data.elbow_point)]],
                  type: "scatter",
                  mode: "markers",
                  marker: { color: colors.success, size: 14, symbol: "star" },
                  name: `Elbow (k=${data.elbow_point})`,
                  yaxis: "y2",
                  hovertemplate: `<b>Elbow Point</b><br><b>Clusters:</b> ${data.elbow_point}<br><b>Inertia:</b> %{y:.2f}<extra></extra>`,
                }
              : {},
            // Highlight selected point
            selectedIndex >= 0
              ? {
                  x: [selectedClusters],
                  y: [data.silhouette_scores[selectedIndex]],
                  type: "scatter",
                  mode: "markers",
                  marker: {
                    color: colors.selected,
                    size: 12,
                    line: { color: colors.bg, width: 2 },
                  },
                  name: `Selected (k=${selectedClusters})`,
                  hovertemplate: `<b>Selected</b><br><b>Clusters:</b> ${selectedClusters}<br><b>Silhouette:</b> %{y:.6f}<extra></extra>`,
                }
              : {},
          ]}
          layout={{
            autosize: true,
            height: CHART_HEIGHT,
            margin: { l: 60, r: 60, t: 30, b: 70 },
            paper_bgcolor: colors.bg,
            plot_bgcolor: colors.bg,
            font: {
              family: "'JetBrains Mono', 'SF Mono', monospace",
              color: colors.text,
              size: 11,
            },
            xaxis: {
              title: { text: "Number of Clusters", font: { size: 11 } },
              dtick: 1,
              gridcolor: colors.grid,
              linecolor: colors.grid,
              tickfont: { size: 10 },
            },
            yaxis: {
              title: { text: "Silhouette Score", font: { size: 11 } },
              side: "left",
              gridcolor: colors.grid,
              linecolor: colors.grid,
              tickfont: { size: 10 },
            },
            yaxis2: {
              title: { text: "Inertia", font: { size: 11 } },
              side: "right",
              overlaying: "y",
              gridcolor: colors.grid,
              linecolor: colors.grid,
              tickfont: { size: 10 },
            },
            showlegend: true,
            legend: {
              x: 0,
              y: 1.15,
              orientation: "h",
              font: { size: 10 },
            },
          }}
          config={{ displayModeBar: false }}
          style={{ width: "100%", height: "100%" }}
          onClick={(event) => {
            if (event.points && event.points[0] && onSelectClusters) {
              const x = event.points[0].x;
              if (typeof x === "number") {
                onSelectClusters(x);
              }
            }
          }}
        />
      </div>
    </div>
  );
}
