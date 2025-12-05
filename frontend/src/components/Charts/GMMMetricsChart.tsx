import Plot from "react-plotly.js";
import type { GMMMetricsResponse } from "../../types/api";
import { useTheme } from "../../hooks/useTheme";

interface GMMMetricsChartProps {
  data: GMMMetricsResponse | null;
  loading: boolean;
  isValidating?: boolean;
  selectedClusters: number;
  onSelectClusters?: (n: number) => void;
}

const CHART_HEIGHT = 350;

export function GMMMetricsChart({
  data,
  loading,
  isValidating,
  selectedClusters,
  onSelectClusters,
}: GMMMetricsChartProps) {
  const { isDark } = useTheme();

  // Theme-aware colors
  const colors = {
    bg: isDark ? "#21262d" : "#ffffff",
    text: isDark ? "#e6edf3" : "#1a1a1a",
    textMuted: isDark ? "#8b949e" : "#6b6b6b",
    grid: isDark ? "#30363d" : "#e5e5e0",
    bic: "#3b82f6", // Blue for BIC
    aic: "#10b981", // Green for AIC
    optimalBic: "#3b82f6",
    optimalAic: "#10b981",
    selected: "#ff6b35",
  };

  // Show full skeleton only on initial load (no cached data)
  if (loading && !data) {
    return (
      <div className="terminal-panel h-[480px] flex flex-col">
        <div className="terminal-panel-header">GMM Metrics</div>
        <div className="terminal-panel-content flex-1 flex flex-col items-center justify-center">
          <div className="terminal-loading">Computing GMM metrics</div>
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
          GMM Metrics
        </div>
        <div className="terminal-panel-content flex-1 flex flex-col items-center justify-center">
          <svg
            className="w-12 h-12 mb-3"
            style={{ color: "var(--status-error)" }}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
            />
          </svg>
          <span
            className="font-mono text-sm"
            style={{ color: "var(--status-error)" }}
          >
            Failed to load GMM metrics
          </span>
        </div>
      </div>
    );
  }

  const selectedIndex = data.cluster_counts.indexOf(selectedClusters);
  const optimalBicIndex = data.cluster_counts.indexOf(data.optimal_bic);
  const optimalAicIndex = data.cluster_counts.indexOf(data.optimal_aic);
  const showValidatingIndicator = isValidating && data;

  return (
    <div
      className={`terminal-panel h-[480px] flex flex-col relative transition-opacity ${
        showValidatingIndicator ? "opacity-80" : ""
      }`}
    >
      {showValidatingIndicator && (
        <div
          className="absolute top-3 right-3 z-10 flex items-center gap-2 font-mono text-xs"
          style={{ color: "var(--text-muted)" }}
        >
          <div
            className="status-dot status-dot--active"
            style={{ width: 6, height: 6 }}
          />
          <span>Updating...</span>
        </div>
      )}
      <div className="terminal-panel-header">
        GMM Optimal Clusters
        <span
          className="ml-2 font-mono text-xs"
          style={{ color: "var(--text-muted)" }}
        >
          {selectedClusters} clusters
        </span>
        <span
          className="ml-2 px-2 py-0.5 rounded text-xs"
          style={{ background: colors.bic, color: "white" }}
        >
          BIC={data.optimal_bic}
        </span>
        <span
          className="ml-1 px-2 py-0.5 rounded text-xs"
          style={{ background: colors.aic, color: "white" }}
        >
          AIC={data.optimal_aic}
        </span>
      </div>
      <div className="terminal-panel-content flex-1 min-h-0">
        <Plot
          key={isDark ? "dark" : "light"}
          data={[
            // BIC curve (primary y-axis)
            {
              x: data.cluster_counts,
              y: data.bic_scores,
              type: "scatter",
              mode: "lines+markers",
              marker: { color: colors.bic, size: 7 },
              line: { color: colors.bic, width: 2 },
              name: "BIC",
              yaxis: "y",
              hovertemplate:
                "<b>Metric:</b> BIC<br><b>Clusters:</b> %{x}<br><b>Value:</b> %{y:.2f}<extra></extra>",
            },
            // AIC curve (primary y-axis)
            {
              x: data.cluster_counts,
              y: data.aic_scores,
              type: "scatter",
              mode: "lines+markers",
              marker: { color: colors.aic, size: 7 },
              line: { color: colors.aic, width: 2 },
              name: "AIC",
              yaxis: "y",
              hovertemplate:
                "<b>Metric:</b> AIC<br><b>Clusters:</b> %{x}<br><b>Value:</b> %{y:.2f}<extra></extra>",
            },
            // Optimal BIC marker
            optimalBicIndex >= 0
              ? {
                  x: [data.optimal_bic],
                  y: [data.bic_scores[optimalBicIndex]],
                  type: "scatter",
                  mode: "markers",
                  marker: { color: colors.optimalBic, size: 14, symbol: "star" },
                  name: `Optimal BIC (k=${data.optimal_bic})`,
                  yaxis: "y",
                  hovertemplate: `<b>Optimal BIC</b><br><b>Clusters:</b> ${data.optimal_bic}<br><b>BIC:</b> %{y:.2f}<extra></extra>`,
                }
              : {},
            // Optimal AIC marker
            optimalAicIndex >= 0
              ? {
                  x: [data.optimal_aic],
                  y: [data.aic_scores[optimalAicIndex]],
                  type: "scatter",
                  mode: "markers",
                  marker: {
                    color: colors.optimalAic,
                    size: 14,
                    symbol: "diamond",
                  },
                  name: `Optimal AIC (k=${data.optimal_aic})`,
                  yaxis: "y",
                  hovertemplate: `<b>Optimal AIC</b><br><b>Clusters:</b> ${data.optimal_aic}<br><b>AIC:</b> %{y:.2f}<extra></extra>`,
                }
              : {},
            // Highlight selected point on BIC
            selectedIndex >= 0
              ? {
                  x: [selectedClusters],
                  y: [data.bic_scores[selectedIndex]],
                  type: "scatter",
                  mode: "markers",
                  marker: {
                    color: colors.selected,
                    size: 12,
                    line: { color: colors.bg, width: 2 },
                  },
                  name: `Selected (k=${selectedClusters})`,
                  yaxis: "y",
                  hovertemplate: `<b>Selected</b><br><b>Clusters:</b> ${selectedClusters}<br><b>BIC:</b> %{y:.2f}<extra></extra>`,
                }
              : {},
          ]}
          layout={{
            autosize: true,
            height: CHART_HEIGHT,
            margin: { l: 70, r: 30, t: 30, b: 70 },
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
              title: { text: "BIC / AIC (lower is better)", font: { size: 11 } },
              side: "left",
              gridcolor: colors.grid,
              linecolor: colors.grid,
              tickfont: { size: 10 },
            },
            showlegend: true,
            legend: {
              x: 0,
              y: 1.15,
              orientation: "h",
              font: { size: 9 },
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
