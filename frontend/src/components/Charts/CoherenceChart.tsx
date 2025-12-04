import Plot from "react-plotly.js";
import type { CoherenceResponse } from "../../types/api";
import { useTheme } from "../../hooks/useTheme";

interface CoherenceChartProps {
  data: CoherenceResponse | null;
  loading: boolean;
  isValidating?: boolean;
  selectedTopics: number;
  onSelectTopics?: (n: number) => void;
}

const CHART_HEIGHT = 350;

export function CoherenceChart({
  data,
  loading,
  isValidating,
  selectedTopics,
  onSelectTopics,
}: CoherenceChartProps) {
  const { isDark } = useTheme();

  // Theme-aware colors
  const colors = {
    bg: isDark ? '#21262d' : '#ffffff',
    text: isDark ? '#e6edf3' : '#1a1a1a',
    textMuted: isDark ? '#8b949e' : '#6b6b6b',
    grid: isDark ? '#30363d' : '#e5e5e0',
    primary: '#58a6ff',
    success: '#00d4aa',
    selected: '#ff6b35',
  };

  // Show full skeleton only on initial load (no cached data)
  if (loading && !data) {
    return (
      <div className="terminal-panel h-[480px] flex flex-col">
        <div className="terminal-panel-header">
          Coherence Analysis
        </div>
        <div className="terminal-panel-content flex-1 flex flex-col items-center justify-center">
          <div className="terminal-loading">Loading coherence data</div>
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
          Coherence Analysis
        </div>
        <div className="terminal-panel-content flex-1 flex flex-col items-center justify-center">
          <svg className="w-12 h-12 mb-3" style={{ color: 'var(--status-error)' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
          <span className="font-mono text-sm" style={{ color: 'var(--status-error)' }}>Failed to load coherence data</span>
        </div>
      </div>
    );
  }

  const selectedIndex = data.topic_counts.indexOf(selectedTopics);
  const hasCoherence = data.coherence && data.coherence.length > 0;
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
        Optimal Topics
        <span className="ml-2 font-mono text-xs" style={{ color: 'var(--text-muted)' }}>
          {selectedTopics} topics
        </span>
        {data.optimal_topics && (
          <span className="ml-2 px-2 py-0.5 rounded text-xs" style={{ background: 'var(--accent-secondary)', color: 'white' }}>
            optimal={data.optimal_topics}
          </span>
        )}
      </div>
      <div className="terminal-panel-content flex-1 min-h-0">
        <Plot
          key={isDark ? 'dark' : 'light'}
          data={[
            // Coherence line
            ...(hasCoherence
              ? [
                  {
                    x: data.topic_counts,
                    y: data.coherence,
                    type: "scatter" as const,
                    mode: "lines+markers" as const,
                    marker: { color: colors.primary, size: 8 },
                    line: { color: colors.primary, width: 2 },
                    name: "Coherence",
                    yaxis: "y",
                    hovertemplate: "<b>Metric:</b> Coherence<br><b>Topics:</b> %{x}<br><b>Value:</b> %{y:.6f}<extra></extra>",
                  },
                ]
              : []),
            // Highlight optimal point
            ...(hasCoherence
              ? [
                  {
                    x: [data.optimal_topics],
                    y: [data.coherence[data.topic_counts.indexOf(data.optimal_topics)]],
                    type: "scatter" as const,
                    mode: "markers" as const,
                    marker: { color: colors.success, size: 14, symbol: "star" },
                    name: `Optimal (k=${data.optimal_topics})`,
                    yaxis: "y",
                    hovertemplate: `<b>Optimal Point</b><br><b>Topics:</b> ${data.optimal_topics}<br><b>Coherence:</b> %{y:.6f}<extra></extra>`,
                  },
                ]
              : []),
            // Highlight selected point
            ...(selectedIndex >= 0 && hasCoherence
              ? [
                  {
                    x: [selectedTopics],
                    y: [data.coherence[selectedIndex]],
                    type: "scatter" as const,
                    mode: "markers" as const,
                    marker: {
                      color: colors.selected,
                      size: 12,
                      line: { color: colors.bg, width: 2 },
                    },
                    name: `Selected (k=${selectedTopics})`,
                    yaxis: "y",
                    hovertemplate: `<b>Selected</b><br><b>Topics:</b> ${selectedTopics}<br><b>Coherence:</b> %{y:.6f}<extra></extra>`,
                  },
                ]
              : []),
          ]}
          layout={{
            autosize: true,
            height: CHART_HEIGHT,
            margin: { l: 60, r: 30, t: 30, b: 70 },
            paper_bgcolor: colors.bg,
            plot_bgcolor: colors.bg,
            font: {
              family: "'JetBrains Mono', 'SF Mono', monospace",
              color: colors.text,
              size: 11,
            },
            xaxis: {
              title: { text: "Number of Topics", font: { size: 11 } },
              dtick: 1,
              gridcolor: colors.grid,
              linecolor: colors.grid,
              tickfont: { size: 10 },
            },
            yaxis: {
              title: { text: "Coherence Score", font: { size: 11 } },
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
              font: { size: 10 },
            },
          }}
          config={{ displayModeBar: false }}
          style={{ width: "100%", height: "100%" }}
          onClick={(event) => {
            if (event.points && event.points[0] && onSelectTopics) {
              const x = event.points[0].x;
              if (typeof x === "number") {
                onSelectTopics(x);
              }
            }
          }}
        />
      </div>
    </div>
  );
}
