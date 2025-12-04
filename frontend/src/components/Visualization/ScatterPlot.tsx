import Plot from "react-plotly.js";
import type { ClusteredVisualizationResponse, DocumentTopicInfo } from "../../types/api";
import { useTheme } from "../../hooks/useTheme";

interface ScatterPlotProps {
  data: ClusteredVisualizationResponse | null;
  loading: boolean;
  isValidating?: boolean;
}

// Industrial color palette for clusters
const CLUSTER_COLORS = [
  "#58a6ff", // blue
  "#f85149", // red
  "#00d4aa", // teal/green
  "#d29922", // amber
  "#a855f7", // purple
  "#ff6b35", // orange
  "#06b6d4", // cyan
  "#84cc16", // lime
  "#ec4899", // pink
  "#6366f1", // indigo
  "#14b8a6", // teal
  "#f59e0b", // yellow
  "#8b5cf6", // violet
  "#64748b", // slate
  "#78716c", // stone
];

export function ScatterPlot({ data, loading, isValidating }: ScatterPlotProps) {
  const { isDark } = useTheme();

  // Theme-aware colors
  const colors = {
    bg: isDark ? '#21262d' : '#ffffff',
    text: isDark ? '#e6edf3' : '#1a1a1a',
    textMuted: isDark ? '#8b949e' : '#6b6b6b',
    grid: isDark ? '#30363d' : '#e5e5e0',
  };

  // Show full skeleton only on initial load (no cached data)
  if (loading && !data) {
    return (
      <div className="terminal-panel h-[550px] flex flex-col">
        <div className="terminal-panel-header">
          Document Clusters
        </div>
        <div className="terminal-panel-content flex-1 flex flex-col items-center justify-center">
          <div className="terminal-loading">Rendering visualization</div>
          <div className="loading-bar w-64 mt-4">
            <div className="loading-bar-progress"></div>
          </div>
          <span className="font-mono text-xs mt-3" style={{ color: 'var(--text-muted)' }}>
            Processing documents...
          </span>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="terminal-panel h-[550px] flex flex-col">
        <div className="terminal-panel-header">
          <span className="status-dot status-dot--error"></span>
          Document Clusters
        </div>
        <div className="terminal-panel-content flex-1 flex flex-col items-center justify-center">
          <svg className="w-12 h-12 mb-3" style={{ color: 'var(--status-error)' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
          <span className="font-mono text-sm" style={{ color: 'var(--status-error)' }}>Failed to load visualization</span>
        </div>
      </div>
    );
  }

  // Group points by cluster with enrichment data
  const clusterGroups: Record<
    number,
    {
      x: number[];
      y: number[];
      ids: number[];
      labels: (string | undefined)[];
      topTopics: (DocumentTopicInfo[] | undefined)[];
      topicWords: (string[] | undefined)[];
    }
  > = {};

  data.projections.forEach((point, idx) => {
    const cluster = data.cluster_labels[idx];
    if (!clusterGroups[cluster]) {
      clusterGroups[cluster] = {
        x: [],
        y: [],
        ids: [],
        labels: [],
        topTopics: [],
        topicWords: [],
      };
    }
    clusterGroups[cluster].x.push(point[0]);
    clusterGroups[cluster].y.push(point[1]);
    clusterGroups[cluster].ids.push(data.document_ids[idx]);

    // Add enrichment data if available
    clusterGroups[cluster].labels.push(data.newsgroup_labels?.[idx]);
    clusterGroups[cluster].topTopics.push(data.top_topics?.[idx]);
    clusterGroups[cluster].topicWords.push(data.dominant_topic_words?.[idx]);
  });

  const traces = Object.entries(clusterGroups).map(([cluster, points]) => {
    // Build rich hover text with all available information
    const hoverText = points.ids.map((id, i) => {
      let text = `<b>Document ${id}</b>`;

      // Original newsgroup label
      if (points.labels[i]) {
        text += `<br><b>Category:</b> ${points.labels[i]}`;
      }

      // Top 3 topics with probabilities
      if (points.topTopics[i] && points.topTopics[i].length > 0) {
        const topicsStr = points.topTopics[i]
          .map((t) => `Topic ${t.topic_id + 1} (${(t.probability * 100).toFixed(1)}%)`)
          .join(", ");
        text += `<br><b>Top Topics:</b> ${topicsStr}`;
      }

      // Top 5 words from dominant topic
      if (points.topicWords[i] && points.topicWords[i].length > 0) {
        text += `<br><b>Topic Words:</b> ${points.topicWords[i].join(", ")}`;
      }

      return text;
    });

    const clusterColor = CLUSTER_COLORS[parseInt(cluster) % CLUSTER_COLORS.length];

    return {
      x: points.x,
      y: points.y,
      type: "scatter" as const,
      mode: "markers" as const,
      marker: {
        color: clusterColor,
        size: 5,
        opacity: 0.7,
      },
      name: `Cluster ${parseInt(cluster) + 1}`,
      text: hoverText,
      hoverinfo: "text" as const,
      hoverlabel: {
        bgcolor: colors.bg,
        bordercolor: clusterColor,
        font: { size: 11, family: "'JetBrains Mono', monospace", color: colors.text },
      },
    };
  });

  // Show subtle indicator during background revalidation
  const showValidatingIndicator = isValidating && data;

  return (
    <div className={`terminal-panel relative transition-opacity ${showValidatingIndicator ? 'opacity-80' : ''}`}>
      {showValidatingIndicator && (
        <div className="absolute top-3 right-3 z-10 flex items-center gap-2 font-mono text-xs" style={{ color: 'var(--text-muted)' }}>
          <div className="status-dot status-dot--active" style={{ width: 6, height: 6 }} />
          <span>Updating...</span>
        </div>
      )}
      <div className="terminal-panel-header">
        Document Clusters
      </div>
      <div className="terminal-panel-content">
        <Plot
          key={isDark ? 'dark' : 'light'}
          data={traces}
          layout={{
            autosize: true,
            height: 500,
            margin: { l: 50, r: 30, t: 10, b: 50 },
            paper_bgcolor: colors.bg,
            plot_bgcolor: colors.bg,
            font: {
              family: "'JetBrains Mono', 'SF Mono', monospace",
              color: colors.text,
              size: 11,
            },
            xaxis: {
              title: { text: "UMAP Dimension 1", font: { size: 11 } },
              zeroline: false,
              gridcolor: colors.grid,
              linecolor: colors.grid,
              tickfont: { size: 10 },
            },
            yaxis: {
              title: { text: "UMAP Dimension 2", font: { size: 11 } },
              zeroline: false,
              gridcolor: colors.grid,
              linecolor: colors.grid,
              tickfont: { size: 10 },
            },
            showlegend: true,
            legend: {
              x: 1,
              y: 1,
              xanchor: "right",
              font: { size: 10 },
              bgcolor: 'rgba(0,0,0,0)',
            },
            hovermode: "closest",
          }}
          config={{
            displayModeBar: true,
            modeBarButtonsToRemove: ["lasso2d", "select2d"],
          }}
          style={{ width: "100%" }}
        />
      </div>
    </div>
  );
}
