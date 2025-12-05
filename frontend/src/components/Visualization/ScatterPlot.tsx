import Plot from "react-plotly.js";
import type { Data as PlotlyData, Shape as PlotlyShape } from "plotly.js";
import type {
  ClusteredVisualizationResponse,
  GMMClusteredVisualizationResponse,
  DocumentTopicInfo,
  ClusterProbability,
} from "../../types/api";
import { useTheme } from "../../hooks/useTheme";

type VisualizationData = ClusteredVisualizationResponse | GMMClusteredVisualizationResponse;

interface ScatterPlotProps {
  data: VisualizationData | null;
  loading: boolean;
  isValidating?: boolean;
  algorithm?: "kmeans" | "gmm";
}

// Type guard to check if data is GMM response
function isGMMData(data: VisualizationData): data is GMMClusteredVisualizationResponse {
  return "cluster_probabilities" in data;
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

/**
 * Generate ellipse path from covariance matrix.
 * Returns SVG path string for Plotly shape.
 */
function generateEllipsePath(
  cx: number,
  cy: number,
  cov: number[][],
  scale: number = 2 // 2 = ~95% confidence interval
): string {
  // Eigenvalue decomposition for 2x2 covariance matrix
  const a = cov[0][0];
  const b = cov[0][1];
  const c = cov[1][0];
  const d = cov[1][1];

  // Calculate eigenvalues
  const trace = a + d;
  const det = a * d - b * c;
  const discriminant = Math.sqrt(Math.max(0, trace * trace / 4 - det));
  const lambda1 = trace / 2 + discriminant;
  const lambda2 = trace / 2 - discriminant;

  // Semi-axes lengths (scaled by sqrt of eigenvalues)
  const rx = Math.sqrt(Math.max(0.01, lambda1)) * scale;
  const ry = Math.sqrt(Math.max(0.01, lambda2)) * scale;

  // Rotation angle from eigenvector
  let angle = 0;
  if (Math.abs(b) > 1e-10) {
    angle = Math.atan2(lambda1 - a, b);
  } else if (Math.abs(a - lambda1) < 1e-10) {
    angle = 0;
  } else {
    angle = Math.PI / 2;
  }

  // Generate ellipse points
  const points: [number, number][] = [];
  const numPoints = 50;
  for (let i = 0; i <= numPoints; i++) {
    const t = (2 * Math.PI * i) / numPoints;
    const x = rx * Math.cos(t);
    const y = ry * Math.sin(t);
    // Rotate and translate
    const xRot = x * Math.cos(angle) - y * Math.sin(angle) + cx;
    const yRot = x * Math.sin(angle) + y * Math.cos(angle) + cy;
    points.push([xRot, yRot]);
  }

  // Convert to SVG path
  const pathParts = points.map((p, i) => (i === 0 ? `M ${p[0]} ${p[1]}` : `L ${p[0]} ${p[1]}`));
  return pathParts.join(" ") + " Z";
}

export function ScatterPlot({ data, loading, isValidating, algorithm: _algorithm = "kmeans" }: ScatterPlotProps) {
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

  // Check if we have GMM data with cluster probabilities
  const hasGMMProbabilities = isGMMData(data);
  const clusterProbabilities = hasGMMProbabilities ? data.cluster_probabilities : null;

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
      clusterProbs: (ClusterProbability[] | undefined)[];
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
        clusterProbs: [],
      };
    }
    clusterGroups[cluster].x.push(point[0]);
    clusterGroups[cluster].y.push(point[1]);
    clusterGroups[cluster].ids.push(data.document_ids[idx]);

    // Add enrichment data if available
    clusterGroups[cluster].labels.push(data.newsgroup_labels?.[idx]);
    clusterGroups[cluster].topTopics.push(data.top_topics?.[idx]);
    clusterGroups[cluster].topicWords.push(data.dominant_topic_words?.[idx]);
    clusterGroups[cluster].clusterProbs.push(clusterProbabilities?.[idx]);
  });

  const traces: PlotlyData[] = Object.entries(clusterGroups).map(([cluster, points]) => {
    // Build rich hover text with all available information
    const hoverText = points.ids.map((id, i) => {
      let text = `<b>Document ${id}</b>`;

      // Original newsgroup label
      if (points.labels[i]) {
        text += `<br><b>Category:</b> ${points.labels[i]}`;
      }

      // GMM cluster probabilities (soft assignments)
      if (points.clusterProbs[i] && points.clusterProbs[i].length > 0) {
        const probsStr = points.clusterProbs[i]
          .map((p) => `Cluster ${p.cluster_id + 1}: ${(p.probability * 100).toFixed(1)}%`)
          .join("<br>  ");
        text += `<br><b>Cluster Probs:</b><br>  ${probsStr}`;
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

  // Get cluster geometry - works for both K-Means and GMM
  const centers = hasGMMProbabilities
    ? data.cluster_means
    : (data as ClusteredVisualizationResponse).cluster_centers;
  const covariances = hasGMMProbabilities
    ? data.cluster_covariances
    : (data as ClusteredVisualizationResponse).cluster_covariances;

  // Add cluster center markers (more distinguishable)
  if (centers && centers.length > 0) {
    traces.push({
      x: centers.map((c) => c[0]),
      y: centers.map((c) => c[1]),
      type: "scatter" as const,
      mode: "markers" as const,
      marker: {
        color: isDark ? "#ffffff" : "#000000",
        size: 16,
        symbol: "diamond",
        line: {
          color: centers.map((_, i) => CLUSTER_COLORS[i % CLUSTER_COLORS.length]),
          width: 3
        },
      },
      name: "Cluster Centers",
      hoverinfo: "text" as const,
      text: centers.map((_, i) => `<b>Cluster ${i + 1} Center</b>`),
      hoverlabel: {
        bgcolor: colors.bg,
        font: { size: 11, family: "'JetBrains Mono', monospace", color: colors.text },
      },
      showlegend: true,
    });
  }

  // Generate ellipse shapes for cluster boundaries (both K-Means and GMM)
  const shapes: Partial<PlotlyShape>[] = [];
  if (centers && covariances) {
    centers.forEach((mean, i) => {
      const cov = covariances[i];
      const color = CLUSTER_COLORS[i % CLUSTER_COLORS.length];

      // 2σ ellipse (outer, lighter)
      shapes.push({
        type: "path",
        path: generateEllipsePath(mean[0], mean[1], cov, 2),
        fillcolor: color + "15", // 15 = ~8% opacity in hex
        line: { color: color, width: 1, dash: "dot" },
      });

      // 1σ ellipse (inner, slightly more visible)
      shapes.push({
        type: "path",
        path: generateEllipsePath(mean[0], mean[1], cov, 1),
        fillcolor: color + "25", // 25 = ~15% opacity in hex
        line: { color: color, width: 1.5 },
      });
    });
  }

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
            shapes: shapes,
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
