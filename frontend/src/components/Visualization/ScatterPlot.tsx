import Plot from "react-plotly.js";
import type { ClusteredVisualizationResponse } from "../../types/api";

interface ScatterPlotProps {
  data: ClusteredVisualizationResponse | null;
  loading: boolean;
}

// Color palette for clusters
const CLUSTER_COLORS = [
  "#3b82f6", // blue
  "#ef4444", // red
  "#22c55e", // green
  "#f59e0b", // amber
  "#8b5cf6", // violet
  "#ec4899", // pink
  "#06b6d4", // cyan
  "#84cc16", // lime
  "#f97316", // orange
  "#6366f1", // indigo
  "#14b8a6", // teal
  "#a855f7", // purple
  "#eab308", // yellow
  "#64748b", // slate
  "#78716c", // stone
];

export function ScatterPlot({ data, loading }: ScatterPlotProps) {
  if (loading) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-6 h-[550px] flex flex-col">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">
          Document Clusters
        </h3>
        <div className="flex-1 flex flex-col items-center justify-center">
          <div className="relative">
            <div className="animate-spin rounded-full h-12 w-12 border-4 border-pink-200"></div>
            <div className="animate-spin rounded-full h-12 w-12 border-4 border-pink-600 border-t-transparent absolute top-0 left-0"></div>
          </div>
          <span className="text-gray-500 text-sm mt-4">Rendering visualization...</span>
          <span className="text-gray-400 text-xs mt-1">Processing ~18,000 documents</span>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-6 h-[550px] flex flex-col">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">
          Document Clusters
        </h3>
        <div className="flex-1 flex flex-col items-center justify-center">
          <svg className="w-12 h-12 text-red-400 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
          <span className="text-red-500 text-sm">Failed to load visualization</span>
        </div>
      </div>
    );
  }

  // Group points by cluster
  const clusterGroups: Record<
    number,
    { x: number[]; y: number[]; ids: number[] }
  > = {};

  data.projections.forEach((point, idx) => {
    const cluster = data.cluster_labels[idx];
    if (!clusterGroups[cluster]) {
      clusterGroups[cluster] = { x: [], y: [], ids: [] };
    }
    clusterGroups[cluster].x.push(point[0]);
    clusterGroups[cluster].y.push(point[1]);
    clusterGroups[cluster].ids.push(data.document_ids[idx]);
  });

  const traces = Object.entries(clusterGroups).map(([cluster, points]) => ({
    x: points.x,
    y: points.y,
    type: "scatter" as const,
    mode: "markers" as const,
    marker: {
      color: CLUSTER_COLORS[parseInt(cluster) % CLUSTER_COLORS.length],
      size: 5,
      opacity: 0.7,
    },
    name: `Cluster ${parseInt(cluster) + 1}`,
    text: points.ids.map((id) => `Document ${id}`),
    hoverinfo: "text" as const,
  }));

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <h3 className="text-lg font-semibold text-gray-800 mb-4">
        Document Clusters
        <span className="text-sm font-normal text-gray-500 ml-2">
          ({data.n_topics} topics, {data.n_clusters} clusters)
        </span>
      </h3>
      <Plot
        data={traces}
        layout={{
          autosize: true,
          height: 500,
          margin: { l: 50, r: 30, t: 10, b: 50 },
          xaxis: {
            title: { text: "UMAP Dimension 1" },
            zeroline: false,
          },
          yaxis: {
            title: { text: "UMAP Dimension 2" },
            zeroline: false,
          },
          showlegend: true,
          legend: {
            x: 1,
            y: 1,
            xanchor: "right",
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
  );
}
