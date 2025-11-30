import Plot from "react-plotly.js";
import type { ClusterMetricsResponse } from "../../types/api";

interface ClusterMetricsChartProps {
  data: ClusterMetricsResponse | null;
  loading: boolean;
  selectedClusters: number;
  onSelectClusters?: (n: number) => void;
}

const CHART_HEIGHT = 350;

export function ClusterMetricsChart({
  data,
  loading,
  selectedClusters,
  onSelectClusters,
}: ClusterMetricsChartProps) {
  if (loading) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-6 h-[480px] flex flex-col">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">
          Optimal Number of Clusters
        </h3>
        <div className="flex-1 flex flex-col items-center justify-center">
          <div className="relative">
            <div className="animate-spin rounded-full h-10 w-10 border-4 border-purple-200"></div>
            <div className="animate-spin rounded-full h-10 w-10 border-4 border-purple-600 border-t-transparent absolute top-0 left-0"></div>
          </div>
          <span className="text-gray-500 text-sm mt-4">Computing cluster metrics...</span>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-6 h-[480px] flex flex-col">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">
          Optimal Number of Clusters
        </h3>
        <div className="flex-1 flex flex-col items-center justify-center">
          <svg className="w-12 h-12 text-red-400 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
          <span className="text-red-500 text-sm">Failed to load cluster metrics</span>
        </div>
      </div>
    );
  }

  const selectedIndex = data.cluster_counts.indexOf(selectedClusters);

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 h-[480px] flex flex-col">
      <h3 className="text-lg font-semibold text-gray-800 mb-4">
        Optimal Number of Clusters
        <span className="text-sm font-normal text-gray-500 ml-2">({data.n_topics} topics)</span>
      </h3>
      <div className="flex-1 min-h-0">
        <Plot
          data={[
            // Silhouette score (primary y-axis)
            {
              x: data.cluster_counts,
              y: data.silhouette_scores,
              type: "scatter",
              mode: "lines+markers",
              marker: { color: "#8b5cf6", size: 8 },
              line: { color: "#8b5cf6", width: 2 },
              name: "Silhouette",
              yaxis: "y",
            },
            // Inertia (secondary y-axis)
            {
              x: data.cluster_counts,
              y: data.inertia_scores,
              type: "scatter",
              mode: "lines+markers",
              marker: { color: "#64748b", size: 6 },
              line: { color: "#64748b", width: 1, dash: "dash" },
              name: "Inertia",
              yaxis: "y2",
            },
            // Highlight elbow point on the Inertia line (that's where elbow method applies)
            data.elbow_point
              ? {
                  x: [data.elbow_point],
                  y: [data.inertia_scores[data.cluster_counts.indexOf(data.elbow_point)]],
                  type: "scatter",
                  mode: "markers",
                  marker: { color: "#22c55e", size: 14, symbol: "star" },
                  name: `Elbow (k=${data.elbow_point})`,
                  yaxis: "y2",
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
                    color: "#f97316",
                    size: 12,
                    line: { color: "#fff", width: 2 },
                  },
                  name: `Selected (k=${selectedClusters})`,
                }
              : {},
          ]}
          layout={{
            autosize: true,
            height: CHART_HEIGHT,
            margin: { l: 60, r: 60, t: 30, b: 70 },
            xaxis: {
              title: { text: "Number of Clusters" },
              dtick: 1,
            },
            yaxis: {
              title: { text: "Silhouette Score" },
              side: "left",
            },
            yaxis2: {
              title: { text: "Inertia" },
              side: "right",
              overlaying: "y",
            },
            showlegend: true,
            legend: {
              x: 0,
              y: 1.15,
              orientation: "h",
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
