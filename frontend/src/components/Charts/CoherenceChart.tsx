import Plot from "react-plotly.js";
import type { CoherenceResponse } from "../../types/api";

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
  // Show full skeleton only on initial load (no cached data)
  if (loading && !data) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-6 h-[480px] flex flex-col">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">
          Optimal Number of Topics
        </h3>
        <div className="flex-1 flex flex-col items-center justify-center">
          <div className="relative">
            <div className="animate-spin rounded-full h-10 w-10 border-4 border-blue-200"></div>
            <div className="animate-spin rounded-full h-10 w-10 border-4 border-blue-600 border-t-transparent absolute top-0 left-0"></div>
          </div>
          <span className="text-gray-500 text-sm mt-4">Calculating coherence scores...</span>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-6 h-[480px] flex flex-col">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">
          Optimal Number of Topics
        </h3>
        <div className="flex-1 flex flex-col items-center justify-center">
          <svg className="w-12 h-12 text-red-400 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
          <span className="text-red-500 text-sm">Failed to load coherence data</span>
        </div>
      </div>
    );
  }

  const selectedIndex = data.topic_counts.indexOf(selectedTopics);

  // Check if validation and test data are available
  const hasValCoherence = data.coherence_val && data.coherence_val.length > 0;
  const hasTestCoherence = data.coherence_test && data.coherence_test.length > 0;

  // Show subtle indicator during background revalidation
  const showValidatingIndicator = isValidating && data;

  return (
    <div className={`bg-white rounded-xl shadow-lg p-6 h-[480px] flex flex-col relative transition-opacity ${showValidatingIndicator ? 'opacity-80' : ''}`}>
      {showValidatingIndicator && (
        <div className="absolute top-4 right-4 z-10 flex items-center gap-2 text-xs text-gray-400">
          <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" />
          <span>Updating...</span>
        </div>
      )}
      <h3 className="text-lg font-semibold text-gray-800 mb-4">
        Optimal Number of Topics
      </h3>
      <div className="flex-1 min-h-0">
        <Plot
          data={[
            // Coherence validation (dashed) - primary y-axis
            ...(hasValCoherence
              ? [
                  {
                    x: data.topic_counts,
                    y: data.coherence_val,
                    type: "scatter" as const,
                    mode: "lines+markers" as const,
                    marker: { color: "#93c5fd", size: 6 },
                    line: { color: "#93c5fd", width: 2, dash: "dash" as const },
                    name: "Coherence (Val)",
                    yaxis: "y",
                  },
                ]
              : []),
            // Coherence test (solid) - primary y-axis
            ...(hasTestCoherence
              ? [
                  {
                    x: data.topic_counts,
                    y: data.coherence_test,
                    type: "scatter" as const,
                    mode: "lines+markers" as const,
                    marker: { color: "#3b82f6", size: 8 },
                    line: { color: "#3b82f6", width: 2 },
                    name: "Coherence (Test)",
                    yaxis: "y",
                  },
                ]
              : []),
            // Highlight optimal point (on test coherence axis)
            ...(hasTestCoherence
              ? [
                  {
                    x: [data.optimal_topics],
                    y: [data.coherence_test[data.topic_counts.indexOf(data.optimal_topics)]],
                    type: "scatter" as const,
                    mode: "markers" as const,
                    marker: { color: "#22c55e", size: 14, symbol: "star" },
                    name: `Optimal (k=${data.optimal_topics})`,
                    yaxis: "y",
                  },
                ]
              : []),
            // Highlight selected point (on test coherence axis)
            ...(selectedIndex >= 0 && hasTestCoherence
              ? [
                  {
                    x: [selectedTopics],
                    y: [data.coherence_test[selectedIndex]],
                    type: "scatter" as const,
                    mode: "markers" as const,
                    marker: {
                      color: "#f97316",
                      size: 12,
                      line: { color: "#fff", width: 2 },
                    },
                    name: `Selected (k=${selectedTopics})`,
                    yaxis: "y",
                  },
                ]
              : []),
          ]}
          layout={{
            autosize: true,
            height: CHART_HEIGHT,
            margin: { l: 60, r: 60, t: 30, b: 70 },
            xaxis: {
              title: { text: "Number of Topics" },
              dtick: 1,
            },
            yaxis: {
              title: { text: "Coherence Score" },
              side: "left",
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
