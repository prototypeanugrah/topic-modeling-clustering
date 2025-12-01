import { useState } from "react";
import Plot from "react-plotly.js";
import type { EDAResponse, StageStats } from "../../types/api";

interface DatasetOverviewProps {
  data: EDAResponse | null;
  loading: boolean;
  isValidating?: boolean;
}

type Stage = "raw" | "tokenized" | "filtered";
type Dataset = "train" | "test";

const STAGE_LABELS: Record<Stage, string> = {
  raw: "Raw (chars)",
  tokenized: "Tokenized",
  filtered: "Filtered",
};

const STAGE_COLORS: Record<Stage, string> = {
  raw: "#94a3b8",
  tokenized: "#60a5fa",
  filtered: "#34d399",
};

export function DatasetOverview({
  data,
  loading,
  isValidating,
}: DatasetOverviewProps) {
  const [selectedStage, setSelectedStage] = useState<Stage>("filtered");
  const [selectedDataset, setSelectedDataset] = useState<Dataset>("train");

  // Loading skeleton
  if (loading && !data) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">
          Dataset Overview
        </h3>
        <div className="flex flex-col items-center justify-center py-12">
          <div className="relative">
            <div className="animate-spin rounded-full h-10 w-10 border-4 border-blue-200"></div>
            <div className="animate-spin rounded-full h-10 w-10 border-4 border-blue-600 border-t-transparent absolute top-0 left-0"></div>
          </div>
          <span className="text-gray-500 text-sm mt-4">Loading dataset statistics...</span>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">
          Dataset Overview
        </h3>
        <div className="flex flex-col items-center justify-center py-12">
          <svg className="w-12 h-12 text-amber-400 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
          <span className="text-gray-500 text-sm">EDA statistics not available</span>
        </div>
      </div>
    );
  }

  // Get stats for selected stage and dataset
  const getStats = (stage: Stage, dataset: Dataset): StageStats => {
    const key = `${stage}_${dataset}` as keyof EDAResponse;
    return data[key] as StageStats;
  };

  const currentStats = getStats(selectedStage, selectedDataset);
  const showValidatingIndicator = isValidating && data;

  // Format numbers with commas
  const formatNum = (n: number) => n.toLocaleString();
  const formatPct = (n: number) => `${n.toFixed(1)}%`;

  return (
    <div className={`bg-white rounded-xl shadow-lg p-6 relative transition-opacity ${showValidatingIndicator ? 'opacity-80' : ''}`}>
      {showValidatingIndicator && (
        <div className="absolute top-4 right-4 z-10 flex items-center gap-2 text-xs text-gray-400">
          <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" />
          <span>Updating...</span>
        </div>
      )}

      <h3 className="text-lg font-semibold text-gray-800 mb-4">
        Dataset Overview
      </h3>

      {/* Summary Cards Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        {/* Train Documents */}
        <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg p-4">
          <div className="text-xs font-medium text-blue-600 uppercase tracking-wide">Train Docs</div>
          <div className="text-2xl font-bold text-blue-900 mt-1">
            {formatNum(data.filtered_train.n_documents)}
          </div>
        </div>

        {/* Test Documents */}
        <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-lg p-4">
          <div className="text-xs font-medium text-purple-600 uppercase tracking-wide">Test Docs</div>
          <div className="text-2xl font-bold text-purple-900 mt-1">
            {formatNum(data.filtered_test.n_documents)}
          </div>
        </div>

        {/* Vocabulary */}
        <div className="bg-gradient-to-br from-amber-50 to-amber-100 rounded-lg p-4">
          <div className="text-xs font-medium text-amber-600 uppercase tracking-wide">Vocabulary</div>
          <div className="text-lg font-bold text-amber-900 mt-1">
            {formatNum(data.vocab_before_filter)} → {formatNum(data.vocab_after_filter)}
          </div>
          <div className="text-xs text-amber-600 mt-0.5">
            -{formatPct(data.vocab_reduction_pct)}
          </div>
        </div>

        {/* Token Reduction */}
        <div className="bg-gradient-to-br from-emerald-50 to-emerald-100 rounded-lg p-4">
          <div className="text-xs font-medium text-emerald-600 uppercase tracking-wide">Token Reduction</div>
          <div className="text-2xl font-bold text-emerald-900 mt-1">
            {formatPct(data.token_reduction_pct)}
          </div>
        </div>
      </div>

      {/* Toggle Controls */}
      <div className="flex flex-wrap gap-4 mb-4">
        {/* Stage Toggle */}
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-500">Stage:</span>
          <div className="flex rounded-lg overflow-hidden border border-gray-200">
            {(["raw", "tokenized", "filtered"] as Stage[]).map((stage) => (
              <button
                key={stage}
                onClick={() => setSelectedStage(stage)}
                className={`px-3 py-1.5 text-xs font-medium transition-colors ${
                  selectedStage === stage
                    ? "bg-gray-800 text-white"
                    : "bg-white text-gray-600 hover:bg-gray-50"
                }`}
              >
                {STAGE_LABELS[stage]}
              </button>
            ))}
          </div>
        </div>

        {/* Dataset Toggle */}
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-500">Dataset:</span>
          <div className="flex rounded-lg overflow-hidden border border-gray-200">
            {(["train", "test"] as Dataset[]).map((ds) => (
              <button
                key={ds}
                onClick={() => setSelectedDataset(ds)}
                className={`px-3 py-1.5 text-xs font-medium transition-colors ${
                  selectedDataset === ds
                    ? "bg-gray-800 text-white"
                    : "bg-white text-gray-600 hover:bg-gray-50"
                }`}
              >
                {ds.charAt(0).toUpperCase() + ds.slice(1)}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Content Grid: Histogram + Stats Table */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Histogram */}
        <div className="lg:col-span-2">
          <Plot
            data={[
              {
                x: currentStats.histogram_bins.slice(0, -1).map((b, i) =>
                  (b + currentStats.histogram_bins[i + 1]) / 2
                ),
                y: currentStats.histogram_counts,
                type: "bar" as const,
                marker: {
                  color: STAGE_COLORS[selectedStage],
                },
                hovertemplate:
                  selectedStage === "raw"
                    ? "%{x:.0f} chars: %{y} docs<extra></extra>"
                    : "%{x:.0f} tokens: %{y} docs<extra></extra>",
              },
            ]}
            layout={{
              autosize: true,
              height: 250,
              margin: { l: 50, r: 20, t: 30, b: 50 },
              xaxis: {
                title: {
                  text: selectedStage === "raw" ? "Document Length (characters)" : "Document Length (tokens)",
                },
              },
              yaxis: {
                title: { text: "Number of Documents" },
              },
              bargap: 0.05,
            }}
            config={{ displayModeBar: false }}
            style={{ width: "100%", height: "100%" }}
          />
        </div>

        {/* Stats Table */}
        <div className="lg:col-span-1">
          <div className="bg-gray-50 rounded-lg p-4">
            <h4 className="text-sm font-semibold text-gray-700 mb-3">
              Statistics Comparison
            </h4>
            <table className="w-full text-sm">
              <thead>
                <tr className="text-xs text-gray-500 uppercase">
                  <th className="text-left pb-2">Metric</th>
                  <th className="text-right pb-2">Raw (chars)</th>
                  <th className="text-right pb-2">Tok</th>
                  <th className="text-right pb-2">Filt</th>
                </tr>
              </thead>
              <tbody className="text-gray-700">
                {[
                  { label: "Mean", key: "avg_length" },
                  { label: "Median", key: "median_length" },
                  { label: "Min", key: "min_length" },
                  { label: "Max", key: "max_length" },
                  { label: "Empty", key: "empty_count" },
                ].map(({ label, key }) => (
                  <tr key={key} className="border-t border-gray-200">
                    <td className="py-1.5 font-medium">{label}</td>
                    <td className="py-1.5 text-right font-mono text-xs">
                      {formatStatValue(getStats("raw", selectedDataset)[key as keyof StageStats] as number, key)}
                    </td>
                    <td className="py-1.5 text-right font-mono text-xs">
                      {formatStatValue(getStats("tokenized", selectedDataset)[key as keyof StageStats] as number, key)}
                    </td>
                    <td className="py-1.5 text-right font-mono text-xs">
                      {formatStatValue(getStats("filtered", selectedDataset)[key as keyof StageStats] as number, key)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

function formatStatValue(value: number, key: string): string {
  if (key === "empty_count") {
    return value.toString();
  }
  if (key === "avg_length" || key === "median_length") {
    return value.toFixed(1);
  }
  return value.toLocaleString();
}
