import { useState } from "react";
import Plot from "react-plotly.js";
import type { EDAResponse, StageStats } from "../../types/api";

interface DatasetOverviewProps {
  data: EDAResponse | null;
  loading: boolean;
  isValidating?: boolean;
}

type Stage = "raw" | "tokenized" | "filtered" | "final";
type Dataset = "train" | "test";

const STAGE_LABELS: Record<Stage, string> = {
  raw: "Raw",
  tokenized: "Tokenized",
  filtered: "Filtered",
  final: "Final",
};

const STAGE_COLORS: Record<Stage, string> = {
  raw: "#94a3b8",
  tokenized: "#60a5fa",
  filtered: "#34d399",
  final: "#f472b6",
};

export function DatasetOverview({
  data,
  loading,
  isValidating,
}: DatasetOverviewProps) {
  const [selectedStage, setSelectedStage] = useState<Stage>("final");
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

      {/* Summary Cards Row - Dynamic based on selected stage */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        {/* Train Documents */}
        <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg p-4">
          <div className="text-xs font-medium text-blue-600 uppercase tracking-wide">Train Docs</div>
          <div className="text-2xl font-bold text-blue-900 mt-1">
            {formatNum(getStats(selectedStage, "train").n_documents)}
          </div>
          {selectedStage === "final" && data.train_docs_removed > 0 ? (
            <div className="text-xs text-red-600 mt-0.5">
              -{formatNum(data.train_docs_removed)} removed
            </div>
          ) : (
            <div className="text-xs text-blue-600 mt-0.5">
              {getStats(selectedStage, "train").empty_count} empty
            </div>
          )}
        </div>

        {/* Test Documents */}
        <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-lg p-4">
          <div className="text-xs font-medium text-purple-600 uppercase tracking-wide">Test Docs</div>
          <div className="text-2xl font-bold text-purple-900 mt-1">
            {formatNum(getStats(selectedStage, "test").n_documents)}
          </div>
          {selectedStage === "final" && data.test_docs_removed > 0 ? (
            <div className="text-xs text-red-600 mt-0.5">
              -{formatNum(data.test_docs_removed)} removed
            </div>
          ) : (
            <div className="text-xs text-purple-600 mt-0.5">
              {getStats(selectedStage, "test").empty_count} empty
            </div>
          )}
        </div>

        {/* Vocabulary - shows before/after based on stage */}
        <div className="bg-gradient-to-br from-amber-50 to-amber-100 rounded-lg p-4">
          <div className="text-xs font-medium text-amber-600 uppercase tracking-wide">Vocabulary</div>
          {selectedStage === "raw" || selectedStage === "tokenized" ? (
            <>
              <div className="text-2xl font-bold text-amber-900 mt-1">
                {formatNum(data.vocab_before_filter)}
              </div>
              <div className="text-xs text-amber-600 mt-0.5">
                before filtering
              </div>
            </>
          ) : (
            <>
              <div className="text-2xl font-bold text-amber-900 mt-1">
                {formatNum(data.vocab_after_filter)}
              </div>
              <div className="text-xs text-amber-600 mt-0.5">
                -{formatPct(data.vocab_reduction_pct)} from {formatNum(data.vocab_before_filter)}
              </div>
            </>
          )}
        </div>

        {/* Empty Docs / Min Tokens */}
        <div className="bg-gradient-to-br from-emerald-50 to-emerald-100 rounded-lg p-4">
          {selectedStage === "final" ? (
            <>
              <div className="text-xs font-medium text-emerald-600 uppercase tracking-wide">Min Tokens</div>
              <div className="text-2xl font-bold text-emerald-900 mt-1">
                {data.min_tokens_threshold}
              </div>
              <div className="text-xs text-emerald-600 mt-0.5">
                filter threshold
              </div>
            </>
          ) : (
            <>
              <div className="text-xs font-medium text-emerald-600 uppercase tracking-wide">Empty Docs</div>
              <div className="text-2xl font-bold text-emerald-900 mt-1">
                {getStats(selectedStage, "train").empty_count + getStats(selectedStage, "test").empty_count}
              </div>
              <div className="text-xs text-emerald-600 mt-0.5">
                {formatPct(
                  ((getStats(selectedStage, "train").empty_count + getStats(selectedStage, "test").empty_count) /
                    (getStats(selectedStage, "train").n_documents + getStats(selectedStage, "test").n_documents)) *
                    100
                )} of total
              </div>
            </>
          )}
        </div>
      </div>

      {/* Toggle Controls */}
      <div className="flex flex-wrap gap-4 mb-4">
        {/* Stage Toggle */}
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-500">Stage:</span>
          <div className="flex rounded-lg overflow-hidden border border-gray-200">
            {(["raw", "tokenized", "filtered", "final"] as Stage[]).map((stage) => (
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

      {/* Stage Description */}
      <div className="mb-4 p-3 rounded-lg bg-gray-50 border border-gray-100">
        <p className="text-xs text-gray-600">
          {selectedStage === "raw" && (
            <>
              <span className="font-semibold">Raw:</span> Original documents with whitespace-split token counts. All {formatNum(data.raw_train.n_documents + data.raw_test.n_documents)} documents included.
            </>
          )}
          {selectedStage === "tokenized" && (
            <>
              <span className="font-semibold">Tokenized:</span> After preprocessing (tokenization, lemmatization, stopword removal). Same document count, but token distribution changes significantly.
            </>
          )}
          {selectedStage === "filtered" && (
            <>
              <span className="font-semibold">Filtered:</span> After dictionary filter_extremes (no_below={data.filter_no_below}, no_above={data.filter_no_above}). Vocabulary reduced by {formatPct(data.vocab_reduction_pct)}, some docs may have 0 tokens.
            </>
          )}
          {selectedStage === "final" && (
            <>
              <span className="font-semibold">Final:</span> After removing documents with {"<"}{data.min_tokens_threshold} tokens. {formatNum(data.train_docs_removed + data.test_docs_removed)} documents removed ({formatNum(data.train_docs_removed)} train, {formatNum(data.test_docs_removed)} test).
            </>
          )}
        </p>
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
                hovertemplate: "%{x:.0f} tokens: %{y} docs<extra></extra>",
              },
            ]}
            layout={{
              autosize: true,
              height: 250,
              margin: { l: 50, r: 20, t: 30, b: 50 },
              xaxis: {
                title: {
                  text: "Document Length (tokens)",
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
                  <th className="text-right pb-2">Raw</th>
                  <th className="text-right pb-2">Tok</th>
                  <th className="text-right pb-2">Filt</th>
                  <th className="text-right pb-2">Final</th>
                </tr>
              </thead>
              <tbody className="text-gray-700">
                {/* Document count row */}
                <tr className="border-t border-gray-200 bg-gray-50">
                  <td className="py-1.5 font-semibold">Docs</td>
                  <td className="py-1.5 text-right font-mono text-xs font-semibold">
                    {formatNum(getStats("raw", selectedDataset).n_documents)}
                  </td>
                  <td className="py-1.5 text-right font-mono text-xs font-semibold">
                    {formatNum(getStats("tokenized", selectedDataset).n_documents)}
                  </td>
                  <td className="py-1.5 text-right font-mono text-xs font-semibold">
                    {formatNum(getStats("filtered", selectedDataset).n_documents)}
                  </td>
                  <td className="py-1.5 text-right font-mono text-xs font-semibold text-pink-600">
                    {formatNum(getStats("final", selectedDataset).n_documents)}
                  </td>
                </tr>
                {[
                  { label: "Mean", key: "mean" },
                  { label: "Median", key: "median" },
                  { label: "Min", key: "min" },
                  { label: "Max", key: "max" },
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
                    <td className="py-1.5 text-right font-mono text-xs">
                      {formatStatValue(getStats("final", selectedDataset)[key as keyof StageStats] as number, key)}
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
  if (key === "mean" || key === "median") {
    return value.toFixed(1);
  }
  return value.toLocaleString();
}
