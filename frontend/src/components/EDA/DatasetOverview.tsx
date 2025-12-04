import { useState } from "react";
import Plot from "react-plotly.js";
import type { EDAResponse, StageStats } from "../../types/api";
import { useTheme } from "../../hooks/useTheme";

interface DatasetOverviewProps {
  data: EDAResponse | null;
  loading: boolean;
  isValidating?: boolean;
}

type Stage = "raw" | "tokenized" | "filtered" | "final";

const STAGE_LABELS: Record<Stage, string> = {
  raw: "Raw",
  tokenized: "Preprocess",
  filtered: "Filtered",
  final: "Final",
};

const STAGE_COLORS: Record<Stage, string> = {
  raw: "#6e7681",
  tokenized: "#58a6ff",
  filtered: "#00d4aa",
  final: "#ff6b35",
};

export function DatasetOverview({
  data,
  loading,
  isValidating,
}: DatasetOverviewProps) {
  const [selectedStage, setSelectedStage] = useState<Stage>("final");
  const { isDark } = useTheme();

  // Theme-aware colors
  const colors = {
    bg: isDark ? '#21262d' : '#ffffff',
    text: isDark ? '#e6edf3' : '#1a1a1a',
    textMuted: isDark ? '#8b949e' : '#6b6b6b',
    grid: isDark ? '#30363d' : '#e5e5e0',
  };

  // Loading skeleton
  if (loading && !data) {
    return (
      <div className="terminal-panel">
        <div className="terminal-panel-header">
          Dataset Overview
        </div>
        <div className="terminal-panel-content flex flex-col items-center justify-center py-12">
          <div className="terminal-loading">Loading dataset statistics</div>
          <div className="loading-bar w-48 mt-4">
            <div className="loading-bar-progress"></div>
          </div>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="terminal-panel">
        <div className="terminal-panel-header">
          <span className="status-dot status-dot--warning"></span>
          Dataset Overview
        </div>
        <div className="terminal-panel-content flex flex-col items-center justify-center py-12">
          <svg className="w-12 h-12 mb-3" style={{ color: 'var(--status-warning)' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
          <span className="font-mono text-sm" style={{ color: 'var(--text-secondary)' }}>EDA statistics not available</span>
        </div>
      </div>
    );
  }

  // Get stats for selected stage
  const getStats = (stage: Stage): StageStats => {
    return data[stage] as StageStats;
  };

  const currentStats = getStats(selectedStage);
  const showValidatingIndicator = isValidating && data;

  // Format numbers with commas
  const formatNum = (n: number) => n.toLocaleString();
  const formatPct = (n: number) => `${n.toFixed(1)}%`;

  return (
    <div className={`terminal-panel relative transition-opacity ${showValidatingIndicator ? 'opacity-80' : ''}`}>
      {showValidatingIndicator && (
        <div className="absolute top-3 right-3 z-10 flex items-center gap-2 font-mono text-xs" style={{ color: 'var(--text-muted)' }}>
          <div className="status-dot status-dot--active" style={{ width: 6, height: 6 }} />
          <span>Updating...</span>
        </div>
      )}

      <div className="terminal-panel-header">
        Dataset Overview
      </div>

      <div className="terminal-panel-content">
        {/* Summary Cards Row */}
        <div className="grid grid-cols-3 gap-4 mb-6">
          {/* Documents */}
          <div
            className="rounded p-4"
            style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-color)' }}
          >
            <div className="data-label" style={{ color: 'var(--accent-tertiary)' }}>Documents</div>
            <div className="data-value text-2xl mt-1" style={{ color: 'var(--text-primary)' }}>
              {formatNum(getStats(selectedStage).n_documents)}
            </div>
            <div className="font-mono text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>
              {selectedStage === "raw" && `${getStats("raw").empty_count} empty`}
              {selectedStage === "tokenized" && `${getStats("tokenized").empty_count - getStats("raw").empty_count} empty`}
              {selectedStage === "filtered" && `${getStats("filtered").empty_count - getStats("tokenized").empty_count} empty`}
              {selectedStage === "final" && `${getStats("final").empty_count} empty`}
            </div>
          </div>

          {/* Vocabulary */}
          <div
            className="rounded p-4"
            style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-color)' }}
          >
            <div className="data-label" style={{ color: 'var(--status-warning)' }}>Vocabulary</div>
            {selectedStage === "raw" || selectedStage === "tokenized" ? (
              <>
                <div className="data-value text-2xl mt-1" style={{ color: 'var(--text-primary)' }}>
                  {formatNum(data.vocab_before_filter)}
                </div>
                <div className="font-mono text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>
                  before filtering
                </div>
              </>
            ) : (
              <>
                <div className="data-value text-2xl mt-1" style={{ color: 'var(--text-primary)' }}>
                  {formatNum(data.vocab_after_filter)}
                </div>
                <div className="font-mono text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>
                  -{formatPct(data.vocab_reduction_pct)} from {formatNum(data.vocab_before_filter)}
                </div>
              </>
            )}
          </div>

          {/* Empty Docs */}
          <div
            className="rounded p-4"
            style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-color)' }}
          >
            <div className="data-label" style={{ color: 'var(--accent-secondary)' }}>Empty Docs</div>
            <div className="data-value text-2xl mt-1" style={{ color: 'var(--text-primary)' }}>
              {formatNum(getStats(selectedStage === "final" ? "filtered" : selectedStage).empty_count)}
            </div>
            <div className="font-mono text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>
              {formatPct(getStats(selectedStage === "final" ? "filtered" : selectedStage).empty_pct)} of total
            </div>
          </div>
        </div>

        {/* Stage Toggle */}
        <div className="flex flex-wrap gap-4 mb-4">
          <div className="flex items-center gap-2">
            <span className="data-label">Stage:</span>
            <div className="flex rounded overflow-hidden" style={{ border: '1px solid var(--border-color)' }}>
              {(["raw", "tokenized", "filtered", "final"] as Stage[]).map((stage) => (
                <button
                  key={stage}
                  onClick={() => setSelectedStage(stage)}
                  className="px-3 py-1.5 font-mono text-xs font-medium transition-colors"
                  style={{
                    background: selectedStage === stage ? 'var(--accent-primary)' : 'var(--bg-card)',
                    color: selectedStage === stage ? 'white' : 'var(--text-secondary)',
                  }}
                >
                  {STAGE_LABELS[stage]}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Stage Description */}
        <div
          className="mb-4 p-3 rounded font-mono text-xs"
          style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-color)', color: 'var(--text-secondary)' }}
        >
          {selectedStage === "raw" && (
            <>
              <span style={{ color: 'var(--accent-primary)' }}>RAW:</span> Original documents with whitespace-split token counts. All {formatNum(data.raw.n_documents)} documents included.
            </>
          )}
          {selectedStage === "tokenized" && (
            <>
              <span style={{ color: 'var(--accent-primary)' }}>PREPROCESS:</span> After tokenization, lemmatization, and stopword removal. Same document count, but token distribution changes significantly.
            </>
          )}
          {selectedStage === "filtered" && (
            <>
              <span style={{ color: 'var(--accent-primary)' }}>FILTERED:</span> After dictionary filter_extremes (no_below={data.filter_no_below}, no_above={data.filter_no_above}). Vocabulary reduced by {formatPct(data.vocab_reduction_pct)}.
            </>
          )}
          {selectedStage === "final" && (
            <>
              <span style={{ color: 'var(--accent-primary)' }}>FINAL:</span> After removing documents with {"<"}{data.min_tokens_threshold} tokens. {formatNum(data.docs_removed)} documents removed.
            </>
          )}
        </div>

        {/* Content Grid: Histogram + Stats Table */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Histogram */}
          <div>
            <Plot
              key={isDark ? 'dark' : 'light'}
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
                  customdata: currentStats.histogram_bins.slice(0, -1).map((b, i, arr) => {
                    const midpoint = Math.round((b + currentStats.histogram_bins[i + 1]) / 2);
                    return i === arr.length - 1 ? `≥${midpoint}` : `${midpoint}`;
                  }),
                  hovertemplate: "%{customdata} tokens: %{y} docs<extra></extra>",
                },
              ]}
              layout={{
                autosize: true,
                height: 340,
                margin: { l: 50, r: 20, t: 30, b: 50 },
                paper_bgcolor: colors.bg,
                plot_bgcolor: colors.bg,
                font: {
                  family: "'JetBrains Mono', monospace",
                  color: colors.text,
                  size: 10,
                },
                xaxis: {
                  title: { text: "Document Length (tokens)", font: { size: 10 } },
                  gridcolor: colors.grid,
                  linecolor: colors.grid,
                },
                yaxis: {
                  title: { text: "Number of Documents", font: { size: 10 } },
                  gridcolor: colors.grid,
                  linecolor: colors.grid,
                },
                bargap: 0.05,
              }}
              config={{ displayModeBar: false }}
              style={{ width: "100%", height: "100%" }}
            />
            <p className="font-mono text-xs text-right -mt-11" style={{ color: 'var(--text-muted)' }}>
              *capped at 95th percentile
            </p>
          </div>

          {/* Stats Table */}
          <div
            className="rounded p-4"
            style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-color)' }}
          >
            <h4 className="data-label mb-3">
              Statistics by Stage <span style={{ color: 'var(--text-muted)' }}>(tokens/doc)</span>
            </h4>
            <table className="terminal-table text-xs">
              <thead>
                <tr>
                  <th className="w-[20%]">Metric</th>
                  <th className="text-right w-[20%]">Raw</th>
                  <th className="text-right w-[20%]">Prep</th>
                  <th className="text-right w-[20%]">Filt</th>
                  <th className="text-right w-[20%]">Final</th>
                </tr>
              </thead>
              <tbody>
                {/* Document count row */}
                <tr>
                  <td className="font-semibold">Docs</td>
                  <td className="text-right">{formatNum(getStats("raw").n_documents)}</td>
                  <td className="text-right">{formatNum(getStats("tokenized").n_documents)}</td>
                  <td className="text-right">{formatNum(getStats("filtered").n_documents)}</td>
                  <td className="text-right" style={{ color: 'var(--accent-primary)' }}>
                    {formatNum(getStats("final").n_documents)}
                  </td>
                </tr>
                {[
                  { label: "Mean", key: "mean" },
                  { label: "Median", key: "median" },
                  { label: "Q1", key: "q1" },
                  { label: "Q3", key: "q3" },
                  { label: "Min", key: "min" },
                  { label: "Max", key: "max" },
                ].map(({ label, key }) => {
                  const getValue = (stage: Stage) => {
                    const stats = getStats(stage);
                    if (key === "q1" || key === "q3") {
                      return getBoxPlotStats(stats)[key as keyof ReturnType<typeof getBoxPlotStats>];
                    }
                    return stats[key as keyof StageStats] as number;
                  };
                  return (
                    <tr key={key}>
                      <td className="font-medium">{label}</td>
                      <td className="text-right">{formatStatValue(getValue("raw"), key)}</td>
                      <td className="text-right">{formatStatValue(getValue("tokenized"), key)}</td>
                      <td className="text-right">{formatStatValue(getValue("filtered"), key)}</td>
                      <td className="text-right">{formatStatValue(getValue("final"), key)}</td>
                    </tr>
                  );
                })}
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
  if (key === "mean" || key === "median" || key === "q1" || key === "q3" || key === "lower_fence" || key === "upper_fence") {
    return value.toFixed(1);
  }
  return value.toLocaleString();
}

// Compute Q1, Q3, and fences from percentiles
function getBoxPlotStats(stats: StageStats) {
  const q1 = stats.percentiles["25"];
  const q3 = stats.percentiles["75"];
  const iqr = q3 - q1;
  const lowerFence = Math.max(stats.min, q1 - 1.5 * iqr);
  const upperFence = Math.min(stats.max, q3 + 1.5 * iqr);
  return { q1, q3, lower_fence: lowerFence, upper_fence: upperFence };
}
