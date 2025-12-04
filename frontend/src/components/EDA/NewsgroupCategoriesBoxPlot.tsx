import Plot from "react-plotly.js";
import type { BoxPlotData } from "../../types/api";
import { useTheme } from "../../hooks/useTheme";

interface NewsgroupCategoriesBoxPlotProps {
  data: BoxPlotData | null;
  loading: boolean;
  isValidating?: boolean;
}

// Abbreviate long newsgroup names while keeping them unique
function abbreviateCategory(name: string): string {
  const parts = name.split(".");
  if (parts.length <= 2) return name;
  if (parts.length === 3) {
    // e.g., comp.windows.x -> comp.win.x, rec.sport.baseball -> rec.sport.bb
    const lastPart = parts[2].length > 4 ? parts[2].substring(0, 4) : parts[2];
    return `${parts[0]}.${parts[1].substring(0, 3)}.${lastPart}`;
  }
  // 4+ parts: comp.sys.ibm.pc.hardware -> comp.sys.ibm
  return `${parts[0]}.${parts[1]}.${parts[2]}`;
}

// Industrial color palette for categories
function getCategoryColor(index: number, total: number): string {
  const hue = (index * 360) / total;
  return `hsl(${hue}, 50%, 55%)`;
}

export function NewsgroupCategoriesBoxPlot({
  data,
  loading,
  isValidating,
}: NewsgroupCategoriesBoxPlotProps) {
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
          Token Distribution by Category
        </div>
        <div className="terminal-panel-content flex flex-col items-center justify-center py-12">
          <div className="terminal-loading">Loading box plot data</div>
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
          Token Distribution by Category
        </div>
        <div className="terminal-panel-content flex flex-col items-center justify-center py-12">
          <svg className="w-12 h-12 mb-3" style={{ color: 'var(--status-warning)' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
          <span className="font-mono text-sm" style={{ color: 'var(--text-secondary)' }}>Box plot data not available</span>
        </div>
      </div>
    );
  }

  const showValidatingIndicator = isValidating && data;

  // Get sorted category names
  const categories = Object.keys(data.category_token_counts).sort();
  const numCategories = categories.length;

  // Build traces - one box per category
  const traces = categories.map((category, idx) => ({
    y: data.category_token_counts[category],
    type: "box" as const,
    name: abbreviateCategory(category),
    marker: { color: getCategoryColor(idx, numCategories) },
    boxpoints: false as const,
    hoverinfo: "y+name" as const,
    hoverlabel: {
      namelength: -1,
      bgcolor: colors.bg,
      font: { family: "'JetBrains Mono', monospace", color: colors.text },
    },
  }));

  return (
    <div className={`terminal-panel relative transition-opacity ${showValidatingIndicator ? 'opacity-80' : ''}`}>
      {showValidatingIndicator && (
        <div className="absolute top-3 right-3 z-10 flex items-center gap-2 font-mono text-xs" style={{ color: 'var(--text-muted)' }}>
          <div className="status-dot status-dot--active" style={{ width: 6, height: 6 }} />
          <span>Updating...</span>
        </div>
      )}

      <div className="terminal-panel-header">
        Token Distribution by Category
        <span className="ml-2 font-mono text-xs" style={{ color: 'var(--text-muted)' }}>
          {numCategories} categories
        </span>
      </div>

      <div className="terminal-panel-content">
        <Plot
          key={isDark ? 'dark' : 'light'}
          data={traces}
          layout={{
            autosize: true,
            height: 350,
            margin: { l: 55, r: 15, t: 15, b: 110 },
            paper_bgcolor: colors.bg,
            plot_bgcolor: colors.bg,
            font: {
              family: "'JetBrains Mono', monospace",
              color: colors.text,
              size: 10,
            },
            yaxis: {
              title: { text: "Tokens per Document", font: { size: 10 } },
              zeroline: false,
              gridcolor: colors.grid,
              linecolor: colors.grid,
            },
            xaxis: {
              tickangle: -45,
              tickfont: { size: 7 },
              gridcolor: colors.grid,
              linecolor: colors.grid,
              automargin: true,
            },
            showlegend: false,
          }}
          useResizeHandler={true}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: "100%", height: "100%" }}
        />
      </div>
    </div>
  );
}
